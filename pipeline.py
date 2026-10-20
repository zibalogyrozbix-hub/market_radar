"""AI Market Radar - end-to-end orchestration.

data -> validation -> features -> engines -> ML -> ensemble -> calibration -> score -> DB
     -> event ranking -> post -> image -> Telegram
Every stage is isolated: a failing symbol/model/publisher is logged and skipped, never fatal.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from config.settings import display_name, get_secrets, resolve_path
from data.loader import MarketDataLoader
from engines import (anomaly_engine, cross_market_engine, momentum_engine, regime_engine, similarity_engine,
                     trend_engine, volatility_engine)
from features.builder import FEATURE_COLUMNS, build_features, make_targets
from models import ensemble as ens
from publishing import priority as prio
from publishing.image_generator import render_card, render_summary
from publishing.post_generator import BUTTON_LABEL, generate_post, guide_usage_url
from publishing.telegram import TelegramPublisher
from scoring import scorer
from training.model_registry import ModelRegistry
from training.train import load_or_train, train_symbol
from utils.logging_setup import get_logger
from utils.market_hours import hours_since_gap
from utils.schema import finalize_result

log = get_logger("pipeline")


def _num(v):
    return None if v is None or (isinstance(v, float) and np.isnan(v)) else v


class RadarPipeline:
    def __init__(self, cfg: dict, db=None, loader: Optional[MarketDataLoader] = None,
                 registry: Optional[ModelRegistry] = None, publisher: Optional[TelegramPublisher] = None):
        self.cfg = cfg
        self.db = db
        self.loader = loader or MarketDataLoader(cfg)
        self.registry = registry or self._make_registry(cfg, db)
        sec = get_secrets()
        tcfg = cfg.get("telegram", {})
        self.publisher = publisher or TelegramPublisher(
            sec["telegram_token"] if tcfg.get("enabled", True) else None, sec["telegram_channel"],
            retries=int(tcfg.get("retries", 3)), timeout=int(tcfg.get("timeout", 20)),
            parse_mode="HTML", button=(BUTTON_LABEL, guide_usage_url(cfg)) if cfg.get("publishing", {}).get("guide_button", True) else None)
        self.last_results: List[Dict[str, Any]] = []
        self._sent_hashes: set = set()  # in-process memory when no DB is available

    @staticmethod
    def _make_registry(cfg: dict, db):
        """Models live in the database (Turso) so they survive between stateless CI runs; local files otherwise."""
        import os

        mode = str(cfg["models"].get("storage", "auto")).lower()
        use_db = mode == "db" or (mode == "auto" and bool(os.getenv("TURSO_DATABASE_URL")))
        if db is not None and use_db:
            from training.db_registry import DBModelRegistry

            return DBModelRegistry(db)
        return ModelRegistry(resolve_path(cfg, cfg["models"].get("registry_dir", "data/models")))

    # ------------------------------------------------------------------ analysis
    def analyze_symbol(self, symbol: str, tf: str) -> Optional[Dict[str, Any]]:
        cfg = self.cfg
        df, rep = self.loader.get(symbol, tf)
        if df is None:
            log.warning("%s %s skipped: no valid data", symbol, tf)
            return None
        log.info("Rows processed: %s %s -> %d", symbol, tf, len(df))

        horizon = int(cfg["prediction"]["horizon"])
        thr = float(cfg["prediction"].get("direction_threshold", 0.0))
        feat = build_features(df, donchian_window=int(cfg.get("regime", {}).get("donchian_window", 20)))
        last = feat.iloc[-1]
        if last[FEATURE_COLUMNS].isna().any():
            log.warning("%s %s skipped: latest features incomplete (%s)", symbol, tf,
                        list(last[FEATURE_COLUMNS].index[last[FEATURE_COLUMNS].isna()])[:4])
            return None
        direction, fut = make_targets(feat["close"], horizon, thr)
        log.info("Features calculated: %s %s (%d cols)", symbol, tf, len(FEATURE_COLUMNS))

        trend = trend_engine.analyze(feat)
        mom = momentum_engine.analyze(feat)
        vol = volatility_engine.analyze(feat)
        peers = self.loader.peers(symbol, tf, cross_market_engine.peers_for(cfg, symbol))
        cross = cross_market_engine.analyze(symbol, df, peers, cfg)
        anom = anomaly_engine.analyze(feat, cfg, cross)
        sim = similarity_engine.analyze(feat, fut, cfg, thr)
        reg = regime_engine.analyze(feat, confirm=int(cfg.get("regime", {}).get("confirm_candles", 2)))

        # ---- ML layer -------------------------------------------------------
        ml = self._ml_predict(symbol, tf, df, feat)
        log.info("Models executed: %s %s -> %s", symbol, tf, list(ml["model_probs"]) or "none")

        # ---- ensemble + calibration ----------------------------------------
        ens_cal = None
        if self.db is not None:
            try:
                from evaluation.tracker import fit_ensemble_calibrator
                with self.db.session() as s:
                    ens_cal = fit_ensemble_calibrator(s, symbol, ml["version"], cfg)
            except Exception as exc:
                log.warning("ensemble calibration unavailable: %s", exc)
        parts = {
            "trend_bias": trend.get("trend_bias"), "momentum_bias": mom.get("momentum_bias"),
            "cross_bias": cross.get("cross_market_bias") if cross.get("cross_market_score") is not None else None,
            "similarity_up_rate": sim.get("up_rate") if sim.get("similarity_score") is not None else None,
            "ml_prob_up": ml["prob_up"], "ml_expected_return": ml["expected_return"],
            "similarity_median_return": sim.get("median_future_return") if sim.get("similarity_score") is not None else None,
            "volatility_score": vol.get("volatility_score"), "volatility_state": vol.get("volatility_state"),
            "anomaly_score": anom.get("anomaly_score"), "model_probs": ml["model_probs"],
        }
        e = ens.combine(parts, cfg["weights"], ens_cal)

        baseline = scorer.baseline_up_rate(direction, int(cfg["prediction"].get("baseline_window", 1000)))
        edge = scorer.compute_edge(e["probability_up"], baseline, float(cfg["thresholds"]["min_edge"]))
        risk = scorer.risk_score(vol.get("volatility_score"), vol.get("volatility_state"), last.get("spread_z"),
                                 last.get("tick_vol_z"), last.get("spread"), last.get("atr"), anom.get("anomaly_score"),
                                 e["probability_up"], e["model_agreement"])
        comps = {"trend": trend.get("trend_score"), "momentum": mom.get("momentum_score"),
                 "volatility": vol.get("volatility_score"), "anomaly": anom.get("anomaly_score"),
                 "similarity": sim.get("similarity_score"), "ml": scorer.ml_component(ml["prob_up"]),
                 "cross_market": cross.get("cross_market_score")}
        score = scorer.market_score(comps, cfg["score_weights"])

        ts = feat.index[-1]
        result = {
            "symbol": symbol, "timeframe": tf, "timestamp": ts.isoformat(), "price": float(last["close"]),
            "horizon": horizon, "hours_since_gap": hours_since_gap(feat.index),
            "has_volume": bool(feat["tick_volume"].tail(100).fillna(0).sum() > 0),
            "trend_score": trend.get("trend_score"), "trend_direction": trend.get("trend_direction"),
            "momentum_score": mom.get("momentum_score"), "momentum_direction": mom.get("momentum_direction"),
            "volatility_score": vol.get("volatility_score"), "volatility_state": vol.get("volatility_state"),
            "atr_ratio": vol.get("atr_ratio"),
            "anomaly_score": anom.get("anomaly_score"), "anomaly_detected": anom.get("anomaly_detected"),
            "anomaly_types": anom.get("anomaly_types"), "anomaly_deviations": anom.get("anomaly_deviations"),
            "anomaly_component_scores": anom.get("anomaly_component_scores"),
            "anomaly_details": {k: anom.get(k) for k in ("anomaly_top_deviation_sigma", "anomaly_isolation_forest")},
            "similarity_score": sim.get("similarity_score"), "similar_states": sim.get("n_similar"),
            "historical_up_rate": sim.get("up_rate"), "similarity_details": sim,
            "cross_market_score": cross.get("cross_market_score"), "correlation_break": cross.get("correlation_break"),
            "divergence_score": cross.get("divergence_score"), "cross_pairs": cross.get("cross_pairs"),
            "probability_up": e["probability_up"], "probability_down": e["probability_down"],
            "probability_kind": e["probability_kind"], "probability_raw": e.get("probability_raw"),
            "probability_sources": e["sources"], "ml_probabilities": ml["model_probs"],
            "expected_return": e["expected_return"], "return_quantiles": ml["quantiles"], "confidence": e["confidence"],
            "regime": reg["regime"], "previous_regime": reg["previous_regime"], "regime_changed": reg["regime_changed"],
            "risk": risk["risk"], "risk_score": risk["risk_score"], "model_agreement": e["model_agreement"],
            "market_score": score["market_score"], "market_score_components": score["market_score_components"],
            "edge": edge["edge"], "edge_side": edge["edge_side"], "baseline_up": edge["baseline_up"],
            "edge_label": edge["edge_label"],
            "model_version": ml["version"], "ml_calibrated": ml["calibrated"], "model_skill": ml["skill"],
            "top_drivers": ml["drivers"],
        }
        result["why"] = self._why(result, trend, mom, vol)
        result = finalize_result(result)
        log.info("Prediction generated: %s %s p_up=%s (%s) edge=%s score=%s regime=%s", symbol, tf,
                 result["probability_up"], result["probability_kind"], result["edge_label"],
                 result["market_score"], result["regime"])

        self._persist(symbol, tf, df, feat, result)
        return result

    def _ml_predict(self, symbol: str, tf: str, df: pd.DataFrame, feat: pd.DataFrame) -> Dict[str, Any]:
        empty = {"prob_up": None, "model_probs": {}, "expected_return": None, "quantiles": {}, "version": "NONE",
                 "calibrated": False, "skill": None, "drivers": []}
        try:
            if self.cfg["models"].get("train_on_scan", True):
                bundle = load_or_train(symbol, tf, df, self.cfg, self.registry)
            else:  # CI: models are trained by the dedicated `train` job; a scan only loads the active model
                bundle = self.registry.load_active(symbol, tf)
            if bundle is None:
                log.warning("%s %s: no ML model available (not enough data)", symbol, tf)
                return empty
            ms, cal = bundle["model_set"], bundle.get("calibrator")
            X = feat[FEATURE_COLUMNS].iloc[[-1]]
            probs = ms.predict_proba_up(X)
            raw = float(np.mean([v[0] for v in probs.values()]))
            prob = float(cal.transform([raw])[0]) if cal is not None else raw
            meta = bundle.get("meta", {})
            skill = (meta.get("summary") or {}).get("brier_skill")
            return {"prob_up": prob, "model_probs": {k: float(v[0]) for k, v in probs.items()},
                    "expected_return": float(ms.predict_return(X)[0]), "quantiles": ms.predict_quantiles(X),
                    "version": bundle["version"], "calibrated": cal is not None, "skill": skill,
                    "drivers": ms.top_drivers(3)}
        except Exception as exc:
            log.warning("ML layer failed for %s %s (continuing without it): %s", symbol, tf, exc)
            return empty

    @staticmethod
    def _why(r: Dict[str, Any], trend: dict, mom: dict, vol: dict) -> List[str]:
        why = []
        if vol.get("volatility_state") in ("EXPANSION", "HIGH"):
            why.append(f"Volatility {vol['volatility_state'].lower()} (ATR x{vol.get('atr_ratio')})")
        labels = {"RETURN_ANOMALY": "Abnormal return", "VOLATILITY_SPIKE": "Volatility spike",
                  "VOLUME_ANOMALY": "Unusual tick volume", "SPREAD_ANOMALY": "Spread anomaly",
                  "VELOCITY_ANOMALY": "Abnormal price velocity", "CORRELATION_BREAK": "Cross-market correlation break"}
        why += [labels.get(t, t) for t in (r.get("anomaly_types") or [])]
        if (r.get("similarity_score") or 0) >= 70:
            why.append(f"Historical similarity {r['similarity_score']:.0f}/100 ({r.get('similar_states')} states)")
        if (r.get("momentum_score") or 0) >= 70:
            why.append(f"Strong {str(mom.get('momentum_direction')).lower()} momentum")
        if r.get("regime_changed"):
            why.append(f"Regime shift {r['previous_regime']} → {r['regime']}")
        if r.get("top_drivers"):
            why.append("Model drivers: " + ", ".join(r["top_drivers"][:2]))
        return why[:5]

    def _persist(self, symbol, tf, df, feat, r) -> None:
        if self.db is None:
            return
        try:
            from database import repository as repo
            with self.db.session() as s:
                repo.ensure_market(s, symbol, display_name(self.cfg, symbol))
                n = repo.save_new_candles(s, symbol, tf, df, self.cfg.get("database", {}).get("initial_candles"))
                repo.save_features(s, symbol, tf, feat.index[-1], {c: _num(float(feat.iloc[-1][c])) for c in FEATURE_COLUMNS})
                saved = repo.save_prediction(s, r)
                if saved is not None:
                    repo.save_scan_artifacts(s, r)
                log.info("Stored %s %s: +%d candles, prediction %s", symbol, tf, n, "saved" if saved else "already existed")
        except Exception as exc:
            log.error("DB persistence failed for %s %s: %s", symbol, tf, exc)

    # ------------------------------------------------------------------ scan + publish
    def scan(self, symbols: Optional[List[str]] = None, timeframes: Optional[List[str]] = None,
             publish: bool = True) -> List[Dict[str, Any]]:
        symbols = self.scannable_symbols(symbols)
        timeframes = timeframes or self.cfg["timeframes"]
        if not symbols:
            log.info("Scan skipped: every requested market is closed or peer-only (nothing to do, 0 credits used)")
            return []
        if self.db is not None:
            try:
                from evaluation.tracker import resolve_predictions
                with self.db.session() as s:
                    resolve_predictions(s)
            except Exception as exc:
                log.warning("prediction resolution failed: %s", exc)
        results = []
        for tf in timeframes:
            for sym in symbols:
                try:
                    r = self.analyze_symbol(sym, tf)
                    if r:
                        results.append(r)
                except Exception as exc:  # one broken symbol must never stop the scan
                    log.exception("analysis failed for %s %s: %s", sym, tf, exc)
        self.last_results = results
        log.info("Scan complete: %d/%d analyses", len(results), len(symbols) * len(timeframes))
        if publish and results:
            self.publish_events(results)
        return results

    def scannable_symbols(self, symbols: Optional[List[str]] = None) -> List[str]:
        """Configured symbols minus peer-only ones and markets that are closed right now."""
        from utils.market_hours import is_market_open

        peer_only = set(self.cfg.get("data", {}).get("peer_only", []))
        out = []
        for s in (symbols or self.cfg["symbols"]):
            if symbols is None and s in peer_only:
                continue
            if not is_market_open(s, self.cfg):
                log.info("%s skipped: market closed", s)
                continue
            out.append(s)
        return out

    def _last_event(self, symbol: str, etype: str) -> Optional[dict]:
        if self.db is None:
            return None
        try:
            from database import repository as repo
            with self.db.session() as s:
                ev = repo.last_published_event(s, symbol, etype)
                return None if ev is None else {"time": repo.aware(ev.created_at), "signature": ev.signature}
        except Exception as exc:
            log.warning("event lookup failed: %s", exc)
            return None

    def _event_known(self, h: str) -> bool:
        if h in self._sent_hashes:
            return True
        if self.db is None:
            return False
        try:
            from database import repository as repo
            with self.db.session() as s:
                return repo.event_exists(s, h)
        except Exception:
            return False

    def _caps_state(self, now, since=None) -> Dict[str, int]:
        """Events already published since `since` (default: today 00:00 UTC), from the DB; no DB => zero counts."""
        if self.db is not None:
            try:
                from database import repository as repo
                with self.db.session() as s:
                    return repo.published_event_counts_since(s, since or now.replace(hour=0, minute=0, second=0, microsecond=0))
            except Exception as exc:
                log.debug("caps lookup failed: %s", exc)
        return {"_total": 0}

    def _log_rejected(self, ev: Dict[str, Any], reason: str) -> None:
        log.info("event %s %s rejected: %s", ev["symbol"], ev["type"], reason)
        if self.db is None:
            return
        try:
            from database import repository as repo
            with self.db.session() as s:
                repo.log_rejected_event(s, ev, reason)
        except Exception as exc:
            log.debug("could not log rejected event: %s", exc)

    def rank_events(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Candidate events -> age/priority/dedupe/cooldown filter -> best event per symbol -> daily caps -> per-run cap."""
        now = datetime.now(timezone.utc)
        pcfg = self.cfg["publishing"]
        max_age = float(pcfg.get("max_event_age_minutes", 120))
        best: Dict[str, Dict[str, Any]] = {}
        for r in results:
            try:
                from utils.market_hours import just_reopened
                ts = datetime.fromisoformat(r["timestamp"])
                quiet_h = float(pcfg.get("quiet_after_open_hours", 3))
                hsg = r.get("hours_since_gap")  # data-driven: hours since the first candle after a weekend/session gap
                if just_reopened(r["symbol"], self.cfg, ts, quiet_h) or (quiet_h > 0 and hsg is not None and hsg < quiet_h):
                    log.info("%s: first candles after the weekly open/gap - no event posts", r["symbol"])
                    continue
                age_min = (now - (ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc))).total_seconds() / 60
                if max_age > 0 and age_min > max_age:
                    log.info("%s: candle is %.0f min old (> %.0f) - no event posts", r["symbol"], age_min, max_age)
                    continue
            except Exception as exc:
                log.debug("reopen/age check skipped: %s", exc)
            for etype in prio.detect_candidate_events(r, self.cfg):
                h = prio.event_hash(r["symbol"], r["timeframe"], etype, r["timestamp"])
                ev = prio.evaluate_event(r, etype, self.cfg, self._last_event(r["symbol"], etype), self._event_known(h), now)
                ev.update({"hash": h, "symbol": r["symbol"], "timeframe": r["timeframe"], "event_time": r["timestamp"],
                           "why": r.get("why", []), "result": r})
                if not ev["publish"]:
                    if "duplicate" not in ev["reason"]:
                        self._log_rejected(ev, ev["reason"])
                    continue
                log.info("Event detected: %s %s priority=%.2f", r["symbol"], etype, ev["priority"])
                if r["symbol"] not in best or ev["priority"] > best[r["symbol"]]["priority"]:
                    best[r["symbol"]] = ev
        ranked = sorted(best.values(), key=lambda e: -e["priority"])
        # channel caps: per UTC day, per rolling window (so a night burst cannot eat the whole day), per symbol, per type
        day_cap, sym_cap = int(pcfg.get("max_posts_per_day", 20)), int(pcfg.get("max_posts_per_symbol_per_day", 4))
        win_h, win_cap = float(pcfg.get("rolling_window_hours", 6)), int(pcfg.get("max_posts_per_rolling_window", 6))
        type_caps = pcfg.get("max_posts_per_type_per_run") or {}
        used = self._caps_state(now)
        used_win = self._caps_state(now, since=now - timedelta(hours=win_h)) if win_cap > 0 and win_h > 0 else {"_total": 0}
        out: List[Dict[str, Any]] = []
        by_type: Dict[str, int] = {}
        for ev in ranked:
            if len(out) >= int(pcfg.get("max_posts_per_run", 3)):
                self._log_rejected(ev, "per-run cap")
            elif day_cap > 0 and used.get("_total", 0) + len(out) >= day_cap:
                self._log_rejected(ev, f"daily channel cap ({day_cap})")
            elif win_cap > 0 and used_win.get("_total", 0) + len(out) >= win_cap:
                self._log_rejected(ev, f"rolling cap ({win_cap} per {win_h:g}h)")
            elif sym_cap > 0 and used.get(ev["symbol"], 0) >= sym_cap:
                self._log_rejected(ev, f"daily cap for {ev['symbol']} ({sym_cap})")
            elif type_caps.get(ev["type"]) and by_type.get(ev["type"], 0) >= int(type_caps[ev["type"]]):
                self._log_rejected(ev, f"{ev['type']} cap per run ({type_caps[ev['type']]})")
            else:
                out.append(ev)
                by_type[ev["type"]] = by_type.get(ev["type"], 0) + 1
        return out

    def publish_events(self, results: List[Dict[str, Any]]) -> int:
        self.flush_pending()  # deliver anything that failed earlier before sending new posts
        sent = 0
        for ev in self.rank_events(results):
            try:
                if self._publish_one(ev["type"], ev["result"], ev):
                    sent += 1
            except Exception as exc:
                log.exception("publishing failed for %s %s: %s", ev["symbol"], ev["type"], exc)
        return sent

    def _enrich(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Add run-state facts used by the post text: models/calibration actually in use, live track record."""
        try:
            sym, tf = data.get("symbol"), data.get("timeframe")
            if sym and tf and "methods_info" not in data:
                meta = self.registry.meta(sym, tf, self.registry.active_version(sym, tf) or "") or {}
                data["methods_info"] = {"models": meta.get("models") or [], "calibration": meta.get("calibration_method"),
                                        "isolation_forest": self.cfg.get("anomaly", {}).get("use_isolation_forest", True)}
            if self.db is not None and "track_record" not in data and data.get("symbol"):
                from database import repository as repo
                with self.db.session() as s:
                    df = repo.resolved_predictions_df(s, symbol=data["symbol"], limit=1000)
                    df = df.dropna(subset=["probability_up"]) if len(df) else df
                    data["track_record"] = self._track_record(data["symbol"], df, int(self.cfg["prediction"]["horizon"]))
        except Exception as exc:
            log.debug("post enrichment skipped: %s", exc)
        return data

    @staticmethod
    def _track_record(symbol: str, df, horizon: int) -> Dict[str, Any]:
        """Live record of ONE symbol. Hourly predictions with a 4h horizon overlap, so only n/horizon of them are
        independent. Shown only with >=100 independent samples AND a Brier score better than the baseline."""
        out: Dict[str, Any] = {"symbol": symbol, "n": int(len(df)), "n_eff": 0, "show": False}
        if not len(df):
            return out
        y, p = df["y"], df["probability_up"]
        brier = float(((p - y) ** 2).mean())
        base_rate = float(y.mean())
        base_brier = base_rate * (1 - base_rate)  # always predicting the observed rate
        n_eff = int(len(df) / max(horizon, 1))
        out.update({"brier": brier, "base_brier": base_brier, "n_eff": n_eff,
                    "show": n_eff >= 100 and brier < base_brier})
        return out

    def _publish_one(self, post_type: str, data: Dict[str, Any], ev: Optional[Dict[str, Any]] = None) -> bool:
        text = generate_post(post_type, self._enrich(data), self.cfg)
        image_path = None
        if self.cfg["publishing"].get("send_images", True):
            try:
                image_path = str(render_summary(data, self.cfg) if post_type == "DAILY_MARKET_SUMMARY"
                                 else render_card(data, post_type, self.cfg))
            except Exception as exc:
                log.warning("image generation failed (text-only post): %s", exc)
        log.info("Post generated: %s %s", post_type, data.get("symbol", ""))
        symbol = data.get("symbol")
        post_row_id = None
        if self.db is not None:
            try:
                from database import repository as repo
                with self.db.session() as s:
                    ev_id = None
                    if ev is not None:
                        ev_id = repo.create_event(s, ev, published=True).id
                    post_row_id = repo.create_post(s, post_type, symbol, text, image_path, ev_id).id
            except Exception as exc:
                log.error("could not store post/event: %s", exc)
        if ev is not None:
            self._sent_hashes.add(ev["hash"])
        ok, err = self.publisher.publish(text, image_path)
        if ok:
            log.info("Telegram sent: %s %s", post_type, symbol or "")
        else:
            log.warning("Telegram not sent (%s) -> stays pending in DB", err)
        if self.db is not None and post_row_id is not None:
            try:
                from database import repository as repo
                from database.models import PostRow
                with self.db.session() as s:
                    repo.mark_post(s, s.get(PostRow, post_row_id), ok, None if ok else err)
            except Exception as exc:
                log.error("could not update post status: %s", exc)
        return ok

    TEST_LABEL = "🧪 TEST POST — real format and live data, not a market event\n\n"

    def publish_test(self, post_type: str = "MARKET_SCAN", symbol: Optional[str] = None,
                     timeframe: Optional[str] = None) -> bool:
        """Send one sample post of `post_type` right now, bypassing priority/cooldown/dedupe, so the channel
        output can be checked without waiting for a real event. Nothing is written to the database."""
        from publishing.post_generator import _BUILDERS

        post_type = post_type.upper()
        if post_type not in _BUILDERS:
            raise ValueError(f"unknown post type {post_type}; choose one of: {', '.join(sorted(_BUILDERS))}")
        tf = (timeframe or self.cfg["timeframes"][0]).upper()
        saved_db, self.db = self.db, None  # analysis must not store predictions/candles for a test
        try:
            if post_type == "DAILY_MARKET_SUMMARY":
                data = summarize([r for r in (self.analyze_symbol(s, tf) for s in self.scannable_symbols()[:6]) if r]
                                 if saved_db is None else [])
                if saved_db is not None:
                    self.db = saved_db
                    data = self.build_summary()
                    self.db = None
                if not data["markets_scanned"]:
                    data = summarize([r for r in (self.analyze_symbol(s, tf) for s in self.scannable_symbols()[:6]) if r])
            elif post_type == "MODEL_UPDATE":
                sym = (symbol or self.cfg["symbols"][0]).upper()
                meta = self.registry.meta(sym, tf, self.registry.active_version(sym, tf) or "") or {}
                data = {"symbol": sym, "timeframe": tf, "version": self.registry.active_version(sym, tf) or "ML_v001",
                        "activated": True, "summary": meta.get("summary") or {}}
            else:
                sym = (symbol or self.cfg["symbols"][0]).upper()
                data = self.analyze_symbol(sym, tf)
                if not data:
                    raise RuntimeError(f"no valid data for {sym} {tf}")
        finally:
            self.db = saved_db
        from utils.version import code_version
        text = self.TEST_LABEL.rstrip("\n") + f" · code {code_version()}\n\n" + generate_post(post_type, self._enrich(data), self.cfg)
        image_path = None
        if self.cfg["publishing"].get("send_images", True):
            try:
                image_path = str(render_summary(data, self.cfg) if post_type == "DAILY_MARKET_SUMMARY"
                                 else render_card(data, post_type, self.cfg))
            except Exception as exc:
                log.warning("image generation failed (text-only test post): %s", exc)
        ok, err = self.publisher.publish(text, image_path)
        log.info("Test post %s %s: %s", post_type, data.get("symbol", ""), "sent" if ok else f"NOT sent ({err})")
        return ok

    def flush_pending(self) -> int:
        """Retry posts that could not be delivered earlier."""
        if self.db is None or not self.publisher.configured:
            return 0
        n = 0
        try:
            from database import repository as repo
            with self.db.session() as s:
                max_age = float(self.cfg["publishing"].get("pending_max_age_minutes", 120))
                nowu = datetime.now(timezone.utc)
                for p in repo.pending_posts(s):
                    created = p.created_at if p.created_at.tzinfo else p.created_at.replace(tzinfo=timezone.utc)
                    if max_age > 0 and (nowu - created).total_seconds() / 60 > max_age:
                        repo.expire_post(s, p, f"expired: older than {max_age:.0f} min, not sent")
                        log.info("pending post %s expired (too old to be useful)", p.id)
                        continue
                    ok, err = self.publisher.publish(p.text, p.image_path)
                    repo.mark_post(s, p, ok, None if ok else err)
                    n += int(ok)
        except Exception as exc:
            log.error("flush_pending failed: %s", exc)
        if n:
            log.info("Telegram sent: %d pending post(s)", n)
        return n

    # ------------------------------------------------------------------ summary / training / evaluation
    def build_summary(self, results: Optional[List[Dict[str, Any]]] = None, hours: int = 24) -> Dict[str, Any]:
        track = None
        event_counts: Dict[str, int] = {}
        if results is None and self.db is not None:
            try:
                from database import repository as repo
                from evaluation.tracker import performance_report
                since = datetime.now(timezone.utc) - timedelta(hours=hours)
                with self.db.session() as s:
                    latest: Dict[str, Dict[str, Any]] = {}
                    for row in repo.scores_since(s, since):
                        latest[row.symbol + row.timeframe] = row.result
                    results = list(latest.values())
                    event_counts = {}
                    for ev in repo.events_since(s, since):
                        event_counts[ev.event_type] = event_counts.get(ev.event_type, 0) + 1
                    rep = performance_report(s, persist=False)
                    if rep.get("n"):
                        track = {"n": rep["n"], "brier": rep["overall"]["brier"], "accuracy": rep["overall"]["accuracy"]}
            except Exception as exc:
                log.warning("summary from DB failed: %s", exc)
        results = results if results is not None else self.last_results
        return summarize(results, event_counts, track)

    def publish_daily_summary(self) -> bool:
        summary = self.build_summary()
        if not summary["markets_scanned"]:
            log.info("No data for daily summary")
            return False
        return self._publish_one("DAILY_MARKET_SUMMARY", summary, None)

    def train(self, symbols=None, timeframes=None, notify: bool = False) -> List[Dict[str, Any]]:
        reports = []
        for tf in timeframes or self.cfg["timeframes"]:
            peer_only = set(self.cfg.get("data", {}).get("peer_only", []))
            for sym in symbols or [s for s in self.cfg["symbols"] if s not in peer_only]:
                df, rep = self.loader.get(sym, tf)
                if df is None:
                    continue
                try:
                    r = train_symbol(sym, tf, df, self.cfg, self.registry)
                except Exception as exc:
                    log.exception("training failed for %s %s: %s", sym, tf, exc)
                    continue
                reports.append(r)
                if r.get("trained"):
                    self._record_training(r)
                    if notify:
                        self._publish_one("MODEL_UPDATE", {"symbol": sym, "timeframe": tf, **{k: r[k] for k in ("version", "activated", "summary")}}, None)
        return reports

    def _record_training(self, r: Dict[str, Any]) -> None:
        if self.db is None:
            return
        try:
            from database import repository as repo
            with self.db.session() as s:
                repo.save_model_version(s, r["symbol"], r["timeframe"], r["version"], r["activated"], r["meta"])
                repo.save_metric(s, f"{r['symbol']}:{r['timeframe']}", "walk_forward", r["summary"], r["version"])
                repo.save_metric(s, f"{r['symbol']}:{r['timeframe']}", "feature_importance",
                                 r["meta"].get("feature_importance", {}), r["version"])
        except Exception as exc:
            log.error("could not record training: %s", exc)

    def evaluate(self) -> Dict[str, Any]:
        if self.db is None:
            return {"n": 0, "message": "database unavailable"}
        from evaluation.tracker import performance_report, resolve_predictions
        with self.db.session() as s:
            resolve_predictions(s)
            return performance_report(s)


def summarize(results: List[Dict[str, Any]], event_counts: Optional[Dict[str, int]] = None,
              track: Optional[dict] = None) -> Dict[str, Any]:
    """Aggregate scan results (latest per symbol/timeframe) into the daily-summary structure."""
    latest: Dict[str, Dict[str, Any]] = {}
    for r in results:
        latest[r["symbol"]] = r
    rs = list(latest.values())
    scored = sorted([r for r in rs if r.get("market_score") is not None], key=lambda r: -r["market_score"])
    unusual = max(rs, key=lambda r: r.get("anomaly_score") or 0, default=None)
    ec = event_counts or {}
    return {
        "markets_scanned": len(rs),
        "anomalies": ec.get("ANOMALY_ALERT", sum(1 for r in rs if r.get("anomaly_detected"))),
        "regime_changes": ec.get("REGIME_CHANGE", sum(1 for r in rs if r.get("regime_changed"))),
        "strong_matches": sum(1 for r in rs if (r.get("similarity_score") or 0) >= 85),
        "high_vol": sum(1 for r in rs if r.get("volatility_state") in ("HIGH", "EXPANSION")),
        "top": [{"symbol": r["symbol"], "score": r["market_score"]} for r in scored[:3]],
        "top_unusual": unusual["symbol"] if unusual and (unusual.get("anomaly_score") or 0) > 0 else None,
        "track_record": track,
    }
