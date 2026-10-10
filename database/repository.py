"""Data-access helpers. All functions take an open SQLAlchemy Session."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import pandas as pd
from sqlalchemy import func, select

from database.models import (AnomalyRow, Candle, EventRow, FeatureRow, Market, MarketScoreRow, ModelMetricRow,
                             ModelVersionRow, PostRow, Prediction, PredictionResult, SimilarityRow)


def aware(dt: datetime | None) -> datetime | None:
    """SQLite drops tzinfo; everything we store is UTC."""
    if dt is None:
        return None
    return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt.astimezone(timezone.utc)


def to_dt(ts) -> datetime:
    return pd.Timestamp(ts).tz_convert("UTC").to_pydatetime() if pd.Timestamp(ts).tzinfo else pd.Timestamp(ts, tz="UTC").to_pydatetime()


# ---- markets / candles / features --------------------------------------------------------
def ensure_market(s, symbol: str, display: str) -> None:
    if s.scalar(select(Market).where(Market.symbol == symbol)) is None:
        s.add(Market(symbol=symbol, display_name=display))


def save_new_candles(s, symbol: str, tf: str, df: pd.DataFrame, initial_limit: Optional[int] = None) -> int:
    """Store candles newer than the last stored one. On the first run only the latest `initial_limit`
    candles are kept (the tracker only needs recent candles; this keeps remote DB writes small)."""
    last = s.scalar(select(func.max(Candle.timestamp)).where(Candle.symbol == symbol, Candle.timeframe == tf))
    last = aware(last)
    d = df if last is None else df[df["timestamp"] > pd.Timestamp(last)]
    if last is None and initial_limit:
        d = d.tail(int(initial_limit))
    rows = [Candle(symbol=symbol, timeframe=tf, timestamp=to_dt(r.timestamp), open=float(r.open), high=float(r.high),
                   low=float(r.low), close=float(r.close), tick_volume=float(r.tick_volume), spread=float(r.spread))
            for r in d.itertuples(index=False)]
    s.add_all(rows)
    return len(rows)


def save_features(s, symbol: str, tf: str, ts, values: Dict[str, Any]) -> None:
    dt = to_dt(ts)
    if s.scalar(select(FeatureRow.id).where(FeatureRow.symbol == symbol, FeatureRow.timeframe == tf, FeatureRow.timestamp == dt)) is None:
        s.add(FeatureRow(symbol=symbol, timeframe=tf, timestamp=dt, values=values))


# ---- predictions -----------------------------------------------------------------------
def save_prediction(s, r: Dict[str, Any]) -> Optional[Prediction]:
    """Insert one immutable prediction; returns None if this candle/model was already predicted."""
    dt = to_dt(r["timestamp"])
    exists = s.scalar(select(Prediction.id).where(
        Prediction.symbol == r["symbol"], Prediction.timeframe == r["timeframe"],
        Prediction.timestamp == dt, Prediction.model_version == r.get("model_version")))
    if exists is not None:
        return None
    p = Prediction(
        timestamp=dt, symbol=r["symbol"], timeframe=r["timeframe"], horizon=r["horizon"], price=r["price"],
        probability_up=r["probability_up"], probability_down=r["probability_down"],
        probability_kind=r["probability_kind"], expected_return=r["expected_return"],
        market_score=r["market_score"], risk_score=r["risk_score"], regime=r["regime"],
        anomaly_score=r["anomaly_score"], similarity_score=r["similarity_score"],
        model_agreement=r["model_agreement"], edge=r["edge"], edge_label=r["edge_label"],
        model_version=r.get("model_version"), payload=r)
    s.add(p)
    s.flush()
    return p


def save_scan_artifacts(s, r: Dict[str, Any]) -> None:
    dt = to_dt(r["timestamp"])
    s.add(AnomalyRow(symbol=r["symbol"], timeframe=r["timeframe"], timestamp=dt, anomaly_score=r["anomaly_score"] or 0.0,
                     detected=bool(r["anomaly_detected"]), types=r["anomaly_types"], details=r.get("anomaly_details", {})))
    s.add(SimilarityRow(symbol=r["symbol"], timeframe=r["timeframe"], timestamp=dt, similarity_score=r["similarity_score"],
                        n_similar=int(r.get("similar_states") or 0), details=r.get("similarity_details", {})))
    if s.scalar(select(MarketScoreRow.id).where(MarketScoreRow.symbol == r["symbol"], MarketScoreRow.timeframe == r["timeframe"],
                                                MarketScoreRow.timestamp == dt)) is None:
        s.add(MarketScoreRow(symbol=r["symbol"], timeframe=r["timeframe"], timestamp=dt, market_score=r["market_score"],
                             components=r.get("market_score_components", {}), result=r))


def unresolved_predictions(s) -> List[Prediction]:
    resolved = select(PredictionResult.prediction_id)
    return list(s.scalars(select(Prediction).where(Prediction.id.not_in(resolved)).order_by(Prediction.timestamp)))


def resolve_prediction(s, pred: Prediction, actual_price: float) -> PredictionResult:
    actual_ret = actual_price / pred.price - 1.0
    direction = "UP" if actual_ret > 0 else "DOWN"
    correct = None
    if pred.probability_up is not None:
        correct = (pred.probability_up >= 0.5) == (direction == "UP")
    err = None if pred.expected_return is None else float(pred.expected_return - actual_ret)
    res = PredictionResult(prediction_id=pred.id, actual_price=actual_price, actual_return=actual_ret,
                           actual_direction=direction, correct=correct, error=err)
    s.add(res)
    return res


def resolved_predictions_df(s, symbol: Optional[str] = None, limit: Optional[int] = None) -> pd.DataFrame:
    q = (select(Prediction, PredictionResult).join(PredictionResult, PredictionResult.prediction_id == Prediction.id)
         .order_by(Prediction.timestamp.desc()))
    if symbol:
        q = q.where(Prediction.symbol == symbol)
    if limit:
        q = q.limit(limit)
    rows = []
    for p, r in s.execute(q).all():
        rows.append({"timestamp": aware(p.timestamp), "symbol": p.symbol, "timeframe": p.timeframe, "horizon": p.horizon,
                     "probability_up": p.probability_up, "probability_kind": p.probability_kind,
                     "probability_raw": (p.payload or {}).get("probability_raw"),
                     "expected_return": p.expected_return, "regime": p.regime, "model_version": p.model_version,
                     "edge_label": p.edge_label, "risk_score": p.risk_score,
                     "actual_return": r.actual_return, "actual_direction": r.actual_direction,
                     "y": 1.0 if r.actual_direction == "UP" else 0.0, "correct": r.correct, "error": r.error})
    return pd.DataFrame(rows).sort_values("timestamp") if rows else pd.DataFrame()


# ---- events / posts --------------------------------------------------------------------
def event_exists(s, event_hash: str) -> bool:
    return s.scalar(select(EventRow.id).where(EventRow.event_hash == event_hash)) is not None


def last_published_event(s, symbol: str, event_type: str) -> Optional[EventRow]:
    return s.scalar(select(EventRow).where(EventRow.symbol == symbol, EventRow.event_type == event_type,
                                           EventRow.published.is_(True)).order_by(EventRow.created_at.desc()).limit(1))


def create_event(s, ev: Dict[str, Any], published: bool) -> EventRow:
    row = EventRow(event_hash=ev["hash"], symbol=ev["symbol"], timeframe=ev["timeframe"], event_type=ev["type"],
                   event_time=to_dt(ev["event_time"]), priority=ev["priority"], published=published,
                   signature=ev.get("signature", {}), why=ev.get("why", []))
    s.add(row)
    s.flush()
    return row


def create_post(s, post_type: str, symbol: Optional[str], text: str, image_path: Optional[str], event_id: Optional[int] = None) -> PostRow:
    row = PostRow(post_type=post_type, symbol=symbol, text=text, image_path=image_path, event_id=event_id, status="pending")
    s.add(row)
    s.flush()
    return row


def pending_posts(s, max_attempts: int = 10) -> List[PostRow]:
    return list(s.scalars(select(PostRow).where(PostRow.status == "pending", PostRow.attempts < max_attempts).order_by(PostRow.created_at)))


def mark_post(s, post: PostRow, ok: bool, error: Optional[str] = None) -> None:
    post.attempts += 1
    if ok:
        post.status, post.sent_at, post.last_error = "sent", datetime.now(timezone.utc), None
    else:
        post.last_error = error
        if post.attempts >= 10:
            post.status = "failed"


def scores_since(s, since: datetime) -> List[MarketScoreRow]:
    return list(s.scalars(select(MarketScoreRow).where(MarketScoreRow.created_at >= since).order_by(MarketScoreRow.created_at)))


def events_since(s, since: datetime) -> List[EventRow]:
    return list(s.scalars(select(EventRow).where(EventRow.created_at >= since)))


# ---- models / metrics ------------------------------------------------------------------
def save_model_version(s, symbol: str, tf: str, version: str, active: bool, meta: Dict[str, Any]) -> None:
    if active:
        for r in s.scalars(select(ModelVersionRow).where(ModelVersionRow.symbol == symbol, ModelVersionRow.timeframe == tf)):
            r.active = False
    if s.scalar(select(ModelVersionRow.id).where(ModelVersionRow.symbol == symbol, ModelVersionRow.timeframe == tf,
                                                 ModelVersionRow.version == version)) is None:
        s.add(ModelVersionRow(symbol=symbol, timeframe=tf, version=version, active=active, meta=meta))


def save_metric(s, scope: str, kind: str, metrics: Dict[str, Any], model_version: Optional[str] = None) -> None:
    s.add(ModelMetricRow(scope=scope, kind=kind, metrics=metrics, model_version=model_version))
