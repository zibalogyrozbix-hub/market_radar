#!/usr/bin/env python
"""AI Market Radar CLI.

  python main.py --once | --scan                 scan all configured markets and publish worthy events
  python main.py --symbol XAUUSD [--timeframe H1]  scan a single market
  python main.py --train                          train / evaluate / (conditionally) activate models
  python main.py --evaluate                       resolve predictions and print performance
  python main.py --publish                        retry pending Telegram posts
  python main.py --summary                        publish the daily summary
  python main.py --daemon                         run continuously
  python main.py --check-data                     test the data provider + database connection (uses ~1 credit/symbol)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config.settings import load_config  # noqa: E402
from utils.logging_setup import setup_logging  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="AI Market Radar")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--once", action="store_true", help="one full scan + publish cycle")
    g.add_argument("--scan", action="store_true", help="scan all markets, rank events, publish")
    g.add_argument("--train", action="store_true", help="train models (walk-forward evaluated)")
    g.add_argument("--evaluate", action="store_true", help="resolve predictions, show performance")
    g.add_argument("--publish", action="store_true", help="send pending posts")
    g.add_argument("--summary", action="store_true", help="publish the daily summary")
    g.add_argument("--daemon", action="store_true", help="run continuously")
    g.add_argument("--check-data", action="store_true", help="test data provider and database, then exit")
    p.add_argument("--symbol", help="restrict to one symbol, e.g. XAUUSD")
    p.add_argument("--timeframe", help="restrict to one timeframe, e.g. H1")
    p.add_argument("--no-publish", action="store_true", help="analyse only; do not post")
    p.add_argument("--notify", action="store_true", help="with --train: post a MODEL_UPDATE")
    p.add_argument("--json", action="store_true", help="print full JSON results")
    p.add_argument("--config", help="path to config.yaml")
    p.add_argument("--log-level", default=None)
    return p


def _make_db(cfg: dict, log):
    import os

    required = os.getenv("REQUIRE_DB", "").lower() in ("1", "true", "yes")
    try:
        from database.session import Database

        db = Database(cfg)
        from sqlalchemy import text

        with db.session() as s:  # fail fast on a bad Turso URL/token instead of at the first write
            s.execute(text("SELECT 1"))
        return db
    except Exception as exc:  # e.g. SQLAlchemy not installed, Turso unreachable
        if required:  # on stateless CI a missing DB means duplicate posts: refuse to run
            log.error("Database unavailable (%s) and REQUIRE_DB is set: aborting", exc)
            raise
        log.warning("Database unavailable (%s): predictions will NOT be tracked in this run", exc)
        return None


def _check_data(cfg: dict, pipe, db, symbols) -> int:
    """Diagnostic: one request per symbol; prints OK/FAIL, rows, last candle and credits used."""
    from utils.market_hours import is_market_open

    print(f"provider: {pipe.loader.provider.name} | database: {'connected' if db is not None else 'NOT connected'} "
          f"| models storage: {type(pipe.registry).__name__}")
    syms = symbols or list(cfg["symbols"])
    tf = (cfg["timeframes"] or ["H1"])[0]
    bad = 0
    for sym in syms:
        if getattr(pipe.loader.provider, "auth_error", None):
            print(f"STOP: {pipe.loader.provider.auth_error}")
            return 1
        df, rep = pipe.loader.get(sym, tf)
        mark = "" if is_market_open(sym, cfg) else " (market closed now)"
        if df is None:
            bad += 1
            print(f"FAIL {sym:<7} {'; '.join(rep.reasons)[:120]}{mark}")
        else:
            print(f"OK   {sym:<7} rows={len(df)} last={df['timestamp'].iloc[-1]}{mark}")
    used = getattr(pipe.loader.provider, "credits_used", None)
    if used is not None:
        print(f"API credits used: {used} (free plan: 800/day)")
    return 1 if bad else 0


def _print_results(results: list, as_json: bool) -> None:
    if as_json:
        print(json.dumps(results, indent=2, default=str))
        return
    print(f"{'SYMBOL':<8} {'TF':<4} {'SCORE':>5} {'TREND':>5} {'MOM':>5} {'VOL':>5} {'ANOM':>5} {'SIM':>5} "
          f"{'P(up)':>6} {'EDGE':<14} {'RISK':<8} {'REGIME'}")
    f = lambda v, fmt="{:.0f}": "  N/A" if v is None else fmt.format(v)
    for r in results:
        p = "  N/A" if r["probability_up"] is None else f"{r['probability_up'] * 100:.0f}%"
        print(f"{r['symbol']:<8} {r['timeframe']:<4} {f(r['market_score']):>5} {f(r['trend_score']):>5} "
              f"{f(r['momentum_score']):>5} {f(r['volatility_score']):>5} {f(r['anomaly_score']):>5} "
              f"{f(r['similarity_score']):>5} {p:>6} {r['edge_label']:<14} {r['risk']:<8} {r['regime']}")


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    cfg = load_config(args.config)
    log = setup_logging(cfg, args.log_level)

    from pipeline import RadarPipeline

    db = _make_db(cfg, log)
    pipe = RadarPipeline(cfg, db=db)
    symbols = [args.symbol.upper()] if args.symbol else None
    tfs = [args.timeframe.upper()] if args.timeframe else None

    if args.check_data:
        return _check_data(cfg, pipe, db, symbols)
    if args.train:
        reports = pipe.train(symbols, tfs, notify=args.notify and not args.no_publish)
        for r in reports:
            if r.get("trained"):
                s = r["summary"]
                print(f"{r['symbol']} {r['timeframe']} {r['version']} activated={r['activated']} "
                      f"brier={s.get('brier')} skill={s.get('brier_skill')} auc={s.get('roc_auc')} ece={s.get('ece')}")
            else:
                print(f"{r['symbol']} {r['timeframe']} not trained: {r.get('reason')}")
        return 0
    if args.evaluate:
        print(json.dumps(pipe.evaluate(), indent=2, default=str))
        return 0
    if args.publish:
        print(f"sent {pipe.flush_pending()} pending post(s)")
        return 0
    if args.summary:
        if db is None and not pipe.last_results:  # with a database the summary is built from stored scores
            pipe.scan(symbols, tfs, publish=False)
        print("summary published" if pipe.publish_daily_summary() else "no summary published")
        return 0
    if args.daemon:
        from scheduler.runner import run_daemon

        try:
            run_daemon(pipe, cfg)
        except KeyboardInterrupt:
            log.info("Daemon stopped")
        return 0

    # --once / --scan / --symbol (default action: scan)
    results = pipe.scan(symbols, tfs, publish=not args.no_publish)
    _print_results(results, args.json)
    return 0 if results else 2


if __name__ == "__main__":
    sys.exit(main())
