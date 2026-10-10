# AI Market Radar

Quantitative market scanner that fetches candles, detects anomalies, compares the current state with
historical look-alikes, estimates probabilities with calibrated ML, tracks every prediction against
reality, and publishes honest, event-driven posts to a Telegram channel.

It is **not** a signal seller: no BUY/SELL wording, and when there is no statistical edge it says
`NO CLEAR EDGE`. LLMs are not used anywhere (fully deterministic, zero API cost).

## Architecture

```
DATA(MT5 | CSV | synthetic) → VALIDATION → FEATURES
  → ENGINES: trend · momentum · volatility · anomaly · similarity · cross-market · regime
  → MODELS: logistic · random forest · xgboost (optional) + return regression + quantiles
  → ENSEMBLE (weights in config) → CALIBRATION → EDGE / RISK / MARKET SCORE
  → DB (SQLAlchemy/SQLite) → EVENT PRIORITY (dedupe + cooldown) → POST → IMAGE → TELEGRAM
```

| Folder | Role |
|---|---|
| `config/` | `config.yaml` (all settings) + loader; secrets only from `.env` |
| `data/` | providers (MT5/CSV/synthetic), validation, loader |
| `features/` | causal indicators, feature builder, future-looking targets (separate) |
| `engines/` | the analysis engines |
| `models/` | ML model set, ensemble, calibration |
| `training/` | dataset, chronological/walk-forward splits, model registry, train pipeline |
| `evaluation/` | metrics, walk-forward backtest, prediction tracker/performance report |
| `scoring/` | edge, risk, Market Intelligence Score |
| `database/` | SQLAlchemy models, session, repository |
| `publishing/` | post generator, image generator, priority/dedupe/cooldown, Telegram |
| `scheduler/` | daemon loop |
| `pipeline.py`, `main.py` | orchestration and CLI |

## Installation

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt                      # dev: requirements-dev.txt
cp .env.example .env                                 # then fill it in
python main.py --scan --no-publish                   # try without Telegram
```

Python 3.10+. XGBoost is optional (automatic fallback to Logistic + RandomForest).

## Data sources

`data.provider` in config (or `DATA_PROVIDER` env): `twelvedata` (default, free API, runs anywhere) or `auto`
(Twelve Data if a key is set → MT5 → CSV → synthetic).

* **Twelve Data** — free key in `TWELVEDATA_API_KEY`; throttled to 8 requests/min, 1 credit per symbol,
  only closed candles are used. Indices are not on the free plan, so US500/NAS100 use SPY/QQQ as peer-only proxies.
  State (events, cooldowns, predictions, models) is stored in **Turso** (`TURSO_DATABASE_URL` + `TURSO_AUTH_TOKEN`),
  which makes stateless GitHub Actions runs possible. Full free-hosting guide (Persian): `docs/SETUP_FA.md`.

* **MT5** — `pip install MetaTrader5` works on **Windows only**; install the MT5 terminal, log in to
  your broker once, keep it running. Optional `MT5_*` vars in `.env`. If your broker names symbols
  `XAUUSD.m`, set `MT5_SYMBOL_SUFFIX=.m`. The still-forming candle is excluded.
* **CSV** (Linux/macOS/Docker/CI) — put `data/csv/<SYMBOL>_<TF>.csv` with columns
  `timestamp,open,high,low,close[,tick_volume,spread]` (e.g. export from MT5 on Windows).
* **synthetic** — generated demo data so everything runs anywhere. **Not real market data**; a loud
  warning is logged and results are meaningless.

DXY/US10Y/VIX can be used for cross-market analysis if your broker provides them: add the symbol
names to `symbols` and to the lists under `cross_market` in `config.yaml`. Missing peers are skipped.

## Running

```bash
python main.py --scan               # = --once: scan all symbols, rank events, publish
python main.py --symbol XAUUSD [--timeframe H1]
python main.py --train [--notify]   # walk-forward evaluate + train; activate only if better
python main.py --evaluate           # resolve predictions, print live performance
python main.py --publish            # retry posts that failed to send
python main.py --summary            # publish the daily summary
python main.py --daemon             # hourly scan + daily summary
python main.py --scan --no-publish --json
```

A first scan trains a model automatically per symbol/timeframe (~20 s each on 2 CPUs).
`logs/market_radar.log` records: data fetched, rows processed, features calculated, models executed,
prediction generated, event detected, post generated, Telegram sent.

## Telegram setup

1. Create a bot with @BotFather → `TELEGRAM_BOT_TOKEN`.
2. Add the bot as **admin** of your channel with permission to post.
3. `TELEGRAM_CHANNEL_ID` = `@channelusername` or the numeric `-100…` id.

If Telegram is unreachable the post is stored as `pending` in the DB and retried on the next run
(`--publish`, or automatically before the next scan's posts).

## Honesty rules built into the system

* Probabilities are labelled **`raw model probability`** until a calibrator (isotonic/Platt, fit on
  chronologically later held-out data) exists; only then they are `calibrated`. The ensemble is
  re-calibrated from the system's own resolved predictions once ≥ `min_calibration_rows` exist.
* Similarity is `N/A` (and not shown) if there are not enough usable historical states.
* **Edge** = probability − that side's historical base rate; below `thresholds.min_edge` ⇒ `NO CLEAR EDGE`.
* Risk is computed separately from probability; Market Score measures how *notable* a state is,
  not direction.
* Prediction rows are immutable; outcomes are written to `prediction_results` after the horizon.
* Leakage control: features are causal (tested), scalers live inside pipelines fit on train only,
  similarity pools only states whose outcome was already known (embargo), walk-forward uses a
  purge gap, random splits are never used. A model becomes active only if its out-of-sample
  Brier/ECE/stability beat the active one (not accuracy alone).
* On random-walk-like markets the honest result is usually Brier skill ≈ 0 ⇒ mostly `NO CLEAR EDGE`.
  That is correct behaviour. Check `--train` output (`brier_skill > 0`) before trusting any number.

## Configuration

Everything lives in `config/config.yaml`: `symbols`, `timeframes`, `prediction.horizon`,
`models.*`, `weights` (ensemble), `score_weights`, `thresholds`, `priority_weights`,
`publishing.cooldown_minutes`, `branding`, `telegram`, `database`, `logging`.
Secrets (`.env`) are never read from or written to the config.

## Database

SQLite at `data/market_radar.db` (tables: markets, candles, features, predictions,
prediction_results, anomalies, similarity_results, market_scores, events, posts, model_versions,
model_metrics). For PostgreSQL set `DATABASE_URL=postgresql+psycopg://user:pass@host/db`
(install `psycopg`). Without SQLAlchemy the scan still runs, but predictions are **not tracked**.

## Backtesting & evaluation

`evaluation/backtest.py::walk_forward` (expanding window, purge gap, per-fold calibration) is run
inside `--train`; metrics: accuracy, precision, recall, F1, ROC-AUC, Brier (+skill vs base rate),
ECE, calibration/probability buckets, MAE, RMSE, directional accuracy, stability across folds.
`--evaluate` computes the same on live resolved predictions by symbol/timeframe/regime/bucket.

## Deployment

* Linux cron: `5 * * * * cd /path && python main.py --scan`
* Daemon: `python main.py --daemon` · Docker: `docker compose up -d` (uses `DATA_PROVIDER=csv`).
* GitHub Actions: `.github/workflows/ci.yml` (tests+lint) and optional `scheduled.yml`. Runners
  have no MT5 and no persistent SQLite — use a self-hosted runner or external DB for track record.

## Tests

`pytest -q` (or `python run_tests.py` if pytest is unavailable). Covers validation, causality of
features, all engines, similarity look-ahead, ML, calibration, walk-forward, ensemble, scoring,
priority/dedupe/cooldown, posts, images, Telegram payload/retry, scheduler, DB tracking.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `MT5 initialize failed` | Windows only; start/log in to the terminal; check `MT5_PATH` |
| `Data provider: SYNTHETIC` | no MT5/CSV found; provide CSVs or run MT5 |
| `Validation failed … abnormal gap` | recent missing candles; refresh/export data |
| `not enough data` / similarity `N/A` | need ≥ `data.min_bars` and `similarity.min_history` candles |
| Posts stuck `pending` | wrong token/channel id or bot not admin; then `--publish` |
| `Database unavailable` | `pip install SQLAlchemy` |
| Too many/few posts | tune `thresholds.post_priority`, `cooldown_minutes`, `max_posts_per_run` |

## Defaults chosen (not specified in the brief)

H1 only, horizon 4 candles, UP = future return > 0, similarity K=500 with an 8-bar embargo, rule-based
regime detector behind a `RegimeDetector` interface (swap in HMM/clustering later), weighted ensemble
(weights from config; a meta-model can replace `models/ensemble.combine`), isotonic calibration.
