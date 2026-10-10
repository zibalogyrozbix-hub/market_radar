# AI Market Radar (رادار هوشمند بازار)

اسکنر کمّی بازار: کندل‌ها را می‌گیرد، ناهنجاری‌ها را پیدا می‌کند، وضعیت فعلی را با «وضعیت‌های مشابه گذشته» مقایسه می‌کند،
با مدل‌های ML کالیبره‌شده احتمال تخمین می‌زند، هر پیش‌بینی را با واقعیت می‌سنجد و پست‌های صادقانه و رویدادمحور در کانال تلگرام منتشر می‌کند.

این پروژه **سیگنال‌فروش نیست**: عبارت خرید/فروش (BUY/SELL) ندارد و وقتی برتری آماری وجود نداشته باشد صریحاً
`NO CLEAR EDGE` می‌نویسد. هیچ LLM‌ای در آن استفاده نشده (کاملاً قطعی و بدون هزینه‌ی API).

## معماری

```
DATA (Twelve Data | MT5 | CSV | synthetic) → VALIDATION → FEATURES
  → ENGINES: trend · momentum · volatility · anomaly · similarity · cross-market · regime
  → MODELS: logistic · random forest · xgboost (اختیاری) + رگرسیون بازده + کوانتایل‌ها
  → ENSEMBLE (وزن‌ها در config) → CALIBRATION → EDGE / RISK / MARKET SCORE
  → DB (SQLAlchemy: SQLite یا Turso) → اولویت‌بندی رویداد (dedupe + cooldown) → POST → IMAGE → TELEGRAM
```

| پوشه | نقش |
|---|---|
| `config/` | `config.yaml` (همه‌ی تنظیمات) + بارگذار؛ رازها فقط از `.env` یا Secrets |
| `data/` | providerها (Twelve Data/MT5/CSV/synthetic)، اعتبارسنجی، loader |
| `features/` | اندیکاتورهای علّی (causal)، سازنده‌ی ویژگی، هدف‌های آینده‌نگر (جدا) |
| `engines/` | موتورهای تحلیل |
| `models/` | مجموعه‌ی مدل ML، ensemble، calibration |
| `training/` | دیتاست، تقسیم زمانی/walk-forward، رجیستری مدل (فایل یا دیتابیس)، pipeline آموزش |
| `evaluation/` | متریک‌ها، بک‌تست walk-forward، ردیاب پیش‌بینی و گزارش عملکرد |
| `scoring/` | edge، ریسک، Market Intelligence Score |
| `database/` | مدل‌های SQLAlchemy، session، repository |
| `publishing/` | تولید پست و تصویر، اولویت/dedupe/cooldown، تلگرام |
| `scheduler/` | حلقه‌ی daemon |
| `utils/` | ابزارها از جمله ساعت بازار (`market_hours.py`) |
| `docs/SETUP_FA.md` | راهنمای گام‌به‌گام اجرای رایگان روی GitHub Actions |
| `pipeline.py`, `main.py` | هماهنگ‌سازی و CLI |

## اجرای رایگان و بدون سرور (روش پیشنهادی)

**cron-job.org** هر ساعت یک درخواست به API گیت‌هاب می‌فرستد ← **GitHub Actions** پروژه را اجرا می‌کند ← کندل‌ها از
**Twelve Data** می‌آیند ← state (رویدادها، cooldown، پیش‌بینی‌ها و مدل‌ها) در **Turso** می‌ماند ← پست به **تلگرام** می‌رود.

راهنمای کامل (ساخت کلیدها، Secretها، درخواست cron-job.org و زمان‌بندی‌ها) در **`docs/SETUP_FA.md`** است.
ترتیب اولین اجرا در تب Actions (workflow با نام `radar`): ‏`check-data` ← `train` ← `scan`.

## نصب (اجرای لوکال)

```bash
python -m venv .venv && source .venv/bin/activate        # ویندوز: .venv\Scripts\activate
pip install -r requirements.txt -r requirements-turso.txt # توسعه: requirements-dev.txt
cp .env.example .env                                      # سپس مقادیر را پر کنید
python main.py --check-data                               # تست provider و دیتابیس
python main.py --scan --no-publish                        # اسکن بدون ارسال به تلگرام
```

پایتون ۳٫۱۰ به بالا. XGBoost اختیاری است (در نبودش خودکار Logistic + RandomForest استفاده می‌شود).
`requirements-turso.txt` فقط برای اتصال به Turso لازم است؛ بدون آن، دیتابیس SQLite محلی استفاده می‌شود.

## منابع داده

مقدار `data.provider` در config (یا متغیر `DATA_PROVIDER`): ‏`twelvedata` (پیش‌فرض) یا `auto`
(اگر کلید Twelve Data باشد همان ← MT5 ← CSV ← synthetic).

* **Twelve Data** — کلید رایگان در `TWELVEDATA_API_KEY`. محدودیت پلن رایگان: ۸ درخواست در دقیقه و ۸۰۰ در روز؛ هر نماد در هر درخواست ۱ credit.
  کد خودش فاصله‌ی درخواست‌ها را رعایت می‌کند و فقط کندل‌های **بسته‌شده** را می‌خواند.
  * شاخص‌ها در پلن رایگان نیستند: `US500` و `NAS100` با ETFهای `SPY` و `QQQ` جایگزین شده‌اند و فقط به‌عنوان peer برای تحلیل cross-market به کار می‌روند (`data.peer_only`).
  * فارکس حجم و اسپرد ندارد؛ این دو ویژگی صفر می‌شوند.
  * اگر داده‌ی یک نماد قدیمی‌تر از `twelvedata.max_stale_minutes` باشد (در ساعات باز بازار) آن نماد رد می‌شود.
* **MT5** — فقط ویندوز؛ `pip install MetaTrader5`، نصب و ورود به ترمینال. متغیرهای اختیاری `MT5_*` در `.env`. اگر بروکر نمادها را مثل `XAUUSD.m` می‌نامد، `MT5_SYMBOL_SUFFIX=.m` بگذارید.
* **CSV** (لینوکس/مک/Docker/CI) — فایل‌های `data/csv/<SYMBOL>_<TF>.csv` با ستون‌های `timestamp,open,high,low,close[,tick_volume,spread]`.
* **synthetic** — داده‌ی ساختگی برای تست. **داده‌ی واقعی بازار نیست** و نتایجش بی‌معناست (هشدار در لاگ ثبت می‌شود).

برای افزودن نماد: آن را به `symbols`، `display_names` و (برای تحلیل cross-market) به فهرست‌های `cross_market` در `config.yaml` اضافه کنید.
اگر برای نمادی peer تعریف نشود از `cross_market.default` استفاده می‌شود. peerهای بدون داده نادیده گرفته می‌شوند.

### ساعت بازار

فارکس و فلزات از جمعه ۲۲:۰۰ تا یکشنبه ۲۲:۰۰ (UTC) بسته‌اند و اسکن نمی‌شوند (بدون مصرف credit). `BTCUSD` همیشه باز است.
نمادهای `session_only` (پروکسی‌های ETF) فقط روزهای کاری داده دارند. تنظیم در بخش `market_hours` فایل config.

## دستورات

```bash
python main.py --scan               # = --once: همه‌ی نمادها را اسکن، رویدادها را رتبه‌بندی و منتشر می‌کند
python main.py --symbol XAUUSD [--timeframe H1]
python main.py --train [--notify]   # ارزیابی walk-forward + آموزش؛ فقط در صورت بهتربودن فعال می‌شود
python main.py --evaluate           # حل پیش‌بینی‌های قدیمی و نمایش عملکرد واقعی
python main.py --publish            # ارسال مجدد پست‌های ناموفق
python main.py --summary            # انتشار خلاصه‌ی روزانه (از دیتابیس؛ بدون مصرف credit)
python main.py --check-data         # تست provider، دیتابیس و وضعیت هر نماد
python main.py --daemon             # اسکن ساعتی + خلاصه‌ی روزانه (برای سرور دائمی)
python main.py --scan --no-publish --json
```

* مدل‌ها در حالت پیش‌فرض (`models.train_on_scan: false`) فقط با `--train` آموزش می‌بینند و اسکن فقط مدل فعال را می‌خواند.
  اگر هنوز مدلی نیست، اسکن بدون لایه‌ی ML ادامه می‌دهد. برای آموزش خودکار در اسکن مقدار را `true` کنید (حدود ۲۰ ثانیه برای هر نماد).
* `logs/market_radar.log` ثبت می‌کند: دریافت داده، ردیف‌های پردازش‌شده، ویژگی‌ها، مدل‌های اجراشده، پیش‌بینی، رویداد، پست و ارسال تلگرام.

## راه‌اندازی تلگرام

1. با @BotFather یک ربات بسازید ← `TELEGRAM_BOT_TOKEN`.
2. ربات را **ادمین** کانال کنید (با دسترسی ارسال پست).
3. `TELEGRAM_CHANNEL_ID` = ‏`@نام_کانال` یا شناسه‌ی عددی `-100…`.

اگر تلگرام در دسترس نباشد، پست با وضعیت `pending` در دیتابیس می‌ماند و در اجرای بعد (یا با `--publish`) دوباره ارسال می‌شود.

## قواعد صداقت (در سیستم تعبیه شده)

* احتمال‌ها تا وقتی calibrator (ایزوتونیک/Platt روی داده‌ی بعدی و جداشده) وجود نداشته باشد با برچسب **`raw model probability`** نمایش داده می‌شوند؛ بعد از آن `calibrated`.
  با رسیدن پیش‌بینی‌های حل‌شده به `min_calibration_rows`، ensemble از روی نتایج خود سیستم دوباره کالیبره می‌شود.
* اگر وضعیت تاریخی قابل‌استفاده‌ی کافی نباشد، شباهت (Similarity) به‌صورت `N/A` است و نمایش داده نمی‌شود.
* **Edge** = احتمال منهای نرخ پایه‌ی تاریخی همان سمت؛ کمتر از `thresholds.min_edge` یعنی `NO CLEAR EDGE`.
* ریسک جدا از احتمال محاسبه می‌شود؛ Market Score میزان «قابل‌توجه بودن» وضعیت را می‌سنجد، نه جهت را.
* ردیف‌های پیش‌بینی تغییرناپذیرند؛ نتیجه پس از افق پیش‌بینی در `prediction_results` نوشته می‌شود.
* کنترل نشت داده (leakage): ویژگی‌ها علّی‌اند (تست شده)، scalerها داخل pipeline و فقط روی train فیت می‌شوند، Similarity فقط حالت‌هایی را به‌کار می‌برد که نتیجه‌شان از قبل معلوم بوده (embargo)،
  walk-forward فاصله‌ی purge دارد و از تقسیم تصادفی استفاده نمی‌شود. مدل جدید فقط اگر Brier/ECE/پایداری out-of-sample بهتری از مدل فعال داشته باشد فعال می‌شود (نه فقط accuracy).
* در بازارهای شبیه به random walk نتیجه‌ی صادقانه معمولاً Brier skill ≈ ۰ و در نتیجه بیشتر `NO CLEAR EDGE` است؛ این رفتار درست است.
  قبل از اعتماد به هر عدد، خروجی `--train` را ببینید (`brier_skill > 0`).

## پیکربندی

همه‌چیز در `config/config.yaml`: ‏`symbols`، `timeframes`، `prediction.horizon`، `twelvedata.*`، `market_hours`، `models.*`،
`weights` (ensemble)، `score_weights`، `thresholds`، `priority_weights`، `publishing.cooldown_minutes`، `branding`، `telegram`، `database`، `logging`.
رازها (`.env` یا GitHub Secrets) هرگز در config خوانده یا نوشته نمی‌شوند.

## دیتابیس

* **Turso (libSQL)** — با تنظیم `TURSO_DATABASE_URL` و `TURSO_AUTH_TOKEN` فعال می‌شود و برای اجرای stateless روی GitHub Actions لازم است.
  با `REQUIRE_DB=1` اگر دیتابیس در دسترس نباشد اجرا متوقف می‌شود (تا پست تکراری ارسال نشود).
  مدل‌ها هم در همین دیتابیس نگهداری می‌شوند (`models.storage: auto`؛ جدول `model_artifacts`).
* **SQLite** (پیش‌فرض لوکال) در `data/market_radar.db`. جدول‌ها: markets، candles، features، predictions، prediction_results، anomalies،
  similarity_results، market_scores، events، posts، model_versions، model_metrics، model_artifacts.
* **PostgreSQL** — `DATABASE_URL=postgresql+psycopg://user:pass@host/db` (نصب `psycopg`).
* بدون SQLAlchemy اسکن اجرا می‌شود ولی پیش‌بینی‌ها **ردیابی نمی‌شوند**.

## بک‌تست و ارزیابی

`evaluation/backtest.py::walk_forward` (پنجره‌ی انبساطی، فاصله‌ی purge، کالیبراسیون در هر fold) داخل `--train` اجرا می‌شود.
متریک‌ها: accuracy، precision، recall، F1، ROC-AUC، Brier (و skill نسبت به نرخ پایه)، ECE، باکت‌های کالیبراسیون، MAE، RMSE، دقت جهت و پایداری بین foldها.
`--evaluate` همین را روی پیش‌بینی‌های واقعیِ حل‌شده به تفکیک نماد/تایم‌فریم/رژیم/باکت حساب می‌کند.

## استقرار

* **پیشنهادی (رایگان):** cron-job.org + GitHub Actions + Twelve Data + Turso؛ فایل `.github/workflows/radar.yml` و راهنمای `docs/SETUP_FA.md`.
* کرون لینوکس: `5 * * * * cd /path && python main.py --scan`
* Daemon: ‏`python main.py --daemon` · Docker: ‏`docker compose up -d`.
* `.github/workflows/ci.yml` تست و lint را اجرا می‌کند.

## تست

`pytest -q` (یا `python run_tests.py` اگر pytest نصب نیست). پوشش: اعتبارسنجی، علّی بودن ویژگی‌ها، همه‌ی موتورها، look-ahead در Similarity،
ML، calibration، walk-forward، ensemble، scoring، اولویت/dedupe/cooldown، پست‌ها، تصاویر، payload و retry تلگرام، scheduler، ردیابی دیتابیس،
provider ‏Twelve Data (rate-limit، کندل در حال تشکیل، stale)، ساعت بازار، رجیستری مدل در دیتابیس و dialect ‏libSQL.

## عیب‌یابی

| نشانه | راه‌حل |
|---|---|
| `FAIL` برای یک نماد در `--check-data` | نماد روی پلن رایگان نیست یا `symbol_map` غلط است؛ از `symbols` حذف یا اصلاح کنید |
| `Twelve Data 429` / تمام شدن credit | سقف ۸ در دقیقه یا ۸۰۰ در روز؛ تعداد نمادها یا دفعات اسکن را کم کنید |
| `stale data` | آخرین کندل بسته قدیمی است (تعطیلی/مشکل API)؛ همان نماد در این اجرا رد می‌شود |
| `Database unavailable … REQUIRE_DB` | آدرس/توکن Turso را بررسی کنید و `sqlalchemy-libsql` را نصب کنید |
| `Data provider: SYNTHETIC` | کلید Twelve Data یا MT5/CSV پیدا نشد |
| `MT5 initialize failed` | فقط ویندوز؛ ترمینال را اجرا و وارد شوید؛ `MT5_PATH` را بررسی کنید |
| `Validation failed … abnormal gap` | کندل‌های اخیر ناقص‌اند؛ داده را تازه کنید |
| `not enough data` / Similarity برابر `N/A` | به `data.min_bars` و `similarity.min_history` کندل نیاز است |
| پست‌ها `pending` می‌مانند | توکن/شناسه‌ی کانال غلط است یا ربات ادمین نیست؛ سپس `--publish` |
| پست‌های زیاد یا کم | `thresholds.post_priority`، `cooldown_minutes` و `max_posts_per_run` را تنظیم کنید |

## پیش‌فرض‌های انتخاب‌شده

فقط H1، افق ۴ کندل، UP یعنی بازده‌ی آینده > ۰، Similarity با K=500 و embargo هشت‌کندلی، تشخیص رژیم قاعده‌محور پشت رابط `RegimeDetector`
(بعداً می‌توان HMM/خوشه‌بندی گذاشت)، ensemble وزنی (وزن‌ها از config؛ meta-model جایگزین `models/ensemble.combine` می‌شود) و کالیبراسیون ایزوتونیک.
