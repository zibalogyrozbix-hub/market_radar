# راه‌اندازی رایگان: Twelve Data + Turso + GitHub Actions + cron-job.org

معماری: **cron-job.org** هر ساعت یک درخواست به API گیت‌هاب می‌فرستد ← **GitHub Actions** پروژه را روی یک runner موقت اجرا می‌کند ←
کندل‌ها از **Twelve Data** می‌آیند ← همه‌ی state (رویدادها، cooldown، پیش‌بینی‌ها، مدل‌ها) در **Turso** می‌ماند ← پست به **تلگرام** می‌رود.
هیچ سرور یا سیستم روشنی لازم نیست.

## ۱) ساخت سرویس‌ها

1. **Twelve Data:** ثبت‌نام رایگان در twelvedata.com و برداشتن API key (پلن Basic: ‏۸ credit در دقیقه، ‏۸۰۰ در روز).
2. **Turso:** ثبت‌نام در turso.tech، ساخت یک دیتابیس (مثلاً `market-radar`) و برداشتن دو مقدار:
   * `TURSO_DATABASE_URL` به شکل `libsql://market-radar-<نام‌کاربری>.turso.io`
   * `TURSO_AUTH_TOKEN` (توکن دیتابیس؛ با CLI: `turso db tokens create market-radar`)
3. **تلگرام:** ساخت ربات با BotFather، ادمین‌کردن ربات در کانال، و `TELEGRAM_CHANNEL_ID` (مثل `@my_channel`).

## ۲) گیت‌هاب

1. پروژه را در یک repo (ترجیحاً **خصوصی**) آپلود کن.
2. در Settings ← Secrets and variables ← Actions پنج Secret بساز:
   `TELEGRAM_BOT_TOKEN`، `TELEGRAM_CHANNEL_ID`، `TWELVEDATA_API_KEY`، `TURSO_DATABASE_URL`، `TURSO_AUTH_TOKEN`.
3. در تب Actions، workflow با نام **radar** را دستی (Run workflow) به‌ترتیب اجرا کن:
   1. `check-data` — اتصال Turso و وضعیت هر نماد (OK/FAIL) را نشان می‌دهد. نمادهای FAIL را از `symbols` حذف یا `symbol_map` را اصلاح کن.
   2. `train` — مدل‌ها را آموزش می‌دهد و در Turso ذخیره می‌کند (بار اول چند دقیقه).
   3. `scan` — اولین اسکن و پست.

## ۳) cron-job.org (تریگر ساعتی)

ابتدا در گیت‌هاب یک **Fine-grained personal access token** بساز (Settings ← Developer settings): فقط همین repo، و دسترسی
*Actions: Read and write*. بعد در cron-job.org برای هر ردیف جدول زیر یک cronjob بساز:

| تنظیم | مقدار |
|---|---|
| URL | `https://api.github.com/repos/<OWNER>/<REPO>/actions/workflows/radar.yml/dispatches` |
| Request method | `POST` (در Advanced) |
| Headers | `Accept: application/vnd.github+json` ، `Authorization: Bearer <TOKEN>` ، `X-GitHub-Api-Version: 2022-11-28` ، `Content-Type: application/json` |
| Body | `{"ref":"main","inputs":{"job":"scan"}}` |
| Time zone | UTC |
| پاسخ موفق | کد `204` |

| cronjob | زمان‌بندی (UTC) | `job` در Body | معادل تهران |
|---|---|---|---|
| اسکن | هر ساعت، دقیقه‌ی ۵ (`5 * * * *`) | `scan` | هر ساعت، دقیقه‌ی ۳۵ |
| خلاصه‌ی روزانه | دوشنبه تا جمعه ۲۱:۲۰ | `summary` | ۰۰:۵۰ بامداد |
| آموزش مجدد + ارزیابی | شنبه ۰۶:۰۰ | `train` | شنبه ۰۹:۳۰ |

* اسکن را **هر ساعت و هر روز** بگذار: کد خودش بازار بسته را تشخیص می‌دهد (فارکس/فلز از جمعه ۲۲:۰۰ تا یکشنبه ۲۲:۰۰ UTC بسته است)
  و آخر هفته فقط BTC را اسکن می‌کند، یعنی بدون مصرف credit برای بازارهای بسته.
* دقیقه‌ی ۵ یعنی کندل H1 قطعاً بسته شده است. تأخیر چند دقیقه‌ای اجرا اشکالی ندارد.
* خلاصه‌ی روزانه از امتیازهای ذخیره‌شده در Turso ساخته می‌شود و credit مصرف نمی‌کند.

## ۴) بودجه‌ی Twelve Data و دقایق گیت‌هاب

* هر اسکن حدود ۱۹ credit مصرف می‌کند (۱۷ نماد + ۲ ETF). روزهای کاری ۲۴×۱۹ ≈ ۴۵۶ از ۸۰۰ credit روزانه؛ هر آموزش هفتگی ۱۹ credit دیگر.
* سقف ۸ credit در دقیقه یعنی هر اسکن حدود ۲ تا ۳ دقیقه صرف دریافت داده می‌شود (کد خودش فاصله می‌گذارد).
* repo خصوصی: حدود ۴ دقیقه × ۲۴ × ۲۲ روز ≈ ۲۱۰۰ دقیقه در ماه که کمی بیشتر از سهمیه‌ی رایگان ۲۰۰۰ دقیقه است. اگر خطر دارد:
  اسکن را هر ۲ ساعت بگذار، یا تعداد نمادها را کم کن، یا repo را عمومی کن (دقیقه‌ی Actions نامحدود است؛ Secrets عمومی نمی‌شوند).

## ۵) نکات مهم

* **US500 / NAS100:** پلن رایگان Twelve Data شاخص ندارد، پس با ETFهای SPY و QQQ جایگزین شده‌اند و فقط به‌عنوان peer
  برای تحلیل cross-market به کار می‌روند (`peer_only`) و پست جداگانه ندارند. برای شاخص واقعی باید پلن پولی بگیری و
  `symbol_map` را به `SPX` و `NDX` برگردانی.
* **طلا/نقره (XAU/XAG):** به‌عنوان جفت‌ارز درخواست می‌شوند. اگر `check-data` برایشان FAIL داد، از `symbols` حذف کن.
* **حجم و اسپرد:** فارکس در Twelve Data حجم ندارد؛ این دو ویژگی صفر می‌شوند و روی نتیجه اثر منفی ندارند، فقط اطلاعاتی هم نمی‌دهند.
* مدل‌ها فقط در job‌ی `train` آموزش می‌بینند (`models.train_on_scan: false`)، و اسکن فقط مدل فعال را از Turso می‌خواند.
* اگر Turso در دسترس نباشد، اجرا متوقف می‌شود (`REQUIRE_DB=1`) تا پست تکراری نرود.
* با workflow_dispatch محدودیت «غیرفعال‌شدن بعد از ۶۰ روز بی‌فعالیتی» که برای `schedule:` وجود دارد اعمال نمی‌شود.

## اجرای لوکال (اختیاری)

```bash
pip install -r requirements.txt -r requirements-turso.txt
cp .env.example .env     # مقادیر را پر کن
python main.py --check-data
python main.py --scan --no-publish
```
