# AI Market Radar — Reader's Guide

**What you are looking at, what every number means, and how to use it responsibly.**

> AI Market Radar publishes **statistical market analysis**. It is **not** a trading signal service, never says BUY or SELL, and does not know your account, costs or risk tolerance. Use it as *context* for your own decisions.

---

## The 60-second version

- The radar scans markets every hour on **closed H1 (1-hour) candles** and posts only when something is *worth your attention* (an anomaly, a regime change, a strong historical match, a correlation break, a volatility spike) plus a daily summary.
- Every post is built from the same few ideas: **scores (0–100)**, a **market state**, a **probability with a statistical edge**, a **risk label**, and **how similar situations behaved in the past**.
- Most of the time the honest answer is **`NO CLEAR EDGE`**. That is a feature, not a bug: it means the model sees no measurable advantage over the long-run base rate.
- Scores describe *how notable or unusual* a market is. They are **not** direction and **not** a probability of profit.

**A safe reading order for any post:** Header → Market state → Risk → Edge/Probability → Model agreement → Historical context → *then* your own chart and plan.

---

## Anatomy of a post

### The Market Scan

```
🧊 BTC/USD · Volatility squeeze — quietest 2% of recent history
H1 · candle 04:00 UTC · 63,420.00 · horizon 4h        ← market · candle · price · look-ahead

TREND      ▰▰▰▱▱▱▱▱▱▱ 38 →     ← strength of the trend; arrow = its direction
MOMENTUM   ▰▰▰▰▱▱▱▱▱▱ 40 ↗     ← strength of the current push
VOLATILITY ▱▱▱▱▱▱▱▱▱▱ 2        ← how large volatility is vs. its own history
ANOMALY    ▰▰▱▱▱▱▱▱▱▱ 20       ← how unusual the behaviour is
SIMILARITY ▰▰▰▰▰▰▰▰▰▱ 94       ← how well this state is represented in history
MKT SCORE  ▰▰▰▰▰▰▰▱▱▱ 70       ← Market Intelligence Score (blend of all the above)

🧊 COMPRESSION                  ← market state (section "Market states") + one-line meaning
Probability over 4h: ↑ 68% · ↓ 32%   ← both sides; bold = the side the models lean to ("raw model" = not calibrated, "no proven skill" = the model has not beaten the baseline)
Statistical edge: +14.0 pp toward ↑ vs baseline   ← ✅ ≥ 5 pp · 🔥 ≥ 10 pp (only for calibrated models with proven skill)
Expected move: +0.05%   ← shown only when it is larger than the normal noise; otherwise "≈0 (within normal ±0.09%)"

🟢 Risk LOW · 🟡 Agreement MEDIUM
🧬 Of 100 similar past situations, 62% were higher 4h later   ← printed only when similarity ≥ 60; "≈ coin flip" when close to 50%
📈 EUR/USD last 500 resolved (≈125 independent): Brier 0.231 vs baseline 0.246   ← per symbol; hidden until there are 100+ independent samples and the model beats the baseline
🔬 Logistic + Random Forest + XGBoost · isotonic calibration · nearest-neighbour history search
⚠️ Statistical analysis · not financial advice
#BTCUSD #MarketScan #Compression
```

Bars: each ▰ is 10 points. Colour dots: 🟢 low / 🟡 medium / 🟠 high / 🔴 extreme risk; for agreement 🟢 high / 🟡 medium / 🟠 low. Labels such as *Probability*, *Statistical edge*, *Risk*, *Agreement* and *similar* in the post are links to the matching section of this guide, and the **🧭 How do I use this?** button under each post opens "Using it in your trading".

⚖️ **Mixed signals** appears when trend/momentum point one way and the models lean the other; the edge is then reported as "none clear". A post can also be skipped for the first hours after the weekly market open, when gap candles look like events.

All values are always shown. When the model has no measurable skill the numbers still appear, but treat them as context, not as an edge (see "Statistical edge").

### Time horizon

All probabilities and "future move" figures refer to **4 candles ahead = 4 hours** on H1 (the operator can change this in `config.yaml`). "UP" means *the close 4 candles from now is higher than now*.

---

## The scores (0–100)

All scores are on a 0–100 scale and are **non-directional** unless stated otherwise.

- **TREND** — **What it measures:** *Strength* of the prevailing trend, from trend-strength (ADX) and alignment of 20/50/100/200 moving averages. **How to read it:** Rough guide: <35 weak/choppy · 40–60 developing · >60 strong. Direction is shown by the state `TREND_UP` / `TREND_DOWN`, not by this number.
- **MOMENTUM** — **What it measures:** *Size* of the current push, from RSI, MACD histogram, rate of change, price velocity, acceleration and divergences. **How to read it:** High = price is moving decisively now; low = drifting. Says nothing alone about direction.
- **VOLATILITY** — **What it measures:** **Percentile** of current volatility vs. the market's own recent history (50 = typical, 90 = more volatile than 90% of the past). **How to read it:** >85 is "HIGH"; <15 is "LOW". High volatility means bigger swings both ways.
- **ANOMALY** — **What it measures:** How statistically unusual the latest candle is: z-scores of return, volatility, volume, spread and price velocity, blended (≈75/25) with an Isolation-Forest outlier score. **How to read it:** Rough guide: 0–40 normal · 40–80 elevated · **≥80 alert**. Also flags when any single deviation is ≥ 2.5σ.
- **SIMILARITY** — **What it measures:** How well the **current state is represented in history**: how close its nearest historical look-alikes are, compared with how close look-alikes usually are. **How to read it:** High = history has many close analogues, so the "historical UP rate" and move range are more informative. Low/`N/A` = a rare or novel state; treat historical stats with caution.
- **MARKET INTELLIGENCE SCORE** — **What it measures:** A weighted blend of all the above (trend, momentum, volatility, anomaly, similarity, ML confidence, cross-market). **How to read it:** Measures **how notable and information-rich the situation is** — *not* how bullish/bearish and *not* a success probability. 60+ is what the scanner treats as noteworthy.

> **Common mistake:** reading "TREND 80" as "go long". It only says the trend is *strong*; check whether the state is `TREND_UP` or `TREND_DOWN`.

---

#### Section Market states

## Market states

Assigned by transparent rules (later rules in this list override earlier ones when several apply):

- **😴 **LOW_VOLATILITY**** — **Meaning:** Weak trend and unusually quiet market (volatility percentile < 15%). **Typical character:** Small ranges; breakouts often follow, but timing is unknown.
- **🧊 **COMPRESSION**** — **Meaning:** Volatility shrinking and Bollinger bands narrow. **Typical character:** Energy building; direction of the eventual exit is not predicted.
- **↔️ **RANGE**** — **Meaning:** Default when none of the others apply: no strong trend. **Typical character:** Mean-reverting, choppy conditions.
- **📈 **TREND_UP** / 📉 **TREND_DOWN**** — **Meaning:** Strong, aligned trend (strength ≥ 40) with positive/negative bias. **Typical character:** Moves tend to persist; pullbacks possible.
- **🌪️ **HIGH_VOLATILITY**** — **Meaning:** Volatility above its 85th percentile. **Typical character:** Wider stops/smaller size are usually appropriate.
- **🔥 **EXPANSION**** — **Meaning:** Current range is > 1.3× its recent average (ATR ratio). **Typical character:** Volatility is actively rising.
- **⚡ **BREAKOUT**** — **Meaning:** Price breaks its recent high/low (Donchian) while range expands (ATR ratio > 1.1). **Typical character:** New range being established; false breakouts occur.
- **🔄 **REVERSAL**** — **Meaning:** Short-term slope flips against the prevailing trend while RSI is stretched (>68 or <32) **and** there is a divergence. **Typical character:** Early warning of exhaustion, not a confirmed turn.

A **REGIME CHANGE** post appears when the state differs from the previous candle's state (e.g. `RANGE → BREAKOUT`).

---

## Probability, edge and expected move

#### Section Probability

### Probability
The ensemble's estimate that price is **higher 4 candles from now**. It combines trend, momentum, historical similarity, a machine-learning model (logistic regression, random forest, optionally XGBoost) and cross-market information, then shrinks toward 50% when volatility or anomalies make conditions unstable.

- **Below 50% means a lean downward** (the post still prints the "up" probability).
- **`(raw model probability)`** means the number has not yet been calibrated against held-out data — treat it as a ranking, not a literal frequency. **`calibrated`** numbers are adjusted so that, historically, "60%" really happened about 60% of the time.

#### Section Statistical edge

### Statistical edge
`Edge = probability of the favoured side − that side's historical base rate` (shown in **percentage points, pp**).

- Edge below the threshold (default **5 pp**) → **`NO CLEAR EDGE`**.
- A "+6 pp" edge is *small*. It is an advantage in a long series of similar situations, not a prediction for one trade, and it can be smaller than your spread and costs.

### Expected move
The model's average expected return over the next 4 candles (e.g. `+0.09%`). It is **not a target**; actual moves are usually much larger or smaller.

#### Section Model agreement

### Model agreement
How closely the individual models agree on the probability.

- **HIGH** — All on the same side, spread ≤ 12 pp.
- **MEDIUM** — Same side with moderate spread, or a clear majority.
- **LOW** — Models disagree — treat any edge with extra scepticism.

#### Section Similarity

### Similarity
From the *nearest historical look-alikes* (up to 500). `Historical UP 58%` means that in 58% of those past situations price was higher 4 candles later. `P25` and `P75` give the typical range of outcomes (the middle 50%): a wide P25–P75 means high uncertainty even if the median is positive.

---

#### Section Risk

## Risk

Risk is computed **separately from direction** (a bullish and a bearish setup can have the same risk).

- **LOW** — **Score:** < 30 **Interpretation:** Calm, orderly conditions.
- **MEDIUM** — **Score:** 30–50 **Interpretation:** Normal conditions.
- **HIGH** — **Score:** 50–70 **Interpretation:** Elevated volatility/uncertainty.
- **EXTREME** — **Score:** ≥ 70 **Interpretation:** Unstable or unusual conditions.

Built from: volatility (30%), anomaly level (20%), directional uncertainty (20%), spread/liquidity stress (15%) and model disagreement (15%). Use it to decide **how much** to risk, not **which way** to trade.

---

## The post types

- **🧠 AI MARKET SCAN** — **Typical trigger (defaults):** Market Intelligence Score ≥ 60 **What to look at:** The full picture ("Anatomy of a post").
- **🚨 MARKET ANOMALY DETECTED** — **Typical trigger (defaults):** Anomaly score ≥ 80, or a deviation ≥ 2.5σ **What to look at:** The **σ values** (1σ ≈ normal; ≥ 3σ is rare). *Why flagged* lists the causes: abnormal return, volatility spike, price velocity, volume, spread, correlation break.
- **🧬 MARKET DNA MATCH** — **Typical trigger (defaults):** Similarity ≥ 85 **What to look at:** How many look-alikes were found, the Historical UP rate and the **P25–P75 range** of what followed.
- **⚡ MARKET REGIME CHANGE** — **Typical trigger (defaults):** State changed vs. the previous candle **What to look at:** `Previous → Current`, plus the four scores.
- **🔗 CROSS-MARKET CORRELATION BREAK** — **Typical trigger (defaults):** Correlation with a peer moved ≥ 0.45 from its normal level **What to look at:** `Normal +0.80 → Current +0.20 ⚠️ BREAK` means two markets that usually move together have decoupled. Check divergence and whether your open positions are really as diversified as you think.
- **🌪️ VOLATILITY ALERT** — **Typical trigger (defaults):** Volatility percentile ≥ 90 **What to look at:** **ATR Expansion** (e.g. `1.6x` = current range is 1.6× normal), state and risk.
- **📊 DAILY SUMMARY** — **Typical trigger (defaults):** Once per weekday (after the US close) **What to look at:** Markets scanned, anomalies, regime changes, strong matches, top-3 markets by score, most unusual market, and the **track record** (see "Honesty and the track record").
- **🔧 MODEL UPDATE** — **Typical trigger (defaults):** After periodic retraining **What to look at:** Whether the new model replaced the old one, its out-of-sample **Brier, ROC-AUC, ECE** ("Metric glossary") and whether it shows measurable skill.

Posting rules: at most 3 posts per run, one best event per market, a **cooldown** per post type (e.g. 60–180 min), and a post is repeated only if the situation has materially changed. **No post does not mean calm** — it only means nothing new passed the filters.

---

#### Section Using it in your trading

## Using it in your trading

The radar is best used as a **context and filter layer** on top of your own method.

### A practical routine
1. **Regime first.** Does the state fit your strategy? (Trend-followers want `TREND_*`/`BREAKOUT`; mean-reversion traders prefer `RANGE`/`LOW_VOLATILITY`; many avoid `HIGH_VOLATILITY` and `REVERSAL` unless they trade them specifically.)
2. **Check risk.** Scale position size down as risk rises; consider standing aside at `EXTREME`.
3. **Check the edge.** `NO CLEAR EDGE` → the radar offers no directional support; rely on your own setup. A `CLEAR EDGE` with **HIGH agreement** is a *tie-breaker*, not a reason to trade by itself.
4. **Check history.** A high similarity score with a tight P25–P75 range is more informative than a low similarity score with a wide range.
5. **Confirm on your chart** (levels, structure, news calendar) and only then apply your own entry, stop and exit rules.

### Reading examples
- **`TREND_UP`, risk MEDIUM, edge +6 pp, agreement HIGH, similarity 85:** conditions that *support* a trend-following long idea you already have. Entry, stop and sizing remain yours.
- **`NO CLEAR EDGE`, `HIGH_VOLATILITY`, risk HIGH:** no statistical support; swings are large. Smaller size or no trade.
- **Anomaly alert (z_vol +3.4σ):** something unusual is happening (news, liquidity gap). Check the calendar; widen stops or stay out until volatility normalises.
- **Correlation break between EUR/USD and GBP/USD:** if you hold both, your "diversification" may have changed; review combined exposure.
- **`REVERSAL` state:** an *early* warning that a trend is stretched; wait for your own confirmation.

### Do not
- Treat any number as a guarantee or a buy/sell instruction.
- Trade every post. Most posts describe conditions, not opportunities.
- Ignore costs: a +5 pp statistical edge over 4 hours can be consumed by spread, commission and slippage.
- Risk more because a post looks "strong". Risk sizing should come from your plan, not from the radar.
- Rely on the numbers before the **track record** ("Honesty and the track record") shows the model has measurable skill.

*Nothing here is financial advice. Trading carries a high risk of loss.*

---

## Honesty and the track record

- Every prediction is stored **before** the outcome is known and is **never edited**. Outcomes are recorded after 4 candles and feed the daily **track record**: `Brier 0.250 · hit rate 51%` over the last N resolved predictions.
- **Brier score** measures probability quality: 0.250 is what you get by always saying 50%; **lower is better**. A model with *Brier skill ≤ 0* adds nothing over the base rate.
- On efficient, random-walk-like markets the honest result is often *no measurable skill*, so many posts will show `NO CLEAR EDGE`. That is the system being truthful.
- Features use only past data; similarity looks only at situations whose outcome was already known; models are validated on later, unseen data (walk-forward), never on random splits.

---

## Metric glossary

- **H1** — 1-hour candles; the radar acts only on *closed* candles.
- **Candle / bar** — One period of price data (open, high, low, close).
- **Percentile** — Rank vs. history: 90 = higher than 90% of past values.
- **σ (sigma) / z-score** — How many standard deviations a value is from its recent norm. ±1σ common, ±2σ uncommon, ±3σ rare.
- **ATR / ATR ratio** — Average True Range = typical candle range; ratio = current vs. average (1.0 = normal).
- **ADX** — Trend-strength indicator (direction-neutral).
- **RSI** — Momentum oscillator; >70 stretched up, <30 stretched down.
- **Correlation** — How similarly two markets move (+1 together, −1 opposite, 0 unrelated).
- **Base rate** — How often price was up historically over the same horizon.
- **pp (percentage points)** — Difference between two percentages (60% − 55% = 5 pp).
- **Calibration** — Whether stated probabilities match real frequencies.
- **Brier / ECE / ROC-AUC** — Out-of-sample model quality: Brier (lower is better; compare with 0.25), ECE = calibration error (lower is better), ROC-AUC = ranking quality (0.5 = random, 1.0 = perfect).
- **Regime** — The market's current behavioural state ("Market states").
- **Divergence** — Price makes a new extreme but momentum does not (possible exhaustion).

---

## Data, timing and limits

- **Markets:** major and cross forex pairs, gold, bitcoin. US indices are used internally as reference markets for cross-market analysis. Forex and gold do not publish during the weekend close (Fri 22:00 → Sun 22:00 UTC); only bitcoin is scanned then.
- **Timing:** analysis runs shortly after each hourly candle closes; posts can arrive a few minutes after the hour. They describe the *last closed candle*, not the live price.
- **Volume/spread:** spot forex has no real volume, so "volume" and "spread" anomaly components are not informative there.
- **Data quality:** candles are validated; markets with gaps, stale or invalid data are skipped rather than guessed.
- **Not covered:** news, fundamentals, your costs and your account. The radar sees price history only.

---

## FAQ

**Why do I see `NO CLEAR EDGE` so often?** Because short-term markets are close to random most of the time. The radar refuses to invent an edge.

**Is a higher Market Intelligence Score better?** No. It means the situation is more *notable*, not more bullish or more profitable.

**The probability is 62% — so I should buy?** No. 62% is a statistical estimate for a 4-hour horizon, compared with a base rate; it does not account for your stop, target, costs or risk.

**Why was there no post for my market?** Cooldowns, thresholds, a closed market, or data that failed validation. Silence is not a signal.

**What does `(raw model probability)` mean?** The probability has not been calibrated yet; use it for ranking and be extra cautious.

**Can I use it to copy trades?** There are no trades to copy: the radar does not issue entries, stops or targets.
