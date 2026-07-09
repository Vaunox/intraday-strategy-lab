"""Point-in-time technical indicators (Phase 1, P1.5).

Every function is a *pure, causal* map from an :class:`OHLCV` to an aligned array:
the value at bar ``i`` uses only bars ``0..i``. Standard indicators come from
TA-Lib (avoiding hand-rolled bugs); VWAP, pivots, opening range, gaps, Donchian
channels, relative volume, and realized volatility are hand-rolled because TA-Lib
lacks them — each with tests and each strictly causal, so the dual-path skew test
(``harness``) passes.

Warmup regions are ``NaN``. Session-aware features (VWAP, pivots, opening range,
gap) reset on the IST trading date carried by the bar timestamps.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta

import numpy as np
import talib
from numpy.lib.stride_tricks import sliding_window_view

from lab.data.features.ohlcv import OHLCV, FloatArray


def _empty(n: int) -> FloatArray:
    return np.full(n, np.nan, dtype=np.float64)


def _rolling(values: FloatArray, period: int, use_max: bool) -> FloatArray:
    """Causal rolling max/min: out[i] over values[i-period+1 .. i], NaN in warmup."""
    out = _empty(len(values))
    if len(values) >= period:
        windows = sliding_window_view(values, period)
        reduced = windows.max(axis=1) if use_max else windows.min(axis=1)
        out[period - 1 :] = reduced
    return out


# --- TA-Lib standard indicators --------------------------------------------- #
def sma(data: OHLCV, period: int) -> FloatArray:
    """Simple moving average of close."""
    result: FloatArray = talib.SMA(data.close, timeperiod=period)
    return result


def ema(data: OHLCV, period: int) -> FloatArray:
    """Exponential moving average of close."""
    result: FloatArray = talib.EMA(data.close, timeperiod=period)
    return result


def kama(data: OHLCV, period: int) -> FloatArray:
    """Kaufman adaptive moving average of close (noise-adaptive)."""
    result: FloatArray = talib.KAMA(data.close, timeperiod=period)
    return result


def rsi(data: OHLCV, period: int) -> FloatArray:
    """Relative strength index of close."""
    result: FloatArray = talib.RSI(data.close, timeperiod=period)
    return result


def adx(data: OHLCV, period: int) -> FloatArray:
    """Average directional index (trend strength)."""
    result: FloatArray = talib.ADX(data.high, data.low, data.close, timeperiod=period)
    return result


def atr(data: OHLCV, period: int) -> FloatArray:
    """Average true range (volatility)."""
    result: FloatArray = talib.ATR(data.high, data.low, data.close, timeperiod=period)
    return result


def atr_ratio(data: OHLCV, short_period: int, long_period: int) -> FloatArray:
    """Dimensionless ATR ratio ``ATR(short) / ATR(long)`` -- a self-normalizing vol regime.

    ``> 1.0`` means the recent (short-window) ATR is above the longer baseline -> volatility
    EXPANDING; ``< 1.0`` -> CONTRACTING. Self-normalizing per symbol (each stock's own current
    ATR over its own baseline), so it is directly comparable across the panel WITHOUT a
    cross-sectional rank -- a ratio of 1.3 means "30% above this stock's own baseline"
    identically for every symbol. Causal: both ATRs are Wilder averages of completed prior
    bars (:func:`atr` / ``talib.ATR``, confirmed prefix-invariant), so the value at bar ``i``
    uses only bars ``0..i``. ``NaN`` until the longer ATR warms up, or where ``ATR(long) == 0``.
    """
    short_atr = atr(data, short_period)
    long_atr = atr(data, long_period)
    with np.errstate(invalid="ignore", divide="ignore"):
        out: FloatArray = np.where(long_atr > 0.0, short_atr / long_atr, np.nan)
    return out


def macd(
    data: OHLCV, fast: int, slow: int, signal: int
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """MACD line, signal line, and histogram of close."""
    macd_line, signal_line, hist = talib.MACD(
        data.close, fastperiod=fast, slowperiod=slow, signalperiod=signal
    )
    return macd_line, signal_line, hist


def bollinger(
    data: OHLCV, period: int, num_std: float
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Bollinger upper, middle, and lower bands of close."""
    upper, middle, lower = talib.BBANDS(
        data.close, timeperiod=period, nbdevup=num_std, nbdevdn=num_std
    )
    return upper, middle, lower


def percent_b(data: OHLCV, period: int, num_std: float) -> FloatArray:
    """Bollinger %B: position of close within the bands."""
    upper, _middle, lower = bollinger(data, period, num_std)
    width = upper - lower
    with np.errstate(invalid="ignore", divide="ignore"):
        out: FloatArray = np.where(width > 0, (data.close - lower) / width, np.nan)
    return out


# --- hand-rolled channels / volume / volatility ----------------------------- #
def donchian(data: OHLCV, period: int) -> tuple[FloatArray, FloatArray]:
    """Donchian channel: rolling N-bar high and low (global window, may cross days)."""
    return _rolling(data.high, period, use_max=True), _rolling(data.low, period, use_max=False)


def intraday_donchian(data: OHLCV, period: int) -> tuple[FloatArray, FloatArray]:
    """Prior N-bar Donchian high/low over CURRENT-DAY bars only (resets each IST day).

    ``out[i]`` is the max-high / min-low over the prior ``period`` bars that share bar
    ``i``'s IST trading date -- window ``[max(day_start_i, i - period) .. i - 1]``,
    EXCLUDING bar ``i`` -- so an intraday breakout is measured against TODAY's range, never
    a level carried across the overnight gap. The first bar of each day is ``NaN`` (no
    prior intraday bar defines a range). Strictly causal; the value at bar ``i`` uses only
    bars ``0..i-1`` of the same day.

    Distinct from :func:`donchian` (a global rolling window that crosses days) AND from an
    expanding day-range: it keeps the N-bar-range concept, so a prior-N-bar-range breakout
    (P3.2) stays a different strategy from an Opening-Range Breakout (P3.11), which trades
    the day's running high/low. Both indicators are kept; callers choose.
    """
    high, low = data.high, data.low
    n = len(high)
    upper, lower = _empty(n), _empty(n)
    day_start = 0
    for i in range(n):
        if i > 0 and data.timestamps[i].date() != data.timestamps[i - 1].date():
            day_start = i  # reset the window at the IST trading-date boundary
        window_start = max(day_start, i - period)
        if window_start < i:  # at least one prior intraday bar defines the range
            upper[i] = float(np.max(high[window_start:i]))
            lower[i] = float(np.min(low[window_start:i]))
    return upper, lower


def intraday_zscore(data: OHLCV, period: int) -> FloatArray:
    """Rolling z-score of close over CURRENT-DAY bars only (resets each IST day).

    ``out[i]`` = ``(close[i] - mean) / std`` over the last ``period`` bars sharing bar
    ``i``'s IST trading date -- window ``[i - period + 1 .. i]``, INCLUDING bar ``i`` -- once a
    full ``period``-bar intraday window has formed since the day's open, else ``NaN``.
    Population std (``ddof=0``), matching the Bollinger convention (:func:`bollinger` /
    TA-Lib ``BBANDS``); the z-score is the statistic Bollinger %B expresses (a band at
    ``k*std`` is a z of ``k``), so this is the intraday-reset analogue of Bollinger used by
    the P3.3 mean-reversion fade.

    Intraday-reset by design (NOT a trailing window): an intraday fade must measure stretch
    against TODAY's mean, never across the overnight gap -- a trailing window would read a
    gap as a huge z-score and fade AGAINST the gap, silently becoming a gap-fade (a
    different strategy). A FULL ``period``-bar window is required (the first ``period - 1``
    bars of each day are ``NaN``): an expanding 2-3 bar window would give a degenerate std
    and an explosive z. Strictly causal -- the value at bar ``i`` uses only bars ``0..i`` of
    the same day -- so it is prefix-invariant (the dual-path skew contract).
    """
    close = data.close
    n = len(close)
    out = _empty(n)
    day_start = 0
    for i in range(n):
        if i > 0 and data.timestamps[i].date() != data.timestamps[i - 1].date():
            day_start = i  # reset the window at the IST trading-date boundary
        window_start = i - period + 1
        if window_start >= day_start:  # a FULL period-bar window has formed within the day
            window = close[window_start : i + 1]
            std = float(np.std(window))  # population std (ddof=0), matches TA-Lib BBANDS
            if std > 0.0:
                out[i] = (float(close[i]) - float(np.mean(window))) / std
    return out


def prior_donchian(data: OHLCV, period: int) -> tuple[FloatArray, FloatArray]:
    """Prior N-bar Donchian high/low EXCLUDING the current bar (GLOBAL; crosses days).

    ``out[i]`` = max high / min low over bars ``[i - period .. i - 1]`` -- the ``period`` bars
    BEFORE bar ``i`` -- so ``close[i] > upper[i]`` is a genuine breakout of the prior N-bar
    high. GLOBAL: the window does NOT reset at the day boundary, so an early-session bar
    references the PRIOR session's extreme (a multi-session level) -- deliberately distinct
    from :func:`intraday_donchian` (day-reset) and from :func:`donchian` (which INCLUDES the
    current bar). ``NaN`` until ``period`` prior bars exist. Strictly causal: the value at bar
    ``i`` uses only bars ``0..i-1``, so it is prefix-invariant (the no-lookahead contract).
    """
    high, low = data.high, data.low
    n = len(high)
    upper, lower = _empty(n), _empty(n)
    if n > period:
        # sliding window w[k] covers bars [k .. k+period-1]; out[i] over [i-period .. i-1]
        # is the window starting at i-period, i.e. w[i-period] for i in [period, n-1].
        high_windows = sliding_window_view(high, period)[: n - period]
        low_windows = sliding_window_view(low, period)[: n - period]
        upper[period:] = high_windows.max(axis=1)
        lower[period:] = low_windows.min(axis=1)
    return upper, lower


def relative_volume(data: OHLCV, period: int) -> FloatArray:
    """Volume divided by the mean of the prior ``period`` volumes (excludes current)."""
    out = _empty(len(data))
    volume = data.volume
    if len(volume) > period:
        windows = sliding_window_view(volume[:-1], period)  # prior-period windows
        means = windows.mean(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            out[period:] = np.where(means > 0, volume[period:] / means, np.nan)
    return out


def realized_volatility(data: OHLCV, period: int) -> FloatArray:
    """Rolling standard deviation of log returns over ``period`` bars."""
    out = _empty(len(data))
    close = data.close
    if len(close) > period:
        log_ret = np.diff(np.log(close))
        windows = sliding_window_view(log_ret, period)
        out[period:] = windows.std(axis=1, ddof=0)
    return out


# --- session-aware features (reset per IST trading date) -------------------- #
def vwap(data: OHLCV) -> FloatArray:
    """Intraday cumulative VWAP of the typical price, reset each trading day."""
    out = _empty(len(data))
    typical = (data.high + data.low + data.close) / 3.0
    cum_pv = 0.0
    cum_v = 0.0
    current_date = None
    for i, ts in enumerate(data.timestamps):
        day = ts.date()
        if day != current_date:
            cum_pv = 0.0
            cum_v = 0.0
            current_date = day
        cum_pv += typical[i] * data.volume[i]
        cum_v += data.volume[i]
        if cum_v > 0:
            out[i] = cum_pv / cum_v
    return out


def vwap_deviation(data: OHLCV) -> FloatArray:
    """Fractional deviation of close from intraday VWAP."""
    reference = vwap(data)
    with np.errstate(invalid="ignore", divide="ignore"):
        out: FloatArray = np.where(reference > 0, data.close / reference - 1.0, np.nan)
    return out


def _daily_hlc(data: OHLCV) -> dict[date, tuple[float, float, float]]:
    """Per-date (high, low, last-close) for the completed prior-day pivot inputs."""
    result: dict[date, tuple[float, float, float]] = {}
    for i, ts in enumerate(data.timestamps):
        day = ts.date()
        high, low, close = float(data.high[i]), float(data.low[i]), float(data.close[i])
        if day not in result:
            result[day] = (high, low, close)
        else:
            prev_high, prev_low, _ = result[day]
            result[day] = (max(prev_high, high), min(prev_low, low), close)
    return result


def pivot(data: OHLCV) -> FloatArray:
    """Classic daily pivot from the PRIOR trading day's high/low/close.

    Constant within a day and known at the open (prior day is complete), so it is
    strictly point-in-time. The first day in the series is NaN (no prior day).
    """
    out = _empty(len(data))
    daily = _daily_hlc(data)
    ordered_dates = list(daily.keys())
    prior_by_date = {ordered_dates[k]: ordered_dates[k - 1] for k in range(1, len(ordered_dates))}
    for i, ts in enumerate(data.timestamps):
        day = ts.date()
        prior = prior_by_date.get(day)
        if prior is not None:
            high, low, close = daily[prior]
            out[i] = (high + low + close) / 3.0
    return out


def opening_range(data: OHLCV, window_minutes: int) -> tuple[FloatArray, FloatArray]:
    """Opening-range high/low over the first ``window_minutes`` of each session.

    While the window is forming it is the running high/low since the open; once
    the window closes it is fixed. Strictly causal.
    """
    or_high = _empty(len(data))
    or_low = _empty(len(data))
    session_open: dict[date, datetime] = {}
    for ts in data.timestamps:
        session_open.setdefault(ts.date(), ts)
    window = timedelta(minutes=window_minutes)
    run_high = -np.inf
    run_low = np.inf
    current_date = None
    for i, ts in enumerate(data.timestamps):
        day = ts.date()
        if day != current_date:
            current_date = day
            run_high = -np.inf
            run_low = np.inf
        if ts < session_open[day] + window:
            run_high = max(run_high, float(data.high[i]))
            run_low = min(run_low, float(data.low[i]))
        or_high[i] = run_high
        or_low[i] = run_low
    return or_high, or_low


def gap(data: OHLCV) -> FloatArray:
    """Overnight gap: (day's first open - prior day's last close) / prior close.

    Constant within a day, known at the open. First day is NaN.
    """
    out = _empty(len(data))
    first_open: dict[date, float] = {}
    last_close: dict[date, float] = {}
    for i, ts in enumerate(data.timestamps):
        day = ts.date()
        if day not in first_open:
            first_open[day] = float(data.open[i])
        last_close[day] = float(data.close[i])
    ordered_dates = list(first_open.keys())
    gap_by_date: dict[date, float] = {}
    for k in range(1, len(ordered_dates)):
        prior_close = last_close[ordered_dates[k - 1]]
        if prior_close > 0:
            gap_by_date[ordered_dates[k]] = first_open[ordered_dates[k]] / prior_close - 1.0
    for i, ts in enumerate(data.timestamps):
        value = gap_by_date.get(ts.date())
        if value is not None:
            out[i] = value
    return out


# --- additional Layer-1 families (P1.5 completion) -------------------------- #
def plus_di(data: OHLCV, period: int) -> FloatArray:
    """Positive directional indicator (+DI) of the DMI."""
    result: FloatArray = talib.PLUS_DI(data.high, data.low, data.close, timeperiod=period)
    return result


def minus_di(data: OHLCV, period: int) -> FloatArray:
    """Negative directional indicator (-DI) of the DMI."""
    result: FloatArray = talib.MINUS_DI(data.high, data.low, data.close, timeperiod=period)
    return result


def parkinson_volatility(data: OHLCV, period: int) -> FloatArray:
    """Rolling Parkinson high-low range volatility estimator."""
    out = _empty(len(data))
    log_hl_sq = np.log(data.high / data.low) ** 2
    if len(data) >= period:
        windows = sliding_window_view(log_hl_sq, period)
        factor = 1.0 / (4.0 * period * np.log(2.0))
        out[period - 1 :] = np.sqrt(factor * windows.sum(axis=1))
    return out


def garman_klass_volatility(data: OHLCV, period: int) -> FloatArray:
    """Rolling Garman-Klass OHLC volatility estimator."""
    out = _empty(len(data))
    term = (
        0.5 * np.log(data.high / data.low) ** 2
        - (2.0 * np.log(2.0) - 1.0) * np.log(data.close / data.open) ** 2
    )
    if len(data) >= period:
        windows = sliding_window_view(term, period)
        out[period - 1 :] = np.sqrt(np.clip(windows.mean(axis=1), 0.0, None))
    return out


def atr_bands(data: OHLCV, period: int, num_atr: float) -> tuple[FloatArray, FloatArray]:
    """ATR-based bands/stops: close +/- num_atr * ATR."""
    band = atr(data, period) * num_atr
    return data.close + band, data.close - band


def momentum(data: OHLCV, period: int) -> FloatArray:
    """Rate of change of close over ``period`` bars."""
    out = _empty(len(data))
    if len(data) > period:
        out[period:] = data.close[period:] / data.close[:-period] - 1.0
    return out


def pullback_depth(data: OHLCV, period: int) -> FloatArray:
    """Fractional pullback of close from the rolling ``period``-bar high."""
    rolling_high = _rolling(data.high, period, use_max=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        out: FloatArray = np.where(
            rolling_high > 0, (rolling_high - data.close) / rolling_high, np.nan
        )
    return out


def _confirmed_swing(values: FloatArray, window: int, *, high: bool) -> FloatArray:
    """Last confirmed swing level (point-in-time: a swing is confirmed ``window`` bars later)."""
    out = _empty(len(values))
    last = float("nan")
    for i in range(len(values)):
        pivot_idx = i - window
        if pivot_idx - window >= 0:
            segment = values[pivot_idx - window : pivot_idx + window + 1]
            center = float(values[pivot_idx])
            extreme = float(segment.max()) if high else float(segment.min())
            if center == extreme and int(np.sum(segment == center)) == 1:
                last = center
        out[i] = last
    return out


def swing_high(data: OHLCV, window: int) -> FloatArray:
    """Most recent confirmed swing-high price (uses only past bars)."""
    return _confirmed_swing(data.high, window, high=True)


def swing_low(data: OHLCV, window: int) -> FloatArray:
    """Most recent confirmed swing-low price (uses only past bars)."""
    return _confirmed_swing(data.low, window, high=False)


def engulfing(data: OHLCV) -> FloatArray:
    """Bullish/bearish engulfing candlestick signal (TA-Lib CDLENGULFING)."""
    result: FloatArray = talib.CDLENGULFING(data.open, data.high, data.low, data.close).astype(
        np.float64
    )
    return result


def doji(data: OHLCV) -> FloatArray:
    """Doji candlestick signal (TA-Lib CDLDOJI)."""
    result: FloatArray = talib.CDLDOJI(data.open, data.high, data.low, data.close).astype(
        np.float64
    )
    return result


def time_of_day_encoding(data: OHLCV) -> tuple[FloatArray, FloatArray]:
    """Cyclical (sin, cos) encoding of the bar's time of day."""
    minutes = np.array([ts.hour * 60 + ts.minute for ts in data.timestamps], dtype=np.float64)
    angle = 2.0 * np.pi * minutes / (24.0 * 60.0)
    return np.sin(angle), np.cos(angle)


def trend_regime(data: OHLCV, fast: int, slow: int) -> FloatArray:
    """Trend-regime sign: +1 fast MA above slow MA, -1 below (NaN in warmup)."""
    result: FloatArray = np.sign(sma(data, fast) - sma(data, slow))
    return result


def _prior_day_lookup(
    data: OHLCV,
) -> tuple[dict[date, tuple[float, float, float]], dict[date, date]]:
    daily = _daily_hlc(data)
    ordered = list(daily)
    prior_by_date = {ordered[k]: ordered[k - 1] for k in range(1, len(ordered))}
    return daily, prior_by_date


def fibonacci_pivot_levels(data: OHLCV) -> tuple[FloatArray, FloatArray]:
    """Fibonacci pivot R1/S1 from the prior day's HLC (point-in-time)."""
    out_r1 = _empty(len(data))
    out_s1 = _empty(len(data))
    daily, prior_by_date = _prior_day_lookup(data)
    for i, ts in enumerate(data.timestamps):
        prior = prior_by_date.get(ts.date())
        if prior is not None:
            high, low, close = daily[prior]
            pivot_point = (high + low + close) / 3.0
            span = high - low
            out_r1[i] = pivot_point + 0.382 * span
            out_s1[i] = pivot_point - 0.382 * span
    return out_r1, out_s1


def camarilla_pivot_levels(data: OHLCV) -> tuple[FloatArray, FloatArray]:
    """Camarilla pivot R1/S1 from the prior day's HLC (point-in-time)."""
    out_r1 = _empty(len(data))
    out_s1 = _empty(len(data))
    daily, prior_by_date = _prior_day_lookup(data)
    for i, ts in enumerate(data.timestamps):
        prior = prior_by_date.get(ts.date())
        if prior is not None:
            high, low, close = daily[prior]
            span = high - low
            out_r1[i] = close + 1.1 * span / 12.0
            out_s1[i] = close - 1.1 * span / 12.0
    return out_r1, out_s1


def classic_pivot_levels(data: OHLCV) -> tuple[FloatArray, FloatArray]:
    """Classic pivot R1/S1 from the prior day's HLC (point-in-time).

    ``R1 = 2P - prevLow``, ``S1 = 2P - prevHigh`` where ``P = (prevH + prevL + prevC) / 3`` --
    the classic support/resistance levels, constant within a day and known at the open
    (prior day complete), first day NaN. Reuses the same prior-completed-day lookup as
    :func:`pivot` / :func:`fibonacci_pivot_levels` / :func:`camarilla_pivot_levels`, so it
    is causal by construction: it reads only the PRIOR day's HLC, never the current day's,
    so there is no same-day leak.
    """
    out_r1 = _empty(len(data))
    out_s1 = _empty(len(data))
    daily, prior_by_date = _prior_day_lookup(data)
    for i, ts in enumerate(data.timestamps):
        prior = prior_by_date.get(ts.date())
        if prior is not None:
            high, low, close = daily[prior]
            pivot_point = (high + low + close) / 3.0
            out_r1[i] = 2.0 * pivot_point - low
            out_s1[i] = 2.0 * pivot_point - high
    return out_r1, out_s1


def cross_sectional_rank(values_by_symbol: dict[str, FloatArray]) -> dict[str, FloatArray]:
    """Point-in-time cross-sectional rank of each symbol at each timestamp.

    Ranks are in [0, 1] per timestamp across the panel (higher value -> higher
    rank); uses only same-timestamp values, so it is not lookahead. Panel arrays
    must be aligned and equal length. NaNs are ignored in the ranking.
    """
    symbols = list(values_by_symbol)
    if not symbols:
        return {}
    matrix = np.vstack([values_by_symbol[s] for s in symbols])  # (n_symbols, T)
    ranks = np.full_like(matrix, np.nan)
    for t in range(matrix.shape[1]):
        column = matrix[:, t]
        valid = ~np.isnan(column)
        count = int(valid.sum())
        if count >= 2:
            order = column[valid].argsort().argsort().astype(np.float64)
            ranks[valid, t] = order / (count - 1)
    return {symbol: ranks[i] for i, symbol in enumerate(symbols)}
