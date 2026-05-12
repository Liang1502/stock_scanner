#!/usr/bin/env python3
"""
AI 產業鏈三策略回測

Strategy A 趨勢跟隨：子題材前3 + 收>MA20 + RSI<72 + 5日漲<15% + 法人20日累積>0
Strategy B 起漲預判：子題材前5 + 法人剛轉買加速(3日買>前期) + 量能初動1.2x + RSI 45-65 + 5日漲<8% + 子題材落後股
Strategy C 趨勢+加速：子題材前5 + 收>MA20 + RSI 50-70 + 5日漲<12% + 法人20日>0 AND 近3日>0 + 量>1.1x + 落後股

共用：
  進場 T+1 開盤 / 出場 trailing stop 8% + 時間停損 20 天
  成本 swing 0.38% / 每檔 100,000 / 上限 20 檔同時持有

執行：
  python3 backtest_ai.py                 # 跑 A + B + C
  python3 backtest_ai.py --strategy A    # 只跑 A
  python3 backtest_ai.py --strategy B
  python3 backtest_ai.py --strategy C
  python3 backtest_ai.py --days 400
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from indicators import (
    atr_pct,
    bollinger_position,
    high_volume_breakout,
    macd_histogram,
    momentum,
    moving_average_slope_pct,
    range_position,
    wilder_rsi,
)
from indicator_trial import indicator_trial_score
from strategy_config import (
    A_EXCLUDE_SECTORS,
    A_MAX_RSI,
    A_MAX_RUN5,
    A_MIN_INST,
    A_SECTOR_TOP,
    B_MAX_RUN5,
    B_RSI_HI,
    B_RSI_LO,
    B_SECTOR_TOP,
    B_VOL_RATIO,
    BUY_FEE_RATE,
    C_MAX_RUN5,
    C_MIN_INST_20D,
    C_RSI_HI,
    C_RSI_LO,
    C_SECTOR_TOP,
    C_VOL_RATIO,
    D_EXCLUDE_SECTORS,
    D_MAX_RUN5,
    D_MIN_INST_20D,
    D_MIN_REV_YOY,
    D_MIN_SECTOR_MOM20,
    D_RSI_HI,
    D_RSI_LO,
    D_SECTOR_TOP,
    D_VOL_RATIO,
    CAPITAL_BY_STRATEGY,
    CAPITAL_PER_TRADE,
    A_ATR_STOP_MULT,
    A_ATR_TRAIL_MULT,
    A_ATR_TIGHTEN_MULT,
    INST_W_DEALER,
    INST_W_FOREIGN,
    INST_W_TRUST,
    ATR_STOP_MULT,
    ATR_TRAIL_MULT,
    ATR_TIGHTEN_MULT,
    INIT_STOP_PCT,
    INST_LOOKBACK,
    INST_PATH,
    KBAR_DIR,
    MAX_CONCURRENT,
    MAX_CONCURRENT_BY_STRATEGY,
    MAX_HOLD_DAYS,
    MIN_AVG_LOTS,
    MIN_REV_YOY,
    MIN_SECTOR_STOCKS,
    MOMENTUM_PERIODS,
    REV_PATH,
    SCORE_DELTA_DAYS,
    SELL_FEE_RATE,
    TAX_RATE,
    TARGET_PCT,
    TRAIL_PCT,
    TRAIL_TIGHTEN_PCT,
    UNIVERSE_PATH,
    W_DELTA,
    W_INST,
    W_MOM,
    W_VOL,
    net_return_rate,
)

load_dotenv()

# ── 路徑 ─────────────────────────────────────────────────────────
OUT_DIR       = Path("backtest_results")
OUT_DIR.mkdir(exist_ok=True)
B_INST_RECENT   = 3     # 法人近 N 日轉買
B_INST_PRIOR_LO = 4     # 前期開始天數
B_INST_PRIOR_HI = 15    # 前期結束天數


# ─────────────────────────────────────────────────────────────────
# 資料結構
# ─────────────────────────────────────────────────────────────────
@dataclass
class Trade:
    strategy:    str
    stock_id:    str
    name:        str
    sub_sector:  str
    entry_date:  str
    entry_price: float
    entry_atr:   float | None = None   # ATR% at entry, used for dynamic stops
    exit_date:   str = ""
    exit_price:  float = 0.0
    exit_reason: str = ""
    gross:       float = 0.0
    net:         float = 0.0
    hold_days:   int = 0


# ─────────────────────────────────────────────────────────────────
# 技術指標
# ─────────────────────────────────────────────────────────────────
def _rsi(closes: np.ndarray, period: int = 14) -> float:
    return wilder_rsi(closes, period)


def _momentum(closes: np.ndarray, n: int) -> float:
    return momentum(closes, n)


# ─────────────────────────────────────────────────────────────────
# 資料載入
# ─────────────────────────────────────────────────────────────────
def load_universe() -> pd.DataFrame:
    u = pd.read_csv(UNIVERSE_PATH)
    u["stock_id"] = u["stock_id"].astype(str)
    # 排除單股子題材
    cnt = u["sub_sector"].value_counts()
    valid = cnt[cnt >= MIN_SECTOR_STOCKS].index
    return u[u["sub_sector"].isin(valid)].copy()


def load_kbars(stocks: list[str]) -> dict[str, pd.DataFrame]:
    kbars = {}
    for sid in stocks:
        p = KBAR_DIR / f"{sid}.parquet"
        if p.exists():
            df = pd.read_parquet(p)
            df["date"] = df["date"].astype(str).str[:10]
            df = df.sort_values("date").reset_index(drop=True)
            df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0)
            kbars[sid] = df
    return kbars


def download_missing(stocks: list[str], kbars: dict) -> None:
    """嘗試用 SDK 下載缺少的 K 線並快取。"""
    missing = [s for s in stocks if s not in kbars]
    if not missing:
        return
    print(f"  補抓缺少 K 線：{len(missing)} 檔...")
    try:
        from fubon_neo.sdk import FubonSDK
        sdk = FubonSDK()
        sdk.login(
            os.environ["FUBON_ID"],
            os.environ["FUBON_PWD"],
            os.environ["FUBON_CERT_PATH"],
            os.environ.get("FUBON_CERT_PWD", ""),
        )
    except Exception as e:
        print(f"  SDK 失敗 ({e})，跳過 {len(missing)} 檔")
        return

    end = date.today()
    mid = end - timedelta(days=350)
    start = end - timedelta(days=700)
    for sid in missing:
        all_rows: list = []
        for (s, e_) in [(start, mid), (mid + timedelta(days=1), end)]:
            try:
                result = sdk.marketdata.rest_client.stock.historical.candles(
                    symbol=sid,
                    **{
                        "from": s.strftime("%Y-%m-%d"),
                        "to": e_.strftime("%Y-%m-%d"),
                        "fields": "open,high,low,close,volume",
                    }
                )
                data = result.get("data") if isinstance(result, dict) else getattr(result, "data", None)
                if data:
                    all_rows.extend(data)
                time.sleep(0.15)
            except Exception:
                pass
        if not all_rows:
            continue
        df = pd.DataFrame(all_rows)
        df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
        df = df.sort_values("date").drop_duplicates("date").reset_index(drop=True)
        df.to_parquet(KBAR_DIR / f"{sid}.parquet", index=False)
        kbars[sid] = df
    print(f"  補抓完成，有效 {len(kbars)}/{len(stocks)} 檔")


# ─────────────────────────────────────────────────────────────────
# 子題材輪動評分（給定某一天的日期）
# ─────────────────────────────────────────────────────────────────
def _raw_sector_scores(
    d_str: str,
    universe: pd.DataFrame,
    kbars: dict[str, pd.DataFrame],
    inst_df: pd.DataFrame,
) -> dict[str, dict]:
    """回傳 {sub_sector: {momentum, vol_ratio, inst_net}}，僅用 d_str 及之前資料。"""
    before = [d for d in sorted(inst_df["date"].unique()) if d <= d_str]
    cutoff = before[-INST_LOOKBACK] if len(before) >= INST_LOOKBACK else (before[0] if before else d_str)

    raw: dict[str, dict] = {}
    for sec, grp in universe.groupby("sub_sector"):
        stocks = grp["stock_id"].tolist()
        mom_vals, vol_vals = [], []

        for sid in stocks:
            df = kbars.get(sid)
            if df is None:
                continue
            sub = df[df["date"] <= d_str]
            if len(sub) < 22:
                continue
            closes  = sub["close"].values.astype(float)
            volumes = sub["volume"].values.astype(float)
            mom = sum(w * _momentum(closes, n)
                      for n, w in MOMENTUM_PERIODS if len(closes) > n)
            v5  = volumes[-5:].mean()
            v20 = volumes[-20:].mean()
            mom_vals.append(mom)
            vol_vals.append(float(v5 / v20) if v20 > 0 else 1.0)

        sec_inst = inst_df[
            (inst_df["stock_id"].isin(stocks)) &
            (inst_df["date"] >= cutoff) &
            (inst_df["date"] <= d_str)
        ]
        inst_net = int(
            INST_W_FOREIGN * sec_inst["foreign_net"].sum()
            + INST_W_TRUST  * sec_inst["trust_net"].sum()
            + INST_W_DEALER * sec_inst["dealer_net"].sum()
        )

        if not mom_vals:
            continue
        raw[sec] = {
            "momentum":  float(np.mean(mom_vals)),
            "vol_ratio": float(np.mean(vol_vals)),
            "inst_net":  inst_net,
        }
    return raw


def sector_scores_on(
    d_idx: int,
    dates: list[str],
    universe: pd.DataFrame,
    kbars: dict[str, pd.DataFrame],
    inst_df: pd.DataFrame,
) -> dict[str, float]:
    """回傳 {sub_sector: score}，包含評分加速度（score_delta）。"""
    d_str     = dates[d_idx]
    past_idx  = max(0, d_idx - SCORE_DELTA_DAYS)
    past_str  = dates[past_idx]

    raw_now  = _raw_sector_scores(d_str,   universe, kbars, inst_df)
    raw_past = _raw_sector_scores(past_str, universe, kbars, inst_df) if past_idx < d_idx else {}

    if not raw_now:
        return {}

    records = []
    for sec, v in raw_now.items():
        records.append({
            "sub_sector": sec,
            "momentum":   v["momentum"],
            "vol_ratio":  v["vol_ratio"],
            "inst_net":   v["inst_net"],
        })

    df_s = pd.DataFrame(records)
    # 基礎分：rank 化後加權
    scale = 1 - W_DELTA
    for col in ("momentum", "vol_ratio", "inst_net"):
        df_s[f"{col}_r"] = df_s[col].rank(pct=True)
    df_s["base_score"] = (W_MOM * df_s["momentum_r"]
                          + W_INST * df_s["inst_net_r"]
                          + W_VOL  * df_s["vol_ratio_r"])

    # 加速度：本週評分 - N 日前評分（用 base_score 近似）
    if raw_past:
        past_records = [{"sub_sector": s, "momentum": v["momentum"],
                         "vol_ratio": v["vol_ratio"], "inst_net": v["inst_net"]}
                        for s, v in raw_past.items()]
        df_past = pd.DataFrame(past_records)
        for col in ("momentum", "vol_ratio", "inst_net"):
            df_past[f"{col}_r"] = df_past[col].rank(pct=True)
        df_past["base_score"] = (W_MOM * df_past["momentum_r"]
                                 + W_INST * df_past["inst_net_r"]
                                 + W_VOL  * df_past["vol_ratio_r"])
        past_map = dict(zip(df_past["sub_sector"], df_past["base_score"]))
        df_s["delta"] = df_s.apply(
            lambda r: r["base_score"] - past_map.get(r["sub_sector"], r["base_score"]), axis=1
        )
        df_s["delta_r"] = df_s["delta"].rank(pct=True)
    else:
        df_s["delta_r"] = 0.5  # 無歷史時中性

    df_s["score"] = scale * df_s["base_score"] + W_DELTA * df_s["delta_r"]
    return dict(zip(df_s["sub_sector"], df_s["score"]))


# ─────────────────────────────────────────────────────────────────
# 個股訊號
# ─────────────────────────────────────────────────────────────────
def _revenue_release_date(year: int, month: int) -> date:
    if month == 12:
        return date(year + 1, 1, 10)
    return date(year, month + 1, 10)


def _available_revenue(rev_df: pd.DataFrame, sid: str, asof: str) -> pd.DataFrame:
    if rev_df.empty:
        return pd.DataFrame()
    s_rev = rev_df[rev_df["stock_id"] == sid].copy()
    if s_rev.empty:
        return s_rev
    asof_date = date.fromisoformat(asof[:10])
    release_dates = [
        _revenue_release_date(int(y), int(m))
        for y, m in zip(s_rev["year"], s_rev["month"])
    ]
    s_rev = s_rev[pd.Series(release_dates, index=s_rev.index) <= asof_date]
    return s_rev.sort_values(["year", "month"])


def _stock_ctx(sid: str, d_str: str, kbars: dict, inst_df: pd.DataFrame, rev_df: pd.DataFrame):
    """取出某股在某日的技術/法人/營收快照，回傳 dict 或 None。"""
    df = kbars.get(sid)
    if df is None:
        return None
    sub = df[df["date"] <= d_str]
    if len(sub) < 22:
        return None

    closes  = sub["close"].values.astype(float)
    volumes = sub["volume"].values.astype(float)
    highs   = sub["high"].values.astype(float)
    lows    = sub["low"].values.astype(float)
    opens   = sub["open"].values.astype(float)

    avg_lots = volumes[-20:].mean() / 1000
    if avg_lots < MIN_AVG_LOTS:
        return None

    ma20    = closes[-20:].mean()
    rsi_val = _rsi(closes)
    run5    = _momentum(closes, 5)
    run20   = _momentum(closes, 20)
    v5      = volumes[-5:].mean()
    v20     = volumes[-20:].mean()
    vol_ratio = float(v5 / v20) if v20 > 0 else 1.0
    close_now = float(closes[-1])
    high_now = float(highs[-1])
    low_now = float(lows[-1])
    close_range_pos = (close_now - low_now) / (high_now - low_now) if high_now > low_now else 0.5
    atr_val = atr_pct(highs, lows, closes)
    macd_val = macd_histogram(closes)
    bb_pos = bollinger_position(closes)
    ma20_slope = moving_average_slope_pct(closes, 20)
    ma60_slope = moving_average_slope_pct(closes, 60) if len(closes) >= 65 else None
    pos_52w = range_position(highs, lows, close_now, 252) if len(sub) >= 252 else None
    high_vol = high_volume_breakout(closes, volumes)
    # 取 T+1 開盤（下一根 K 棒的 open 與日期）
    nxt_open = None
    nxt_date = None
    full_sub = df[df["date"] > d_str]
    if not full_sub.empty:
        nxt_open = float(full_sub.iloc[0]["open"])
        nxt_date = str(full_sub.iloc[0]["date"])[:10]

    # 法人
    d_inst = inst_df[inst_df["stock_id"] == sid]

    def _inst_sum(from_d: str, to_d: str) -> int:
        mask = (d_inst["date"] >= from_d) & (d_inst["date"] <= to_d)
        s = d_inst[mask]
        return int(
            INST_W_FOREIGN * s["foreign_net"].sum()
            + INST_W_TRUST  * s["trust_net"].sum()
            + INST_W_DEALER * s["dealer_net"].sum()
        )

    # 日期計算輔助（只用已知 trading dates）
    dates_before = sorted(d_inst[d_inst["date"] <= d_str]["date"].unique())

    def _nth_date(n: int) -> str | None:
        """n 天前的法人資料日期"""
        return dates_before[-n] if len(dates_before) >= n else None

    d_3  = _nth_date(3)
    d_4  = _nth_date(4)
    d_15 = _nth_date(15)
    d_20 = _nth_date(20)

    inst_3d    = _inst_sum(d_3,  d_str) if d_3  else 0
    inst_4_15  = _inst_sum(d_15, d_4)   if d_4 and d_15 else 0
    inst_20d   = _inst_sum(d_20, d_str) if d_20 else _inst_sum(dates_before[0], d_str) if dates_before else 0

    # 是否剛突破 MA20（3~7 天前在 MA20 以下）
    close_7ago = closes[-8] if len(closes) >= 8 else closes[0]
    ma20_7ago  = closes[-27:-7].mean() if len(closes) >= 27 else ma20
    crossed_ma = (close_now > ma20) and (close_7ago < ma20_7ago)

    # 營收 YoY：上月營收最晚次月 10 日公布，回測只能使用 as-of 已發布資料。
    s_rev = _available_revenue(rev_df, sid, d_str)
    rev_yoy = float(s_rev["yoy_pct"].iloc[-1]) if not s_rev.empty else None

    return {
        "close":       close_now,
        "ma20":        float(ma20),
        "rsi":         rsi_val,
        "run5":        run5,
        "run20":       run20,
        "vol_ratio":   vol_ratio,
        "avg_lots":    avg_lots,
        "close_range_pos": close_range_pos,
        "atr_pct":     atr_val,
        "macd_hist":   macd_val,
        "bb_pos":      bb_pos,
        "ma20_slope_pct": ma20_slope,
        "ma60_slope_pct": ma60_slope,
        "pos_52w":     pos_52w,
        "high_vol_breakout": high_vol,
        "inst_3d":     inst_3d,
        "inst_4_15":   inst_4_15,
        "inst_20d":    inst_20d,
        "crossed_ma":  crossed_ma,
        "rev_yoy":     rev_yoy,
        "nxt_open":    nxt_open,
        "nxt_date":    nxt_date,
    }


def signal_A(ctx: dict) -> bool:
    """Strategy A：趨勢跟隨"""
    if ctx["close"] < ctx["ma20"]:
        return False
    if ctx["rsi"] > A_MAX_RSI:
        return False
    if ctx["run5"] > A_MAX_RUN5:
        return False
    if ctx["inst_20d"] < A_MIN_INST:
        return False
    if ctx["rev_yoy"] is not None and ctx["rev_yoy"] < MIN_REV_YOY:
        return False
    return True


def signal_B(ctx: dict, sector_avg_run20: float) -> bool:
    """Strategy B：起漲預判"""
    # RSI 中段，未超買
    if not (B_RSI_LO <= ctx["rsi"] <= B_RSI_HI):
        return False
    # 尚未追高
    if ctx["run5"] > B_MAX_RUN5:
        return False
    # 收盤要在 MA20 以上（基本趨勢確認）
    if ctx["close"] < ctx["ma20"]:
        return False
    # 量能初動
    if ctx["vol_ratio"] < B_VOL_RATIO:
        return False
    # 法人剛轉買或加速：近3日>0
    if ctx["inst_3d"] <= 0:
        return False
    # 前期若也在買，要求近期買超至少2倍（加速）才算有效訊號
    if ctx["inst_4_15"] > 0 and ctx["inst_3d"] <= ctx["inst_4_15"] * 2:
        return False
    # 子題材相對落後股（個股漲幅 < 子題材平均）
    if ctx["run20"] >= sector_avg_run20:
        return False
    # 營收
    if ctx["rev_yoy"] is not None and ctx["rev_yoy"] < MIN_REV_YOY:
        return False
    return True


def signal_C(ctx: dict, sector_avg_run20: float) -> bool:
    """Strategy C：趨勢+加速（A 的趨勢確認 + B 的法人加速）"""
    # 趨勢確認
    if ctx["close"] < ctx["ma20"]:
        return False
    # RSI 中高段，未超買
    if not (C_RSI_LO <= ctx["rsi"] <= C_RSI_HI):
        return False
    # 未大幅追高
    if ctx["run5"] > C_MAX_RUN5:
        return False
    # 量能初動
    if ctx["vol_ratio"] < C_VOL_RATIO:
        return False
    # 法人20日整體偏多（趨勢面）
    if ctx["inst_20d"] < C_MIN_INST_20D:
        return False
    # 近3日也在買（動能面）
    if ctx["inst_3d"] <= 0:
        return False
    # 子題材相對落後股
    if ctx["run20"] >= sector_avg_run20:
        return False
    # 營收
    if ctx["rev_yoy"] is not None and ctx["rev_yoy"] < MIN_REV_YOY:
        return False
    return True


def signal_D(ctx: dict, sector_avg_run20: float) -> bool:
    """Strategy D：C 訊號條件 + 限定 top-3 題材（汰弱留強集中火力）

    sector_top=3 由 run_backtest 控制，signal_D 條件同 C。
    """
    # 子題材絕對動能：整個題材需正在上漲，避免選空頭題材中的「最強弱雞」
    if sector_avg_run20 < D_MIN_SECTOR_MOM20:
        return False
    if ctx["close"] < ctx["ma20"]:
        return False
    if not (D_RSI_LO <= ctx["rsi"] <= D_RSI_HI):
        return False
    if ctx["run5"] > D_MAX_RUN5:
        return False
    if ctx["vol_ratio"] < D_VOL_RATIO:
        return False
    if ctx["inst_20d"] < D_MIN_INST_20D:
        return False
    if ctx["inst_3d"] <= 0:
        return False
    if ctx["run20"] >= sector_avg_run20:
        return False
    if ctx["rev_yoy"] is not None and ctx["rev_yoy"] < D_MIN_REV_YOY:
        return False
    return True


# ─────────────────────────────────────────────────────────────────
# 交易模擬
# ─────────────────────────────────────────────────────────────────
def _atr_mults(strategy: str) -> tuple[float, float, float]:
    """Return (stop_mult, trail_mult, tighten_mult) for the given strategy."""
    if strategy == "A":
        return A_ATR_STOP_MULT, A_ATR_TRAIL_MULT, A_ATR_TIGHTEN_MULT
    return ATR_STOP_MULT, ATR_TRAIL_MULT, ATR_TIGHTEN_MULT


def simulate_hold(
    trade: Trade,
    entry_idx: int,
    dates: list[str],
    kbars: dict[str, pd.DataFrame],
    max_hold_days: int = MAX_HOLD_DAYS,
) -> Trade:
    """從 entry_idx（T+1 開盤）起持有，套用追蹤停損 + 時間停損。"""
    df = kbars[trade.stock_id]
    sub = df[df["date"] > trade.entry_date].reset_index(drop=True)
    if sub.empty:
        trade.exit_date   = trade.entry_date
        trade.exit_price  = trade.entry_price
        trade.exit_reason = "data_end"
        return _calc_pnl(trade)

    s_mult, tr_mult, tg_mult = _atr_mults(trade.strategy)
    atr_w = (trade.entry_atr or 0.0) * trade.entry_price  # ATR in price units
    if atr_w > 0:
        stop = trade.entry_price - s_mult * atr_w
    else:
        stop = trade.entry_price * (1 - INIT_STOP_PCT)
    peak  = trade.entry_price

    for i, row in sub.iterrows():
        hold = i + 1
        low_  = float(row["low"])
        high_ = float(row["high"])
        close = float(row["close"])
        open_ = float(row["open"])

        # 跳空停損
        if open_ < stop:
            trade.exit_date   = str(row["date"])[:10]
            trade.exit_price  = open_
            trade.exit_reason = "gap_stop"
            return _calc_pnl(trade)

        # 追蹤停損（ATR-based，達目標後收緊）
        if high_ > peak:
            peak = high_
        if atr_w > 0:
            trail_stop = peak - (tg_mult if peak >= trade.entry_price * (1 + TARGET_PCT) else tr_mult) * atr_w
        else:
            trail_pct  = TRAIL_TIGHTEN_PCT if peak >= trade.entry_price * (1 + TARGET_PCT) else TRAIL_PCT
            trail_stop = peak * (1 - trail_pct)
        stop = max(stop, trail_stop)

        if low_ < stop:
            trade.exit_date   = str(row["date"])[:10]
            trade.exit_price  = round(stop, 1)
            trade.exit_reason = "trail_stop"
            return _calc_pnl(trade)

        # 時間停損
        if hold >= max_hold_days:
            trade.exit_date   = str(row["date"])[:10]
            trade.exit_price  = close
            trade.exit_reason = "time_stop"
            return _calc_pnl(trade)

    # 資料結尾
    last = sub.iloc[-1]
    trade.exit_date   = str(last["date"])[:10]
    trade.exit_price  = float(last["close"])
    trade.exit_reason = "data_end"
    return _calc_pnl(trade)


def _calc_pnl(trade: Trade) -> Trade:
    gross = (trade.exit_price - trade.entry_price) / trade.entry_price
    net = net_return_rate(trade.entry_price, trade.exit_price)

    trade.gross     = round(gross, 4)
    trade.net       = round(net, 4)
    trade.hold_days = max(
        1,
        (date.fromisoformat(trade.exit_date) - date.fromisoformat(trade.entry_date)).days
    )

    return trade

# ─────────────────────────────────────────────────────────────────
# 主回測迴圈
# ─────────────────────────────────────────────────────────────────
def run_backtest(
    strategy: Literal["A", "B", "C", "D"],
    universe: pd.DataFrame,
    kbars: dict[str, pd.DataFrame],
    inst_df: pd.DataFrame,
    rev_df: pd.DataFrame,
    start_date: str,
    end_date: str = "",
    silent: bool = False,
    max_hold_days: int = MAX_HOLD_DAYS,
    indicator_trial: bool = False,
) -> list[Trade]:

    # 全部交易日
    all_dates = sorted(set(
        d for df in kbars.values() for d in df["date"].tolist()
    ))
    all_dates = [d for d in all_dates if d >= start_date]
    if end_date:
        all_dates = [d for d in all_dates if d <= end_date]
    if not all_dates:
        return []

    name_map = universe.set_index("stock_id")["name"].to_dict()
    sec_map  = universe.set_index("stock_id")["sub_sector"].to_dict()

    # 快取當日評分（避免每 tick 重算）
    sector_top = A_SECTOR_TOP if strategy == "A" else (D_SECTOR_TOP if strategy == "D" else (C_SECTOR_TOP if strategy == "C" else B_SECTOR_TOP))
    max_conc   = MAX_CONCURRENT_BY_STRATEGY.get(strategy, MAX_CONCURRENT)
    score_cache: dict[str, dict] = {}

    open_positions: dict[str, Trade] = {}   # stock_id → Trade
    closed_trades:  list[Trade] = []

    if not silent:
        print(f"\n[Strategy {strategy}] 回測 {start_date} ~ {all_dates[-1]}，共 {len(all_dates)} 個交易日")

    for day_i, d_str in enumerate(all_dates):
        # ── 更新未平倉（提前平倉）─────────────────────────────────
        to_close = []
        for sid, tr in open_positions.items():
            df = kbars[sid]
            row = df[df["date"] == d_str]
            if row.empty:
                continue
            open_ = float(row.iloc[0]["open"])
            low_ = float(row.iloc[0]["low"])
            close = float(row.iloc[0]["close"])

            # 跳空停損：與昨日有效停損（初始停損 or 前日追蹤停損）比較
            s_mult, tr_mult, tg_mult = _atr_mults(tr.strategy)
            atr_w = (tr.entry_atr or 0.0) * tr.entry_price
            prev_rows = df[(df["date"] >= tr.entry_date) & (df["date"] < d_str)]
            if not prev_rows.empty:
                prev_peak = float(prev_rows["high"].max())
                if atr_w > 0:
                    prev_mult = tg_mult if prev_peak >= tr.entry_price * (1 + TARGET_PCT) else tr_mult
                    prev_stop = max(tr.entry_price - s_mult * atr_w, prev_peak - prev_mult * atr_w)
                else:
                    prev_trail_pct = TRAIL_TIGHTEN_PCT if prev_peak >= tr.entry_price * (1 + TARGET_PCT) else TRAIL_PCT
                    prev_stop = max(tr.entry_price * (1 - INIT_STOP_PCT), prev_peak * (1 - prev_trail_pct))
            else:
                prev_stop = (tr.entry_price - s_mult * atr_w) if atr_w > 0 else tr.entry_price * (1 - INIT_STOP_PCT)
            if open_ < prev_stop:
                tr.exit_date   = d_str
                tr.exit_price  = open_
                tr.exit_reason = "gap_stop"
                to_close.append(sid)
                continue

            # 追蹤停損（以本日收盤更新）
            peak = float(df[
                (df["date"] >= tr.entry_date) & (df["date"] <= d_str)
            ]["high"].max())
            if atr_w > 0:
                trail_stop = peak - (tg_mult if peak >= tr.entry_price * (1 + TARGET_PCT) else tr_mult) * atr_w
                init_stop  = tr.entry_price - s_mult * atr_w
            else:
                trail_pct  = TRAIL_TIGHTEN_PCT if peak >= tr.entry_price * (1 + TARGET_PCT) else TRAIL_PCT
                trail_stop = peak * (1 - trail_pct)
                init_stop  = tr.entry_price * (1 - INIT_STOP_PCT)
            stop       = max(init_stop, trail_stop)

            if low_ < stop:
                tr.exit_date   = d_str
                tr.exit_price  = round(stop, 1)
                tr.exit_reason = "trail_stop"
                to_close.append(sid)
                continue

            # 時間停損
            held = (date.fromisoformat(d_str) - date.fromisoformat(tr.entry_date)).days
            if held >= max_hold_days:
                tr.exit_date   = d_str
                tr.exit_price  = close
                tr.exit_reason = "time_stop"
                to_close.append(sid)

        for sid in to_close:
            tr = open_positions.pop(sid)
            _calc_pnl(tr)
            closed_trades.append(tr)

        if len(open_positions) >= max_conc:
            continue

        # ── 子題材評分（快取，每日計算一次）──────────────────────
        if d_str not in score_cache:
            scores = sector_scores_on(
                all_dates.index(d_str), all_dates, universe, kbars, inst_df
            )
            top_secs = {
                s for s, sc in sorted(scores.items(), key=lambda x: -x[1])[:sector_top]
            }
            score_cache[d_str] = {"scores": scores, "top": top_secs}
        top_secs = score_cache[d_str]["top"]

        # ── 訊號掃描 ─────────────────────────────────────────────
        # 子題材平均 run20（Strategy B / C 用）
        sec_avg_run20: dict[str, float] = {}
        if strategy in ("B", "C", "D"):
            for sec in top_secs:
                stks = universe[universe["sub_sector"] == sec]["stock_id"].tolist()
                runs = []
                for sid in stks:
                    df = kbars.get(sid)
                    if df is None: continue
                    sub = df[df["date"] <= d_str]
                    if len(sub) >= 22:
                        runs.append(_momentum(sub["close"].values.astype(float), 20))
                sec_avg_run20[sec] = float(np.mean(runs)) if runs else 0.0

        daily_candidates: list[tuple[float, float, str, dict, str]] = []

        for _, row in universe.iterrows():
            sid = str(row["stock_id"])
            sec = str(row["sub_sector"])
            if sec not in top_secs:
                continue
            if strategy == "A" and sec in A_EXCLUDE_SECTORS:
                continue
            if strategy == "D" and sec in D_EXCLUDE_SECTORS:
                continue
            if sid in open_positions:
                continue
            if len(open_positions) >= max_conc:
                break

            ctx = _stock_ctx(sid, d_str, kbars, inst_df, rev_df)
            if ctx is None or ctx["nxt_open"] is None:
                continue

            triggered = False
            if strategy == "A":
                triggered = signal_A(ctx)
            elif strategy == "B":
                triggered = signal_B(ctx, sec_avg_run20.get(sec, 0.0))
            elif strategy == "D":
                triggered = signal_D(ctx, sec_avg_run20.get(sec, 0.0))
            else:
                triggered = signal_C(ctx, sec_avg_run20.get(sec, 0.0))

            if triggered:
                if indicator_trial:
                    daily_candidates.append(
                        (
                            float(indicator_trial_score(ctx)),
                            float(score_cache[d_str]["scores"].get(sec, 0.0)),
                            sid,
                            ctx,
                            sec,
                        )
                    )
                else:
                    tr = Trade(
                        strategy    = strategy,
                        stock_id    = sid,
                        name        = name_map.get(sid, ""),
                        sub_sector  = sec,
                        entry_date  = ctx["nxt_date"],   # T+1 實際進場日，而非訊號日
                        entry_price = ctx["nxt_open"],
                        entry_atr   = ctx.get("atr_pct"),
                    )
                    open_positions[sid] = tr

        if indicator_trial and daily_candidates:
            daily_candidates.sort(key=lambda x: (-x[0], -x[1], x[2]))
            for _, _, sid, ctx, sec in daily_candidates:
                if sid in open_positions:
                    continue
                if len(open_positions) >= MAX_CONCURRENT:
                    break
                tr = Trade(
                    strategy    = strategy,
                    stock_id    = sid,
                    name        = name_map.get(sid, ""),
                    sub_sector  = sec,
                    entry_date  = ctx["nxt_date"],
                    entry_price = ctx["nxt_open"],
                    entry_atr   = ctx.get("atr_pct"),
                )
                open_positions[sid] = tr

        if not silent and (day_i + 1) % 50 == 0:
            print(f"  進度 {day_i+1}/{len(all_dates)}，已入場 {len(closed_trades)+len(open_positions)} 筆")

    # 強制平倉未平倉
    last_d = all_dates[-1]
    for sid, tr in open_positions.items():
        df = kbars[sid]
        last_row = df[df["date"] <= last_d]
        if not last_row.empty:
            tr.exit_date   = str(last_row.iloc[-1]["date"])[:10]   # 用實際最後K棒日期
            tr.exit_price  = float(last_row.iloc[-1]["close"])
            tr.exit_reason = "data_end"
            _calc_pnl(tr)
            closed_trades.append(tr)

    return closed_trades


# ─────────────────────────────────────────────────────────────────
# 績效報告
# ─────────────────────────────────────────────────────────────────
def report(trades: list[Trade], strategy: str, start_date: str) -> dict:
    if not trades:
        print(f"\n[Strategy {strategy}] 無交易紀錄")
        return {}

    df = pd.DataFrame([t.__dict__ for t in trades])
    n     = len(df)
    wins  = df[df["net"] > 0]
    loss  = df[df["net"] <= 0]
    wr    = len(wins) / n
    avg_w = wins["net"].mean() * 100 if len(wins) else 0
    avg_l = loss["net"].mean() * 100 if len(loss) else 0
    ev    = df["net"].mean() * 100
    pf    = (wins["net"].sum() / -loss["net"].sum()) if loss["net"].sum() < 0 else float("inf")
    cap_per_trade = CAPITAL_BY_STRATEGY.get(strategy, CAPITAL_PER_TRADE)
    max_conc      = MAX_CONCURRENT_BY_STRATEGY.get(strategy, MAX_CONCURRENT)
    total_pnl = df["net"].sum() * cap_per_trade

    days  = max((date.fromisoformat(df["exit_date"].max()) -
                 date.fromisoformat(start_date)).days, 1)
    ann   = ((1 + total_pnl / (cap_per_trade * max_conc)) ** (365 / days) - 1) * 100

    # 最大回撤（從初始資本起算，prepend 0 使首筆虧損能正確計算）
    capital = cap_per_trade * max_conc
    df_s = df.sort_values("exit_date")
    cum = pd.concat([pd.Series([0.0]), (df_s["net"] * cap_per_trade).cumsum()]).reset_index(drop=True)
    roll_max = cum.cummax()
    dd = ((cum - roll_max) / capital * 100).min()

    print(f"\n{'═'*60}")
    print(f"  Strategy {strategy} 績效報告  ({start_date} ~ {df['exit_date'].max()})")
    print(f"{'═'*60}")
    print(f"  交易筆數     : {n}")
    print(f"  勝率         : {wr*100:.1f}%")
    print(f"  平均獲利     : {avg_w:+.2f}%")
    print(f"  平均虧損     : {avg_l:+.2f}%")
    print(f"  期望值/筆    : {ev:+.2f}%")
    print(f"  獲利因子     : {pf:.2f}")
    print(f"  累積 PnL     : {total_pnl:+,.0f} 元")
    print(f"  年化報酬     : {ann:+.2f}%")
    print(f"  最大回撤     : {dd:.2f}%")
    print(f"  平均持倉天數 : {df['hold_days'].mean():.1f}")

    print(f"\n  出場原因分佈：")
    for reason, cnt in df["exit_reason"].value_counts().items():
        print(f"    {reason:<20}: {cnt}")

    print(f"\n  子題材績效：")
    sec_g = df.groupby("sub_sector").agg(
        n=("net","count"), wr=("net", lambda x: (x>0).mean()),
        avg=("net","mean")
    ).sort_values("avg", ascending=False)
    for sec, row in sec_g.iterrows():
        print(f"    {sec:<12} n={row['n']:>3}  勝率{row['wr']*100:.0f}%  均報酬{row['avg']*100:+.2f}%")

    return {
        "strategy": strategy, "n": n, "win_rate": wr,
        "ev_pct": ev, "pf": pf, "total_pnl": total_pnl, "ann_pct": ann, "dd_pct": dd,
    }


# ─────────────────────────────────────────────────────────────────
# Walk-Forward 驗證
# ─────────────────────────────────────────────────────────────────
def walk_forward(
    strategy: str,
    universe: pd.DataFrame,
    kbars: dict,
    inst_df: pd.DataFrame,
    rev_df: pd.DataFrame,
    fold_months: int = 3,
    warmup_months: int = 3,
    max_hold_days: int = MAX_HOLD_DAYS,
    indicator_trial: bool = False,
) -> None:
    """
    滾動驗證：每 fold_months 個月為一個測試區間，
    前 warmup_months 個月為暖機期（不計入績效）。
    """
    from calendar import monthrange

    # 找出全資料的起訖月份
    all_dates = sorted({d for df in kbars.values() for d in df["date"].tolist()})
    if not all_dates:
        return
    data_start = date.fromisoformat(all_dates[0])
    data_end   = date.fromisoformat(all_dates[-1])

    # 暖機結束後才開始第一個 fold
    fold_start = date(data_start.year, data_start.month, 1)
    for _ in range(warmup_months):
        y, m = fold_start.year, fold_start.month + 1
        if m > 12: y, m = y + 1, 1
        fold_start = date(y, m, 1)

    print(f"\n{'═'*60}")
    print(f"  Walk-Forward  Strategy {strategy}  (fold={fold_months}月, 暖機={warmup_months}月)")
    print(f"{'═'*60}")
    print(f"  {'區間':<24} {'筆':>4}  {'勝率':>6}  {'EV/筆':>7}  {'PF':>5}  {'PnL':>10}")
    print(f"  {'─'*58}")

    fold_results = []
    while fold_start <= data_end:
        # fold 結束月
        fe = fold_start
        for _ in range(fold_months):
            y, m = fe.year, fe.month + 1
            if m > 12: y, m = y + 1, 1
            fe = date(y, m, 1)
        fold_end = date(fe.year, fe.month, 1) - timedelta(days=1)
        fold_end = min(fold_end, data_end)

        fs_str = fold_start.strftime("%Y-%m-%d")
        fe_str = fold_end.strftime("%Y-%m-%d")

        trades = run_backtest(strategy, universe, kbars, inst_df, rev_df,
                              start_date=fs_str, end_date=fe_str, silent=True,
                              max_hold_days=max_hold_days,
                              indicator_trial=indicator_trial)

        if trades:
            df_t = pd.DataFrame([t.__dict__ for t in trades])
            n_  = len(df_t)
            wr_ = (df_t["net"] > 0).mean()
            ev_ = df_t["net"].mean() * 100
            wins_ = df_t[df_t["net"] > 0]["net"].sum()
            loss_ = df_t[df_t["net"] <= 0]["net"].sum()
            pf_ = wins_ / -loss_ if loss_ < 0 else float("inf")
            pnl_ = df_t["net"].sum() * CAPITAL_BY_STRATEGY.get(strategy, CAPITAL_PER_TRADE)
            print(f"  {fs_str} ~ {fe_str}  {n_:>4}  {wr_*100:>5.1f}%  {ev_:>+6.2f}%  "
                  f"{min(pf_,99.9):>5.1f}  {pnl_:>+10,.0f}")
            fold_results.append({"period": f"{fs_str[:7]}~{fe_str[:7]}",
                                  "n": n_, "wr": wr_, "ev": ev_, "pf": pf_, "pnl": pnl_})
        else:
            print(f"  {fs_str} ~ {fe_str}  （無交易）")

        fold_start = fe  # 下一個 fold 從本 fold 結束月開始

    if fold_results:
        df_r = pd.DataFrame(fold_results)
        pos_ev = (df_r["ev"] > 0).sum()
        print(f"\n  EV>0 的 fold：{pos_ev}/{len(fold_results)}")
        print(f"  平均 EV/筆：{df_r['ev'].mean():+.2f}%   平均 PF：{df_r['pf'].replace([float('inf')], 10).mean():.2f}")


# ─────────────────────────────────────────────────────────────────
# 主程式
# ─────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategy",    choices=["A","B","C","D","all"], default="all")
    ap.add_argument("--days",        type=int, default=400, help="回測天數")
    ap.add_argument("--end-date",    type=str, default="", help="回測結束日（YYYY-MM-DD），預設用最新 K 線")
    ap.add_argument("--walkforward", action="store_true", help="執行 walk-forward 驗證")
    ap.add_argument("--fold",        type=int, default=3, help="walk-forward fold 大小（月）")
    ap.add_argument("--hold",        type=int, default=MAX_HOLD_DAYS, help=f"最大持倉天數（預設 {MAX_HOLD_DAYS}）")
    ap.add_argument("--indicator-trial", action="store_true", help="試跑指標條件：保留 not-overextended / breakout")
    args = ap.parse_args()

    start_date = (date.today() - timedelta(days=args.days)).strftime("%Y-%m-%d")
    # 法人資料從 2024-09-03 起，加熱身
    start_date = max(start_date, "2024-10-01")

    print("=" * 60)
    print(f"  AI 產業鏈三策略回測  ({start_date} ~ {date.today()})")
    print("=" * 60)

    universe = load_universe()
    all_stocks = universe["stock_id"].tolist()
    print(f"宇宙：{len(all_stocks)} 檔 / {universe['sub_sector'].nunique()} 個子題材")

    print("載入 K 線...")
    kbars = load_kbars(all_stocks)
    download_missing(all_stocks, kbars)
    print(f"  有效 {len(kbars)}/{len(all_stocks)} 檔")

    print("載入法人 / 營收...")
    inst_df = pd.read_parquet(INST_PATH)
    inst_df["stock_id"] = inst_df["stock_id"].astype(str)
    inst_df["date"] = inst_df["date"].astype(str).str[:10]
    inst_df = inst_df[inst_df["stock_id"].isin(all_stocks)].copy()

    rev_df = pd.read_parquet(REV_PATH) if REV_PATH.exists() else pd.DataFrame()
    if not rev_df.empty:
        rev_df["stock_id"] = rev_df["stock_id"].astype(str)
        rev_df["ym"] = rev_df["year"].astype(int) * 100 + rev_df["month"].astype(int)

    # ── Walk-Forward ──────────────────────────────────────────────
    if args.walkforward:
        strategies = ["A", "B", "C", "D"] if args.strategy == "all" else [args.strategy]
        for s in strategies:
            walk_forward(
                s,
                universe,
                kbars,
                inst_df,
                rev_df,
                fold_months=args.fold,
                max_hold_days=args.hold,
                indicator_trial=args.indicator_trial,
            )
        print()
        return

    # ── 執行回測 ──────────────────────────────────────────────────
    results = {}
    strategies = ["A", "B", "C"] if args.strategy == "all" else [args.strategy]

    for s in strategies:
        trades = run_backtest(
            s,
            universe,
            kbars,
            inst_df,
            rev_df,
            start_date,
            end_date=args.end_date,
            max_hold_days=args.hold,
            indicator_trial=args.indicator_trial,
        )
        r = report(trades, s, start_date)
        results[s] = (trades, r)

        # 儲存明細
        if trades:
            today_str = date.today().strftime("%Y%m%d")
            suffix = "_trial" if args.indicator_trial else ""
            out_path = OUT_DIR / f"ai_strategy_{s}{suffix}_{today_str}.csv"
            pd.DataFrame([t.__dict__ for t in trades]).to_csv(out_path, index=False)
            print(f"\n  明細 → {out_path}")

    # ── 對比摘要 ──────────────────────────────────────────────────
    active = [s for s in strategies if results.get(s, (None, {}))[1]]
    if len(active) >= 2:
        cols = active
        header = "  " + f"{'指標':<16}" + "".join(f"{'Strategy '+s:>14}" for s in cols)
        print(f"\n{'═'*60}")
        print("  策略對比")
        print(f"{'═'*60}")
        print(header)
        print(f"  {'─'*56}")
        keys = [("n","交易筆數","d"), ("win_rate","勝率","%"), ("ev_pct","期望值/筆","+.2f%"),
                ("pf","獲利因子",".2f"), ("total_pnl","累積PnL(元)","+,.0f"),
                ("ann_pct","年化報酬","+.2f%"), ("dd_pct","最大回撤",".2f%")]

        def _fmt(v, fmt):
            if v == "-": return f"{'−':>14}"
            if fmt == "d": return f"{int(v):>14}"
            if fmt == "%": return f"{v*100:>13.1f}%"
            if fmt == "+.2f%": return f"{v:>+13.2f}%"
            if fmt == ".2f": return f"{v:>14.2f}"
            if fmt == "+,.0f": return f"{v:>+13,.0f}"
            if fmt == ".2f%": return f"{v:>13.2f}%"
            return str(v)

        for key, label, fmt in keys:
            row = "  " + f"{label:<16}"
            for s in cols:
                v = results[s][1].get(key, "-")
                row += _fmt(v, fmt)
            print(row)
    print()


if __name__ == "__main__":
    main()
