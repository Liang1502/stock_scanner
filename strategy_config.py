from __future__ import annotations

from pathlib import Path

UNIVERSE_PATH = Path("ai_universe.csv")
INST_PATH = Path("backtest_cache/institutional.parquet")
REV_PATH = Path("backtest_cache/revenues.parquet")
KBAR_DIR = Path("backtest_cache")

CAPITAL_PER_TRADE = 250_000
MAX_CONCURRENT = 20

# 依策略分配資金（總資金 500萬）
# C 主力：50萬/筆 × 最多 6 檔 = 300萬上限，平均動用 ~125萬
# A 輔助：25萬/筆 × 最多 8 檔 = 200萬上限，平均動用 ~210萬
CAPITAL_BY_STRATEGY: dict[str, int] = {"A": 250_000, "B": 500_000, "C": 500_000, "D": 500_000}
MAX_CONCURRENT_BY_STRATEGY: dict[str, int] = {"A": 8, "B": 4, "C": 6, "D": 6}

INIT_STOP_PCT = 0.05
TRAIL_PCT = 0.08
TRAIL_TIGHTEN_PCT = 0.05   # 達到目標後收緊，保護獲利
MAX_HOLD_DAYS = 999
TARGET_PCT = 0.10

# ATR 動態停損（取代固定百分比停損）
# initial stop = entry_price - ATR_STOP_MULT * atr_price
# trailing stop = peak - ATR_TRAIL_MULT * atr_price
# 達目標後收緊為 ATR_TIGHTEN_MULT
ATR_STOP_MULT    = 2.0
ATR_TRAIL_MULT   = 2.5
ATR_TIGHTEN_MULT = 1.5

# 策略 A 的標的波動較高，用較小乘數避免停損過遠
A_ATR_STOP_MULT    = 1.5
A_ATR_TRAIL_MULT   = 2.0
A_ATR_TIGHTEN_MULT = 1.2

BUY_FEE_RATE = 0.000399
SELL_FEE_RATE = 0.000399
TAX_RATE = 0.003


def net_return_rate(entry_price: float, exit_price: float) -> float:
    gross = (exit_price - entry_price) / entry_price
    cost_rate = BUY_FEE_RATE + (exit_price / entry_price) * (SELL_FEE_RATE + TAX_RATE)
    return gross - cost_rate

# 三大法人訊號權重（研究依據：投信訊號對中型股最可靠，自營商雜訊高）
INST_W_TRUST   = 0.60
INST_W_FOREIGN = 0.30
INST_W_DEALER  = 0.10

MIN_SECTOR_STOCKS = 2
MIN_AVG_LOTS = 200
MIN_REV_YOY = -5.0
INST_LOOKBACK = 15
SCORE_DELTA_DAYS = 5
MOMENTUM_PERIODS = [(5, 0.50), (10, 0.30), (20, 0.20)]
W_MOM = 0.40
W_INST = 0.40
W_VOL = 0.20
W_DELTA = 0.20

A_SECTOR_TOP = 3
A_MAX_RSI = 72
A_MAX_RUN5 = 0.15
A_MIN_INST = 1
A_EXCLUDE_SECTORS = {"IC設計"}

B_SECTOR_TOP = 5
B_RSI_LO = 45
B_RSI_HI = 65
B_MAX_RUN5 = 0.08
B_VOL_RATIO = 1.2

C_SECTOR_TOP = 5
C_RSI_LO = 50
C_RSI_HI = 70
C_MAX_RUN5 = 0.12
C_VOL_RATIO = 1.1
C_MIN_INST_20D = 1

# Strategy D：C + 題材集中（top-3）
# 基於股癌哲學：「汰弱留強」——集中在最強的 3 個題材，而非廣撒 5 個
D_SECTOR_TOP = 3
D_RSI_LO = 50
D_RSI_HI = 70
D_MAX_RUN5 = 0.12
D_VOL_RATIO = 1.1
D_MIN_INST_20D = 1
D_MIN_MA60_SLOPE = 0.0      # 保留欄位（未使用）
D_MIN_REV_YOY = -5.0        # 同 C 策略
D_MIN_SECTOR_MOM20 = 0.03   # 子題材 20 日平均漲幅 > 3%，避免選空頭題材中的「最強弱雞」
D_EXCLUDE_SECTORS: set[str] = {"PCB上游材料"}

INTRADAY_VOL_SCALE_CAP = 6.0
INTRADAY_VOLUME_START_MINUTE = 30
