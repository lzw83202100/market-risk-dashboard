#!/usr/bin/env python3
"""
美股市场风险仪表盘 — 数据获取与指标计算
每日由 GitHub Actions 调用，输出 data/latest.json
"""
import json
import os
import sys
import traceback
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import yfinance as yf

# ---------------------------------------------------------------------------
# 全局配置
# ---------------------------------------------------------------------------
FRED_API_KEY = os.environ.get("FRED_API_KEY", "")
HEADERS = {"User-Agent": "MarketRiskDashboard/1.0"}
TIMEOUT = 30
OUTPUT_PATH = Path(__file__).resolve().parent.parent / "data" / "latest.json"

# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------

INDICATOR_META = {
    "qqq_2y_ma": ("QQQ 2年均线乘数", "> 1.45x"),
    "spy_200d_dev": ("SPY 200日均线乖离率", "> +20%"),
    "spy_monthly_rsi": ("月线RSI(SPY)", "> 80"),
    "rainbow_valuation": ("美股彩虹图估值热图", "> +40%"),
    "presidential_cycle": ("总统周期第3-4年顶部", "选举年+高估值"),
    "cnn_fear_greed": ("CNN恐慌贪婪指数", "> 80"),
    "cboe_put_call": ("CBOE Put/Call比率", "< 0.50"),
    "vix": ("VIX波动率", "< 11"),
    "aaii_bull": ("AAII散户牛市情绪", "> 60%"),
    "naaim_exposure": ("NAAIM机构风险敞口", "> 85%"),
    "pct_above_200d": ("200日均线以上股票占比", "< 60%"),
    "hindenburg_omen": ("兴登堡凶兆", ">= 3次/30天"),
    "mcclellan_sum": ("麦克莱伦加总指数", "死叉0轴或高位回落"),
    "ad_line_divergence": ("腾落线(A/D Line)背离", "指数新高但A/D未新高"),
    "nyse_high_low_pct": ("NYSE新高/新低比率", "< 20%"),
    "pct_above_50d": ("50日均线以上股票占比", "> 80%"),
    "margin_debt_ratio": ("FINRA保证金债务/总市值", "> 2.5%"),
    "etf_volume_momentum": ("ETF成交额动量(4周)", "> +30%"),
    "insider_sell_buy": ("内部人卖出/买入比率", "> 3x"),
    "spy_rsp_divergence": ("SPY vs RSP等权重背离", "> 15%"),
    "vix_term_structure": ("投机者情绪(VIX9D/VIX)", "< 0.85"),
    "buffett_indicator": ("巴菲特指标(全球化调整)", "> 180%"),
    "shiller_cape": ("席勒CAPE(Margin调整)", "> 42x"),
    "ndx_forward_pe": ("纳斯达克100前瞻市盈率", "> 38x"),
    "tobins_q": ("Tobin's Q(无形资产调整)", "> 2.2"),
    "household_equity_pct": ("家庭股票配置(上调阈值)", "> 50%"),
    "yield_curve_10y2y": ("美债收益率曲线(10Y-2Y)", "> +0.50%"),
    "fed_funds_rate": ("美联储政策拐点", "停止降息转升息"),
    "m2_liquidity": ("M2流动性比率", "> 3.25"),
    "wei": ("Weekly Economic Index", "< 0"),
    "hy_oas": ("高收益信用利差(HY OAS)", "> 400bps"),
    "retail_defense": ("零售销售+防御板块强度", "< 2% YoY + 防御板块跑赢"),
}


def safe(func):
    """装饰器：捕获异常时返回带 error 的 fallback indicator"""
    def wrapper(*a, **kw):
        try:
            val = func(*a, **kw)
            return (val, None)
        except Exception as e:
            print(f"[ERROR] {func.__name__}: {e}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            # 从函数名推断 indicator id
            id_ = func.__name__.replace("calc_", "")
            meta = INDICATOR_META.get(id_)
            if meta:
                fallback = make_indicator(id_, meta[0], None, "N/A", meta[1], False, error=str(e))
                return (fallback, str(e))
            return (None, str(e))
    return wrapper


def yf_download(tickers, period="2y", **kw):
    """批量下载 yfinance 数据"""
    df = yf.download(tickers, period=period, progress=False, **kw)
    return df


def fred_get(series_id, limit=500):
    """从 FRED API 获取数据"""
    if not FRED_API_KEY:
        raise ValueError("FRED_API_KEY not set")
    url = "https://api.stlouisfed.org/fred/series/observations"
    params = {
        "series_id": series_id,
        "api_key": FRED_API_KEY,
        "file_type": "json",
        "sort_order": "desc",
        "limit": limit,
    }
    r = requests.get(url, params=params, timeout=TIMEOUT)
    r.raise_for_status()
    obs = r.json()["observations"]
    records = []
    for o in obs:
        try:
            records.append({"date": o["date"], "value": float(o["value"])})
        except (ValueError, KeyError):
            continue
    df = pd.DataFrame(records)
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)
    return df


def sma(series, window):
    return series.rolling(window=window, min_periods=window).mean()


def ema(series, span):
    return series.ewm(span=span, adjust=False).mean()


def rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss
    return 100 - 100 / (1 + rs)


def get_sp500_tickers():
    """从 Wikipedia 获取 S&P 500 成分股列表"""
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    r = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
    r.raise_for_status()
    tables = pd.read_html(r.text)
    df = tables[0]
    tickers = df["Symbol"].tolist()
    # 清理：BRK.B → BRK-B (yfinance 格式)
    tickers = [t.replace(".", "-") for t in tickers]
    return tickers


def calc_pct_above_ma(tickers, ma_window):
    """批量计算股票在 N 日均线以上的占比
    下载足够的历史数据，计算每只股票当前价格 vs N日均线
    """
    # 需要至少 ma_window 个交易日的数据
    period = "1y" if ma_window <= 200 else "2y"
    print(f"  Downloading {len(tickers)} stocks for {ma_window}-day MA check...", file=sys.stderr)

    # 分批下载（每批50只，避免超时）
    batch_size = 50
    above = 0
    total_valid = 0

    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i+batch_size]
        try:
            df = yf.download(batch, period=period, progress=False)
            if df.empty:
                continue
            close = df["Close"]
            if isinstance(close, pd.Series):
                # 单只股票
                close = close.to_frame(batch[0])

            for ticker in batch:
                try:
                    if ticker not in close.columns:
                        continue
                    series = close[ticker].dropna()
                    if len(series) < ma_window:
                        continue
                    ma_val = series.rolling(ma_window).mean().iloc[-1]
                    if pd.isna(ma_val):
                        continue
                    total_valid += 1
                    if series.iloc[-1] > ma_val:
                        above += 1
                except Exception:
                    continue
        except Exception as e:
            print(f"  Batch {i//batch_size} failed: {e}", file=sys.stderr)
            continue

    if total_valid == 0:
        raise ValueError("No valid stocks found")

    pct = round(above / total_valid * 100, 1)
    print(f"  {ma_window}-day MA: {above}/{total_valid} = {pct}%", file=sys.stderr)
    return pct


def to_python(val):
    """numpy/pandas 类型转 Python 原生类型，确保 JSON 可序列化"""
    if val is None:
        return None
    if isinstance(val, (np.bool_,)):
        return bool(val)
    if isinstance(val, (np.integer,)):
        return int(val)
    if isinstance(val, (np.floating,)):
        return float(val)
    return val


def make_indicator(id_, name, current, current_display, threshold, hit, error=None):
    status = "hit" if hit else "normal"
    if current is None:
        status = "error"
        hit = False
    return {
        "id": id_,
        "name": name,
        "current": to_python(current),
        "current_display": str(current_display),
        "threshold": threshold,
        "hit": bool(hit),
        "status": status,
        "error": error,
    }


# ---------------------------------------------------------------------------
# 批量预拉取 yfinance 数据（减少调用次数）
# ---------------------------------------------------------------------------

def prefetch_yfinance():
    """一次性拉取所有需要的 yfinance 数据"""
    tickers_2y = ["QQQ", "SPY", "RSP", "^VIX", "^GSPC", "XLP", "XLY"]
    tickers_short = ["^VIX9D"]

    print("Fetching yfinance data (2y)...", file=sys.stderr)
    df_2y = yf_download(tickers_2y, period="2y")

    # QQQ 需要3年数据来算504日均线
    print("Fetching QQQ 3y...", file=sys.stderr)
    df_qqq_3y = yf_download("QQQ", period="3y")

    # 长周期数据用于彩虹图
    print("Fetching SPY long history...", file=sys.stderr)
    df_spy_long = yf_download("SPY", period="max")

    # VIX9D 可能不可用
    print("Fetching VIX9D...", file=sys.stderr)
    try:
        df_vix9d = yf_download(tickers_short, period="3mo")
    except Exception:
        df_vix9d = pd.DataFrame()

    return df_2y, df_qqq_3y, df_spy_long, df_vix9d


# ---------------------------------------------------------------------------
# Category 1: 技术面与周期预警
# ---------------------------------------------------------------------------

@safe
def calc_qqq_2y_ma(df_2y, df_qqq_3y):
    """用3年数据计算504日均线"""
    if isinstance(df_qqq_3y.columns, pd.MultiIndex):
        close = df_qqq_3y["Close"].iloc[:, 0].dropna()
    else:
        close = df_qqq_3y["Close"].dropna()
    sma_504 = sma(close, 504)
    current = float(close.iloc[-1])
    ma_val = float(sma_504.iloc[-1])
    if np.isnan(ma_val):
        raise ValueError("Not enough data for 504-day SMA")
    ratio = round(current / ma_val, 3)
    return make_indicator(
        "qqq_2y_ma", "QQQ 2年均线乘数",
        ratio, f"{ratio}x", "> 1.45x", ratio > 1.45
    )


@safe
def calc_spy_200d_dev(df_2y):
    close = df_2y["Close"]["SPY"].dropna()
    sma_200 = sma(close, 200)
    current = close.iloc[-1]
    ma_val = sma_200.iloc[-1]
    dev = round((current - ma_val) / ma_val * 100, 2)
    return make_indicator(
        "spy_200d_dev", "SPY 200日均线乖离率",
        dev, f"{dev:+.2f}%", "> +20%", dev > 20
    )


@safe
def calc_spy_monthly_rsi(df_2y):
    close = df_2y["Close"]["SPY"].dropna()
    monthly = close.resample("ME").last().dropna()
    rsi_val = rsi(monthly, 14)
    current = round(rsi_val.iloc[-1], 1)
    return make_indicator(
        "spy_monthly_rsi", "月线RSI(SPY)",
        current, f"{current}", "> 80", current > 80
    )


@safe
def calc_rainbow_valuation(df_spy_long):
    # 处理多层列索引
    if isinstance(df_spy_long.columns, pd.MultiIndex):
        close = df_spy_long["Close"]
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
    else:
        close = df_spy_long["Close"]
    close = close.dropna()

    # 对数回归
    x = np.arange(len(close), dtype=float)
    log_close = np.log(close.values)
    coeffs = np.polyfit(x, log_close, 1)
    trend = np.exp(np.polyval(coeffs, x))
    dev_pct = round((close.iloc[-1] / trend[-1] - 1) * 100, 1)
    return make_indicator(
        "rainbow_valuation", "美股彩虹图估值热图",
        dev_pct, f"{dev_pct:+.1f}%", "> +40%", dev_pct > 40
    )


@safe
def calc_presidential_cycle(spy_dev_val):
    year = datetime.now().year
    cycle_year = year % 4  # 0=选举年
    labels = {0: "选举年", 1: "后选举年", 2: "中期年", 3: "前选举年"}
    label = labels[cycle_year]
    # 只有选举年且SPY高估值才警报
    hit = (cycle_year == 0 and spy_dev_val is not None and spy_dev_val > 10)
    return make_indicator(
        "presidential_cycle", "总统周期第3-4年顶部",
        cycle_year, label, "选举年+高估值", hit
    )


# ---------------------------------------------------------------------------
# Category 2: 市场情绪与行为
# ---------------------------------------------------------------------------

@safe
def calc_cnn_fear_greed():
    today = datetime.now().strftime("%Y-%m-%d")
    url = f"https://production.dataviz.cnn.io/index/fearandgreed/graphdata/{today}"
    r = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
    r.raise_for_status()
    data = r.json()
    score = round(data["fear_and_greed"]["score"], 1)
    return make_indicator(
        "cnn_fear_greed", "CNN恐慌贪婪指数",
        score, f"{score}", "> 80", score > 80
    )


@safe
def calc_vix(df_2y):
    close = df_2y["Close"]["^VIX"].dropna()
    current = round(close.iloc[-1], 2)
    return make_indicator(
        "vix", "VIX波动率",
        current, f"{current}", "< 11", current < 11
    )


@safe
def calc_cboe_put_call():
    """获取 CBOE equity put/call 比率"""
    import re
    # 方法1: yfinance
    for ticker in ["^PCCE", "^PCALL"]:
        try:
            df = yf_download(ticker, period="1mo")
            if isinstance(df.columns, pd.MultiIndex):
                close = df["Close"].iloc[:, 0].dropna()
            else:
                close = df["Close"].dropna()
            if len(close) > 0:
                current = round(float(close.iloc[-1]), 3)
                return make_indicator(
                    "cboe_put_call", "CBOE Put/Call比率",
                    current, f"{current}", "< 0.50", current < 0.50
                )
        except Exception:
            continue
    # 方法2: CBOE 官网 CSV
    try:
        url = "https://www.cboe.com/us/options/market_statistics/daily/"
        r = requests.get(url, headers={**HEADERS, "Accept": "text/html"}, timeout=TIMEOUT)
        r.raise_for_status()
        # 尝试找到 equity put/call ratio
        match = re.search(r'(?:equity|EQUITY).*?put.*?call.*?(\d+\.?\d*)', r.text, re.DOTALL | re.IGNORECASE)
        if match:
            val = round(float(match.group(1)), 3)
            if 0.1 < val < 3.0:  # 合理范围
                return make_indicator(
                    "cboe_put_call", "CBOE Put/Call比率",
                    val, f"{val}", "< 0.50", val < 0.50
                )
    except Exception:
        pass
    # 方法3: 从 FRED 获取 total P/C ratio
    try:
        df = fred_get("PCOTTM", limit=5)
        if not df.empty:
            val = round(df["value"].iloc[-1], 3)
            return make_indicator(
                "cboe_put_call", "CBOE Put/Call比率",
                val, f"{val}", "< 0.50", val < 0.50
            )
    except Exception:
        pass
    raise ValueError("data source unavailable")


@safe
def calc_aaii_bull():
    """获取 AAII 散户情绪数据"""
    import re
    # 尝试多个 URL 和解析方式
    urls = [
        "https://www.aaii.com/sentimentsurvey",
        "https://www.aaii.com/sentimentsurvey/sent_results",
    ]
    for url in urls:
        try:
            r = requests.get(url, headers={
                **HEADERS,
                "Accept": "text/html",
                "Referer": "https://www.aaii.com/",
            }, timeout=TIMEOUT)
            if r.status_code != 200:
                continue
            # 尝试多种模式匹配 Bullish 百分比
            patterns = [
                r'[Bb]ullish.*?(\d+\.?\d*)\s*%',
                r'(\d+\.?\d*)\s*%\s*[Bb]ullish',
                r'bullish.*?(\d{2}\.?\d*)',
            ]
            for pat in patterns:
                match = re.search(pat, r.text, re.DOTALL)
                if match:
                    val = round(float(match.group(1)), 1)
                    if 5 <= val <= 95:  # 合理范围
                        return make_indicator(
                            "aaii_bull", "AAII散户牛市情绪",
                            val, f"{val}%", "> 60%", val > 60
                        )
        except Exception:
            continue
    raise ValueError("data source unavailable")


@safe
def calc_naaim_exposure():
    """尝试获取 NAAIM 机构风险敞口"""
    try:
        url = "https://www.naaim.org/programs/naaim-exposure-index/"
        r = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
        r.raise_for_status()
        import re
        # 查找较大的数字（NAAIM 指数通常在 20-120 之间）
        numbers = re.findall(r'(\d{2,3}\.?\d{0,2})\s*%?\s*</td>', r.text)
        for n in numbers:
            val = float(n)
            if 10 <= val <= 200:  # 合理范围
                val = round(val, 1)
                return make_indicator(
                    "naaim_exposure", "NAAIM机构风险敞口",
                    val, f"{val}%", "> 85%", val > 85
                )
    except Exception:
        pass
    return make_indicator(
        "naaim_exposure", "NAAIM机构风险敞口",
        None, "N/A", "> 85%", False, error="data source unavailable"
    )


# ---------------------------------------------------------------------------
# Category 3: 市场广度与内部健康
# ---------------------------------------------------------------------------

@safe
def calc_pct_above_200d(sp500_tickers=None):
    """S&P 500 中在200日均线以上的股票占比"""
    # 优先尝试 yfinance 指标 ticker
    for ticker in ["MMTH", "^SPXA200R"]:
        try:
            df = yf_download(ticker, period="1mo")
            if isinstance(df.columns, pd.MultiIndex):
                close = df["Close"].iloc[:, 0].dropna()
            else:
                close = df["Close"].dropna()
            if len(close) > 0:
                current = round(float(close.iloc[-1]), 1)
                return make_indicator(
                    "pct_above_200d", "200日均线以上股票占比",
                    current, f"{current}%", "< 60%", current < 60
                )
        except Exception:
            continue
    # Fallback: 手动计算
    if sp500_tickers:
        current = calc_pct_above_ma(sp500_tickers, 200)
        return make_indicator(
            "pct_above_200d", "200日均线以上股票占比",
            current, f"{current}%", "< 60%", current < 60
        )
    raise ValueError("pct_above_200d data unavailable")


@safe
def calc_pct_above_50d(sp500_tickers=None):
    """S&P 500 中在50日均线以上的股票占比"""
    for ticker in ["MMFI", "^SPXA50R"]:
        try:
            df = yf_download(ticker, period="1mo")
            if isinstance(df.columns, pd.MultiIndex):
                close = df["Close"].iloc[:, 0].dropna()
            else:
                close = df["Close"].dropna()
            if len(close) > 0:
                current = round(float(close.iloc[-1]), 1)
                return make_indicator(
                    "pct_above_50d", "50日均线以上股票占比",
                    current, f"{current}%", "> 80%", current > 80
                )
        except Exception:
            continue
    # Fallback: 手动计算
    if sp500_tickers:
        current = calc_pct_above_ma(sp500_tickers, 50)
        return make_indicator(
            "pct_above_50d", "50日均线以上股票占比",
            current, f"{current}%", "> 80%", current > 80
        )
    raise ValueError("pct_above_50d data unavailable")


@safe
def calc_hindenburg_omen():
    """兴登堡凶兆 — 简化版（需要 NYSE 新高新低数据，通常难以免费获取）"""
    # 尝试获取 NYSE 新高新低
    return make_indicator(
        "hindenburg_omen", "兴登堡凶兆",
        None, "N/A", ">= 3次/30天", False,
        error="NYSE new high/low data not freely available"
    )


@safe
def calc_mcclellan_sum():
    """麦克莱伦加总指数 — 需要 NYSE advance/decline 数据"""
    return make_indicator(
        "mcclellan_sum", "麦克莱伦加总指数",
        None, "N/A", "死叉0轴或高位回落", False,
        error="NYSE advance/decline data not freely available"
    )


@safe
def calc_ad_line_divergence(df_2y):
    """A/D Line 背离检测 — 简化版"""
    return make_indicator(
        "ad_line_divergence", "腾落线(A/D Line)背离",
        None, "N/A", "指数新高但A/D未新高", False,
        error="NYSE advance/decline data not freely available"
    )


@safe
def calc_nyse_high_low_pct():
    """NYSE 新高/新低比率"""
    return make_indicator(
        "nyse_high_low_pct", "NYSE新高/新低比率",
        None, "N/A", "< 20%", False,
        error="NYSE new high/low data not freely available"
    )


# ---------------------------------------------------------------------------
# Category 4: 杠杆与资金结构风险
# ---------------------------------------------------------------------------

@safe
def calc_margin_debt_ratio():
    """FINRA保证金债务/总市值"""
    margin_val = None
    for series in ["BOGZ1FL663067003Q", "BOGZ1FL663067003A"]:
        try:
            margin = fred_get(series, limit=5)
            if not margin.empty:
                margin_val = margin["value"].iloc[-1]
                break
        except Exception:
            continue

    wilshire_val = None
    for series in ["WILL5000PRFC", "WILL5000PR"]:
        try:
            wilshire = fred_get(series, limit=5)
            if not wilshire.empty:
                wilshire_val = wilshire["value"].iloc[-1]
                break
        except Exception:
            continue

    if margin_val is None or wilshire_val is None:
        raise ValueError("FRED data unavailable")

    # 保证金债务单位: 百万美元; Wilshire: 指数点 ≈ 十亿美元
    ratio = round(margin_val / (wilshire_val * 1000) * 100, 2)
    return make_indicator(
        "margin_debt_ratio", "FINRA保证金债务/总市值",
        ratio, f"{ratio}%", "> 2.5%", ratio > 2.5
    )


@safe
def calc_etf_volume_momentum(df_2y):
    """ETF成交额动量(4周)"""
    close = df_2y["Close"]["SPY"].dropna()
    volume = df_2y["Volume"]["SPY"].dropna()
    # 对齐索引
    idx = close.index.intersection(volume.index)
    close = close.loc[idx]
    volume = volume.loc[idx]

    turnover = close * volume
    avg_5d = turnover.iloc[-5:].mean()
    avg_20d = turnover.iloc[-20:].mean()
    momentum = round((avg_5d / avg_20d - 1) * 100, 1)
    return make_indicator(
        "etf_volume_momentum", "ETF成交额动量(4周)",
        momentum, f"{momentum:+.1f}%", "> +30%", momentum > 30
    )


@safe
def calc_insider_sell_buy():
    return make_indicator(
        "insider_sell_buy", "内部人卖出/买入比率",
        None, "N/A", "> 3x", False,
        error="free insider data source unavailable"
    )


@safe
def calc_spy_rsp_divergence(df_2y):
    """SPY vs RSP 等权重背离"""
    spy = df_2y["Close"]["SPY"].dropna()
    rsp = df_2y["Close"]["RSP"].dropna()
    idx = spy.index.intersection(rsp.index)
    spy = spy.loc[idx]
    rsp = rsp.loc[idx]

    if len(spy) < 60:
        raise ValueError("Not enough data")

    spy_ret = (spy.iloc[-1] / spy.iloc[-60] - 1) * 100
    rsp_ret = (rsp.iloc[-1] / rsp.iloc[-60] - 1) * 100
    div = round(spy_ret - rsp_ret, 1)
    return make_indicator(
        "spy_rsp_divergence", "SPY vs RSP等权重背离",
        div, f"{div:+.1f}%", "> 15%", div > 15
    )


@safe
def calc_vix_term_structure(df_2y, df_vix9d):
    """VIX期限结构 (VIX9D/VIX)"""
    vix = df_2y["Close"]["^VIX"].dropna()
    try:
        if isinstance(df_vix9d.columns, pd.MultiIndex):
            vix9d = df_vix9d["Close"].iloc[:, 0].dropna()
        else:
            vix9d = df_vix9d["Close"].dropna()
        if len(vix9d) == 0:
            raise ValueError("empty")
    except Exception:
        raise ValueError("VIX9D data unavailable")

    idx = vix.index.intersection(vix9d.index)
    if len(idx) == 0:
        raise ValueError("No overlapping dates")

    ratio = round(vix9d.loc[idx[-1]] / vix.loc[idx[-1]], 3)
    return make_indicator(
        "vix_term_structure", "投机者情绪(VIX9D/VIX)",
        ratio, f"{ratio}", "< 0.85", ratio < 0.85
    )


# ---------------------------------------------------------------------------
# Category 5: 核心估值泡沫区
# ---------------------------------------------------------------------------

@safe
def calc_buffett_indicator():
    """巴菲特指标 (Wilshire 5000 / GDP)"""
    # 尝试多个 FRED series
    w_val = None
    for series in ["WILL5000PRFC", "WILL5000PR"]:
        try:
            wilshire = fred_get(series, limit=5)
            if not wilshire.empty:
                w_val = wilshire["value"].iloc[-1]
                break
        except Exception:
            continue

    # yfinance fallback
    if w_val is None:
        try:
            df = yf_download("^W5000", period="5d")
            if isinstance(df.columns, pd.MultiIndex):
                close = df["Close"].iloc[:, 0].dropna()
            else:
                close = df["Close"].dropna()
            if len(close) > 0:
                w_val = float(close.iloc[-1])
        except Exception:
            pass

    gdp = fred_get("GDP", limit=5)
    if w_val is None or gdp.empty:
        raise ValueError("data unavailable")

    g_val = gdp["value"].iloc[-1]
    ratio = round(w_val / g_val * 100, 1)
    return make_indicator(
        "buffett_indicator", "巴菲特指标(全球化调整)",
        ratio, f"{ratio}%", "> 180%", ratio > 180
    )


@safe
def calc_shiller_cape():
    """席勒CAPE — 从 multpl.com 抓取"""
    import re
    try:
        url = "https://www.multpl.com/shiller-pe"
        r = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
        r.raise_for_status()
        match = re.search(r'Current Shiller PE.*?(\d+\.?\d*)', r.text, re.DOTALL)
        if match:
            val = round(float(match.group(1)), 1)
            return make_indicator(
                "shiller_cape", "席勒CAPE(Margin调整)",
                val, f"{val}x", "> 42x", val > 42
            )
    except Exception:
        pass
    # FRED fallback
    try:
        cape = fred_get("CAPE10", limit=5)
        if not cape.empty:
            val = round(cape["value"].iloc[-1], 1)
            return make_indicator(
                "shiller_cape", "席勒CAPE(Margin调整)",
                val, f"{val}x", "> 42x", val > 42
            )
    except Exception:
        pass
    raise ValueError("data source unavailable")


@safe
def calc_ndx_forward_pe():
    """纳斯达克100前瞻/追踪市盈率"""
    try:
        info = yf.Ticker("QQQ").info
        fpe = info.get("forwardPE") or info.get("trailingPE")
        if fpe:
            val = round(float(fpe), 1)
            label = "前瞻PE" if "forwardPE" in info and info["forwardPE"] else "追踪PE"
            return make_indicator(
                "ndx_forward_pe", f"纳斯达克100{label}",
                val, f"{val}x", "> 38x", val > 38
            )
    except Exception:
        pass
    return make_indicator(
        "ndx_forward_pe", "纳斯达克100前瞻市盈率",
        None, "N/A", "> 38x", False, error="data source unavailable"
    )


@safe
def calc_tobins_q():
    """Tobin's Q"""
    try:
        assets = fred_get("BOGZ1FL102090005Q", limit=5)
        equity = fred_get("WILL5000PRFC", limit=5)
        if assets.empty or equity.empty:
            raise ValueError("FRED data unavailable")
        q = round(equity["value"].iloc[-1] * 1000 / assets["value"].iloc[-1], 2)
        return make_indicator(
            "tobins_q", "Tobin's Q(无形资产调整)",
            q, f"{q}", "> 2.2", q > 2.2
        )
    except Exception:
        return make_indicator(
            "tobins_q", "Tobin's Q(无形资产调整)",
            None, "N/A", "> 2.2", False, error="data source unavailable"
        )


@safe
def calc_household_equity_pct():
    """家庭股票配置"""
    try:
        eq = fred_get("BOGZ1FL153064105Q", limit=5)
        total_assets = fred_get("BOGZ1FL154090005Q", limit=5)
        if eq.empty or total_assets.empty:
            raise ValueError("FRED data unavailable")
        pct = round(eq["value"].iloc[-1] / total_assets["value"].iloc[-1] * 100, 1)
        return make_indicator(
            "household_equity_pct", "家庭股票配置(上调阈值)",
            pct, f"{pct}%", "> 50%", pct > 50
        )
    except Exception:
        return make_indicator(
            "household_equity_pct", "家庭股票配置(上调阈值)",
            None, "N/A", "> 50%", False, error="data source unavailable"
        )


# ---------------------------------------------------------------------------
# Category 6: 宏观流动性与政策
# ---------------------------------------------------------------------------

@safe
def calc_yield_curve_10y2y():
    """10Y-2Y 收益率曲线"""
    df = fred_get("T10Y2Y", limit=30)
    if df.empty:
        raise ValueError("FRED data unavailable")
    current = round(df["value"].iloc[-1], 2)
    return make_indicator(
        "yield_curve_10y2y", "美债收益率曲线(10Y-2Y)",
        current, f"{current:+.2f}%", "> +0.50%", current > 0.50
    )


@safe
def calc_fed_funds_rate():
    """美联储政策拐点"""
    df = fred_get("DFEDTARU", limit=30)
    if df.empty:
        raise ValueError("FRED data unavailable")
    rates = df["value"].values
    current = rates[-1]
    # 判断方向
    if len(rates) >= 2:
        prev = rates[-2]
        if current > prev:
            direction = "加息中"
        elif current < prev:
            direction = "降息中"
        else:
            direction = "按兵不动"
    else:
        direction = "数据不足"

    # 警报: 从降息转加息
    hit = False
    if len(rates) >= 3:
        prev_dir = rates[-2] - rates[-3]
        curr_dir = rates[-1] - rates[-2]
        hit = (prev_dir < 0 and curr_dir > 0)

    return make_indicator(
        "fed_funds_rate", "美联储政策拐点",
        current, f"{current}% ({direction})", "停止降息转升息", hit
    )


@safe
def calc_m2_liquidity():
    """M2流动性比率"""
    m2 = fred_get("M2SL", limit=5)
    gdp = fred_get("GDP", limit=5)
    if m2.empty or gdp.empty:
        raise ValueError("FRED data unavailable")
    # M2: 十亿美元, GDP: 十亿美元
    ratio = round(m2["value"].iloc[-1] / gdp["value"].iloc[-1], 2)
    return make_indicator(
        "m2_liquidity", "M2流动性比率",
        ratio, f"{ratio}", "> 3.25", ratio > 3.25
    )


@safe
def calc_wei():
    """Weekly Economic Index"""
    df = fred_get("WEI", limit=10)
    if df.empty:
        raise ValueError("FRED data unavailable")
    current = round(df["value"].iloc[-1], 2)
    return make_indicator(
        "wei", "Weekly Economic Index",
        current, f"{current}", "< 0", current < 0
    )


@safe
def calc_hy_oas():
    """高收益信用利差"""
    df = fred_get("BAMLH0A0HYM2", limit=10)
    if df.empty:
        raise ValueError("FRED data unavailable")
    pct_val = df["value"].iloc[-1]  # FRED 给的是百分比，如 3.5 = 350bps
    bps = round(pct_val * 100, 0)
    return make_indicator(
        "hy_oas", "高收益信用利差(HY OAS)",
        bps, f"{int(bps)}bps", "> 400bps", bps > 400
    )


@safe
def calc_retail_defense(df_2y):
    """零售销售+防御板块强度"""
    # 零售销售 YoY
    try:
        retail = fred_get("RSAFS", limit=24)
        if len(retail) >= 13:
            current_val = retail["value"].iloc[-1]
            year_ago = retail["value"].iloc[-13]
            yoy = round((current_val / year_ago - 1) * 100, 1)
        else:
            yoy = None
    except Exception:
        yoy = None

    # XLP vs XLY 相对强度
    try:
        xlp = df_2y["Close"]["XLP"].dropna()
        xly = df_2y["Close"]["XLY"].dropna()
        idx = xlp.index.intersection(xly.index)
        xlp = xlp.loc[idx]
        xly = xly.loc[idx]
        if len(xlp) >= 60:
            xlp_ret = xlp.iloc[-1] / xlp.iloc[-60] - 1
            xly_ret = xly.iloc[-1] / xly.iloc[-60] - 1
            defense_outperform = xlp_ret > xly_ret
        else:
            defense_outperform = False
    except Exception:
        defense_outperform = False

    if yoy is not None:
        hit = (yoy < 2 and defense_outperform)
        display = f"零售YoY {yoy}%, {'防御>进攻' if defense_outperform else '进攻>防御'}"
    else:
        hit = False
        display = "N/A"

    return make_indicator(
        "retail_defense", "零售销售+防御板块强度",
        yoy, display, "< 2% YoY + 防御板块跑赢", hit
    )


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def build_output():
    # 批量拉取 yfinance 数据
    df_2y, df_qqq_3y, df_spy_long, df_vix9d = prefetch_yfinance()

    # ---- Category 1: 技术面与周期预警 ----
    qqq_2y, e1 = calc_qqq_2y_ma(df_2y, df_qqq_3y)
    spy_dev, e2 = calc_spy_200d_dev(df_2y)
    spy_rsi, e3 = calc_spy_monthly_rsi(df_2y)
    rainbow, e4 = calc_rainbow_valuation(df_spy_long)

    # 总统周期需要 spy_dev 的值
    spy_dev_val = spy_dev["current"] if spy_dev else None
    pres, e5 = calc_presidential_cycle(spy_dev_val)

    cat1 = {
        "id": "technical",
        "name": "技术面与周期预警",
        "color": "amber",
        "indicators": [x for x in [qqq_2y, spy_dev, spy_rsi, rainbow, pres] if x],
    }

    # ---- Category 2: 市场情绪与行为 ----
    cnn, _ = calc_cnn_fear_greed()
    cboe, _ = calc_cboe_put_call()
    vix_val, _ = calc_vix(df_2y)
    aaii, _ = calc_aaii_bull()
    naaim, _ = calc_naaim_exposure()

    cat2 = {
        "id": "sentiment",
        "name": "市场情绪与行为",
        "color": "purple",
        "indicators": [x for x in [cnn, cboe, vix_val, aaii, naaim] if x],
    }

    # ---- Category 3: 市场广度与内部健康 ----
    # 获取 S&P 500 成分股（供 200d/50d 手动计算用）
    try:
        print("Fetching S&P 500 constituents...", file=sys.stderr)
        sp500 = get_sp500_tickers()
        print(f"  Got {len(sp500)} tickers", file=sys.stderr)
    except Exception as e:
        print(f"[WARN] Failed to get S&P 500 list: {e}", file=sys.stderr)
        sp500 = None

    pct200, _ = calc_pct_above_200d(sp500)
    hind, _ = calc_hindenburg_omen()
    mccl, _ = calc_mcclellan_sum()
    ad_div, _ = calc_ad_line_divergence(df_2y)
    nyse_hl, _ = calc_nyse_high_low_pct()
    pct50, _ = calc_pct_above_50d(sp500)

    cat3 = {
        "id": "breadth",
        "name": "市场广度与内部健康",
        "color": "blue",
        "indicators": [x for x in [pct200, hind, mccl, ad_div, nyse_hl, pct50] if x],
    }

    # ---- Category 4: 杠杆与资金结构风险 ----
    margin, _ = calc_margin_debt_ratio()
    etf_mom, _ = calc_etf_volume_momentum(df_2y)
    insider, _ = calc_insider_sell_buy()
    spy_rsp, _ = calc_spy_rsp_divergence(df_2y)
    vix_ts, _ = calc_vix_term_structure(df_2y, df_vix9d)

    cat4 = {
        "id": "leverage",
        "name": "杠杆与资金结构风险",
        "color": "orange",
        "indicators": [x for x in [margin, etf_mom, insider, spy_rsp, vix_ts] if x],
    }

    # ---- Category 5: 核心估值泡沫区 ----
    buffett, _ = calc_buffett_indicator()
    cape, _ = calc_shiller_cape()
    ndx_pe, _ = calc_ndx_forward_pe()
    tobin, _ = calc_tobins_q()
    household, _ = calc_household_equity_pct()

    cat5 = {
        "id": "valuation",
        "name": "核心估值泡沫区",
        "color": "red",
        "indicators": [x for x in [buffett, cape, ndx_pe, tobin, household] if x],
    }

    # ---- Category 6: 宏观流动性与政策 ----
    yc, _ = calc_yield_curve_10y2y()
    fed, _ = calc_fed_funds_rate()
    m2, _ = calc_m2_liquidity()
    wei_val, _ = calc_wei()
    hy, _ = calc_hy_oas()
    retail, _ = calc_retail_defense(df_2y)

    cat6 = {
        "id": "macro",
        "name": "宏观流动性与政策",
        "color": "red",
        "indicators": [x for x in [yc, fed, m2, wei_val, hy, retail] if x],
    }

    categories = [cat1, cat2, cat3, cat4, cat5, cat6]

    # 计算总分
    all_indicators = []
    for cat in categories:
        all_indicators.extend(cat["indicators"])

    valid = [i for i in all_indicators if i["current"] is not None]
    hits = sum(1 for i in valid if i["hit"])
    total = len(valid)
    pct = round(hits / total * 100, 1) if total > 0 else 0

    if pct <= 15:
        level = f"低风险({pct}%) 安全区"
    elif pct <= 30:
        level = f"轻度预警({pct}%)"
    elif pct <= 50:
        level = f"中度风险({pct}%)"
    elif pct <= 70:
        level = f"高风险({pct}%)"
    else:
        level = f"极端风险({pct}%)"

    output = {
        "updated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "market_date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "score": {
            "hits": hits,
            "total": total,
            "total_all": len(all_indicators),
            "pct": pct,
            "level": level,
        },
        "categories": categories,
    }

    return output


def main():
    print("=" * 60, file=sys.stderr)
    print("Market Risk Dashboard — Data Fetch", file=sys.stderr)
    print("=" * 60, file=sys.stderr)

    output = build_output()

    # 写入 JSON
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    # 打印摘要
    score = output["score"]
    print(f"\nScore: {score['hits']}/{score['total']} ({score['pct']}%)", file=sys.stderr)
    print(f"Level: {score['level']}", file=sys.stderr)
    print(f"\nOutput: {OUTPUT_PATH}", file=sys.stderr)

    for cat in output["categories"]:
        print(f"\n  {cat['name']}:", file=sys.stderr)
        for ind in cat["indicators"]:
            flag = "⚠️" if ind["hit"] else "  "
            err = f" [ERR: {ind['error']}]" if ind.get("error") else ""
            print(f"    {flag} {ind['name']}: {ind['current_display']} (阈值: {ind['threshold']}){err}", file=sys.stderr)


if __name__ == "__main__":
    main()
