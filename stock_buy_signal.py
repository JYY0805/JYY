#!/usr/bin/env python3
"""A 股买点辅助分析器（仅作研究，不构成投资建议）。"""

from __future__ import annotations

import argparse
import sys
import time
import warnings
from dataclasses import dataclass
from datetime import date, timedelta

warnings.filterwarnings("ignore", message="urllib3 v2 only supports OpenSSL.*")

import akshare as ak
import numpy as np
import pandas as pd
import requests


@dataclass(frozen=True)
class Signal:
    score: int
    verdict: str
    reasons: list[str]
    warnings: list[str]


@dataclass(frozen=True)
class TradePlan:
    buy_text: str
    sell_text: str
    stop: float
    target1: float
    target2: float


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(100)


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    close, volume = out["close"], out["volume"]
    out["ma20"] = close.rolling(20).mean()
    out["ma60"] = close.rolling(60).mean()
    out["rsi14"] = rsi(close)
    out["vol_ma20"] = volume.rolling(20).mean()
    out["high_20"] = close.rolling(20).max()
    out["atr14"] = pd.concat(
        [out["high"] - out["low"], (out["high"] - close.shift()).abs(),
         (out["low"] - close.shift()).abs()], axis=1
    ).max(axis=1).rolling(14).mean()
    return out


def evaluate(df: pd.DataFrame) -> Signal:
    if len(df) < 80:
        raise ValueError("至少需要 80 个交易日的数据")
    x, prev = df.iloc[-1], df.iloc[-2]
    score, reasons, warnings = 0, [], []

    if x.close > x.ma20 > x.ma60:
        score += 30; reasons.append("价格位于 20/60 日均线上方，趋势偏多（+30）")
    elif x.close > x.ma60:
        score += 15; reasons.append("价格仍在 60 日均线上方（+15）")
    else:
        warnings.append("价格低于 60 日均线，中期趋势偏弱")

    if 45 <= x.rsi14 <= 65:
        score += 25; reasons.append(f"RSI14={x.rsi14:.1f}，动量健康且未明显过热（+25）")
    elif 35 <= x.rsi14 < 45:
        score += 12; reasons.append(f"RSI14={x.rsi14:.1f}，处于修复区（+12）")
    elif x.rsi14 > 75:
        warnings.append(f"RSI14={x.rsi14:.1f}，短线可能过热")

    distance = x.close / x.ma20 - 1
    if -0.02 <= distance <= 0.03 and x.close >= prev.close:
        score += 25; reasons.append("价格靠近 20 日线并止跌，符合回踩候选（+25）")
    elif distance > 0.08:
        warnings.append("价格偏离 20 日线超过 8%，追高风险较大")

    volume_ratio = x.volume / x.vol_ma20 if x.vol_ma20 else np.nan
    if x.close > prev.close and 1.0 <= volume_ratio <= 2.5:
        score += 20; reasons.append(f"上涨且量比={volume_ratio:.2f}，量价配合（+20）")
    elif volume_ratio > 3:
        warnings.append(f"量比={volume_ratio:.2f}，异常放量需辨别出货风险")

    verdict = "候选买点" if score >= 70 else "继续观察" if score >= 45 else "暂不买"
    return Signal(min(score, 100), verdict, reasons, warnings)


def make_trade_plan(df: pd.DataFrame, signal: Signal) -> TradePlan:
    """根据趋势和波动生成观察价位，不预测必然成交或盈利。"""
    x = df.iloc[-1]
    recent_low = float(df["low"].tail(10).min())
    recent_high = float(df["high"].tail(20).max())
    stop = max(0.01, recent_low - 0.3 * x.atr14)
    if x.close < x.ma20 and x.close < x.ma60:
        target1 = x.close + x.atr14
        target2 = max(x.ma20, x.close + 2 * x.atr14)
    else:
        target1 = max(recent_high, x.close + 1.5 * x.atr14)
        target2 = max(target1, x.close + 3 * x.atr14)

    if signal.score >= 70:
        buy_text = f"回踩买入观察区 {x.ma20 * 0.98:.2f}～{x.ma20 * 1.02:.2f} 元（需出现止跌）"
    else:
        confirm = max(x.ma20, x.ma60)
        buy_text = f"当前不追买；等待日收盘站上 {confirm:.2f} 元后再观察"

    if x.rsi14 > 75 or x.close > x.ma20 * 1.10:
        sell_text = "短线偏热，接近止盈位时宜锁定部分利润"
    elif x.close < x.ma20 and x.close < x.ma60:
        sell_text = "趋势偏弱；持仓以止损位控制风险，反弹到压力位重新评估"
    else:
        sell_text = "趋势尚未破坏；跌破止损位退出，接近目标位分批止盈"
    return TradePlan(buy_text, sell_text, stop, target1, target2)


def fetch_tencent(symbol: str, start: str, end: str) -> pd.DataFrame:
    """备用数据源：腾讯证券前复权日线。"""
    market = "sh" if symbol.startswith(("5", "6", "9")) else "sz"
    try:
        query = (
            f"{market}{symbol},day,{start[:4]}-{start[4:6]}-{start[6:]},"
            f"{end[:4]}-{end[4:6]}-{end[6:]},800,qfq"
        )
        response = requests.get(
            "https://web.ifzq.gtimg.cn/appstock/app/fqkline/get",
            params={"param": query}, timeout=15,
        )
        response.raise_for_status()
        payload = response.json()
        stock_data = payload["data"][f"{market}{symbol}"]
        rows = stock_data.get("qfqday") or stock_data.get("day") or []
    except Exception as exc:
        raise ConnectionError("主、备用行情源均无法连接，请检查网络后重试") from exc
    if not rows:
        raise ValueError(f"未取得 {symbol} 的行情，请检查股票代码")
    # 腾讯字段顺序：日期、开盘、收盘、最高、最低、成交量。
    frame = pd.DataFrame(
        [row[:6] for row in rows],
        columns=["date", "open", "close", "high", "low", "volume"],
    )
    for column in ["open", "high", "low", "close", "volume"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame.dropna().query("volume > 0").reset_index(drop=True)


def fetch_a_share(symbol: str, start: str, end: str) -> pd.DataFrame:
    # 腾讯接口通常更快；若不可用，再尝试 AKShare/东方财富。
    try:
        return fetch_tencent(symbol, start, end)
    except (ValueError, ConnectionError):
        print("快速行情源连接失败，正在切换备用行情源……", file=sys.stderr)
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            raw = ak.stock_zh_a_hist(
                symbol=symbol, period="daily", start_date=start, end_date=end, adjust="qfq",
                timeout=15,
            )
            break
        except Exception as exc:
            last_error = exc
            if attempt < 2:
                time.sleep(1.5 * (attempt + 1))
    else:
        raise ConnectionError("两个行情源均无法连接，请检查网络后重试") from last_error
    if raw.empty:
        raise ValueError(f"未取得 {symbol} 的行情，请检查股票代码或网络")
    rename = {"日期": "date", "开盘": "open", "收盘": "close", "最高": "high",
              "最低": "low", "成交量": "volume"}
    return raw.rename(columns=rename)[list(rename.values())].sort_values("date").reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="A 股买卖点辅助分析器")
    parser.add_argument("symbol", nargs="?", help="6 位 A 股代码，例如 600519")
    parser.add_argument("--years", type=int, default=3, help="取历史数据年数（默认 3）")
    args = parser.parse_args()
    symbol = args.symbol or input("请输入6位股票代码（例如 600105）：").strip()
    if not (symbol.isdigit() and len(symbol) == 6):
        parser.error("股票代码必须是 6 位数字")

    end = date.today()
    start = end - timedelta(days=365 * args.years)
    data = add_indicators(fetch_a_share(symbol, start.strftime("%Y%m%d"), end.strftime("%Y%m%d")))
    signal = evaluate(data)
    plan = make_trade_plan(data, signal)
    x = data.iloc[-1]

    print(f"\n{symbol}  |  数据日期 {x.date}  |  收盘价 {x.close:.2f}")
    print(f"结论：{signal.verdict}  |  评分：{signal.score}/100")
    for reason in signal.reasons:
        print(f"  ✓ {reason}")
    for warning in signal.warnings:
        print(f"  ! {warning}")
    print("\n【买点】", plan.buy_text)
    print("【卖点】", plan.sell_text)
    print(f"  风险止损观察位：{plan.stop:.2f} 元")
    print(f"  第一止盈观察位：{plan.target1:.2f} 元")
    print(f"  第二止盈观察位：{plan.target2:.2f} 元")
    print("提示：价位由历史日线规则计算，不保证未来收益；财报、公告和仓位仍需人工判断。\n")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, ConnectionError) as exc:
        print(f"错误：{exc}", file=sys.stderr)
        raise SystemExit(1)
