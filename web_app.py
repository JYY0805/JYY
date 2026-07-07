from datetime import date, timedelta

import streamlit as st

from stock_buy_signal import add_indicators, evaluate, fetch_a_share, make_trade_plan


st.set_page_config(page_title="A股买卖点分析器", page_icon="📈", layout="centered")
st.title("A股买卖点分析器")
st.caption("输入6位股票代码，查看基于日线趋势、动量与波动率的规则化观察价位。")


@st.cache_data(ttl=1800, show_spinner=False)
def load_stock(symbol: str):
    end = date.today()
    start = end - timedelta(days=365 * 3)
    return add_indicators(
        fetch_a_share(symbol, start.strftime("%Y%m%d"), end.strftime("%Y%m%d"))
    )


symbol = st.text_input("股票代码", placeholder="例如：600105", max_chars=6)
analyze = st.button("开始分析", type="primary", use_container_width=True)

if analyze:
    symbol = symbol.strip()
    if not (symbol.isdigit() and len(symbol) == 6):
        st.error("请输入6位数字股票代码。")
    else:
        try:
            with st.spinner("正在获取行情并计算……"):
                data = load_stock(symbol)
                signal = evaluate(data)
                plan = make_trade_plan(data, signal)
            latest = data.iloc[-1]

            st.subheader(f"{symbol} · {latest.date}")
            col1, col2, col3 = st.columns(3)
            col1.metric("收盘价", f"{latest.close:.2f} 元")
            col2.metric("买点评分", f"{signal.score}/100")
            col3.metric("当前结论", signal.verdict)

            st.line_chart(
                data.tail(120).set_index("date")[["close", "ma20", "ma60"]],
                x_label="日期", y_label="价格",
            )

            st.markdown("### 买卖观察计划")
            st.info(f"买点：{plan.buy_text}")
            st.warning(f"卖点：{plan.sell_text}")
            p1, p2, p3 = st.columns(3)
            p1.metric("风险止损位", f"{plan.stop:.2f}")
            p2.metric("第一止盈位", f"{plan.target1:.2f}")
            p3.metric("第二止盈位", f"{plan.target2:.2f}")

            with st.expander("查看评分依据", expanded=True):
                for reason in signal.reasons:
                    st.write(f"✅ {reason}")
                for warning in signal.warnings:
                    st.write(f"⚠️ {warning}")
        except Exception as exc:
            st.error(f"分析失败：{exc}")

st.divider()
st.caption("仅供学习研究。观察价位来自历史日线规则，不构成投资建议，也不保证收益。")
