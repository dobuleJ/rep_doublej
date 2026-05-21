import datetime as dt
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Tuple

import feedparser
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
from apscheduler.schedulers.background import BackgroundScheduler
from sklearn.linear_model import LinearRegression

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)
SUMMARY_FILE = DATA_DIR / "briefings.json"

KEYWORDS_POS = {"상승", "호재", "수주", "증가", "개선", "흑자", "강세", "성장"}
KEYWORDS_NEG = {"하락", "악재", "감소", "적자", "약세", "리스크", "소송", "규제", "충격"}
KEYWORDS_UNCERTAIN = {"불확실", "변동성", "급락", "급등", "경고", "우려", "긴장", "혼란"}

TICKERS = {
    "삼성전자": "005930.KS",
    "SK하이닉스": "000660.KS",
}


@dataclass
class PredictionResult:
    symbol: str
    forecast: pd.DataFrame
    confidence: float
    blocked: bool
    reason: str


def fetch_news(query: str, limit: int = 20) -> List[Dict]:
    rss_url = f"https://news.google.com/rss/search?q={query}&hl=ko&gl=KR&ceid=KR:ko"
    feed = feedparser.parse(rss_url)
    items = []
    for entry in feed.entries[:limit]:
        title = entry.get("title", "")
        summary = entry.get("summary", "")
        published = entry.get("published", "")
        link = entry.get("link", "")
        items.append({"title": title, "summary": summary, "published": published, "link": link})
    return items


def score_news(items: List[Dict]) -> Dict[str, float]:
    text = " ".join([f"{i['title']} {i['summary']}" for i in items])
    pos = sum(1 for k in KEYWORDS_POS if k in text)
    neg = sum(1 for k in KEYWORDS_NEG if k in text)
    uncertain = sum(1 for k in KEYWORDS_UNCERTAIN if k in text)
    total = max(1, pos + neg + uncertain)
    return {
        "sentiment": (pos - neg) / total,
        "uncertainty": uncertain / total,
        "volume": len(items),
    }


def fetch_prices(ticker: str, period: str = "6mo") -> pd.DataFrame:
    df = yf.Ticker(ticker).history(period=period, interval="1d")
    if df.empty:
        raise ValueError(f"No price data for {ticker}")
    return df[["Open", "High", "Low", "Close", "Volume"]].dropna()


def compute_volatility(price_df: pd.DataFrame, window: int = 20) -> float:
    returns = price_df["Close"].pct_change().dropna()
    if len(returns) < window:
        return float(returns.std() or 0)
    return float(returns.tail(window).std())


def should_block_prediction(volatility: float, uncertainty: float,
                            vol_threshold: float = 0.04,
                            uncertainty_threshold: float = 0.45) -> Tuple[bool, str]:
    if volatility >= vol_threshold:
        return True, f"변동성 임계치 초과 (vol={volatility:.2%}, 기준={vol_threshold:.2%})"
    if uncertainty >= uncertainty_threshold:
        return True, f"뉴스 혼란도 임계치 초과 (u={uncertainty:.2f}, 기준={uncertainty_threshold:.2f})"
    return False, "정상"


def forecast_3days(symbol: str, ticker: str, news_score: Dict[str, float]) -> PredictionResult:
    prices = fetch_prices(ticker)
    volatility = compute_volatility(prices)
    blocked, reason = should_block_prediction(volatility, news_score["uncertainty"])
    if blocked:
        return PredictionResult(symbol, pd.DataFrame(), 0.0, True, reason)

    close = prices["Close"].reset_index(drop=True)
    X = np.arange(len(close)).reshape(-1, 1)
    y = close.values
    model = LinearRegression().fit(X, y)

    future_x = np.arange(len(close), len(close) + 3).reshape(-1, 1)
    pred = model.predict(future_x)

    residual = y - model.predict(X)
    rmse = float(np.sqrt(np.mean(residual ** 2)))
    conf = max(0.05, 1 - rmse / max(np.mean(y), 1))

    future_dates = pd.bdate_range(start=dt.date.today() + dt.timedelta(days=1), periods=3)
    forecast = pd.DataFrame({"date": future_dates, "pred_close": pred})

    return PredictionResult(symbol, forecast, conf, False, "정상")


def recommend_split_strategy(price_df: pd.DataFrame, sentiment: float) -> Dict[str, str]:
    close = price_df["Close"]
    ma5 = close.rolling(5).mean().iloc[-1]
    ma20 = close.rolling(20).mean().iloc[-1]
    current = close.iloc[-1]

    buy_msg = "관망"
    sell_msg = "보유"

    if current < ma5 < ma20 and sentiment > -0.2:
        buy_msg = "3회 분할매수 권장 (40%-30%-30%)"
    elif current > ma5 > ma20 and sentiment < 0:
        sell_msg = "3회 분할매도 권장 (30%-30%-40%)"
    elif current > ma20 and sentiment > 0.2:
        sell_msg = "수익보전형 2회 분할매도 (50%-50%)"

    risk = "중립"
    vol = compute_volatility(price_df)
    if vol > 0.05:
        risk = "높음"
    elif vol < 0.02:
        risk = "낮음"

    return {
        "buy_timing": buy_msg,
        "sell_timing": sell_msg,
        "risk_level": risk,
        "reference": f"현재가={current:.0f}, MA5={ma5:.0f}, MA20={ma20:.0f}, 변동성={vol:.2%}",
    }


def generate_briefing(time_slot: str) -> Dict:
    all_news = []
    for name in TICKERS.keys():
        all_news.extend(fetch_news(name, limit=12))

    score = score_news(all_news)
    headlines = [f"- {n['title']}" for n in all_news[:10]]

    summary = {
        "timestamp": dt.datetime.now().isoformat(),
        "time_slot": time_slot,
        "headline_count": len(all_news),
        "sentiment": score["sentiment"],
        "uncertainty": score["uncertainty"],
        "highlights": headlines,
        "comment": (
            "혼란도 높음: 예측/매매 신중" if score["uncertainty"] > 0.45
            else "정상 구간: 리스크 규칙 기반 대응"
        )
    }

    existing = []
    if SUMMARY_FILE.exists():
        existing = json.loads(SUMMARY_FILE.read_text(encoding="utf-8"))
    existing.append(summary)
    SUMMARY_FILE.write_text(json.dumps(existing, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def setup_schedule() -> BackgroundScheduler:
    scheduler = BackgroundScheduler(timezone="Asia/Seoul")
    scheduler.add_job(lambda: generate_briefing("아침"), "cron", hour=8, minute=30)
    scheduler.add_job(lambda: generate_briefing("점심"), "cron", hour=15, minute=30)
    scheduler.add_job(lambda: generate_briefing("저녁"), "cron", hour=22, minute=0)
    scheduler.start()
    return scheduler


def main():
    st.set_page_config(page_title="반도체 2종목 트레이딩 보조", layout="wide")
    st.title("삼성전자/SK하이닉스 전용 트레이딩 보조 프로그램")
    st.caption("뉴스 브리핑 · 리스크 관리 · 3일 예측 (조건부)")

    if "scheduler_started" not in st.session_state:
        st.session_state["scheduler"] = setup_schedule()
        st.session_state["scheduler_started"] = True

    c1, c2 = st.columns(2)

    with c1:
        st.subheader("1) 뉴스·이슈 브리핑")
        if st.button("지금 브리핑 생성"):
            b = generate_briefing("수동")
            st.success("브리핑 저장 완료")
            st.json(b)

        if SUMMARY_FILE.exists():
            logs = json.loads(SUMMARY_FILE.read_text(encoding="utf-8"))
            st.write(f"최근 브리핑 수: {len(logs)}")
            st.dataframe(pd.DataFrame(logs).tail(5), use_container_width=True)

    with c2:
        st.subheader("2) 리스크 관리 + 분할매수/매도 권고")
        selected = st.selectbox("종목", list(TICKERS.keys()))
        ticker = TICKERS[selected]
        prices = fetch_prices(ticker)
        news = fetch_news(selected, limit=15)
        score = score_news(news)
        rec = recommend_split_strategy(prices, score["sentiment"])

        st.metric("리스크 수준", rec["risk_level"])
        st.write("매수 타이밍:", rec["buy_timing"])
        st.write("매도 타이밍:", rec["sell_timing"])
        st.caption(rec["reference"])

    st.subheader("3) 3일 시계열 예측")
    for name, ticker in TICKERS.items():
        news = fetch_news(name, limit=12)
        score = score_news(news)
        result = forecast_3days(name, ticker, score)

        st.markdown(f"#### {name}")
        if result.blocked:
            st.error(f"예측 불가: {result.reason}")
        else:
            st.success(f"예측 가능 (신뢰도 추정: {result.confidence:.2f})")
            st.dataframe(result.forecast, use_container_width=True)

    st.markdown("---")
    st.markdown(
        "**보완 권장사항**: 백테스트 모듈, 주문 실행 API 분리, 손실 제한(예: 일손실 -2%) 규칙 추가를 권장합니다."
    )


if __name__ == "__main__":
    main()
