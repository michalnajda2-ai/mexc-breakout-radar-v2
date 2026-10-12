"""MEXC Breakout Radar v2.0.2 — research, paper simulation, and in-app alerts only."""
import json
import math
import sqlite3
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import streamlit as st

st.set_page_config(page_title="MEXC Breakout Radar v2.0.2", page_icon="📡", layout="wide")
BASE = "https://api.mexc.com"
TIMEOUT = 12
DB = Path("radar_paper.sqlite3")


def utcnow():
    return datetime.now(timezone.utc)


def safe_float(v, default=float("nan")):
    try:
        n = float(v)
        return n if math.isfinite(n) else default
    except (TypeError, ValueError):
        return default


class MarketDataHTTPError(RuntimeError):
    def __init__(self, status, endpoint, detail="", retry_after=""):
        self.status = status
        self.endpoint = endpoint
        self.retry_after = retry_after
        super().__init__(f"HTTP {status} | {endpoint} | {detail[:180]}" + (f" | Retry-After: {retry_after}s" if retry_after else ""))


def get_json(path, params=None):
    # No API keys, no retries on 403/418/429, no full URL or secret logging.
    try:
        r = requests.get(BASE + path, params=params, timeout=TIMEOUT,
                         headers={"User-Agent": "MEXC-Radar-Research/2.0.1", "Accept": "application/json"})
    except requests.RequestException as e:
        raise RuntimeError(f"Błąd sieci {path}: {type(e).__name__}: {str(e)[:160]}") from e
    if not r.ok:
        detail = r.text[:180].replace("\n", " ").replace("\r", " ")
        raise MarketDataHTTPError(r.status_code, path, detail, r.headers.get("Retry-After", ""))
    try:
        return r.json()
    except ValueError as e:
        raise RuntimeError(f"Odpowiedź nie jest JSON: {path}, HTTP {r.status_code}") from e


@st.cache_data(ttl=120, show_spinner=False)
def spot_tickers():
    data = get_json("/api/v3/ticker/24hr")
    if not isinstance(data, list):
        raise ValueError("Nieprawidłowa odpowiedź ticker spot")
    return data


@st.cache_data(ttl=120, show_spinner=False)
def futures_tickers():
    # Public MEXC contract endpoints; failures are visible, not silently replaced with spot data.
    data = get_json("/api/v1/contract/ticker")
    if not isinstance(data, dict) or not data.get("success"):
        raise ValueError("API kontraktów nie zwróciło poprawnej odpowiedzi")
    return data.get("data") or []


@st.cache_data(ttl=120, show_spinner=False)
def candles(symbol, interval="60m", limit=200):
    # MEXC Spot uses 60m for hourly candles; 1h yields HTTP 400 / -1121.
    interval = {"1h": "60m", "1H": "60m", "4H": "4h"}.get(interval, interval)
    data = get_json("/api/v3/klines", {"symbol": symbol, "interval": interval, "limit": limit})
    if not isinstance(data, list) or not data:
        return pd.DataFrame()
    if any(not isinstance(r, list) or len(r) < 8 for r in data):
        raise ValueError("Niepoprawny format świec spot")
    df = pd.DataFrame([r[:8] for r in data], columns=["ts", "open", "high", "low", "close", "volume", "close_ts", "quote_volume"])
    for c in ["open", "high", "low", "close", "volume", "quote_volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["ts"] = pd.to_datetime(df.ts, unit="ms", utc=True)
    df["close_ts"] = pd.to_datetime(pd.to_numeric(df.close_ts, errors="coerce"), unit="ms", utc=True)
    # Never score an incomplete candle.
    df = df[df.close_ts < pd.Timestamp.now(tz="UTC")]
    return df.dropna().sort_values("ts").drop_duplicates("ts")


@st.cache_data(ttl=900, show_spinner=False)
def age_from_spot_history(symbol):
    # 1000 daily candles reaches far enough back to distinguish >=50 days.
    df = candles(symbol, "1d", 120)
    if df.empty:
        return None, "Brak historii"
    age = (utcnow().date() - df.iloc[0].ts.date()).days
    if len(df) >= 119:
        return 50, "50+ (starsza historia; dokładna data nieznana)"
    return age, f"~{age} dni od pierwszej dostępnej świecy"


def compute_signal(symbol, df, fut, cohort, min_turnover, max_spread, max_funding):
    if len(df) < 26:
        return None
    close = df.close
    price = float(close.iloc[-1])
    if price <= 0:
        return None
    ema9 = close.ewm(span=9, adjust=False).mean()
    ema21 = close.ewm(span=21, adjust=False).mean()
    high20 = float(df.high.iloc[-21:-1].max())
    q = df.quote_volume
    prev = float(q.iloc[-25:-1].mean())
    vol_ratio = float(q.iloc[-1] / prev) if prev > 0 else np.nan
    prev_close = close.shift(1)
    tr = pd.concat([(df.high - df.low), (df.high - prev_close).abs(), (df.low - prev_close).abs()], axis=1).max(axis=1)
    atr = float(tr.tail(14).mean())
    if not np.isfinite(atr) or atr <= 0:
        return None
    bid = safe_float(fut.get("bid1"))
    ask = safe_float(fut.get("ask1"))
    spread = (ask / bid - 1) * 100 if bid > 0 and ask >= bid else np.nan
    funding = safe_float(fut.get("fundingRate"))
    turnover = safe_float(fut.get("amount24"))
    if not all(np.isfinite(x) for x in [spread, funding, turnover]):
        return None
    if turnover < min_turnover or spread > max_spread or funding * 100 > max_funding:
        return None
    trend = bool(ema9.iloc[-1] > ema21.iloc[-1] and price > ema21.iloc[-1])
    breakout = bool(price > high20)
    volume_ok = bool(np.isfinite(vol_ratio) and vol_ratio >= 1.4)
    # Score is descriptive, never a calibrated success probability.
    score = 25 + (25 if trend else 0) + (20 if breakout else 0) + (20 if volume_ok else 0)
    score += 10 if spread < 0.15 else 0
    score = min(100, score)
    entry = float(ask)  # indicative quote, not an executable fill
    stop = entry - 1.5 * atr
    target = entry + 2.0 * (entry - stop)
    if stop <= 0:
        return None
    status = "GOTOWY DO OBSERWACJI" if trend and breakout and volume_ok else "CZEKAJ"
    return {"symbol": symbol, "contract": str(fut.get("symbol")), "group": cohort, "score": score,
            "status": status, "spot_close": price, "entry": entry, "stop": stop, "target": target,
            "atr_pct": 100 * atr / price, "volume_ratio": vol_ratio, "spread_pct": spread,
            "funding_pct": 100 * funding, "turnover": turnover, "trend": trend, "breakout": breakout,
            "quote_time": utcnow().isoformat()}


def db_connect():
    con = sqlite3.connect(DB, timeout=8)
    con.execute("""CREATE TABLE IF NOT EXISTS paper (
        id INTEGER PRIMARY KEY AUTOINCREMENT, created TEXT NOT NULL, symbol TEXT NOT NULL,
        contract TEXT NOT NULL, cohort TEXT NOT NULL, entry REAL NOT NULL,
        stop REAL NOT NULL, target REAL NOT NULL, quantity REAL NOT NULL,
        fee_pct REAL NOT NULL, slippage_pct REAL NOT NULL, status TEXT NOT NULL,
        exit_price REAL, closed TEXT, pnl REAL, reason TEXT)""")
    con.commit()
    return con


def paper_open(row, capital, risk_pct, fee_pct, slippage_pct):
    # Conservative entry estimate; still not a guaranteed fill.
    entry = float(row["entry"]) * (1 + slippage_pct / 100)
    stop = float(row["stop"])
    distance = entry - stop
    if distance <= 0:
        raise ValueError("Niepoprawna odległość stop-loss")
    risk_budget = capital * risk_pct / 100
    quantity = risk_budget / distance
    # Cap notional exposure to the simulated account equity.
    quantity = min(quantity, capital / entry)
    if quantity <= 0:
        raise ValueError("Zbyt mała wielkość pozycji")
    con = db_connect()
    try:
        exists = con.execute("SELECT COUNT(*) FROM paper WHERE symbol=? AND status='OPEN'", (row["symbol"],)).fetchone()[0]
        if exists:
            raise ValueError("Istnieje już otwarta symulacja tej pary")
        con.execute("""INSERT INTO paper(created,symbol,contract,cohort,entry,stop,target,quantity,fee_pct,slippage_pct,status)
                       VALUES(?,?,?,?,?,?,?,?,?,?,?)""", (utcnow().isoformat(), row["symbol"], row["contract"], row["group"], entry,
                        stop, row["target"], quantity, fee_pct, slippage_pct, "OPEN"))
        con.commit()
    finally:
        con.close()


def paper_update():
    """Evaluate closed 1h spot candles as a proxy; never claim futures execution accuracy."""
    con = db_connect()
    try:
        rows = con.execute("SELECT id,created,symbol,entry,stop,target,quantity,fee_pct,slippage_pct FROM paper WHERE status='OPEN'").fetchall()
        for rid, created, symbol, entry, stop, target, qty, fee, slip in rows:
            try:
                df = candles(symbol, "1h", 500)
                if df.empty:
                    continue
                start = pd.Timestamp(created)
                subset = df[df.ts >= start]
                if subset.empty:
                    continue
                expiry = start + pd.Timedelta(hours=24)
                reason = None
                exit_price = None
                exit_time = None
                for _, bar in subset.iterrows():
                    if bar.ts > expiry:
                        break
                    if bar.low <= stop and bar.high >= target:
                        reason, exit_price = "SL/TP w tej samej świecy — konserwatywnie SL", stop
                    elif bar.low <= stop:
                        reason, exit_price = "SL", stop
                    elif bar.high >= target:
                        reason, exit_price = "TP", target
                    if reason:
                        exit_time = bar.close_ts
                        break
                if reason is None:
                    expired = subset[subset.close_ts >= expiry]
                    if not expired.empty:
                        last = expired.iloc[0]
                        reason, exit_price, exit_time = "24h", float(last.close), last.close_ts
                if reason is None:
                    continue
                exit_net = exit_price * (1 - slip / 100)
                gross = (exit_net - entry) * qty
                fees = (entry + exit_net) * qty * fee / 100
                pnl = gross - fees
                con.execute("UPDATE paper SET status='CLOSED',exit_price=?,closed=?,pnl=?,reason=? WHERE id=?",
                            (exit_net, exit_time.isoformat(), pnl, reason, rid))
            except Exception:
                # One missing market must not invalidate the other simulations.
                continue
        con.commit()
    finally:
        con.close()


st.title("📡 MEXC Breakout Radar v2.0.2")
st.caption("Badanie setupów futures LONG • osobne grupy 3–49 i 50+ dni • symulacje i alerty w aplikacji • bez zleceń")
st.warning("Nie ma gwarancji zysków. Sygnały są heurystyczne, a symulacje używają świec spot jako przybliżenia ruchu futures. Bez kluczy API i bez realnego handlu.")

with st.sidebar:
    st.header("⚙️ Ustawienia")
    limit = st.slider("Liczba kontraktów do sprawdzenia", 10, 150, 40, 10)
    min_turnover = st.number_input("Min. obrót futures 24h (USDT)", min_value=0.0, value=1_000_000.0, step=250_000.0)
    max_spread = st.slider("Maksymalny spread (%)", 0.05, 2.0, 0.5, 0.05)
    max_funding = st.slider("Maksymalny funding dodatni (%)", 0.0, 0.5, 0.1, 0.01)
    min_score = st.slider("Minimalna ocena setupu", 0, 100, 65, 5)
    capital = st.number_input("Kapitał symulacyjny (USDT)", 100.0, 1_000_000.0, 1000.0, 100.0)
    risk_pct = st.slider("Ryzyko na pozycję (% kapitału)", 0.1, 2.0, 0.5, 0.1)
    fee_pct = st.number_input("Prowizja za jedną stronę (%)", 0.0, 1.0, 0.05, 0.01)
    slip_pct = st.number_input("Szacowany poślizg na stronę (%)", 0.0, 2.0, 0.1, 0.05)
    scan = st.button("🔎 Skanuj futures LONG", type="primary", use_container_width=True)

scan_tab, manual_tab, paper_tab, about_tab = st.tabs(["🔎 Dwa skanery", "🧪 Analiza ręczna", "📒 Symulacje i alerty", "ℹ️ Metodologia"])

with scan_tab:
    st.subheader("Młode (3–49 dni) i starsze (50+ dni)")
    st.caption("Ranking nie oznacza prawdopodobieństwa zysku. Status CZEKAJ nie jest sygnałem wejścia.")
    if scan:
        st.session_state["results"] = []
        st.session_state["scan_blocked"] = None
        try:
            raw = futures_tickers()
            futs = []
            for f in raw:
                name = str(f.get("symbol", ""))
                turnover = safe_float(f.get("amount24"))
                if name.endswith("_USDT") and np.isfinite(turnover) and turnover >= min_turnover:
                    futs.append((turnover, f))
            futs.sort(key=lambda x: x[0], reverse=True)
            futs = futs[:limit]
            # Futures symbols do not always have a matching MEXC spot market.
            spot_symbols = {str(t.get("symbol", "")) for t in spot_tickers() if isinstance(t, dict)}
            skipped_no_spot = 0
            found = []
            errors = []
            blocked = None
            bar = st.progress(0)
            for i, (_, f) in enumerate(futs):
                symbol = str(f["symbol"]).replace("_", "")
                if symbol not in spot_symbols:
                    skipped_no_spot += 1
                    bar.progress((i + 1) / max(1, len(futs)))
                    continue
                try:
                    age, age_note = age_from_spot_history(symbol)
                    if age is None or age < 3:
                        continue
                    cohort = "3–49 dni" if age <= 49 else "50+ dni"
                    df = candles(symbol, "1h", 160)
                    result = compute_signal(symbol, df, f, cohort, min_turnover, max_spread, max_funding)
                    if result and result["score"] >= min_score:
                        result["wiek_danych"] = age_note
                        found.append(result)
                except MarketDataHTTPError as e:
                    errors.append(f"{symbol}: {e}")
                    if e.status in (403, 418, 429, 451):
                        blocked = str(e)
                        break
                except Exception as e:
                    errors.append(f"{symbol}: {type(e).__name__}: {str(e)[:160]}")
                finally:
                    bar.progress((i + 1) / max(1, len(futs)))
            bar.empty()
            st.session_state["results"] = found
            st.session_state["scan_blocked"] = blocked
            if blocked:
                st.error("DANE NIEDOSTĘPNE — MEXC odrzucił żądanie. Skan przerwany, aby nie powtarzać blokowanych zapytań. " + blocked)
            st.info(f"Sprawdzono {len(futs)} kontraktów. Kandydatów: {len(found)}. "
                    f"Błędy danych: {len(errors)}. Bez odpowiednika spot: {skipped_no_spot}.")
            if errors:
                st.warning("Nie wszystkie kontrakty udało się przeanalizować. Brak kandydatów nie oznacza pełnego skanu rynku.")
            if errors:
                with st.expander("Błędy pobierania danych"):
                    st.write(errors[:25])
        except Exception as e:
            st.session_state["results"] = []
            st.session_state["scan_blocked"] = str(e)
            st.error(f"DANE NIEDOSTĘPNE — skanowanie nie powiodło się: {e}")
    results = st.session_state.get("results", [])
    for cohort in ["3–49 dni", "50+ dni"]:
        st.markdown(f"### {'🆕' if cohort.startswith('3') else '🏛️'} {cohort}")
        group = sorted([x for x in results if x["group"] == cohort], key=lambda x: x["score"], reverse=True)
        if not group:
            if st.session_state.get("scan_blocked"):
                st.error("DANE NIEDOSTĘPNE — nie można ocenić tej grupy.")
            else:
                st.info("BRAK SYGNAŁU w tej grupie przy aktualnych ustawieniach.")
            continue
        cols = ["symbol", "score", "status", "entry", "stop", "target", "volume_ratio", "spread_pct", "funding_pct", "wiek_danych"]
        st.dataframe(pd.DataFrame(group)[cols], use_container_width=True, hide_index=True)
        st.download_button(f"Pobierz CSV {cohort}", pd.DataFrame(group).to_csv(index=False), f"radar_{cohort.replace(' ','_')}.csv", "text/csv")
        ready = [x for x in group if x["status"] == "GOTOWY DO OBSERWACJI"]
        if ready:
            names = [x["symbol"] for x in ready]
            chosen = st.selectbox(f"Symulacja LONG — {cohort}", names, key=f"select_{cohort}")
            if st.button(f"Dodaj symulację {chosen}", key=f"add_{cohort}"):
                try:
                    paper_open(next(x for x in ready if x["symbol"] == chosen), capital, risk_pct, fee_pct, slip_pct)
                    st.success("Dodano symulację. Nie wysłano żadnego zlecenia.")
                except Exception as e:
                    st.error(str(e))

with manual_tab:
    symbol = st.text_input("Para spot, np. TIAUSDT", value="TIAUSDT").strip().upper()
    if st.button("Analizuj parę"):
        try:
            age, note = age_from_spot_history(symbol)
            df = candles(symbol, "1h", 160)
            st.write("Historia:", note)
            if len(df) >= 26:
                st.line_chart(df.set_index("ts")["close"])
                st.metric("Ostatnia zamknięta świeca 1h", f"{df.close.iloc[-1]:.8g} USDT")
                st.metric("Wolumen ostatniej zamkniętej świecy (USDT)", f"{df.quote_volume.iloc[-1]:,.0f}")
            else:
                st.warning("Za mało zamkniętych świec 1h")
            st.caption("Analiza spot nie oznacza, że istnieje aktywny kontrakt futures dla tej pary.")
        except Exception as e:
            st.error(f"Błąd analizy: {e}")

with paper_tab:
    st.subheader("📒 Dziennik symulowanych pozycji")
    st.caption("Wyniki aktualizują się po naciśnięciu przycisku. Na Streamlit Community Cloud lokalna baza może zniknąć po restarcie lub redeployu.")
    if st.button("🔄 Aktualizuj wyniki i alerty"):
        paper_update()
    con = db_connect()
    try:
        trades = pd.read_sql_query("SELECT * FROM paper ORDER BY id DESC", con)
    finally:
        con.close()
    if trades.empty:
        st.info("Brak symulacji. Uruchom skaner i wybierz setup do obserwacji.")
    else:
        st.dataframe(trades, use_container_width=True, hide_index=True)
        closed = trades[trades.status == "CLOSED"]
        a, b, c = st.columns(3)
        a.metric("Zamknięte symulacje", len(closed))
        b.metric("Łączny wynik netto (USDT)", f"{closed.pnl.sum():.2f}")
        c.metric("Udział zyskownych", f"{100*(closed.pnl > 0).mean():.1f}%" if len(closed) else "—")
        st.download_button("⬇️ Eksport dziennika CSV", trades.to_csv(index=False), "radar_paper_trades.csv", "text/csv")
        opened = trades[trades.status == "OPEN"]
        if not opened.empty:
            st.warning(f"🔔 Alert w aplikacji: {len(opened)} otwartych symulacji do monitorowania. Powiadomienia push 24/7 nie są jeszcze aktywne.")

with about_tab:
    st.markdown("""**Ograniczenia v2.0**\n\n- Brak prawdziwych zleceń i brak kluczy prywatnych.\n- Cena wejścia pochodzi z orientacyjnego ask futures, ale historia SL/TP jest sprawdzana na świecach **spot**. To nie jest wierny backtest futures.\n- Rozliczenie jest przybliżone: prowizje i poślizg są parametrami użytkownika, funding nie jest księgowany.\n- Przebieg ceny wewnątrz świecy jest nieznany; przy jednoczesnym SL i TP zakładamy najgorszy przypadek.\n- Wiek pary jest oszacowany na podstawie dostępnej historii spot, nie oficjalnej daty listingu futures.\n- Skan obejmuje wybrane kontrakty o największym obrocie, a nie cały rynek.\n- Brak samodzielnych alertów w tle. Streamlit Community Cloud nie jest trwałą bazą danych.\n- Score nie jest skalibrowanym prawdopodobieństwem zysku. Wyniki symulacji nie dowodzą przewagi rynkowej.\n\n**Następny etap:** osobny worker 24/7, trwała baza, świece kontraktów futures, funding i rygorystyczne testy historyczne.""")
