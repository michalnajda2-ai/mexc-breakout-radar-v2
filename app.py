"""MEXC Breakout Radar v2.0.4 — research, paper simulation, and in-app alerts only."""
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

st.set_page_config(page_title="MEXC Breakout Radar v2.0.4", page_icon="📡", layout="wide")
BASE = "https://api.mexc.com"
FUTURES_BASE = "https://contract.mexc.com"
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


def get_json(path, params=None, base=None):
    # No API keys, no retries on 403/418/429, no full URL or secret logging.
    try:
        r = requests.get((base or BASE) + path, params=params, timeout=TIMEOUT,
                         headers={"User-Agent": "MEXC-Radar-Research/2.0.3", "Accept": "application/json"})
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
    data = get_json("/api/v1/contract/ticker", base=FUTURES_BASE)
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


@st.cache_data(ttl=120, show_spinner=False)
def futures_candles(contract, interval="Min60", limit=180):
    """Public contract candles, timestamps in seconds, no spot fallback."""
    if not contract.endswith("_USDT") or not contract.replace("_", "").isalnum():
        raise ValueError("Niepoprawny symbol kontraktu")
    seconds = {"Min60": 3600, "Day1": 86400}.get(interval)
    if seconds is None:
        raise ValueError("Nieobsługiwany interwał futures")
    end = int(time.time())
    start = end - (limit + 3) * seconds
    payload = get_json(f"/api/v1/contract/kline/{contract}",
                       {"interval": interval, "start": start, "end": end}, base=FUTURES_BASE)
    if not isinstance(payload, dict) or not payload.get("success"):
        raise ValueError(f"Błąd API świec futures: {str(payload.get('code', 'brak kodu'))[:40] if isinstance(payload, dict) else 'format'}")
    data = payload.get("data")
    required = ("time", "open", "high", "low", "close", "amount")
    if not isinstance(data, dict) or any(not isinstance(data.get(k), list) for k in required):
        raise ValueError("Niepoprawny format świec futures")
    n = len(data["time"])
    if not n or any(len(data[k]) != n for k in required):
        return pd.DataFrame()
    df = pd.DataFrame({"ts": data["time"], "open": data["open"], "high": data["high"],
                       "low": data["low"], "close": data["close"], "quote_volume": data["amount"]})
    for c in ("open", "high", "low", "close", "quote_volume"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["ts"] = pd.to_datetime(pd.to_numeric(df.ts, errors="coerce"), unit="s", utc=True)
    df["close_ts"] = df.ts + pd.Timedelta(seconds=seconds)
    df = df[(df.close_ts <= pd.Timestamp.now(tz="UTC"))].dropna()
    return df.sort_values("ts").drop_duplicates("ts").tail(limit).reset_index(drop=True)


@st.cache_data(ttl=900, show_spinner=False)
def age_from_futures_history(contract):
    # Only approximate: API window is 120 days, not official listing date.
    df = futures_candles(contract, "Day1", 120)
    if df.empty:
        return None, "Brak historii futures"
    age = (utcnow().date() - df.iloc[0].ts.date()).days
    if len(df) >= 118:
        return 50, "50+ dni (historia futures ograniczona do 120 dni)"
    return age, f"~{age} dni według dostępnych świec futures"


def rsi14(close):
    """Wilder RSI (EMA alpha=1/14), 0..100; insufficient data => NaN."""
    if len(close) < 30:
        return float("nan")
    delta = close.diff()
    up = delta.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
    down = (-delta.clip(upper=0)).ewm(alpha=1/14, adjust=False).mean()
    if down.iloc[-1] == 0:
        return 100.0 if up.iloc[-1] > 0 else 50.0
    return float(100 - 100 / (1 + up.iloc[-1] / down.iloc[-1]))


def compute_signal(symbol, df, fut, cohort, min_turnover, max_spread, max_funding, daily=None):
    """Separate trend strength from entry quality; hard risk gates override scores."""
    if len(df) < 55 or daily is None or len(daily) < 30:
        return None
    close = df.close.astype(float)
    price = float(close.iloc[-1])
    if not np.isfinite(price) or price <= 0:
        return None
    ema9 = close.ewm(span=9, adjust=False).mean()
    ema21 = close.ewm(span=21, adjust=False).mean()
    high20 = float(df.high.iloc[-21:-1].max())
    q = df.quote_volume
    prev = float(q.iloc[-25:-1].mean())
    vol_ratio = float(q.iloc[-1] / prev) if prev > 0 else float("nan")
    prev_close = close.shift(1)
    tr = pd.concat([df.high-df.low, (df.high-prev_close).abs(), (df.low-prev_close).abs()], axis=1).max(axis=1)
    atr = float(tr.tail(14).mean())
    if not np.isfinite(atr) or atr <= 0:
        return None
    bid, ask = safe_float(fut.get("bid1")), safe_float(fut.get("ask1"))
    spread = (ask/bid-1)*100 if bid > 0 and ask >= bid else float("nan")
    funding = safe_float(fut.get("fundingRate"))
    turnover = safe_float(fut.get("amount24"))
    if not all(np.isfinite(x) for x in (spread, funding, turnover)):
        return None
    if turnover < min_turnover or spread > max_spread or funding*100 > max_funding:
        return None
    dclose = daily.close.astype(float)
    daily_ema20 = float(dclose.ewm(span=20, adjust=False).mean().iloc[-1])
    rsi = rsi14(dclose)
    if not np.isfinite(rsi) or daily_ema20 <= 0:
        return None
    # 3d/7d return measured on CLOSED daily candles; unavailable history blocks readiness.
    change3 = (float(dclose.iloc[-1]/dclose.iloc[-4])-1)*100 if len(dclose)>=4 else float("nan")
    change7 = (float(dclose.iloc[-1]/dclose.iloc[-8])-1)*100 if len(dclose)>=8 else float("nan")
    distance_ema = (price/daily_ema20-1)*100
    trend = bool(ema9.iloc[-1]>ema21.iloc[-1] and price>ema21.iloc[-1] and dclose.iloc[-1]>daily_ema20)
    breakout = bool(price>high20)
    volume_ok = bool(np.isfinite(vol_ratio) and vol_ratio>=1.4)
    # These are explicit heuristics, NOT probability estimates or backtested thresholds.
    trend_score = (40 if trend else 0) + (30 if breakout else 0) + (20 if volume_ok else 0) + (10 if spread<0.15 else 0)
    overheated = rsi >= 80 or change3 >= 30 or change7 >= 55 or distance_ema >= 25
    warm = rsi >= 70 or change3 >= 15 or change7 >= 30 or distance_ema >= 12
    entry_score = 100
    reasons = []
    if rsi >= 80:
        entry_score -= 45; reasons.append(f"RSI 1D {rsi:.1f} ≥ 80")
    elif rsi >= 70:
        entry_score -= 20; reasons.append(f"RSI 1D {rsi:.1f} ≥ 70")
    if change3 >= 30:
        entry_score -= 35; reasons.append(f"wzrost 3D +{change3:.1f}% ≥ 30%")
    elif change3 >= 15:
        entry_score -= 15; reasons.append(f"wzrost 3D +{change3:.1f}% ≥ 15%")
    if change7 >= 55:
        entry_score -= 35; reasons.append(f"wzrost 7D +{change7:.1f}% ≥ 55%")
    elif change7 >= 30:
        entry_score -= 15; reasons.append(f"wzrost 7D +{change7:.1f}% ≥ 30%")
    if distance_ema >= 25:
        entry_score -= 35; reasons.append(f"cena {distance_ema:+.1f}% od EMA20 1D (≥25%)")
    elif distance_ema >= 12:
        entry_score -= 15; reasons.append(f"cena {distance_ema:+.1f}% od EMA20 1D (≥12%)")
    if not trend:
        entry_score -= 25; reasons.append("brak potwierdzonego trendu 1H/1D")
    if not breakout:
        entry_score -= 10; reasons.append("brak wybicia maksimum 20 świec 1H")
    if not volume_ok:
        entry_score -= 10; reasons.append(f"wolumen 1H {vol_ratio:.2f}×, wymagane ≥1.4×")
    entry_score = max(0, min(100, entry_score))
    if overheated:
        status = "ODRZUCONY — PRZEGRZANY"
    elif warm:
        status = "CZEKAJ NA KOREKTĘ"
    elif not trend or not breakout or not volume_ok:
        status = "CZEKAJ NA POTWIERDZENIE"
    elif entry_score < 75:
        status = "CZEKAJ NA POTWIERDZENIE"
    else:
        status = "GOTOWY DO OBSERWACJI"
    if not reasons:
        reasons.append("trend 1H/1D, wybicie i wolumen potwierdzone; bez progów przegrzania")
    # No actionable entry/SL/TP for rejected or waiting setups.
    entry = float(ask) if status == "GOTOWY DO OBSERWACJI" else None
    stop = entry - 1.5*atr if entry is not None else None
    target = entry + 2.0*(entry-stop) if entry is not None else None
    if entry is not None and (stop <= 0 or (entry-stop)/entry*100 > 8):
        status = "ODRZUCONY — RYZYKO"
        reasons.append("stop-loss zbyt daleko (>8%) lub niepoprawny")
        entry = stop = target = None
    return {"symbol":symbol,"contract":str(fut.get("symbol")),"group":cohort,
            "score":entry_score,"trend_score":trend_score,"entry_score":entry_score,
            "status":status,"powod":"; ".join(reasons),"rsi_1d":round(rsi,1),
            "zmiana_3d_pct":round(change3,1),"zmiana_7d_pct":round(change7,1),
            "od_ema20_1d_pct":round(distance_ema,1),
            "entry":entry,"stop":stop,"target":target,"spot_close":price,
            "atr_pct":100*atr/price,"volume_ratio":vol_ratio,"spread_pct":spread,
            "funding_pct":100*funding,"turnover":turnover,"trend":trend,"breakout":breakout,
            "quote_time":utcnow().isoformat()}


def db_connect():
    con = sqlite3.connect(DB, timeout=8)
    con.execute("""CREATE TABLE IF NOT EXISTS paper (
        id INTEGER PRIMARY KEY AUTOINCREMENT, created TEXT NOT NULL, symbol TEXT NOT NULL,
        contract TEXT NOT NULL, cohort TEXT NOT NULL, entry REAL NOT NULL,
        stop REAL NOT NULL, target REAL NOT NULL, quantity REAL NOT NULL,
        fee_pct REAL NOT NULL, slippage_pct REAL NOT NULL, status TEXT NOT NULL,
        exit_price REAL, closed TEXT, pnl REAL, reason TEXT)""")
    # Existing positions were opened using spot-based analysis: never silently migrate them.
    cols = {row[1] for row in con.execute("PRAGMA table_info(paper)")}
    if "data_source" not in cols:
        con.execute("ALTER TABLE paper ADD COLUMN data_source TEXT NOT NULL DEFAULT 'spot_legacy'")
    con.commit()
    return con


def paper_open(row, capital, risk_pct, fee_pct, slippage_pct):
    ask = float(row["entry"])
    entry = ask * (1 + slippage_pct / 100)
    stop = float(row["stop"])
    if stop <= 0 or entry <= stop:
        raise ValueError("Niepoprawny stop-loss")
    # Risk includes entry fee and an adverse stop exit with fee and slippage.
    stop_exit = stop * (1 - slippage_pct / 100)
    loss_per_unit = (entry - stop_exit) + (entry + stop_exit) * fee_pct / 100
    if loss_per_unit <= 0:
        raise ValueError("Niepoprawny budżet ryzyka")
    risk_budget = capital * risk_pct / 100
    qty = min(risk_budget / loss_per_unit, capital / (entry * (1 + fee_pct / 100)))
    if not math.isfinite(qty) or qty <= 0:
        raise ValueError("Niepoprawna wielkość pozycji")
    con = db_connect()
    try:
        exists = con.execute("SELECT COUNT(*) FROM paper WHERE contract=? AND status='OPEN'", (row["contract"],)).fetchone()[0]
        if exists:
            raise ValueError("Istnieje już otwarta symulacja tego kontraktu")
        con.execute("""INSERT INTO paper(created,symbol,contract,cohort,entry,stop,target,quantity,fee_pct,slippage_pct,status,data_source)
                       VALUES(?,?,?,?,?,?,?,?,?,?,?,?)""", (utcnow().isoformat(), row["symbol"], row["contract"],
                        row["group"], entry, stop, row["target"], qty, fee_pct, slippage_pct, "OPEN", "futures"))
        con.commit()
    finally:
        con.close()


def paper_update():
    """Closed futures candles; conservative SL when TP and SL share a bar."""
    con = db_connect()
    errors = []
    try:
        rows = con.execute("""SELECT id,created,contract,entry,stop,target,quantity,fee_pct,slippage_pct
                              FROM paper WHERE status='OPEN' AND data_source='futures'""").fetchall()
        for rid, created, contract, entry, stop, target, qty, fee, slip in rows:
            try:
                start = pd.Timestamp(created)
                expiry = start + pd.Timedelta(hours=24)
                df = futures_candles(contract, "Min60", 60)
                if df.empty:
                    errors.append(f"{contract}: brak zamkniętych świec futures")
                    continue
                # Exclude the partially observed entry hour to avoid look-ahead.
                subset = df[df.ts >= start]
                if subset.empty:
                    continue
                reason = None
                exit_price = None
                exit_time = None
                for _, bar in subset.iterrows():
                    if bar.ts >= expiry:
                        break
                    if bar.low <= stop:
                        reason, exit_price = "SL (lub SL i TP w jednej świecy)", stop
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
                pnl = (exit_net - entry) * qty - (entry + exit_net) * qty * fee / 100
                con.execute("UPDATE paper SET status='CLOSED',exit_price=?,closed=?,pnl=?,reason=? WHERE id=?",
                            (exit_net, exit_time.isoformat(), pnl, reason, rid))
            except Exception as e:
                errors.append(f"{contract}: {type(e).__name__}: {str(e)[:120]}")
        con.commit()
    finally:
        con.close()
    return errors


st.title("📡 MEXC Breakout Radar v2.0.4")
st.caption("Badanie setupów futures LONG • osobne grupy 3–49 i 50+ dni • symulacje i alerty w aplikacji • bez zleceń")
st.warning("Brak gwarancji zysków. Oceny to nie prawdopodobieństwa; progi przegrzania nie są zweryfikowane backtestem. Analiza używa zamkniętych świec futures. Bez zleceń.")

with st.sidebar:
    st.header("⚙️ Ustawienia")
    limit = st.slider("Liczba kontraktów do sprawdzenia", 10, 150, 40, 10)
    min_turnover = st.number_input("Min. obrót futures 24h (USDT)", min_value=0.0, value=1_000_000.0, step=250_000.0)
    max_spread = st.slider("Maksymalny spread (%)", 0.05, 2.0, 0.5, 0.05)
    max_funding = st.slider("Maksymalny funding dodatni (%)", 0.0, 0.5, 0.1, 0.01)
    min_score = st.slider("Minimalna jakość wejścia do pokazania (odrzucone widoczne diagnostycznie)", 0, 100, 65, 5)
    capital = st.number_input("Kapitał symulacyjny (USDT)", 100.0, 1_000_000.0, 1000.0, 100.0)
    risk_pct = st.slider("Ryzyko na pozycję (% kapitału)", 0.1, 2.0, 0.5, 0.1)
    fee_pct = st.number_input("Prowizja za jedną stronę (%)", 0.0, 1.0, 0.05, 0.01)
    slip_pct = st.number_input("Szacowany poślizg na stronę (%)", 0.0, 2.0, 0.1, 0.05)
    scan = st.button("🔎 Skanuj futures LONG", type="primary", use_container_width=True)

scan_tab, manual_tab, paper_tab, about_tab = st.tabs(["🔎 Dwa skanery", "🧪 Analiza ręczna", "📒 Symulacje i alerty", "ℹ️ Metodologia"])

with scan_tab:
    st.subheader("Młode (3–49 dni) i starsze (50+ dni)")
    st.caption("Ocena wejścia ≠ prawdopodobieństwo zysku. ODRZUCONY i CZEKAJ nie są sygnałami wejścia. Progi są eksperymentalne.")
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
            skipped_no_spot = 0
            found = []
            errors = []
            blocked = None
            bar = st.progress(0)
            for i, (_, f) in enumerate(futs):
                symbol = str(f["symbol"]).replace("_", "")
                try:
                    age, age_note = age_from_futures_history(str(f["symbol"]))
                    if age is None or age < 3:
                        continue
                    cohort = "3–49 dni" if age <= 49 else "50+ dni"
                    df = futures_candles(str(f["symbol"]), "Min60", 160)
                    daily = futures_candles(str(f["symbol"]), "Day1", 85)
                    result = compute_signal(symbol, df, f, cohort, min_turnover, max_spread, max_funding, daily=daily)
                    if result and (result["score"] >= min_score or result["status"].startswith("ODRZUCONY")):
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
            st.info(f"Sprawdzono {len(futs)} kontraktów. Pokazanych ocen: {len(found)} (w tym odrzucone). "
                    f"Błędy danych: {len(errors)}. Dane: świece futures.")
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
        cols = ["symbol", "entry_score", "trend_score", "status", "powod", "rsi_1d", "zmiana_3d_pct", "zmiana_7d_pct", "od_ema20_1d_pct", "entry", "stop", "target", "wiek_danych"]
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
        update_errors = paper_update()
        if update_errors:
            st.warning("Nie wszystkie symulacje można zaktualizować.")
            with st.expander("Błędy aktualizacji"):
                st.write(update_errors[:20])
    con = db_connect()
    try:
        trades = pd.read_sql_query("SELECT * FROM paper ORDER BY id DESC", con)
    finally:
        con.close()
    if trades.empty:
        st.info("Brak symulacji. Uruchom skaner i wybierz setup do obserwacji.")
    else:
        if ((trades.status == "OPEN") & (trades.data_source == "spot_legacy")).any():
            st.warning("Starsze symulacje spot pozostają w dzienniku, ale nie będą automatycznie rozliczane na świecach futures. Eksportuj je do CSV.")
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
    st.markdown("""**Ograniczenia v2.0**\n\n- Brak prawdziwych zleceń i brak kluczy prywatnych.\n- Cena wejścia pochodzi z orientacyjnego ask futures, a nowe symulacje sprawdzają SL/TP na **zamkniętych świecach futures**. To nadal nie jest wierny backtest wykonania zleceń.\n- Ryzyko nowej symulacji uwzględnia założone prowizje i poślizg przy SL, ale nie gwarantuje limitu straty; funding, luki i likwidacja nie są księgowane.\n- Przebieg ceny wewnątrz świecy jest nieznany; przy jednoczesnym SL i TP zakładamy najgorszy przypadek.\n- Wiek kontraktu jest oszacowany na podstawie ograniczonej historii świec futures, nie oficjalnej daty listingu.\n- Skan obejmuje wybrane kontrakty o największym obrocie, a nie cały rynek.\n- Brak samodzielnych alertów w tle. Streamlit Community Cloud nie jest trwałą bazą danych.\n- Score nie jest skalibrowanym prawdopodobieństwem zysku. Wyniki symulacji nie dowodzą przewagi rynkowej.\n\n**Następny etap:** osobny worker 24/7, trwała baza, funding i rygorystyczne testy historyczne.""")
