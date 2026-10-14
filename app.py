"""MEXC Breakout Radar v2.0.7 — research, paper simulation, and in-app alerts only."""
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

st.set_page_config(page_title="MEXC Breakout Radar v2.0.7", page_icon="📡", layout="wide")
BASE = "https://api.mexc.com"
FUTURES_BASE = "https://api.mexc.com"
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
                         headers={"User-Agent": "MEXC-Radar-Research/2.0.7", "Accept": "application/json"})
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
    """Early-base watchlist. No momentum-chasing scores or future-return predictions."""
    if len(df) < 65 or daily is None or len(daily) < 35:
        return None
    close = df.close.astype(float)
    dclose = daily.close.astype(float)
    price = float(close.iloc[-1])
    if not np.isfinite(price) or price <= 0:
        return None
    bid, ask = safe_float(fut.get("bid1")), safe_float(fut.get("ask1"))
    spread = (ask / bid - 1) * 100 if bid > 0 and ask >= bid else float("nan")
    funding, turnover = safe_float(fut.get("fundingRate")), safe_float(fut.get("amount24"))
    if not all(np.isfinite(x) for x in (spread, funding, turnover)):
        return None
    liquidity_reasons = []
    if turnover < min_turnover: liquidity_reasons.append(f"obrót {turnover:,.0f} USDT < {min_turnover:,.0f}")
    if spread > max_spread: liquidity_reasons.append(f"spread {spread:.2f}% > {max_spread:.2f}%")
    if funding * 100 > max_funding: liquidity_reasons.append(f"funding {funding*100:.3f}% > {max_funding:.3f}%")
    rsi_d = rsi14(dclose)
    rsi_h = rsi14(close)
    ema20 = float(dclose.ewm(span=20, adjust=False).mean().iloc[-1])
    change3 = (float(dclose.iloc[-1] / dclose.iloc[-4]) - 1) * 100
    change7 = (float(dclose.iloc[-1] / dclose.iloc[-8]) - 1) * 100
    change24h = (price / float(close.iloc[-25]) - 1) * 100
    distance_ema = (price / ema20 - 1) * 100
    high20 = float(df.high.iloc[-21:-1].max())
    base = df.iloc[-49:-1]
    base_high, base_low = float(base.high.max()), float(base.low.min())
    base_width = (base_high / base_low - 1) * 100 if base_low > 0 else float("inf")
    recent_range = (float(df.high.iloc[-13:-1].max()) / float(df.low.iloc[-13:-1].min()) - 1) * 100
    previous_range = (float(df.high.iloc[-37:-13].max()) / float(df.low.iloc[-37:-13].min()) - 1) * 100
    contraction = recent_range < previous_range * 0.8 if previous_range > 0 else False
    q = df.quote_volume.astype(float)
    vol_base = float(q.iloc[-25:-1].mean())
    vol_ratio = float(q.iloc[-1] / vol_base) if vol_base > 0 else float("nan")
    ema9 = close.ewm(span=9, adjust=False).mean()
    ema21 = close.ewm(span=21, adjust=False).mean()
    rising = bool(ema9.iloc[-1] > ema21.iloc[-1] and ema21.iloc[-1] >= ema21.iloc[-5])
    # Never rank a developed rally as an early-stage setup.
    gates = []
    if rsi_h >= 72: gates.append(f"RSI 1H {rsi_h:.1f} ≥ 72")
    if rsi_d >= 70: gates.append(f"RSI 1D {rsi_d:.1f} ≥ 70")
    if change24h >= 8: gates.append(f"wzrost 24H +{change24h:.1f}% ≥ 8%")
    if change3 >= 12: gates.append(f"wzrost 3D +{change3:.1f}% ≥ 12%")
    if change7 >= 22: gates.append(f"wzrost 7D +{change7:.1f}% ≥ 22%")
    if distance_ema >= 9: gates.append(f"cena {distance_ema:+.1f}% od EMA20 1D ≥ 9%")
    if price > high20 * 1.025: gates.append("wybicie >2.5% ponad wcześniejsze maksimum 20H")
    # Pre-breakout: base under resistance, not a candle already racing away.
    near_resistance = 0.96 * base_high <= price <= 1.012 * base_high
    base_ok = base_width <= 14
    daily_not_weak = 43 <= rsi_d < 70
    early_volume = np.isfinite(vol_ratio) and 0.65 <= vol_ratio <= 2.5
    features = {
        "baza_48h": (base_ok, 25, f"baza 48H {base_width:.1f}% (cel ≤14%)"),
        "kompresja": (contraction, 20, f"kompresja zmienności 12H/24H: {recent_range:.1f}% / {previous_range:.1f}%"),
        "blisko_oporu": (near_resistance, 20, f"cena do oporu 48H: {(price/base_high-1)*100:+.1f}%"),
        "wczesny_trend": (rising, 15, "EMA9 > EMA21 i EMA21 rośnie (1H)"),
        "RSI_umiarkowane": (daily_not_weak, 10, f"RSI 1D {rsi_d:.1f} w przedziale 43–70"),
        "wolumen": (early_volume, 10, f"wolumen ostatniej 1H {vol_ratio:.2f}× średniej"),
    }
    score = sum(weight for passed, weight, _ in features.values() if passed)
    positives = [desc for passed, _, desc in features.values() if passed]
    missing = [desc for passed, _, desc in features.values() if not passed]
    critical = ["baza_48h", "kompresja", "blisko_oporu", "wczesny_trend", "RSI_umiarkowane"]
    missing_critical = [k for k in critical if not features[k][0]]
    if liquidity_reasons:
        status = "ODRZUCONE — PŁYNNOŚĆ"
    elif gates:
        status = "ODRZUCONE — RAJD / PRZEGRZANIE"
    elif not missing_critical and score >= 75:
        status = "PRZED WYBICIEM"
    elif len(missing_critical) <= 2 and score >= 55:
        status = "BLISKO SETUPU"
    else:
        status = "ODRZUCONE — BRAK BAZY"
    reason_codes = list(liquidity_reasons) + gates + ["brak: " + k for k in missing_critical]

    # A watchlist is not an order signal: no actionable entry, stop or target.
    return {"symbol": symbol, "contract": str(fut.get("symbol")), "group": cohort,
            "score": score, "entry_score": score,
            "trend_score": None, "status": status,
            "powod": ("BLOKADA: " + "; ".join(liquidity_reasons + gates) + ". " if liquidity_reasons or gates else "") +
                     "SPEŁNIONE: " + ("; ".join(positives) or "brak") +
                     ". BRAKUJE: " + ("; ".join(missing) or "nic"),
            "brakujace_warunki": len(missing_critical), "powody_odrzucenia": "; ".join(reason_codes) or "—",
            "rsi_1d": round(rsi_d, 1), "rsi_1h": round(rsi_h, 1),
            "zmiana_24h_pct": round(change24h, 1),
            "zmiana_3d_pct": round(change3, 1), "zmiana_7d_pct": round(change7, 1),
            "od_ema20_1d_pct": round(distance_ema, 1), "szerokosc_bazy_48h_pct": round(base_width, 1),
            "entry": None, "stop": None, "target": None, "spot_close": price,
            "volume_ratio": vol_ratio, "spread_pct": spread,
            "funding_pct": funding * 100, "turnover": turnover,
            "quote_time": utcnow().isoformat()}


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


st.title("📡 MEXC Breakout Radar v2.0.7")
st.caption("Badanie setupów futures LONG • osobne grupy 3–49 i 50+ dni • symulacje i alerty w aplikacji • bez zleceń")
st.warning("Wersja eksperymentalna: szuka konsolidacji przed możliwym wybiciem, nie przewiduje rajdów. Brak sygnałów wejścia i nowych symulacji z listy. Bez zleceń.")

with st.sidebar:
    st.header("⚙️ Ustawienia")
    limit = st.slider("Liczba kontraktów do sprawdzenia", 10, 150, 40, 10)
    min_turnover = st.number_input("Min. obrót futures 24h (USDT)", min_value=0.0, value=1_000_000.0, step=250_000.0)
    max_spread = st.slider("Maksymalny spread (%)", 0.05, 2.0, 0.5, 0.05)
    max_funding = st.slider("Maksymalny funding dodatni (%)", 0.0, 0.5, 0.1, 0.01)
    min_score = st.slider("Minimalna ocena bazy do listy obserwacyjnej", 0, 100, 65, 5)
    capital = st.number_input("Kapitał symulacyjny (USDT)", 100.0, 1_000_000.0, 1000.0, 100.0)
    risk_pct = st.slider("Ryzyko na pozycję (% kapitału)", 0.1, 2.0, 0.5, 0.1)
    fee_pct = st.number_input("Prowizja za jedną stronę (%)", 0.0, 1.0, 0.05, 0.01)
    slip_pct = st.number_input("Szacowany poślizg na stronę (%)", 0.0, 2.0, 0.1, 0.05)
    scan = st.button("🔎 Skanuj futures LONG", type="primary", use_container_width=True)

scan_tab, manual_tab, paper_tab, traders_tab, about_tab = st.tabs(["🔎 Dwa skanery", "🧪 Analiza ręczna", "📒 Symulacje i alerty", "🏆 Traderzy futures", "ℹ️ Metodologia"])

with scan_tab:
    st.subheader("Młode (3–49 dni) i starsze (50+ dni)")
    st.caption("Trzy kategorie: PRZED WYBICIEM, BLISKO SETUPU, ODRZUCONE. Przyczyny i liczniki filtrów. Progi eksperymentalne.")
    if scan:
        st.session_state["classified"] = []
        st.session_state["counters"] = {}
        st.session_state["errors"] = []
        st.session_state["results"] = []
        st.session_state["scan_blocked"] = None
        try:
            raw = futures_tickers()
            futs = []
            for f in raw:
                name = str(f.get("symbol", ""))
                turnover = safe_float(f.get("amount24"))
                # Exclude stocks, leveraged ETFs, commodities and index proxies; ticker alone
                # cannot classify every synthetic instrument, so also reject known families.
                base_asset = name[:-5] if name.endswith("_USDT") else ""
                banned = ("SOXS", "SOXL", "USOIL", "XAU", "XAG", "SILVER", "GOLD", "MUSTOCK", "TSLA", "NVDA", "AAPL", "SPY", "QQQ", "STOCK", "OIL")
                if (name.endswith("_USDT") and not any(x in base_asset for x in banned)
                        and np.isfinite(turnover) and turnover >= min_turnover):
                    futs.append((turnover, f))
            futs.sort(key=lambda x: x[0], reverse=True)
            futs = futs[:limit]
            classified = []
            errors = []
            blocked = None
            counters = {"przed": 0, "blisko": 0, "rajd": 0, "plynnosc": 0, "brak_bazy": 0,
                        "brak_historii": 0, "brak_wskaznikow": 0}
            bar = st.progress(0)
            for i, (_, f) in enumerate(futs):
                symbol = str(f["symbol"]).replace("_", "")
                try:
                    age, age_note = age_from_futures_history(str(f["symbol"]))
                    if age is None or age < 3:
                        counters["brak_historii"] += 1
                        continue
                    cohort = "3–49 dni" if age <= 49 else "50+ dni"
                    df = futures_candles(str(f["symbol"]), "Min60", 160)
                    daily = futures_candles(str(f["symbol"]), "Day1", 85)
                    result = compute_signal(symbol, df, f, cohort, min_turnover, max_spread, max_funding, daily=daily)
                    if result is None:
                        counters["brak_wskaznikow"] += 1
                        continue
                    result["wiek_danych"] = age_note
                    classified.append(result)
                    status = result["status"]
                    if status == "PRZED WYBICIEM": counters["przed"] += 1
                    elif status == "BLISKO SETUPU": counters["blisko"] += 1
                    elif "RAJD" in status: counters["rajd"] += 1
                    elif "PŁYNNOŚĆ" in status: counters["plynnosc"] += 1
                    else: counters["brak_bazy"] += 1
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
            st.session_state["classified"] = classified
            st.session_state["counters"] = counters
            st.session_state["errors"] = errors
            st.session_state["scan_blocked"] = blocked
            if blocked:
                st.error("DANE NIEDOSTĘPNE — skan przerwany. " + blocked)
            st.info(f"Wybrano {len(futs)} kontraktów według obrotu. Przeanalizowano: {len(classified)}. "
                    f"Przed wybiciem: {counters['przed']}; blisko setupu: {counters['blisko']}; "
                    f"odrzucone: {counters['rajd']+counters['plynnosc']+counters['brak_bazy']}. "
                    f"Błędy API: {len(errors)}. Świece futures.")
            if errors:
                st.warning("Nie wszystkie kontrakty udało się przeanalizować. Brak kandydatów nie oznacza pełnego skanu rynku.")
            if errors:
                with st.expander("Błędy pobierania danych"):
                    st.write(errors[:25])
        except Exception as e:
            st.session_state["classified"] = []
            st.session_state["counters"] = {}
            st.session_state["scan_blocked"] = str(e)
            st.error(f"DANE NIEDOSTĘPNE — skanowanie nie powiodło się: {e}")
    classified = st.session_state.get("classified", [])
    counters = st.session_state.get("counters", {})
    if counters:
        st.markdown("### 🔍 Diagnostyka filtrów")
        st.dataframe(pd.DataFrame([
            {"Filtr / wynik": "PRZED WYBICIEM", "Liczba": counters.get("przed", 0)},
            {"Filtr / wynik": "BLISKO SETUPU", "Liczba": counters.get("blisko", 0)},
            {"Filtr / wynik": "Rajd / przegrzanie", "Liczba": counters.get("rajd", 0)},
            {"Filtr / wynik": "Płynność / spread / funding", "Liczba": counters.get("plynnosc", 0)},
            {"Filtr / wynik": "Brak bazy / warunków", "Liczba": counters.get("brak_bazy", 0)},
            {"Filtr / wynik": "Za krótka historia / <3 dni", "Liczba": counters.get("brak_historii", 0)},
            {"Filtr / wynik": "Brak wymaganych wskaźników", "Liczba": counters.get("brak_wskaznikow", 0)},
            {"Filtr / wynik": "Błędy API", "Liczba": len(st.session_state.get("errors", []))},
        ]), hide_index=True, use_container_width=True)
        st.caption("Kategorie rozłączne: każdy kontrakt trafia do pierwszej właściwej kategorii. Odrzucone instrumenty pozostają widoczne tylko w diagnostyce.")
    cols = ["symbol", "entry_score", "status", "powod", "brakujace_warunki", "rsi_1h", "rsi_1d", "zmiana_24h_pct", "zmiana_3d_pct", "zmiana_7d_pct", "szerokosc_bazy_48h_pct", "od_ema20_1d_pct", "wiek_danych"]
    for cohort in ["3–49 dni", "50+ dni"]:
        st.markdown(f"### {'🆕' if cohort.startswith('3') else '🏛️'} {cohort}")
        group = [x for x in classified if x["group"] == cohort]
        for status, heading in [("PRZED WYBICIEM", "🟢 PRZED WYBICIEM"), ("BLISKO SETUPU", "🟡 BLISKO SETUPU")]:
            st.markdown(f"**{heading}**")
            rows = sorted([x for x in group if x["status"] == status and (status != "PRZED WYBICIEM" or x["score"] >= min_score)], key=lambda x: (-x["score"], x["brakujace_warunki"]))
            if rows:
                st.dataframe(pd.DataFrame(rows)[cols], use_container_width=True, hide_index=True)
                st.download_button(f"Pobierz CSV {heading} {cohort}", pd.DataFrame(rows).to_csv(index=False), f"radar_{status.replace(' ', '_')}_{cohort.replace(' ','_')}.csv", "text/csv", key=f"csv_{status}_{cohort}")
            else:
                st.info("Brak instrumentów w tej kategorii.")
        rejected = [x for x in group if x["status"].startswith("ODRZUCONE")]
        with st.expander(f"🔴 ODRZUCONE — diagnostyka ({len(rejected)})"):
            if rejected:
                st.dataframe(pd.DataFrame(rejected)[cols], use_container_width=True, hide_index=True)
                st.download_button(f"Pobierz odrzucone CSV {cohort}", pd.DataFrame(rejected).to_csv(index=False), f"odrzucone_{cohort.replace(' ','_')}.csv", "text/csv", key=f"rej_{cohort}")
            else:
                st.caption("Brak odrzuconych z dostępnymi wskaźnikami.")
    st.caption("Żadna kategoria nie stanowi rekomendacji transakcji. Bez automatycznych zleceń i nowych symulacji z listy.")

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

with traders_tab:
    st.subheader("🏆 Publiczni traderzy futures — lista obserwacyjna")
    st.warning("To nie są zweryfikowane konta o trwałej przewadze. Dane publiczne bywają opóźnione, a wysoki ROI lub win rate nie oznacza niskiego ryzyka. Bez automatycznego kopiowania transakcji.")
    st.markdown("[Otwórz ranking MEXC Copy Trading](https://www.mexc.com/futures/copyTrade/home) · [Porównaj traderów na MEXC](https://www.mexc.com/futures/copyTrade/leaderVS)")
    st.caption("Na giełdzie scentralizowanej publiczny profil tradera nie jest jego adresem portfela on-chain. Nie przypisujemy anonimowym adresom niepotwierdzonych wyników.")
    st.markdown("**Przykładowy publiczny profil do sprawdzenia — nie rekomendacja:** [MEXC DD0516](https://www.mexc.com/futures/copyTrade/leader/DD0516). Sprawdź aktualne wyniki na giełdzie; nie są pobierane automatycznie.")
    st.markdown("#### Własna lista profili i porównanie wyników")
    st.caption("Wpisz dane widoczne na profilu tradera (najlepiej 90–180 dni). Puste lub niepełne dane nie otrzymują wysokiej oceny. Dane przechowywane tylko w bieżącej sesji Streamlit — pobierz CSV, aby ich nie stracić.")
    if "trader_watch" not in st.session_state:
        st.session_state.trader_watch = []
    with st.form("trader_add", clear_on_submit=True):
        name = st.text_input("Nazwa publiczna / identyfikator profilu")
        url = st.text_input("Pełny link do profilu MEXC", placeholder="https://www.mexc.com/futures/copyTrade/leader/...")
        c1, c2, c3 = st.columns(3)
        with c1:
            days = st.number_input("Okres oceny (dni)", 0, 365, 90, 1)
            trades_n = st.number_input("Liczba zamkniętych transakcji", 0, 100000, 0, 1)
        with c2:
            pnl = st.number_input("PNL (USDT) w okresie", value=0.0, step=100.0)
            roi = st.number_input("ROI (%) w okresie", value=0.0, step=1.0)
        with c3:
            win = st.number_input("Wygrane (%)", 0.0, 100.0, 0.0, 1.0)
            dd = st.number_input("Maks. obsunięcie (%) — 0 jeśli nieznane", 0.0, 100.0, 0.0, 1.0)
        notes = st.text_input("Uwagi / źródło danych / data odczytu")
        submitted = st.form_submit_button("➕ Dodaj do obserwowanych")
    if submitted:
        if not name.strip() or not (url.startswith("https://www.mexc.com/futures/copyTrade/leader/") or url.startswith("https://www.mexc.co/futures/copyTrade/leader/")):
            st.error("Podaj nazwę i poprawny publiczny link profilu MEXC (ścieżka /futures/copyTrade/leader/...).")
        else:
            enough = days >= 90 and trades_n >= 30 and dd > 0
            if not enough:
                verdict = "NIEZWERYFIKOWANY — za krótki okres, za mało transakcji lub brak drawdown"
                quality = None
            elif pnl <= 0 or roi <= 0 or dd >= 35:
                verdict = "WYSOKIE RYZYKO / BRAK POTWIERDZONEJ RENTOWNOŚCI"
                quality = 0
            else:
                quality = round(max(0, min(100, 45 + min(15, trades_n / 20) + min(15, days / 12) + min(10, win / 10) - dd * 1.2)))
                verdict = "DO DALSZEJ WERYFIKACJI" if quality >= 65 else "OSTROŻNIE — SŁABE PARAMETRY"
            st.session_state.trader_watch.append({"profil": name.strip(), "url": url.strip(), "dni": days, "transakcje": trades_n, "PNL_USDT": pnl, "ROI_pct": roi, "win_rate_pct": win, "max_drawdown_pct": dd if dd else None, "ocena_wstepna": quality, "status": verdict, "uwagi": notes, "odczyt_UTC": utcnow().isoformat(timespec="minutes")})
            st.success("Dodano do listy obserwacyjnej. Ocena jest heurystyczna, nie rankingiem potwierdzonych zwycięzców.")
    if st.session_state.trader_watch:
        df_traders = pd.DataFrame(st.session_state.trader_watch)
        st.dataframe(df_traders, use_container_width=True, hide_index=True, column_config={"url": st.column_config.LinkColumn("Profil MEXC")})
        st.download_button("⬇️ Eksport profili CSV", df_traders.to_csv(index=False).encode("utf-8-sig"), "mexc_trader_watchlist.csv", "text/csv")
        if st.button("🗑️ Wyczyść listę sesji"):
            st.session_state.trader_watch = []
            st.rerun()
    else:
        st.info("Lista jest pusta. Otwórz ranking MEXC, wybierz publiczny profil i wpisz dane z jego statystyk.")
    st.caption("Ocena wstępna nie uwzględnia dźwigni, ryzyka likwidacji, wielkości kapitału ani survivorship bias. Sam win rate nie wystarcza. Żadnych kluczy API, zleceń ani kopiowania pozycji.")

with about_tab:
    st.markdown("""**Ograniczenia v2.0**\n\n- Brak prawdziwych zleceń i brak kluczy prywatnych.\n- Cena wejścia pochodzi z orientacyjnego ask futures, a nowe symulacje sprawdzają SL/TP na **zamkniętych świecach futures**. To nadal nie jest wierny backtest wykonania zleceń.\n- Ryzyko nowej symulacji uwzględnia założone prowizje i poślizg przy SL, ale nie gwarantuje limitu straty; funding, luki i likwidacja nie są księgowane.\n- Przebieg ceny wewnątrz świecy jest nieznany; przy jednoczesnym SL i TP zakładamy najgorszy przypadek.\n- Wiek kontraktu jest oszacowany na podstawie ograniczonej historii świec futures, nie oficjalnej daty listingu.\n- Skan obejmuje wybrane kontrakty o największym obrocie, a nie cały rynek.\n- Brak samodzielnych alertów w tle. Streamlit Community Cloud nie jest trwałą bazą danych.\n- Score nie jest skalibrowanym prawdopodobieństwem zysku. Wyniki symulacji nie dowodzą przewagi rynkowej.\n\n**Następny etap:** osobny worker 24/7, trwała baza, funding i rygorystyczne testy historyczne.""")
