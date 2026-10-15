"""MEXC new crypto listings: separate spot/futures, public market data only."""
from datetime import datetime, timezone
import math
import time
import requests
import pandas as pd
import streamlit as st

st.set_page_config(page_title="MEXC Nowe Kryptowaluty 3.1", page_icon="🌱", layout="wide")
MIN_AGE, MAX_AGE, MIN_TURNOVER = 3, 77, 300_000
S = requests.Session()
S.headers.update({"Accept": "application/json", "User-Agent": "Mozilla/5.0 MEXC-Listings-Research/3.1"})
# Conservative exclusion of common non-crypto tokenized instruments. This list is not exhaustive.
NON_CRYPTO = {"SOXS", "SOXL", "USOIL", "UKOIL", "XAU", "XAG", "SILVER", "GOLD", "XAUUSD", "XAGUSD", "SPX", "SPY", "QQQ", "NDX", "DJI", "DXY", "TSLA", "AAPL", "NVDA", "AMZN", "GOOGL", "GOOG", "META", "MSFT", "MSTR", "COIN", "HOOD", "PLTR", "MU", "TSM", "INTC", "AMD", "NFLX", "BABA", "US500", "NAS100", "XAUT"}
NON_CRYPTO_WORDS = ("stock", "equity", "share", "commodity", "precious metal", "forex", "index", "indices", "etf", "tokenized stock", "oil", "silver", "gold")

def num(value):
    try:
        n=float(value)
        return n if math.isfinite(n) else None
    except (ValueError, TypeError):
        return None

def get_json(url, params=None):
    r=S.get(url, params=params, timeout=18)
    r.raise_for_status()
    obj=r.json()
    if isinstance(obj, dict) and (obj.get("success") is False or ("code" in obj and str(obj["code"]) not in ("0", "200", "200000"))):
        raise ValueError(f"MEXC API: {str(obj.get('msg', obj.get('message', obj.get('code'))))[:110]}")
    return obj

def dt_from_timestamp(value):
    v=num(value)
    if v is None or v<=1e9: return None
    if v>1e12: v/=1000
    try: return datetime.fromtimestamp(v, timezone.utc)
    except (ValueError, OverflowError, OSError): return None

def crypto_only(base, metadata=None):
    if not base or base.upper() in NON_CRYPTO: return False
    if not base.replace('_','').isalnum(): return False
    if metadata:
        s=' '.join(str(metadata.get(k,'')) for k in ('displayName','name','symbolName','category','type','underlyingType','description')).lower()
        if any(word in s for word in NON_CRYPTO_WORDS): return False
    return True

@st.cache_data(ttl=180, show_spinner=False)
def spot_universe():
    info=get_json('https://api.mexc.com/api/v3/exchangeInfo')
    ticks=get_json('https://api.mexc.com/api/v3/ticker/24hr')
    if not isinstance(info,dict) or not isinstance(info.get('symbols'),list) or not isinstance(ticks,list):
        raise ValueError('Nieprawidłowe dane SPOT')
    return info['symbols'], {x.get('symbol'):x for x in ticks if isinstance(x,dict)}

@st.cache_data(ttl=180, show_spinner=False)
def futures_universe():
    details=get_json('https://contract.mexc.com/api/v1/contract/detail')
    ticks=get_json('https://contract.mexc.com/api/v1/contract/ticker')
    if not isinstance(details,dict) or not isinstance(ticks,dict): raise ValueError('Nieprawidłowe dane FUTURES')
    ds,ts=details.get('data'),ticks.get('data')
    if not isinstance(ds,list) or not isinstance(ts,list): raise ValueError('Brak list FUTURES')
    return ds,{x.get('symbol'):x for x in ts if isinstance(x,dict)}

@st.cache_data(ttl=86400, show_spinner=False)
def spot_first_candle(symbol):
    # Earliest available daily candle is a proxy for spot listing, not an official listing timestamp.
    data=get_json('https://api.mexc.com/api/v3/klines', {'symbol':symbol,'interval':'1d','startTime':0,'limit':1})
    if not isinstance(data,list) or not data or not isinstance(data[0],list): return None
    return dt_from_timestamp(data[0][0])

def future_date(d):
    for key in ('createTime','create_time','onlineTime','online_time','listingTime','launchTime'):
        date=dt_from_timestamp(d.get(key))
        if date: return date
    return None

def row(symbol, age, date, turnover, price, source):
    return {'Symbol':symbol,'Wiek (dni)':int(age),'Data początku (UTC)':date.strftime('%Y-%m-%d'),
            'Obrót 24h (USDT)':round(turnover,2),'Cena (USDT)':price,'Źródło daty':source}

st.title('🌱 MEXC — nowe kryptowaluty SPOT i FUTURES')
st.write('**Wiek: 3–77 dni · obrót 24h: co najmniej 300 000 USDT · wyłącznie kryptowaluty.**')
st.caption('Dwie osobne tabele. Brak warunku narastającego wolumenu. Obrót to wartość transakcji w USDT w ostatnich 24 godzinach, a nie liczba tokenów.')
st.warning('Odrzucamy znane akcje, ETF-y, indeksy i surowce, lecz MEXC nie dostarcza jednej pewnej klasyfikacji wszystkich instrumentów. Filtr nie daje 100% gwarancji wykluczenia każdego aktywa niekryptowalutowego.')
spot_limit=st.slider('Ile symboli SPOT sprawdzić pod kątem wieku?', 50, 1000, 300, 50, help='SPOT może wymagać pobrania pierwszej świecy dla każdego symbolu. Zwiększanie limitu trwa dłużej.')

if st.button('🔎 Skanuj SPOT i FUTURES',type='primary'):
    now=datetime.now(timezone.utc)
    output={'spot':[],'futures':[],'warnings':[], 'diagnostics':{}, 'time':now.strftime('%Y-%m-%d %H:%M UTC')}
    try:
        with st.spinner('Pobieram rynek SPOT…'):
            symbols,tickers=spot_universe()
            candidates=[]
            stats={'Wszystkie symbole':len(symbols),'Odrzucone: nie-USDT / status / filtr kryptowalut':0,'Obrót poniżej 300 000 USDT':0,'Bez daty':0,'Poza wiekiem':0,'Wyniki':0,'Sprawdzone daty':0}
            for d in symbols:
                sym=d.get('symbol','');base=d.get('baseAsset','')
                if d.get('quoteAsset')!='USDT' or d.get('status') not in (None,'1',1,'TRADING','ENABLED') or not crypto_only(base,d):
                    stats['Odrzucone: nie-USDT / status / filtr kryptowalut']+=1;continue
                t=tickers.get(sym,{})
                vol=num(t.get('quoteVolume'))
                if vol is None or vol<MIN_TURNOVER:
                    stats['Obrót poniżej 300 000 USDT']+=1;continue
                candidates.append((sym,d,t,vol))
            candidates.sort(key=lambda x:x[3],reverse=True)
            if len(candidates)>spot_limit:
                output['warnings'].append(f'SPOT: ograniczenie do {spot_limit} z {len(candidates)} symboli z wystarczającym obrotem. Wyniki nie obejmują całego rynku.')
            bar=st.progress(0)
            for i,(sym,d,t,vol) in enumerate(candidates[:spot_limit]):
                try:
                    date=None
                    for key in ('onboardDate','listingTime','listTime'):
                        date=dt_from_timestamp(d.get(key))
                        if date:break
                    source='metadane MEXC' if date else 'pierwsza dostępna świeca 1D (przybliżenie)'
                    if not date: date=spot_first_candle(sym)
                    stats['Sprawdzone daty']+=1
                    if not date: stats['Bez daty']+=1;continue
                    age=(now-date).total_seconds()/86400
                    if not MIN_AGE<=age<=MAX_AGE: stats['Poza wiekiem']+=1;continue
                    output['spot'].append(row(sym,age,date,vol,num(t.get('lastPrice')),source))
                except Exception as exc:
                    stats['Bez daty']+=1
                    if len(output['warnings'])<8:output['warnings'].append(f'SPOT {sym}: {str(exc)[:100]}')
                    if isinstance(exc,requests.HTTPError) and exc.response is not None and exc.response.status_code in (403,418,429):
                        output['warnings'].append('SPOT: limit API, przerwano skanowanie — wyniki niepełne.');break
                bar.progress((i+1)/max(1,min(spot_limit,len(candidates))))
            stats['Wyniki']=len(output['spot'])
            output['diagnostics']['SPOT']=stats
    except Exception as exc:
        output['warnings'].append('Nie udało się pobrać SPOT: '+str(exc)[:200])
    try:
        with st.spinner('Pobieram rynek FUTURES…'):
            details,tickers=futures_universe()
            stats={'Wszystkie kontrakty':len(details),'Odrzucone: nie-USDT / filtr kryptowalut':0,'Obrót poniżej 300 000 USDT':0,'Bez daty futures':0,'Poza wiekiem':0,'Wyniki':0}
            for d in details:
                sym=d.get('symbol','')
                if not isinstance(sym,str) or not sym.endswith('_USDT') or not crypto_only(sym[:-5],d):
                    stats['Odrzucone: nie-USDT / filtr kryptowalut']+=1;continue
                t=tickers.get(sym,{})
                vol=num(t.get('amount24'))
                if vol is None or vol<MIN_TURNOVER:
                    stats['Obrót poniżej 300 000 USDT']+=1;continue
                date=future_date(d)
                if not date: stats['Bez daty futures']+=1;continue
                age=(now-date).total_seconds()/86400
                if not MIN_AGE<=age<=MAX_AGE: stats['Poza wiekiem']+=1;continue
                output['futures'].append(row(sym.replace('_',''),age,date,vol,num(t.get('lastPrice')),'data kontraktu futures'))
            stats['Wyniki']=len(output['futures'])
            output['diagnostics']['FUTURES']=stats
    except Exception as exc:
        output['warnings'].append('Nie udało się pobrać FUTURES: '+str(exc)[:200])
    st.session_state['v31']=output

if 'v31' in st.session_state:
    r=st.session_state['v31']
    st.caption('Ostatni skan: '+r['time'])
    for key,heading in [('spot','🟢 SPOT — nowe kryptowaluty'),('futures','🔵 FUTURES — nowe kryptowaluty')]:
        st.subheader(heading)
        rows=r[key]
        if rows:
            df=pd.DataFrame(rows).sort_values('Obrót 24h (USDT)',ascending=False)
            st.dataframe(df,use_container_width=True,hide_index=True)
            st.download_button('⬇️ Pobierz CSV '+key.upper(),df.to_csv(index=False).encode('utf-8-sig'),file_name='mexc_'+key+'_v3_1.csv',mime='text/csv',key='download_'+key)
        else:st.info('Brak potwierdzonych wyników w sprawdzonym zakresie. Zobacz diagnostykę.')
    with st.expander('Diagnostyka filtrów i pokrycie rynku',expanded=True):
        for market,stats in r['diagnostics'].items():
            st.markdown('**'+market+'**')
            st.dataframe(pd.DataFrame([{'Kryterium':k,'Liczba':v} for k,v in stats.items()]),hide_index=True,use_container_width=True)
        for w in r['warnings']:st.warning(w)

st.caption('SPOT: wiek liczony od pierwszej dostępnej świecy, jeśli brak daty oficjalnej. FUTURES: od daty uruchomienia kontraktu. To różne daty. Bez API keys i bez składania zleceń.')
