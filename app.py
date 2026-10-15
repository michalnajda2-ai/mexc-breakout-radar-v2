"""MEXC New Coins Volume Radar v3.0: public USDT perpetual data, no orders."""
from datetime import datetime, timezone, timedelta
import math
import time
import requests
import pandas as pd
import streamlit as st

st.set_page_config(page_title='MEXC Nowe Kryptowaluty | Wolumen', page_icon='🌱', layout='wide')
BASES = ('https://contract.mexc.com', 'https://api.mexc.com')
MIN_DAYS, MAX_DAYS = 3, 77
MIN_USDT = 300_000
SESSION = requests.Session()
SESSION.headers.update({'Accept':'application/json', 'User-Agent':'MEXC-New-Listings-Volume-Radar/3.0'})

def api(path, params=None):
    errors=[]
    for base in BASES:
        try:
            response=SESSION.get(base+path, params=params, timeout=12)
            if response.status_code in (403,418,429):
                raise RuntimeError(f'HTTP {response.status_code}: API ogranicza dostęp. Spróbuj później.')
            response.raise_for_status()
            obj=response.json()
            if not isinstance(obj,dict) or obj.get('success') is not True:
                raise ValueError(f'Niepoprawna odpowiedź API ({obj.get("code") if isinstance(obj,dict) else "format"})')
            return obj.get('data')
        except RuntimeError:
            raise
        except (requests.RequestException,ValueError) as exc:
            errors.append(f'{base}: {str(exc)[:95]}')
    raise RuntimeError(' | '.join(errors))

def number(x):
    try:
        v=float(x)
        return v if math.isfinite(v) else None
    except (TypeError, ValueError):
        return None

def listing_date(detail):
    # Metadata are not always published by the exchange. Never substitute spot age.
    for key in ('createTime','create_time','onlineTime','online_time','listingTime','launchTime'):
        v=number(detail.get(key))
        if v and v>1_000_000_000:
            if v>1e12: v/=1000
            try: return datetime.fromtimestamp(v,timezone.utc),key
            except (ValueError,OverflowError,OSError): pass
    return None,None

@st.cache_data(ttl=180,show_spinner=False)
def universe():
    details=api('/api/v1/contract/detail')
    tickers=api('/api/v1/contract/ticker')
    if not isinstance(details,list) or not isinstance(tickers,list):
        raise ValueError('API nie zwróciło listy kontraktów i tickerów')
    return details,tickers

@st.cache_data(ttl=300,show_spinner=False)
def daily(symbol):
    if not isinstance(symbol,str) or not symbol.endswith('_USDT') or not symbol.replace('_','').isalnum():
        raise ValueError('Nieprawidłowy symbol')
    end=int(time.time())
    data=api(f'/api/v1/contract/kline/{symbol}', {'interval':'Day1','start':end-17*86400,'end':end})
    if not isinstance(data,dict) or not all(isinstance(data.get(k),list) for k in ('time','amount','close')):
        raise ValueError('Niepoprawne świece dzienne')
    n=len(data['time'])
    if not n or len(data['amount'])!=n or len(data['close'])!=n:
        return pd.DataFrame()
    df=pd.DataFrame({'ts':pd.to_datetime(pd.to_numeric(data['time'],errors='coerce'),unit='s',utc=True),
                     'usdt':pd.to_numeric(data['amount'],errors='coerce'),
                     'close':pd.to_numeric(data['close'],errors='coerce')})
    df=df.dropna().sort_values('ts').drop_duplicates('ts')
    # Exclude today's unfinished UTC daily bar.
    return df[df.ts+pd.Timedelta(days=1)<=pd.Timestamp.now(tz='UTC')].tail(10).reset_index(drop=True)

def analyze(df):
    if len(df)<6: return None
    q=df.usdt.to_numpy(dtype=float)
    if any(x<0 or not math.isfinite(x) for x in q): return None
    last3=q[-3:]; previous3=q[-6:-3]
    avg_now=float(last3.mean()); avg_before=float(previous3.mean())
    ratio=avg_now/avg_before if avg_before>0 else None
    # Strong and interpretable rule: each of the last 3 closed days increased.
    increasing=bool(last3[0]<last3[1]<last3[2])
    return {'vol_d1':float(q[-1]),'vol_d2':float(q[-2]),'vol_d3':float(q[-3]),
            'avg3':avg_now,'avg_prev3':avg_before,'growth_pct':(ratio-1)*100 if ratio is not None else None,
            'increasing':increasing}

st.title('🌱 MEXC — nowe kryptowaluty z rosnącym wolumenem')
st.caption('Tylko kontrakty futures USDT. Wiek 3–77 dni (11 tygodni), obrót 24h ≥ 300 000 USDT, narastający wolumen z 3 zakończonych dni UTC. Bez składania zleceń.')
st.info('Wiek liczony od daty uruchomienia kontraktu futures, jeśli MEXC ją publikuje. Kontrakty bez wiarygodnej daty nie są zgadywane ani wyświetlane jako spełniające filtr.')
with st.expander('Jak działa filtr rosnącego wolumenu?',expanded=False):
    st.write('Wolumen obrotu w USDT musi rosnąć każdego z trzech ostatnich zakończonych dni: dzień −3 < dzień −2 < dzień −1. Dodatkowo średni dzienny obrót z tych 3 dni musi być wyższy niż z poprzednich 3 dni. Minimum 300 000 USDT dotyczy bieżącego obrotu z ostatnich 24 godzin (rolling 24h).')
max_requests=st.slider('Maksymalna liczba kontraktów do sprawdzenia świec',20,300,150,20,
                       help='Większy limit zwiększa pokrycie, ale także czas skanowania i ryzyko limitów API.')
if st.button('🔎 Skanuj nowe kryptowaluty',type='primary'):
    try:
        details,tickers=universe()
        now=datetime.now(timezone.utc)
        ticker_map={t.get('symbol'):t for t in tickers if isinstance(t,dict)}
        eligible=[]; counts={'Wszystkie kontrakty':len(details),'Brak daty notowania':0,'Poza wiekiem 3–77 dni':0,
                             'Poza futures USDT':0,'Obrót 24h poniżej 300 000 USDT / brak danych':0,
                             'Odrzucone: zbyt krótka historia wolumenu':0,'Odrzucone: wolumen nie narasta':0,
                             'Błędy pobierania świec':0,'Spełniające wszystkie warunki':0}
        for d in details:
            if not isinstance(d,dict):continue
            sym=d.get('symbol','')
            if not isinstance(sym,str) or not sym.endswith('_USDT'):
                counts['Poza futures USDT']+=1;continue
            date,key=listing_date(d)
            if date is None:
                counts['Brak daty notowania']+=1;continue
            age=(now-date).total_seconds()/86400
            if not MIN_DAYS<=age<=MAX_DAYS:
                counts['Poza wiekiem 3–77 dni']+=1;continue
            t=ticker_map.get(sym,{})
            turnover=number(t.get('amount24'))
            if turnover is None or turnover<MIN_USDT:
                counts['Obrót 24h poniżej 300 000 USDT / brak danych']+=1;continue
            eligible.append((sym,age,date,turnover,t))
        eligible.sort(key=lambda r:r[3],reverse=True)
        checked=eligible[:max_requests]
        rows=[];errors=[]
        progress=st.progress(0)
        for i,(sym,age,date,turnover,t) in enumerate(checked):
            try:
                df=daily(sym)
                metric=analyze(df)
                if metric is None:
                    counts['Odrzucone: zbyt krótka historia wolumenu']+=1
                elif not metric['increasing'] or metric['avg3']<=metric['avg_prev3']:
                    counts['Odrzucone: wolumen nie narasta']+=1
                else:
                    rows.append({'Symbol':sym.replace('_',''),'Wiek (dni)':int(age),
                                 'Data futures (UTC)':date.strftime('%Y-%m-%d'),
                                 'Obrót 24h (USDT)':round(turnover),
                                 'Wolumen dzień -3 (USDT)':round(metric['vol_d3']),
                                 'Wolumen dzień -2 (USDT)':round(metric['vol_d2']),
                                 'Wolumen dzień -1 (USDT)':round(metric['vol_d1']),
                                 'Wzrost średniego wolumenu 3d vs poprzednie 3d (%)':round(metric['growth_pct'],1),
                                 'Cena':number(t.get('lastPrice')),
                                 'Dlaczego':'3 kolejne dni wzrostu wolumenu; średnia 3d > poprzednie 3d; obrót 24h ≥ 300 tys. USDT'})
            except Exception as exc:
                counts['Błędy pobierania świec']+=1
                errors.append(f'{sym}: {str(exc)[:110]}')
                if 'HTTP 429' in str(exc) or 'HTTP 418' in str(exc) or 'HTTP 403' in str(exc):
                    st.warning('API ograniczyło dostęp; przerwano skanowanie, wynik jest niepełny.')
                    break
            progress.progress((i+1)/max(1,len(checked)))
        counts['Spełniające wszystkie warunki']=len(rows)
        st.session_state['last_scan']={'rows':rows,'counts':counts,'eligible':len(eligible),
                                      'checked':len(checked),'errors':errors,'time':now.strftime('%Y-%m-%d %H:%M UTC')}
    except Exception as exc:
        st.error(f'Nie udało się pobrać danych z MEXC: {str(exc)[:350]}')

if 'last_scan' in st.session_state:
    result=st.session_state['last_scan']
    st.subheader('Kryptowaluty spełniające wszystkie warunki')
    st.caption(f"Skan: {result['time']}. Kandydaci po wieku i obrocie: {result['eligible']}. Sprawdzono świece: {result['checked']}.")
    if result['checked']<result['eligible']:
        st.warning(f"Wynik niepełny: limit skanowania. Pominięto {result['eligible']-result['checked']} kwalifikujących się kontraktów.")
    if result['rows']:
        table=pd.DataFrame(result['rows']).sort_values('Wzrost średniego wolumenu 3d vs poprzednie 3d (%)',ascending=False)
        st.dataframe(table,use_container_width=True,hide_index=True)
        st.download_button('⬇️ Pobierz CSV',table.to_csv(index=False).encode('utf-8-sig'),file_name='mexc_nowe_rosnacy_wolumen.csv',mime='text/csv')
    else:
        st.info('Brak kontraktów spełniających wszystkie filtry w sprawdzonym zakresie. To nie oznacza, że na całej giełdzie nie ma takich kontraktów.')
    st.subheader('Diagnostyka — dlaczego odpadają kontrakty')
    st.dataframe(pd.DataFrame([{'Filtr':k,'Liczba':v} for k,v in result['counts'].items()]),hide_index=True,use_container_width=True)
    if result['errors']:
        with st.expander(f"Błędy świec ({len(result['errors'])})"):
            st.code('\n'.join(result['errors'][:30]))

st.caption('Wzrost wolumenu nie gwarantuje wzrostu ceny. Dane z publicznego API MEXC; żadnych kluczy API, zleceń ani automatycznego handlu.')
