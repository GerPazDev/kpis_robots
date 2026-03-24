# app.py
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="openpyxl")

import streamlit as st
import pandas as pd
import numpy as np
import re
from io import BytesIO
from bs4 import BeautifulSoup

st.set_page_config(page_title="KPIs por Robot — MT4 & MT5", layout="wide")
st.title("📊 KPIs por Robot 🦅 Aguila Trading (MT4 & MT5)")

# ── Preservar pestaña activa entre reruns ──────────────────────────────────
if "active_tab" not in st.session_state:
    st.session_state.active_tab = 0

# ========= Métricas =========
def max_drawdown(equity: pd.Series) -> float:
    if equity.empty:
        return 0.0
    roll_max = equity.cummax()
    dd = roll_max - equity
    return float(dd.max())

def profit_factor(pnl: pd.Series) -> float:
    gains = pnl[pnl > 0].sum()
    losses = pnl[pnl < 0].sum()
    return float(gains / abs(losses)) if losses < 0 else (np.inf if gains > 0 else 0.0)

def expectancy(pnl: pd.Series) -> float:
    return float(pnl.mean()) if len(pnl) else 0.0

def sharpe_per_trade(pnl: pd.Series) -> float:
    if len(pnl) < 2:
        return 0.0
    std = pnl.std(ddof=1)
    if std == 0 or np.isnan(std):
        return 0.0
    return float(pnl.mean() / std * np.sqrt(len(pnl)))

def stability_r2(equity: pd.Series) -> float:
    y = equity.values.astype(float)
    if len(y) < 2 or np.allclose(y.std(), 0):
        return 0.0
    x = np.arange(len(y), dtype=float)
    x_mean, y_mean = x.mean(), y.mean()
    cov = ((x - x_mean) * (y - y_mean)).sum()
    var_x = ((x - x_mean) ** 2).sum()
    if var_x == 0:
        return 0.0
    beta = cov / var_x
    alpha = y_mean - beta * x_mean
    y_hat = alpha + beta * x
    sse = ((y - y_hat) ** 2).sum()
    sst = ((y - y_mean) ** 2).sum()
    return float(max(0.0, 1.0 - sse / sst)) if sst > 0 else 0.0

def max_stagnation(times: pd.Series | None, equity: pd.Series) -> str:
    e = equity.values.astype(float)
    if len(e) == 0:
        return "0"
    peaks = np.maximum.accumulate(e)
    uw = e < peaks
    if not uw.any():
        return "0"
    longest = 0
    curr = 0
    start_idx = best_start = best_end = None
    for i, flag in enumerate(uw):
        if flag:
            curr += 1
            if curr == 1:
                start_idx = i
            if curr > longest:
                longest = curr
                best_start, best_end = start_idx, i
        else:
            curr = 0
    if times is not None and pd.api.types.is_datetime64_any_dtype(times):
        t0, t1 = times.iloc[best_start], times.iloc[best_end]
        if pd.isna(t0) or pd.isna(t1):
            return f"{longest} trades"
        return f"{(t1 - t0).days} d"
    return f"{longest} trades"

# ========= Helpers =========
def _rename_dupes(cols):
    seen, out = {}, []
    for c in cols:
        if c != c:  # NaN
            out.append(None); continue
        if c not in seen:
            seen[c] = 0; out.append(c)
        else:
            seen[c] += 1; out.append(f"{c}_{seen[c]}")
    return out

def _ensure_fee_cols(df: pd.DataFrame) -> pd.DataFrame:
    for c in ["commission", "swap", "profit"]:
        if c not in df.columns:
            df[c] = 0.0
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    df["real_profit"] = df["profit"] + df["commission"] + df["swap"]
    return df

def _merge_deals_to_trades(df: pd.DataFrame) -> pd.DataFrame:
    if "profit" not in df.columns or len(df) < 2:
        return df
        
    zeros = (df["profit"] == 0).sum()
    if zeros < len(df) * 0.3:
        return df

    trades = []
    group_col = "Magic" if "Magic" in df.columns and df["Magic"].notna().any() else "Comment"
    open_deals = {}
    
    time_col = "close_time" if "close_time" in df.columns else "open_time"
    if time_col not in df.columns:
        return df
        
    df = df.sort_values(time_col)
    
    for idx, row in df.iterrows():
        if "type" in df.columns and str(row["type"]).lower() in ["balance", "deposit", "withdrawal"]:
            continue
            
        grp = row.get(group_col, "UNKNOWN")
        sym = row.get("symbol", "UNKNOWN")
        key = (grp, sym)
        
        if key not in open_deals:
            open_deals[key] = []
            
        is_out = False
        
        if float(row.get("profit", 0) or 0) != 0:
            is_out = True
        elif isinstance(row.get("Comment"), str) and any(x in str(row["Comment"]).lower() for x in ["[tp", "[sl", "close"]):
            is_out = True
        elif "direction" in df.columns and str(row["direction"]).lower() == "out":
            is_out = True
            
        if not is_out and len(open_deals[key]) > 0:
            last_deal = open_deals[key][0]
            if "type" in row and "type" in last_deal:
                t1, t2 = str(last_deal["type"]).lower(), str(row["type"]).lower()
                if (t1 == "buy" and t2 == "sell") or (t1 == "sell" and t2 == "buy"):
                    is_out = True
                    
        if not is_out:
            open_deals[key].append(row)
        else:
            if len(open_deals[key]) > 0:
                in_deal = open_deals[key].pop(0)
                trade = row.to_dict()
                trade["open_time"] = in_deal.get(time_col, row.get(time_col))
                trade["close_time"] = row.get(time_col)
                trade["open_price"] = in_deal.get("open_price", row.get("open_price"))
                trade["close_price"] = row.get("close_price", row.get("open_price"))
                trade["commission"] = float(in_deal.get("commission", 0) or 0) + float(row.get("commission", 0) or 0)
                trade["swap"] = float(in_deal.get("swap", 0) or 0) + float(row.get("swap", 0) or 0)
                trade["profit"] = float(row.get("profit", 0) or 0)
                trade["Comment"] = in_deal.get("Comment", row.get("Comment"))
                trade["Magic"] = in_deal.get("Magic", row.get("Magic"))
                trades.append(trade)
            else:
                trades.append(row.to_dict())
                
    if not trades:
        return df
        
    return pd.DataFrame(trades)

# ========= Parsing MT5 (XLSX) =========
@st.cache_data(show_spinner=False)
def parse_mt5_xlsx(xlsx_bytes: bytes, time_tolerance="10min", vol_tol=1e-6) -> pd.DataFrame:
    xls = pd.ExcelFile(BytesIO(xlsx_bytes))
    df_raw = pd.read_excel(xls, xls.sheet_names[0])
    col0 = df_raw.columns[0]

    idx_pos = df_raw[df_raw[col0].astype(str).eq("Positions")].index
    idx_ord = df_raw[df_raw[col0].astype(str).eq("Orders")].index
    idx_deals = df_raw[df_raw[col0].astype(str).eq("Deals")].index
    if len(idx_pos) == 0 or len(idx_deals) == 0:
        raise ValueError("No se encontraron secciones 'Positions' y/o 'Deals'.")

    pos_header = idx_pos[0] + 1
    deals_header = idx_deals[0] + 1
    pos_end = idx_ord[0] if len(idx_ord) else idx_deals[0]

    pos_headers = df_raw.iloc[pos_header].tolist()
    df_pos = df_raw.iloc[pos_header + 1 : pos_end].copy()
    df_pos.columns = _rename_dupes(pos_headers)
    rename_map = {
        "Time": "open_time", "Time_1": "close_time", "Position": "position",
        "Symbol": "symbol", "Type": "type", "Volume": "volume",
        "Price": "open_price", "Price_1": "close_price", "S / L": "sl", "T / P": "tp",
        "Commission": "commission", "Swap": "swap", "Profit": "profit",
    }
    df_pos = df_pos.rename(columns=rename_map)

    for c in ["open_time", "close_time"]:
        if c in df_pos.columns: df_pos[c] = pd.to_datetime(df_pos[c], errors="coerce")
    for c in ["position","volume","open_price","close_price","sl","tp","commission","swap","profit"]:
        if c in df_pos.columns: df_pos[c] = pd.to_numeric(df_pos[c], errors="coerce")
    if "symbol" in df_pos.columns: df_pos["symbol"] = df_pos["symbol"].astype(str)
    if "profit" in df_pos.columns: df_pos = df_pos[~df_pos["profit"].isna()].copy()

    deals_headers = df_raw.iloc[deals_header].tolist()
    df_deals = df_raw.iloc[deals_header + 1 :].copy()
    df_deals.columns = deals_headers
    if "Order" in df_deals.columns: df_deals["Order"] = pd.to_numeric(df_deals["Order"], errors="coerce")
    if "Time" in df_deals.columns: df_deals["Time"] = pd.to_datetime(df_deals["Time"], errors="coerce")
    if "Volume" in df_deals.columns: df_deals["Volume"] = pd.to_numeric(df_deals["Volume"], errors="coerce")
    if "Symbol" in df_deals.columns: df_deals["Symbol"] = df_deals["Symbol"].astype(str)

    if {"Order","Direction"}.issubset(df_deals.columns):
        df_in = df_deals[df_deals["Direction"].astype(str).str.lower().eq("in")].copy()
        df_in = df_in.dropna(subset=["Order"]).drop_duplicates("Order")
        
        if "Comment" in df_in.columns and "position" in df_pos.columns:
            robot_map_c = df_in.set_index("Order")["Comment"].to_dict()
            df_pos["Comment"] = df_pos["position"].map(robot_map_c)
            
        if "Magic" in df_in.columns and "position" in df_pos.columns:
            robot_map_m = df_in.set_index("Order")["Magic"].to_dict()
            df_pos["Magic"] = df_pos["position"].map(robot_map_m)

    if "Comment" not in df_pos.columns: df_pos["Comment"] = np.nan
    if "Magic" not in df_pos.columns: df_pos["Magic"] = np.nan

    need = df_pos["Comment"].isna() & df_pos["Magic"].isna()
    if need.any() and {"Time","Symbol","Direction"}.issubset(df_deals.columns):
        df_in2 = df_deals[df_deals["Direction"].astype(str).str.lower().eq("in")].copy()
        cols_in2 = ["Time","Symbol","Volume"]
        if "Comment" in df_in2.columns: cols_in2.append("Comment")
        if "Magic" in df_in2.columns: cols_in2.append("Magic")
        
        df_in2 = df_in2[cols_in2].dropna(subset=["Time","Symbol"]).sort_values("Time")
        df_pos_sorted = df_pos.sort_values("open_time")
        
        matched_c, matched_m = [], []
        for sym, grp_pos in df_pos_sorted[need].groupby("symbol"):
            grp_in = df_in2[df_in2["Symbol"] == str(sym)]
            if grp_in.empty: continue
            
            m = pd.merge_asof(
                grp_pos[["open_time","volume"]].sort_values("open_time"),
                grp_in.sort_values("Time"),
                left_on="open_time", right_on="Time",
                direction="backward", tolerance=pd.Timedelta(time_tolerance)
            )
            vol_match = np.isfinite(m["volume"]) & np.isfinite(m["Volume"])
            close_vol = (np.abs(m["volume"] - m["Volume"]) <= vol_tol) | (~vol_match)
            
            if "Comment" in m.columns:
                mc = m["Comment"].where(close_vol)
                mc.index = grp_pos.sort_values("open_time").index
                matched_c.append(mc)
            if "Magic" in m.columns:
                mm = m["Magic"].where(close_vol)
                mm.index = grp_pos.sort_values("open_time").index
                matched_m.append(mm)

        if matched_c:
            matched_c = pd.concat(matched_c).sort_index()
            df_pos.loc[matched_c.index, "Comment"] = df_pos.loc[matched_c.index, "Comment"].fillna(matched_c)
        if matched_m:
            matched_m = pd.concat(matched_m).sort_index()
            df_pos.loc[matched_m.index, "Magic"] = df_pos.loc[matched_m.index, "Magic"].fillna(matched_m)

    return _ensure_fee_cols(df_pos)

# ========= Parsing MT5/Custom (CSV) =========
@st.cache_data(show_spinner=False)
def parse_csv(uploaded_csv: BytesIO) -> pd.DataFrame:
    file_bytes = uploaded_csv.read() if hasattr(uploaded_csv, 'read') else uploaded_csv
    
    try:
        df_pos = pd.read_csv(BytesIO(file_bytes), sep=None, engine='python', encoding='utf-16')
    except UnicodeError:
        try:
            df_pos = pd.read_csv(BytesIO(file_bytes), sep=None, engine='python', encoding='utf-8-sig')
        except UnicodeError:
            df_pos = pd.read_csv(BytesIO(file_bytes), sep=None, engine='python', encoding='latin-1')

    rename_try = {
        'Open Time':'open_time', 'Close Time':'close_time', 'Time':'close_time', 'Date':'close_time',
        'Symbol':'symbol', 'Item':'symbol',
        'Type':'type', 'Action':'type',
        'Direction':'direction', 
        'Volume':'volume', 'Size':'volume',
        'Open Price':'open_price', 'Close Price':'close_price', 'Price':'close_price',
        'Commission':'commission', 'Swap':'swap', 'Profit':'profit',
        'Comment':'Comment', 'Magic':'Magic', 'MagicNumber':'Magic', 'Magic Number':'Magic',
        'Ticket':'position', 'Order':'position'
    }
    
    col_map = {}
    for col in df_pos.columns:
        col_lower = str(col).strip().lower()
        for k, v in rename_try.items():
            if col_lower == k.lower():
                col_map[col] = v
                break
                
    df_pos = df_pos.rename(columns=col_map)
    
    for c in ['open_time','close_time']:
        if c in df_pos.columns: df_pos[c] = pd.to_datetime(df_pos[c], errors='coerce')
    for c in ['volume','open_price','close_price','commission','swap','profit']:
        if c in df_pos.columns: df_pos[c] = pd.to_numeric(df_pos[c], errors='coerce')
        
    if 'Comment' not in df_pos.columns: df_pos['Comment'] = np.nan
    if 'Magic' not in df_pos.columns: df_pos['Magic'] = np.nan
    if 'symbol' not in df_pos.columns: df_pos['symbol'] = "UNKNOWN"
    
    df_pos = _merge_deals_to_trades(df_pos)
    
    return _ensure_fee_cols(df_pos)

# ========= Parsing MT5 Deals CSV (Magic Number export) =========
@st.cache_data(show_spinner=False)
def parse_mt5_deals_csv(file_bytes: bytes) -> pd.DataFrame:
    """
    Parser para el formato de exportación de Deals de MT5 con Magic Number.
    Formato: CSV separado por ';', UTF-16, columnas:
    Time;Deal;Symbol;Type;Direction;Volume;Price;Order;Commission;Fee;Swap;Profit;Balance;Magic;Comment
    Cada trade tiene 2 filas: entrada (Profit=0) y salida (Profit≠0).
    Se fusionan con FIFO por (Magic, Symbol).
    """
    # Intentar leer con distintos encodings
    for enc in ["utf-16", "utf-8-sig", "utf-8", "latin-1"]:
        try:
            df_raw = pd.read_csv(BytesIO(file_bytes), sep=";", encoding=enc)
            break
        except Exception:
            continue
    else:
        raise ValueError("No se pudo leer el archivo CSV de Deals. Verificá el encoding.")

    # Filtrar filas de balance/depósito/retiro
    if "Type" in df_raw.columns:
        df_raw = df_raw[~df_raw["Type"].astype(str).str.lower().isin(
            ["balance", "deposit", "withdrawal", "credit"]
        )].copy()

    # Parsear fechas
    if "Time" in df_raw.columns:
        df_raw["Time"] = pd.to_datetime(df_raw["Time"], errors="coerce")

    # Convertir numéricos
    for c in ["Volume", "Price", "Commission", "Fee", "Swap", "Profit", "Balance", "Magic", "Order"]:
        if c in df_raw.columns:
            df_raw[c] = pd.to_numeric(df_raw[c], errors="coerce")

    df_raw = df_raw.dropna(subset=["Time"]).sort_values("Time").reset_index(drop=True)

    # ── FIFO merge: entrada (profit=0) + salida (profit≠0) ────
    trades = []
    open_deals: dict = {}

    for _, row in df_raw.iterrows():
        magic  = row.get("Magic", 0)
        sym    = str(row.get("Symbol", "UNKNOWN"))
        key    = (magic, sym)

        if key not in open_deals:
            open_deals[key] = []

        profit_val = float(row.get("Profit", 0) or 0)
        comment    = str(row.get("Comment", ""))
        is_out     = (profit_val != 0) or comment.startswith("[")

        if not is_out:
            open_deals[key].append(row)
        else:
            if open_deals[key]:
                entry = open_deals[key].pop(0)
                comm_in  = float(entry.get("Commission", 0) or 0)
                comm_out = float(row.get("Commission",  0) or 0)
                swap_in  = float(entry.get("Swap", 0) or 0)
                swap_out = float(row.get("Swap",  0) or 0)
                trades.append({
                    "open_time":   entry["Time"],
                    "close_time":  row["Time"],
                    "symbol":      sym,
                    "type":        str(entry.get("Direction", "")),
                    "volume":      float(entry.get("Volume", 0) or 0),
                    "open_price":  float(entry.get("Price", 0) or 0),
                    "close_price": float(row.get("Price", 0) or 0),
                    "commission":  comm_in + comm_out,
                    "swap":        swap_in + swap_out,
                    "profit":      profit_val,
                    "Magic":       magic,
                    "Comment":     str(entry.get("Comment", "")),
                    "position":    float(entry.get("Order", 0) or 0),
                })
            else:
                # Salida sin entrada registrada — añadir de todas formas
                trades.append({
                    "open_time":   row["Time"],
                    "close_time":  row["Time"],
                    "symbol":      sym,
                    "type":        str(row.get("Direction", "")),
                    "volume":      float(row.get("Volume", 0) or 0),
                    "open_price":  float(row.get("Price", 0) or 0),
                    "close_price": float(row.get("Price", 0) or 0),
                    "commission":  float(row.get("Commission", 0) or 0),
                    "swap":        float(row.get("Swap", 0) or 0),
                    "profit":      profit_val,
                    "Magic":       magic,
                    "Comment":     str(row.get("Comment", "")),
                    "position":    float(row.get("Order", 0) or 0),
                })

    if not trades:
        return pd.DataFrame()

    df = pd.DataFrame(trades)
    df["Comment"] = df["Comment"].replace("nan", np.nan)
    return _ensure_fee_cols(df)


def _is_deals_csv(file_bytes: bytes) -> bool:
    """Detecta si un CSV tiene el formato de Deals de MT5 (columna Magic + separador ';')."""
    for enc in ["utf-16", "utf-8-sig", "utf-8", "latin-1"]:
        try:
            sample = file_bytes[:1000].decode(enc)
            first_line = sample.splitlines()[0] if sample.splitlines() else ""
            cols = [c.strip().lower() for c in first_line.split(";")]
            return "magic" in cols and "deal" in cols and "direction" in cols
        except Exception:
            continue
    return False


# ========= Parsing MT4 (HTML) =========
@st.cache_data(show_spinner=False)
def parse_mt4_html(html_bytes: bytes) -> pd.DataFrame:
    soup = BeautifulSoup(html_bytes, "html.parser")
    rows = soup.find_all("tr", attrs={"align": "right"})
    data = []

    for tr in rows:
        tds = tr.find_all("td")
        if len(tds) < 14: continue

        profit_txt = tds[13].get_text(strip=True)
        try:
            profit_val = float(profit_txt.replace(",", ""))
        except ValueError:
            continue

        ticket_td = tds[0]
        ticket = ticket_td.get_text(strip=True)
        comment = ""
        title_attr = ticket_td.get("title")
        if title_attr:
            m = re.match(r"#\s*\d+\s*(.*)", title_attr.strip(), flags=re.I)
            if m:
                comment = m.group(1).strip()
                comment = re.sub(r"\[.*?\]$", "", comment).strip()

        def to_float(x):
            try: return float(str(x).replace(",", ""))
            except Exception: return np.nan
        def to_time(x): return pd.to_datetime(x, errors="coerce")

        data.append({
            "open_time": to_time(tds[1].get_text(strip=True)),
            "close_time": to_time(tds[8].get_text(strip=True)),
            "position": pd.to_numeric(ticket, errors="coerce"),
            "symbol": tds[4].get_text(strip=True),
            "type": tds[2].get_text(strip=True),
            "volume": to_float(tds[3].get_text(strip=True)),
            "open_price": to_float(tds[5].get_text(strip=True)),
            "sl": to_float(tds[6].get_text(strip=True)),
            "tp": to_float(tds[7].get_text(strip=True)),
            "close_price": to_float(tds[9].get_text(strip=True)),
            "commission": to_float(tds[10].get_text(strip=True)),
            "swap": to_float(tds[12].get_text(strip=True)),
            "profit": profit_val,
            "Comment": comment if comment else np.nan,
            "Magic": np.nan 
        })

    df = pd.DataFrame(data)
    if not df.empty:
        for c in ["open_time", "close_time"]: df[c] = pd.to_datetime(df[c], errors="coerce")
        for c in ["position","volume","open_price","close_price","sl","tp","commission","swap","profit"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df["symbol"] = df["symbol"].astype(str)

    return _ensure_fee_cols(df)

# ========= KPIs =========
@st.cache_data(show_spinner=False)
def kpis_por_robot(df_pos: pd.DataFrame):
    key = "robot_id"
    df = df_pos.copy()
    df[key] = df[key].astype(str)

    if "close_time" in df.columns and df["close_time"].notna().any():
        tcol = "close_time"
    elif "open_time" in df.columns and df["open_time"].notna().any():
        tcol = "open_time"
    else:
        tcol = "close_time"
        if "close_time" not in df.columns:
            df["close_time"] = pd.NaT

    df = df.sort_values(tcol)

    rows = []
    for rid, g in df.groupby(key):
        pnl_real = g["real_profit"].fillna(0.0)
        equity = pnl_real.cumsum()
        mdd = max_drawdown(equity)
        net_real = float(pnl_real.sum())
        ret_dd = float(net_real / mdd) if mdd > 0 else np.nan

        rows.append({
            "Robot ID": rid,
            "Net profit (real)": net_real,
            "# Trades": int((~pnl_real.isna()).sum()),
            "% Wins": float((pnl_real > 0).mean() * 100.0) if len(pnl_real) else 0.0,
            "PF": profit_factor(pnl_real),
            "Expectancy": expectancy(pnl_real),
            "Ret/DD": ret_dd,
            "Max DD": mdd,
            "Desde": g[tcol].min(),
            "Hasta": g[tcol].max(),
            "Símbolos": ", ".join(sorted(set(g["symbol"].dropna().astype(str)))) if "symbol" in g else "",
        })
    out = pd.DataFrame(rows).sort_values("Net profit (real)", ascending=False)
    return out, key, tcol

def summary_kpis_robot(g: pd.DataFrame, tcol: str) -> pd.DataFrame:
    pnl_real = g["real_profit"].fillna(0.0)
    equity = pnl_real.cumsum()
    wins = int((pnl_real > 0).sum())
    losses = int((pnl_real < 0).sum())
    mdd = max_drawdown(equity)
    net_bruto = float(g["profit"].sum())
    comm_total = float(g["commission"].sum())
    swap_total = float(g["swap"].sum())
    net_real = float(pnl_real.sum())
    ret_dd = float(net_real / mdd) if mdd > 0 else np.nan
    sharp = sharpe_per_trade(pnl_real)
    stab = stability_r2(equity)
    stagn = max_stagnation(g[tcol] if tcol in g.columns else None, equity)

    data = {
        "Robot ID": [g["robot_id"].iloc[0] if "robot_id" in g.columns else ""],
        "trades": [int(len(pnl_real))],
        "net profit (bruto)": [net_bruto],
        "commissions total": [comm_total],
        "swaps total": [swap_total],
        "net profit (real)": [net_real],
        "max dd": [mdd],
        "ret/dd": [ret_dd],
        "winrate": [float((pnl_real > 0).mean() * 100.0) if len(pnl_real) else 0.0],
        "wins": [wins],
        "loss": [losses],
        "profit factor": [profit_factor(pnl_real)],
        "sharpe ratio": [sharp],
        "expectancy": [expectancy(pnl_real)],
        "stability": [stab],
        "stagnation": [stagn],
    }
    return pd.DataFrame(data)


# ========= EDGE ANALYTICS =========

def compute_edge_score(expectancy_current: float, baseline: float) -> int:
    """
    Convierte expectancy actual vs baseline en score -3 a +3.
    Thresholds basados en % de desviación del baseline.
    """
    if baseline == 0:
        if expectancy_current > 0:
            return 3
        elif expectancy_current < 0:
            return -3
        return 0

    ratio = expectancy_current / abs(baseline)

    if expectancy_current < 0:
        return -3
    elif ratio >= 1.3:
        return 3
    elif ratio >= 1.1:
        return 2
    elif ratio >= 0.9:
        return 1
    elif ratio >= 0.7:
        return 0
    elif ratio >= 0.5:
        return -1
    elif ratio >= 0.2:
        return -2
    else:
        return -3


def edge_score_label(score: int) -> str:
    labels = {
        3:  "🟢 Edge muy por encima del baseline",
        2:  "🟢 Edge por encima del baseline",
        1:  "🟢 Edge estable",
        0:  "🟡 Edge levemente por debajo — monitorear",
        -1: "🟠 Edge en decay moderado",
        -2: "🔴 Edge en decay severo",
        -3: "🔴 Edge negativo",
    }
    return labels.get(score, "❓ Sin datos")


def momentum_score(blocks: list[dict]) -> int | None:
    """
    Score de momentum basado en Δ% entre los últimos bloques consecutivos.
    Independiente del baseline — mide la dirección actual del edge.
    """
    if len(blocks) < 2:
        return None

    # Usar hasta los últimos 3 bloques para calcular momentum
    recent = blocks[-3:] if len(blocks) >= 3 else blocks
    deltas = []
    for i in range(1, len(recent)):
        prev_e = recent[i-1]["expectancy"]
        curr_e = recent[i]["expectancy"]
        if prev_e != 0:
            pct = ((curr_e - prev_e) / abs(prev_e)) * 100
            deltas.append(pct)

    if not deltas:
        return None

    avg_delta = sum(deltas) / len(deltas)

    if avg_delta >= 15:
        return 3
    elif avg_delta >= 5:
        return 2
    elif avg_delta >= -5:
        return 1
    elif avg_delta >= -15:
        return 0
    elif avg_delta >= -30:
        return -1
    elif avg_delta >= -50:
        return -2
    else:
        return -3


def momentum_label(score: int | None) -> str:
    if score is None:
        return "❓ Sin datos suficientes"
    labels = {
        3:  "🟢 Acelerando fuerte",
        2:  "🟢 Mejorando",
        1:  "🟢 Estable",
        0:  "🟡 Desacelerando levemente",
        -1: "🟠 Decayendo",
        -2: "🔴 Decayendo rápido",
        -3: "🔴 Caída pronunciada",
    }
    return labels.get(score, "❓")


def confluence_signal(vs_baseline: int | None, mom: int | None) -> tuple[str, str]:
    """
    Combina score vs baseline y momentum en una señal de confluencia.
    Retorna (emoji + texto, color css).
    """
    if vs_baseline is None or mom is None:
        return "❓ Datos insuficientes", "#888"

    both_positive = vs_baseline >= 1 and mom >= 1
    both_strong_positive = vs_baseline >= 2 and mom >= 2
    both_negative = vs_baseline <= -1 and mom <= -1
    both_strong_negative = vs_baseline <= -2 and mom <= -2
    diverging = (vs_baseline >= 1 and mom <= -1) or (vs_baseline <= -1 and mom >= 1)

    if both_strong_positive:
        return "🌟 Excelente — edge sólido y en crecimiento", "#00d4aa"
    elif both_positive:
        return "✅ Positivo — edge sobre baseline y mejorando", "#00d4aa"
    elif both_strong_negative:
        return "🚨 Atención — decay confirmado en ambas métricas", "#ff4d6d"
    elif both_negative:
        return "⚠️ Precaución — señales de debilitamiento", "#f0a040"
    elif diverging:
        return "🔍 Investigar — señales contradictorias", "#f0c040"
    elif vs_baseline >= 1:
        return "🟡 Sobre baseline, momentum neutral", "#f0c040"
    elif mom >= 1:
        return "🟡 Mejorando, aún bajo baseline", "#f0c040"
    else:
        return "🟡 Neutral — monitorear evolución", "#f0c040"


def compute_rolling_expectancy(pnl: pd.Series, window: int) -> pd.Series:
    """Expectancy rolling de ventana móvil."""
    return pnl.rolling(window=window, min_periods=window).mean()


def compute_period_blocks(pnl: pd.Series, block_size: int) -> list[dict]:
    """
    Divide el PnL en bloques fijos y calcula expectancy + score por bloque.
    Retorna lista con los últimos 3 bloques como máximo.
    """
    blocks = []
    n = len(pnl)
    num_blocks = n // block_size

    if num_blocks == 0:
        return blocks

    for i in range(num_blocks):
        start = i * block_size
        end = start + block_size
        chunk = pnl.iloc[start:end]
        exp = float(chunk.mean())
        wr = float((chunk > 0).mean() * 100)
        pf = profit_factor(chunk)
        blocks.append({
            "bloque": i + 1,
            "desde_trade": start + 1,
            "hasta_trade": end,
            "trades": block_size,
            "expectancy": exp,
            "winrate": wr,
            "profit_factor": pf,
        })

    # Bloque parcial al final si hay suficientes trades (>= 50% del bloque)
    remainder = n % block_size
    if remainder >= block_size // 2:
        chunk = pnl.iloc[num_blocks * block_size:]
        exp = float(chunk.mean())
        wr = float((chunk > 0).mean() * 100)
        pf = profit_factor(chunk)
        blocks.append({
            "bloque": num_blocks + 1,
            "desde_trade": num_blocks * block_size + 1,
            "hasta_trade": n,
            "trades": remainder,
            "expectancy": exp,
            "winrate": wr,
            "profit_factor": pf,
        })

    return blocks


def edge_trend_arrow(scores: list[int]) -> str:
    """Determina tendencia basada en los últimos scores."""
    if len(scores) < 2:
        return "➡️ Sin tendencia (pocos datos)"
    delta = scores[-1] - scores[0]
    if delta >= 2:
        return "📈 En aceleración"
    elif delta == 1:
        return "↗️ Mejorando"
    elif delta == 0:
        return "➡️ Estable"
    elif delta == -1:
        return "↘️ Decayendo"
    else:
        return "📉 Decaimiento acelerado"


def render_edge_comparison_table(
    df_pos: pd.DataFrame,
    tcol: str,
    robots: list,
    cp_size: int,
    mp_size: int,
    lp_size: int,
    baseline_trades: int,
    baseline_manual: float | None,
):
    """Tabla comparativa de edge para todos los robots. Reutilizable en múltiples tabs."""

    # Nombres de columnas para scores numéricos (usados internamente para ordenar)
    col_b_cp  = f"B·CP ({cp_size})"
    col_b_mp  = f"B·MP ({mp_size})"
    col_b_lp  = f"B·LP ({lp_size})"
    col_m_cp  = f"M·CP ({cp_size})"
    col_m_mp  = f"M·MP ({mp_size})"
    col_m_lp  = f"M·LP ({lp_size})"
    score_cols = [col_b_cp, col_b_mp, col_b_lp, col_m_cp, col_m_mp, col_m_lp]

    summary_rows = []
    for rid in robots:
        g = df_pos[df_pos["robot_id"].astype(str) == rid].copy()
        g = g.sort_values(tcol) if tcol in g.columns else g
        pnl = g["real_profit"].fillna(0.0).reset_index(drop=True)
        n = len(pnl)

        if n < cp_size:
            continue

        if baseline_manual is not None:
            b_exp = float(baseline_manual)
        else:
            b_n = min(baseline_trades, n)
            b_exp = float(pnl.iloc[:b_n].mean())

        def last_score(size):
            blocks = compute_period_blocks(pnl, size)
            if not blocks:
                return None, []
            return compute_edge_score(blocks[-1]["expectancy"], b_exp), blocks

        cp_s, cp_bl = last_score(cp_size)
        mp_s, mp_bl = last_score(mp_size)
        lp_s, lp_bl = last_score(lp_size)

        cp_m = momentum_score(cp_bl)
        mp_m = momentum_score(mp_bl)
        lp_m = momentum_score(lp_bl)

        ref_b = lp_s if lp_s is not None else (mp_s if mp_s is not None else cp_s)
        ref_m = lp_m if lp_m is not None else (mp_m if mp_m is not None else cp_m)
        conf_text, _ = confluence_signal(ref_b, ref_m)
        conf_short = conf_text.split("—")[-1].strip() if "—" in conf_text else conf_text.split(" ", 1)[-1]

        # Guardar valores numéricos reales para ordenamiento correcto
        # None → pd.NA para que Streamlit los trate como faltantes al ordenar
        def nv(s): return s if s is not None else pd.NA

        summary_rows.append({
            "Robot":       rid,
            "Trades":      n,
            col_b_cp:      nv(cp_s),
            col_b_mp:      nv(mp_s),
            col_b_lp:      nv(lp_s),
            col_m_cp:      nv(cp_m),
            col_m_mp:      nv(mp_m),
            col_m_lp:      nv(lp_m),
            "Confluencia": conf_short,
            "Exp.Base":    round(b_exp, 3),
        })

    if not summary_rows:
        st.info(f"Se necesitan al menos {cp_size} trades por robot para la tabla de edge.")
        return

    df_summary = pd.DataFrame(summary_rows)

    # Asegurar que columnas de score sean Int64 nullable (soporta pd.NA y ordena correctamente)
    for c in score_cols:
        if c in df_summary.columns:
            df_summary[c] = pd.array(df_summary[c], dtype="Int64")

    def color_score_cell(val):
        try:
            v = int(val)
            if v >= 2:  return "color: #00d4aa; font-weight: bold"
            if v == 1:  return "color: #00d4aa"
            if v == 0:  return "color: #f0c040"
            if v == -1: return "color: #f0a040"
            return "color: #ff4d6d; font-weight: bold"
        except:
            return "color: #888"

    def color_confluence(val):
        v = str(val).lower()
        if any(k in v for k in ["excelente", "positivo", "sólido"]):
            return "color: #00d4aa; font-weight: bold"
        if any(k in v for k in ["atención", "decay", "precaución"]):
            return "color: #ff4d6d; font-weight: bold"
        if any(k in v for k in ["investigar", "contradictorias", "neutral", "monitorear"]):
            return "color: #f0c040"
        return ""

    present_score_cols = [c for c in score_cols if c in df_summary.columns]

    def fmt_score(val):
        try:
            v = int(val)
            return f"+{v}" if v > 0 else str(v)
        except:
            return "—"

    fmt_map = {c: fmt_score for c in present_score_cols}

    st.dataframe(
        df_summary.style
            .map(color_score_cell, subset=present_score_cols)
            .map(color_confluence, subset=["Confluencia"])
            .format(fmt_map, na_rep="—"),
        use_container_width=True,
        hide_index=True,
        column_order=["Robot", "Trades"] + present_score_cols + ["Confluencia", "Exp.Base"],
    )


def render_edge_tab(df_pos: pd.DataFrame, tcol: str):
    """Renderiza la pestaña completa de Edge Analytics."""

    st.markdown("""
    <style>
    .edge-card {
        background: #0f1117;
        border: 1px solid #2a2d3e;
        border-radius: 12px;
        padding: 16px 20px;
        margin-bottom: 12px;
    }
    .score-badge {
        font-size: 2rem;
        font-weight: 900;
        letter-spacing: -1px;
    }
    .score-pos { color: #00d4aa; }
    .score-neu { color: #f0c040; }
    .score-neg { color: #ff4d6d; }
    .period-label {
        font-size: 0.75rem;
        color: #888;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 4px;
    }
    </style>
    """, unsafe_allow_html=True)

    st.subheader("🎯 Edge Analytics — Detección y Medición de Vida del Edge")
    st.caption(
        "Monitorea si el edge de cada sistema está en crecimiento, estable o en decaimiento. "
        "Basado en expectancy rolling comparada contra el baseline histórico del sistema."
    )

    robots = sorted(df_pos["robot_id"].astype(str).unique())

    if not robots:
        st.warning("No hay robots disponibles.")
        return

    # ── Selector de robot ──────────────────────────────────────────────────
    st.markdown("#### 🤖 Seleccionar Robot")

    # Construir opciones con info rápida de trades
    robot_options = {}
    for rid in robots:
        g_tmp = df_pos[df_pos["robot_id"].astype(str) == rid]
        n_tmp = len(g_tmp)
        robot_options[f"🤖 {rid}  ({n_tmp} trades)"] = rid

    selected_robot_label = st.selectbox(
        "Robot a analizar:",
        options=list(robot_options.keys()),
        key="edge_robot_selector",
        help="Seleccioná el robot que querés analizar en detalle."
    )
    selected_rid = robot_options[selected_robot_label]

    st.markdown("---")

    # ── Configuración de períodos ──────────────────────────────────────────
    st.markdown("#### ⚙️ Configuración de períodos")
    c1, c2, c3 = st.columns(3)
    cp_size = c1.number_input("Corto Plazo (trades)", min_value=5, max_value=50, value=10, step=5,
                               key="edge_cp_size",
                               help="Número de trades por bloque de corto plazo")
    mp_size = c2.number_input("Medio Plazo (trades)", min_value=10, max_value=100, value=20, step=5,
                               key="edge_mp_size",
                               help="Número de trades por bloque de medio plazo")
    lp_size = c3.number_input("Largo Plazo (trades)", min_value=20, max_value=200, value=50, step=10,
                               key="edge_lp_size",
                               help="Número de trades por bloque de largo plazo")

    # ── Configuración de Baseline ──────────────────────────────────────────
    st.markdown("#### 📐 Baseline de Expectancy")
    st.caption("El baseline es la expectancy de referencia contra la que se mide el edge actual.")

    baseline_mode = st.radio(
        "Fuente del baseline:",
        options=["📊 Automático (promedio de los primeros N trades)", "✏️ Manual (ingreso desde backtesting u otra fuente)"],
        horizontal=True,
        key="edge_baseline_mode",
        help="Automático calcula el baseline desde los datos cargados. Manual permite ingresar un valor conocido."
    )

    if baseline_mode.startswith("📊"):
        bc1, bc2 = st.columns([1, 3])
        baseline_trades = bc1.number_input(
            "Primeros N trades para baseline",
            min_value=10, max_value=200, value=20, step=10,
            key="edge_baseline_trades",
            help="Se usarán los primeros N trades del historial para calcular el baseline"
        )
        baseline_manual = None

        # Calcular y mostrar el baseline resultante para el robot seleccionado
        _g_preview = df_pos[df_pos["robot_id"].astype(str) == selected_rid]
        _pnl_preview = _g_preview["real_profit"].fillna(0.0).reset_index(drop=True)
        _n_preview = len(_pnl_preview)
        if _n_preview > 0:
            _bn = min(baseline_trades, _n_preview)
            _baseline_calc = float(_pnl_preview.iloc[:_bn].mean())
            _color = "#00d4aa" if _baseline_calc > 0 else "#f87171"
            bc2.markdown(
                f"<div style='background:{_color}18;border:1px solid {_color};"
                f"border-radius:6px;padding:8px 14px;margin-top:4px'>"
                f"<span style='font-size:.75rem;color:#8b949e'>Baseline calculado "
                f"(primeros {_bn} de {_n_preview} trades)</span><br>"
                f"<span style='font-size:1.3rem;font-weight:900;color:{_color}'>"
                f"${_baseline_calc:.2f}</span>"
                f"<span style='font-size:.8rem;color:#8b949e'> / trade</span>"
                f"</div>",
                unsafe_allow_html=True
            )
        else:
            bc2.info("Sin datos para calcular el baseline.")
    else:
        bc1, bc2 = st.columns([1, 3])
        baseline_manual = bc1.number_input(
            "Expectancy baseline (valor por trade en $)",
            min_value=-1000.0, max_value=10000.0, value=0.0, step=0.01,
            format="%.2f",
            key="baseline_manual_input",
            help="Ingresá la expectancy promedio por trade obtenida en tu backtesting o período de referencia anterior"
        )
        baseline_trades = 20  # valor por defecto, no se usará
        bc2.info(
            f"📌 Usando baseline manual de **${baseline_manual:.2f}** por trade. "
            "Este valor reemplaza el cálculo automático desde los datos cargados."
        )

    st.markdown("---")

    # ── Panel del robot seleccionado ──────────────────────────────────────
    rid = selected_rid
    if True:  # bloque único reemplaza el for loop
        g = df_pos[df_pos["robot_id"].astype(str) == rid].copy()
        g = g.sort_values(tcol) if tcol in g.columns else g
        pnl = g["real_profit"].fillna(0.0).reset_index(drop=True)
        n_trades = len(pnl)

        st.markdown(f"### 🤖 {rid} — {n_trades} trades totales")

        if n_trades < cp_size:
            st.info(f"⚠️ Mínimo {cp_size} trades necesarios para análisis de edge. Trades actuales: {n_trades}")
        else:

            # Baseline: automático o manual
            if baseline_manual is not None:
                baseline_exp = float(baseline_manual)
                baseline_n = 0
                baseline_desc = f"Manual — ${baseline_exp:.2f} / trade"
            else:
                baseline_n = min(baseline_trades, n_trades)
                baseline_exp = float(pnl.iloc[:baseline_n].mean())
                baseline_desc = f"Automático ({baseline_n} trades) — ${baseline_exp:.2f} / trade"

            # ── Scores por período ─────────────────────────────────────────
            def get_last_blocks(size, max_blocks=3):
                blocks = compute_period_blocks(pnl, size)
                return blocks[-max_blocks:] if len(blocks) >= 1 else []

            cp_blocks = get_last_blocks(cp_size)
            mp_blocks = get_last_blocks(mp_size)
            lp_blocks = get_last_blocks(lp_size)

            def blocks_to_scores(blocks):
                return [compute_edge_score(b["expectancy"], baseline_exp) for b in blocks]

            cp_scores = blocks_to_scores(cp_blocks)
            mp_scores = blocks_to_scores(mp_blocks)
            lp_scores = blocks_to_scores(lp_blocks)

            # Score actual = último bloque de cada período
            cp_now = cp_scores[-1] if cp_scores else None
            mp_now = mp_scores[-1] if mp_scores else None
            lp_now = lp_scores[-1] if lp_scores else None

            # Momentum scores
            cp_mom = momentum_score(cp_blocks)
            mp_mom = momentum_score(mp_blocks)
            lp_mom = momentum_score(lp_blocks)

            # Confluencia (usar LP si disponible, sino MP, sino CP)
            ref_baseline = lp_now if lp_now is not None else (mp_now if mp_now is not None else cp_now)
            ref_mom = lp_mom if lp_mom is not None else (mp_mom if mp_mom is not None else cp_mom)
            conf_text, conf_color = confluence_signal(ref_baseline, ref_mom)

            # ── Señal de confluencia destacada ────────────────────────────
            st.markdown(
                f"<div style='background:{conf_color}22; border-left: 4px solid {conf_color}; "
                f"padding: 12px 16px; border-radius: 8px; margin-bottom: 16px;'>"
                f"<span style='font-size:1.05rem; font-weight:600; color:{conf_color};'>{conf_text}</span>"
                f"<span style='color:#888; font-size:0.8rem; margin-left:12px;'>Señal de confluencia</span>"
                f"</div>",
                unsafe_allow_html=True
            )

            # ── Tabla de scores por horizonte ──────────────────────────────
            st.markdown("##### 📊 Score por horizonte")

            def fmt_score(s):
                if s is None: return "—"
                sign = "+" if s > 0 else ""
                return f"{sign}{s}"

            score_data = {
                "Horizonte": [f"Corto ({cp_size})", f"Medio ({mp_size})", f"Largo ({lp_size})"],
                "vs Baseline": [fmt_score(cp_now), fmt_score(mp_now), fmt_score(lp_now)],
                "Estado Baseline": [
                    edge_score_label(cp_now).split(" ", 1)[-1] if cp_now is not None else "—",
                    edge_score_label(mp_now).split(" ", 1)[-1] if mp_now is not None else "—",
                    edge_score_label(lp_now).split(" ", 1)[-1] if lp_now is not None else "—",
                ],
                "Momentum": [fmt_score(cp_mom), fmt_score(mp_mom), fmt_score(lp_mom)],
                "Estado Momentum": [
                    momentum_label(cp_mom).split(" ", 1)[-1] if cp_mom is not None else "—",
                    momentum_label(mp_mom).split(" ", 1)[-1] if mp_mom is not None else "—",
                    momentum_label(lp_mom).split(" ", 1)[-1] if lp_mom is not None else "—",
                ],
            }

            df_scores = pd.DataFrame(score_data)

            def color_score_cell(val):
                try:
                    v = int(str(val).replace("+", ""))
                    if v >= 2: return "color: #00d4aa; font-weight: bold"
                    if v == 1: return "color: #00d4aa"
                    if v == 0: return "color: #f0c040"
                    if v == -1: return "color: #f0a040"
                    return "color: #ff4d6d; font-weight: bold"
                except:
                    return "color: #888"

            st.dataframe(
                df_scores.style
                    .map(color_score_cell, subset=["vs Baseline", "Momentum"]),
                use_container_width=True,
                hide_index=True,
            )

            st.caption(f"Baseline: {baseline_desc}")
            st.markdown("---")

            # ── Tabla de bloques históricos ────────────────────────────────
            st.markdown("##### 📋 Historial de bloques por período")
            tab_cp, tab_mp, tab_lp = st.tabs([
                f"Corto Plazo ({cp_size} trades/bloque)",
                f"Medio Plazo ({mp_size} trades/bloque)",
                f"Largo Plazo ({lp_size} trades/bloque)",
            ])

            def render_blocks_table(tab, blocks, scores, baseline_exp_val, rapid_decay_pct=-20.0):
                with tab:
                    if not blocks:
                        st.info("No hay suficientes trades para este período.")
                        return

                    rows = []
                    prev_score = None
                    prev_exp = None
                    rapid_decay_alerts = []

                    for i, (b, s) in enumerate(zip(blocks, scores)):
                        # Δ Score entre bloques
                        if prev_score is None:
                            delta_score = "—"
                        else:
                            d = s - prev_score
                            delta_score = f"{'↑' if d > 0 else '↓' if d < 0 else '→'} {d:+d}"

                        # Δ% de expectancy entre bloques consecutivos
                        if prev_exp is None or prev_exp == 0:
                            delta_pct = "—"
                            delta_pct_val = None
                        else:
                            pct = ((b["expectancy"] - prev_exp) / abs(prev_exp)) * 100
                            delta_pct_val = pct
                            arrow = "📈" if pct > 0 else "📉"
                            delta_pct = f"{arrow} {pct:+.1f}%"
                            # Detectar decay rápido
                            if pct <= rapid_decay_pct:
                                rapid_decay_alerts.append(
                                    f"⚡ **Decay rápido en bloque #{b['bloque']}**: "
                                    f"expectancy cayó **{pct:.1f}%** en un bloque "
                                    f"({prev_exp:.3f} → {b['expectancy']:.3f})"
                                )

                        # Δ% vs baseline
                        if baseline_exp_val != 0:
                            vs_base = ((b["expectancy"] - baseline_exp_val) / abs(baseline_exp_val)) * 100
                            vs_base_str = f"{vs_base:+.1f}%"
                        else:
                            vs_base_str = "—"

                        rows.append({
                            "Bloque": f"#{b['bloque']}",
                            "Trades": f"{b['desde_trade']}–{b['hasta_trade']}",
                            "N": b["trades"],
                            "Expectancy": round(b["expectancy"], 3),
                            "Δ% Exp": delta_pct,
                            "vs Baseline": vs_base_str,
                            "Win Rate": f"{b['winrate']:.1f}%",
                            "PF": round(b["profit_factor"], 2),
                            "Score": f"{'+' if s > 0 else ''}{s}",
                            "Δ Score": delta_score,
                            "Estado": edge_score_label(s),
                        })
                        prev_score = s
                        prev_exp = b["expectancy"]

                    df_blocks = pd.DataFrame(rows)

                    def color_score(val):
                        try:
                            v = int(str(val).replace("+", ""))
                            if v >= 1: return "color: #00d4aa; font-weight: bold"
                            if v <= -1: return "color: #ff4d6d; font-weight: bold"
                            return "color: #f0c040; font-weight: bold"
                        except:
                            return ""

                    def color_delta_pct(val):
                        """Verde si sube, rojo si baja más del umbral, amarillo si baja menos."""
                        if val == "—" or not isinstance(val, str):
                            return ""
                        try:
                            num = float(val.replace("📈","").replace("📉","").replace("%","").strip())
                            if num >= 0: return "color: #00d4aa"
                            if num <= rapid_decay_pct: return "color: #ff4d6d; font-weight: bold"
                            return "color: #f0a040"
                        except:
                            return ""

                    def color_vs_baseline(val):
                        """Verde si está sobre baseline, rojo si está muy por debajo."""
                        if val == "—":
                            return ""
                        try:
                            num = float(str(val).replace("%","").strip())
                            if num >= 10: return "color: #00d4aa"
                            if num >= -10: return "color: #f0c040"
                            return "color: #ff4d6d"
                        except:
                            return ""

                    st.dataframe(
                        df_blocks.style
                            .map(color_score, subset=["Score"])
                            .map(color_delta_pct, subset=["Δ% Exp"])
                            .map(color_vs_baseline, subset=["vs Baseline"]),
                        use_container_width=True,
                        hide_index=True,
                    )

                    # Alertas de decay rápido dentro de la tabla
                    if rapid_decay_alerts:
                        for alert in rapid_decay_alerts:
                            st.error(alert)

                    # Mini chart de expectancy por bloque
                    if len(blocks) > 1:
                        chart_data = pd.DataFrame({
                            "Bloque": [f"#{b['bloque']}" for b in blocks],
                            "Expectancy": [b["expectancy"] for b in blocks],
                            "Baseline": [baseline_exp_val] * len(blocks),
                        }).set_index("Bloque")
                        st.line_chart(chart_data, height=180)
                        st.caption("Expectancy por bloque vs Baseline histórico")

            render_blocks_table(tab_cp, cp_blocks, cp_scores, baseline_exp)
            render_blocks_table(tab_mp, mp_blocks, mp_scores, baseline_exp)
            render_blocks_table(tab_lp, lp_blocks, lp_scores, baseline_exp)

            st.markdown("---")

            # ── Expectancy rolling continua ────────────────────────────────
            st.markdown("##### 📈 Expectancy rolling continua")
            roll_col1, roll_col2 = st.columns(2)

            with roll_col1:
                roll_window = st.slider(
                    "Ventana rolling (trades)",
                    min_value=5, max_value=min(100, n_trades),
                    value=min(mp_size, n_trades),
                    key="edge_roll_window"
                )

            roll_exp = compute_rolling_expectancy(pnl, roll_window).dropna()

            if not roll_exp.empty:
                chart_df = pd.DataFrame({
                    "Expectancy Rolling": roll_exp.values,
                    "Baseline": [baseline_exp] * len(roll_exp),
                    "Cero": [0.0] * len(roll_exp),
                }, index=range(1, len(roll_exp) + 1))
                st.line_chart(chart_df, height=220)
                st.caption(
                    f"Rolling de {roll_window} trades. "
                    f"Cuando la línea azul cae bajo el baseline (naranja) → señal de decay."
                )
            else:
                st.info(f"Se necesitan al menos {roll_window} trades para el gráfico rolling.")

            # ── Resumen de alertas ─────────────────────────────────────────
            st.markdown("##### 🔔 Estado del sistema")
            good_signals = []
            watch_signals = []
            alert_signals = []

            # Señales positivas
            if lp_now is not None and lp_now >= 2:
                good_signals.append(f"🌟 **Largo plazo sobre baseline** (score {fmt_score(lp_now)}): el sistema está rindiendo por encima de su referencia histórica")
            if lp_mom is not None and lp_mom >= 2:
                good_signals.append(f"📈 **Momentum positivo en LP**: la expectancy viene mejorando en los últimos bloques")
            if ref_baseline is not None and ref_baseline >= 2 and ref_mom is not None and ref_mom >= 2:
                good_signals.append("🌟 **Confluencia positiva**: edge sobre baseline y con momentum al alza — sistema en buen estado")

            # Señales de monitoreo
            if cp_now is not None and cp_now <= -1 and (mp_now is None or mp_now >= 0):
                watch_signals.append(f"🔍 **Corto plazo bajo baseline** (score {fmt_score(cp_now)}): puede ser ruido — observar evolución en próximos bloques")
            if mp_mom is not None and mp_mom <= -1 and (lp_mom is None or lp_mom >= 0):
                watch_signals.append("🔍 **Momentum de medio plazo desacelerando**: la tendencia reciente es descendente")
            if len(mp_scores) >= 3 and all(mp_scores[-3+i] > mp_scores[-3+i+1] for i in range(2)):
                watch_signals.append("📉 **3 bloques MP consecutivos en descenso** — prestar atención a la evolución")

            # Alertas
            if mp_now is not None and mp_now <= -2:
                alert_signals.append(f"⚠️ **Medio plazo en decay severo** (score {fmt_score(mp_now)}): la expectancy está significativamente bajo el baseline")
            if lp_now is not None and lp_now <= -2:
                alert_signals.append(f"🚨 **Largo plazo en decay severo** (score {fmt_score(lp_now)}): el sistema muestra debilitamiento estructural del edge")
            if len(lp_scores) >= 3 and all(lp_scores[-3+i] > lp_scores[-3+i+1] for i in range(2)):
                alert_signals.append("🚨 **3 bloques LP consecutivos en descenso** — patrón de decay estructural")
            if ref_baseline is not None and ref_baseline <= -1 and ref_mom is not None and ref_mom <= -1:
                alert_signals.append("🚨 **Confluencia negativa**: edge bajo baseline y con momentum descendente — revisar el sistema")

            # Decay rápido
            RAPID_DECAY_THRESHOLD = -20.0

            def check_rapid_decay(blocks, period_name):
                rapid = []
                for i in range(1, len(blocks)):
                    prev_e = blocks[i-1]["expectancy"]
                    curr_e = blocks[i]["expectancy"]
                    if prev_e != 0:
                        pct = ((curr_e - prev_e) / abs(prev_e)) * 100
                        if pct <= RAPID_DECAY_THRESHOLD:
                            rapid.append(
                                f"⚡ **Caída brusca en {period_name} bloque #{blocks[i]['bloque']}**: "
                                f"expectancy cayó **{pct:.1f}%** en un solo bloque "
                                f"({prev_e:.3f} → {curr_e:.3f})"
                            )
                return rapid

            for r in check_rapid_decay(mp_blocks, "MP"):
                alert_signals.append(r)
            for r in check_rapid_decay(lp_blocks, "LP"):
                alert_signals.append(r)

            # Render
            if good_signals:
                for s in good_signals:
                    st.success(s)
            if watch_signals:
                for s in watch_signals:
                    st.warning(s)
            if alert_signals:
                for s in alert_signals:
                    st.error(s)
            if not good_signals and not watch_signals and not alert_signals:
                st.info("ℹ️ Sistema dentro de parámetros normales — sin señales destacadas")

    # ── Tabla comparativa global ───────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### 🏆 Comparativa de Edge — Todos los Robots")
    render_edge_comparison_table(
        df_pos, tcol, robots,
        cp_size, mp_size, lp_size,
        baseline_trades, baseline_manual,
    )

    # ── Leyenda ───────────────────────────────────────────────────────────
    with st.expander("📖 Leyenda de scores y metodología"):
        st.markdown("""
        ### Sistema de scoring dual

        Cada robot se evalúa con **dos métricas independientes** que responden preguntas distintas:

        #### Score vs Baseline
        Compara la expectancy actual contra el baseline de referencia (backtest SQX o automático).
        Responde: *¿El sistema está rindiendo como se esperaba históricamente?*

        | Score | Condición | Significado |
        |-------|-----------|-------------|
        | **+3** | Expectancy ≥ 130% del baseline | Edge muy por encima del baseline |
        | **+2** | Expectancy ≥ 110% del baseline | Edge por encima del baseline |
        | **+1** | Expectancy 90–110% del baseline | Edge estable |
        | **0**  | Expectancy 70–90% del baseline | Levemente por debajo — monitorear |
        | **-1** | Expectancy 50–70% del baseline | Decay moderado |
        | **-2** | Expectancy < 50% del baseline | Decay severo |
        | **-3** | Expectancy negativa | Edge negativo |

        #### Score Momentum
        Mide la dirección actual del edge comparando bloques consecutivos entre sí.
        Responde: *¿El edge está mejorando o empeorando ahora mismo?*

        | Score | Δ% promedio entre bloques | Significado |
        |-------|--------------------------|-------------|
        | **+3** | ≥ +15% | Acelerando fuerte |
        | **+2** | +5% a +15% | Mejorando |
        | **+1** | ±5% | Estable |
        | **0**  | -5% a -15% | Desacelerando |
        | **-1** | -15% a -30% | Decayendo |
        | **-2** | -30% a -50% | Decayendo rápido |
        | **-3** | < -50% | Caída pronunciada |

        #### Señal de Confluencia
        Combina ambos scores en una señal única. Las señales más confiables son cuando ambas métricas apuntan en la misma dirección.

        | Confluencia | Significado |
        |-------------|-------------|
        | 🌟 Excelente | Ambos positivos y fuertes |
        | ✅ Positivo | Ambos positivos |
        | ⚠️ Precaución | Ambos negativos |
        | 🚨 Atención | Ambos negativos y fuertes |
        | 🔍 Investigar | Señales contradictorias |

        #### Columnas de la tabla de bloques
        | Columna | Significado |
        |---------|-------------|
        | **Expectancy** | Ganancia promedio por trade en ese bloque |
        | **Δ% Exp** | Cambio porcentual vs bloque anterior. Rojo si cae ≥ 20% |
        | **vs Baseline** | Diferencia % respecto al baseline del sistema |
        | **Score** | Score vs baseline (-3 a +3) |
        | **Δ Score** | Cambio en puntos respecto al bloque anterior |
        """)


# ==========================================
# ⚙️ SIDEBAR: Controles y Filtros Globales
# ==========================================
st.sidebar.header("📁 Carga de Datos")

uploaded_files = st.sidebar.file_uploader(
    "📥 Subí tus archivos (MT5: .xlsx/.csv | MT4: .htm/.html)", 
    type=["xlsx","csv","htm","html"], 
    accept_multiple_files=True
)

if not uploaded_files:
    st.info("👈 Por favor, arrastrá y soltá tus historiales en la barra lateral para comenzar.")
    st.stop()

st.sidebar.markdown("---")
st.sidebar.header("⚙️ Configuración")

agrupar_por = st.sidebar.radio(
    "Agrupar robots por:", 
    options=["Comment", "Magic Number"],
    help="Elige qué campo usar para identificar a cada robot."
)

fusionar_archivos = st.sidebar.checkbox(
    "🔗 Fusionar historiales del mismo robot", 
    value=False, 
    help="Si activado, suma los trades de robots con el mismo nombre en diferentes archivos."
)

def clean_id(x):
    if pd.isna(x) or str(x).strip() == "": return "UNKNOWN"
    sx = str(x).strip()
    if sx.endswith(".0"): return sx[:-2]
    return sx

all_dfs = []

with st.spinner("⏳ Procesando archivos..."):
  for uploaded in uploaded_files:
    suffix = uploaded.name.lower().split(".")[-1]
    try:
        file_bytes = uploaded.read()

        if suffix == "xlsx":
            df_temp = parse_mt5_xlsx(file_bytes)
        elif suffix == "csv":
            # Detectar si es el formato Deals de MT5 (Magic + separador ;)
            if _is_deals_csv(file_bytes):
                df_temp = parse_mt5_deals_csv(file_bytes)
                # Para Deals CSV siempre agrupar por Magic Number automáticamente
                if "Magic" in df_temp.columns:
                    df_temp["robot_id"] = df_temp["Magic"].apply(clean_id)
                    mask_unknown = df_temp["robot_id"] == "UNKNOWN"
                    if mask_unknown.any():
                        df_temp.loc[mask_unknown, "robot_id"] = (
                            df_temp.loc[mask_unknown, "symbol"].astype(str) + "_UNKNOWN"
                        )
                    if not fusionar_archivos:
                        df_temp["robot_id"] = df_temp["robot_id"].astype(str) + f" [{uploaded.name}]"
                    df_temp["source_file"] = uploaded.name
                    all_dfs.append(df_temp)
                    st.sidebar.success(f"✅ Deals CSV detectado: {uploaded.name} — agrupado por Magic Number")
                    continue  # skip the generic robot_id logic below
            else:
                df_temp = parse_csv(BytesIO(file_bytes))
        elif suffix in ("htm", "html"):
            df_temp = parse_mt4_html(file_bytes)
        else:
            st.sidebar.error(f"Extensión no soportada: {uploaded.name}")
            continue

        if df_temp.empty:
            st.sidebar.warning(f"No hay operaciones válidas en {uploaded.name}.")
            continue

        col_target = "Magic" if agrupar_por == "Magic Number" else "Comment"
        
        if col_target in df_temp.columns:
            df_temp["robot_id"] = df_temp[col_target].apply(clean_id)
        else:
            df_temp["robot_id"] = "UNKNOWN"
            
        mask_unknown = df_temp["robot_id"] == "UNKNOWN"
        if mask_unknown.any():
            df_temp.loc[mask_unknown, "robot_id"] = df_temp.loc[mask_unknown, "symbol"].astype(str) + "_UNKNOWN"

        if not fusionar_archivos:
            df_temp["robot_id"] = df_temp["robot_id"].astype(str) + f" [{uploaded.name}]"
        
        df_temp["source_file"] = uploaded.name
        
        all_dfs.append(df_temp)

    except Exception as e:
        st.sidebar.error(f"No se pudo leer {uploaded.name}. Verificá el formato.")
        st.sidebar.exception(e)

if not all_dfs:
    st.stop()

df_pos_full = pd.concat(all_dfs, ignore_index=True)

st.sidebar.markdown("---")
st.sidebar.header("🔎 Filtros Globales")

all_symbols = sorted(df_pos_full["symbol"].dropna().astype(str).unique())

selected_symbols = st.sidebar.multiselect(
    "Filtrar por Símbolo(s)", 
    options=all_symbols, 
    default=all_symbols,
    help="Los KPIs y gráficos se calcularán SOLO con los trades de estos símbolos."
)

if not selected_symbols:
    st.warning("⚠️ Debes seleccionar al menos un símbolo en la barra lateral para ver resultados.")
    st.stop()

st.sidebar.markdown("---")
st.sidebar.header("🤖 Filtros de Robots")

# ── Mínimo de trades ──────────────────────────────────────────
min_trades_filter = st.sidebar.number_input(
    "Mínimo de trades por robot",
    min_value=1, max_value=500, value=1, step=1,
    help="Robots con menos trades que este valor no se mostrarán."
)

# ── Activo desde fecha ────────────────────────────────────────
tcol_full = "close_time" if ("close_time" in df_pos_full.columns and df_pos_full["close_time"].notna().any()) else "open_time"
all_robot_dates = df_pos_full.groupby("robot_id")[tcol_full].max().dropna()

# Fecha más antigua de última actividad para el slider
global_min_activity = all_robot_dates.min().date() if not all_robot_dates.empty else None
global_max_activity = all_robot_dates.max().date() if not all_robot_dates.empty else None

if global_min_activity and global_max_activity and global_min_activity < global_max_activity:
    last_activity_after = st.sidebar.date_input(
        "Último trade después de",
        value=global_min_activity,
        min_value=global_min_activity,
        max_value=global_max_activity,
        help="Solo muestra robots que hayan operado después de esta fecha. Útil para ocultar robots inactivos."
    )
else:
    last_activity_after = None

df_pos = df_pos_full[df_pos_full["symbol"].isin(selected_symbols)].copy()

if df_pos.empty:
    st.warning("No hay operaciones para los símbolos seleccionados.")
    st.stop()

# ── Aplicar filtros de robots ─────────────────────────────────
# Calcular trades por robot y última actividad
robot_trade_counts  = df_pos.groupby("robot_id").size()
robot_last_activity = df_pos.groupby("robot_id")[tcol_full].max() if tcol_full in df_pos.columns else None

robots_to_keep = set(robot_trade_counts[robot_trade_counts >= min_trades_filter].index)

if last_activity_after is not None and robot_last_activity is not None:
    cutoff = pd.Timestamp(last_activity_after)
    active_robots = set(robot_last_activity[robot_last_activity >= cutoff].index)
    robots_to_keep = robots_to_keep & active_robots

if robots_to_keep:
    df_pos = df_pos[df_pos["robot_id"].isin(robots_to_keep)].copy()
else:
    st.warning("⚠️ Ningún robot cumple los filtros actuales. Ajustá los criterios.")
    st.stop()

n_filtered = len(df_pos_full["robot_id"].unique()) - len(robots_to_keep)
if n_filtered > 0:
    st.sidebar.caption(f"🔍 {n_filtered} robot(s) ocultados por los filtros aplicados.")

# ==========================================
# 📈 MAIN AREA: Tabs
# ==========================================

kpis_df, key_used, tcol = kpis_por_robot(df_pos)

# ── Preservar pestaña activa con query_params ─────────────────────────────
_TAB_NAMES  = ["📊 KPIs Globales", "🔎 Análisis por Robot", "🎯 Edge Analytics"]
_TAB_KEYS   = ["kpis", "robot", "edge"]

# Leer pestaña activa desde query params
_qp = st.query_params.get("tab", "kpis")
_default_tab = _TAB_KEYS.index(_qp) if _qp in _TAB_KEYS else 0

# JavaScript para detectar cambio de pestaña y guardar en query_params
st.markdown("""
<script>
(function() {
    function watchTabs() {
        const tabs = document.querySelectorAll('[data-baseweb="tab"]');
        tabs.forEach(function(tab, i) {
            tab.addEventListener('click', function() {
                const keys = ['kpis', 'robot', 'edge'];
                const url = new URL(window.location.href);
                url.searchParams.set('tab', keys[i] || 'kpis');
                window.history.replaceState({}, '', url.toString());
            });
        });
    }
    setTimeout(watchTabs, 800);
    const obs = new MutationObserver(function() { setTimeout(watchTabs, 300); });
    obs.observe(document.body, { childList: true, subtree: true });
})();
</script>
""", unsafe_allow_html=True)

tab_kpis, tab_robot, tab_edge = st.tabs(_TAB_NAMES)

# ── Tab 1: KPIs Globales ──────────────────────────────────────────────────
with tab_kpis:
    st.subheader(f"📊 KPIs por Robot (Agrupado por {agrupar_por})")
    st.dataframe(
        kpis_df.style.format({
            "Net profit (real)": "{:.2f}",
            "% Wins": "{:.2f}%",
            "PF": "{:.2f}",
            "Expectancy": "{:.2f}",
            "Ret/DD": "{:.2f}",
            "Max DD": "{:.2f}",
        }),
        use_container_width=True
    )

    st.markdown("---")
    st.markdown("#### 🎯 Edge — Comparativa rápida de todos los robots")
    st.caption("Baseline automático (primeros 20 trades). Para ajustar parámetros o ingresar baseline manual, usá la pestaña **Edge Analytics**.")
    robots_all = sorted(kpis_df["Robot ID"].astype(str).unique())
    render_edge_comparison_table(
        df_pos, tcol, robots_all,
        cp_size=10, mp_size=20, lp_size=50,
        baseline_trades=20, baseline_manual=None,
    )

# ── Tab 2: Análisis por Robot ─────────────────────────────────────────────
with tab_robot:
    st.subheader("🔎 Curva de equity por trade (PnL real)")
    robots = sorted(kpis_df['Robot ID'].astype(str).unique())
    selected = st.selectbox("Seleccioná un robot para analizar a fondo:", robots, key="tab2_robot_selector")

    if selected:
        g = df_pos.copy()
        g['robot_id'] = g['robot_id'].astype(str)
        sel = g[g['robot_id'] == str(selected)].copy()

        orden_cols = []
        if tcol in sel.columns: orden_cols.append(tcol)
        if "close_time" in sel.columns and "close_time" != tcol: orden_cols.append("close_time")
        if "open_time" in sel.columns and "open_time" != tcol: orden_cols.append("open_time")
        if "position" in sel.columns: orden_cols.append("position")

        sel = sel.sort_values(orden_cols if orden_cols else [tcol], kind="mergesort")

        sel["pnl_real"] = sel["real_profit"].fillna(0.0)
        sel["#"] = np.arange(1, len(sel) + 1)
        sel["equity_trade"] = sel["pnl_real"].cumsum()

        serie_equity_trade = pd.Series(sel["equity_trade"].values, index=sel["#"])
        st.line_chart(serie_equity_trade, height=280)
        st.caption(f"Equity acumulada ({', '.join(selected_symbols)}) por operación.")

        st.markdown("### 📐 KPIs del robot seleccionado")
        st.dataframe(
            summary_kpis_robot(sel, tcol).style.format({
                "net profit (bruto)": "{:.2f}",
                "commissions total": "{:.2f}",
                "swaps total": "{:.2f}",
                "net profit (real)": "{:.2f}",
                "max dd": "{:.2f}",
                "ret/dd": "{:.2f}",
                "winrate": "{:.2f}%",
                "profit factor": "{:.2f}",
                "sharpe ratio": "{:.2f}",
                "expectancy": "{:.2f}",
                "stability": "{:.2f}",
            }),
            use_container_width=True
        )

        st.markdown("### 🧾 Historial de trades")

        with st.expander("Filtros de historial (Aplica solo a esta tabla)", expanded=False):
            min_dt = pd.to_datetime(sel[tcol].min()) if (tcol in sel.columns and not sel.empty) else None
            max_dt = pd.to_datetime(sel[tcol].max()) if (tcol in sel.columns and not sel.empty) else None
            c1, c2, c3 = st.columns([1,1,1])

            if (min_dt is not None) and (max_dt is not None):
                f_ini = c1.date_input("Desde", value=min_dt.date(), min_value=min_dt.date(), max_value=max_dt.date())
                f_fin = c2.date_input("Hasta", value=max_dt.date(), min_value=min_dt.date(), max_value=max_dt.date())
            else:
                f_ini, f_fin = None, None

            resultado = c3.selectbox("Resultado de la operación", ["Todos", "Ganadores", "Perdedores"])

        hist = sel.copy()
        hist["pnl_real"] = hist["real_profit"].fillna(0.0)

        if f_ini and f_fin and (tcol in hist.columns):
            mask_fecha = (hist[tcol].dt.date >= f_ini) & (hist[tcol].dt.date <= f_fin)
            hist = hist[mask_fecha]
        
        if resultado == "Ganadores": hist = hist[hist["pnl_real"] > 0]
        elif resultado == "Perdedores": hist = hist[hist["pnl_real"] < 0]

        hist = hist.sort_values(orden_cols if orden_cols else [tcol], kind="mergesort")
        hist["equity_cum"] = hist["pnl_real"].cumsum()
        hist["#"] = np.arange(1, len(hist) + 1)

        columnas = ["#"]
        if tcol in hist.columns: columnas.append(tcol)
        otra_time = "open_time" if tcol == "close_time" else "close_time"
        if otra_time in hist.columns and otra_time != tcol: columnas.append(otra_time)

        for col in ["symbol", "type", "volume", "open_price", "close_price",
                    "commission", "swap", "profit", "pnl_real", "equity_cum",
                    "position", "robot_id", "Comment", "Magic", "source_file"]:
            if col in hist.columns:
                columnas.append(col)

        columnas = list(dict.fromkeys(columnas))
        hist_view = hist[columnas].copy()

        if hist_view.columns.duplicated().any():
            from pandas.io.parsers import ParserBase
            hist_view.columns = ParserBase({'names': hist_view.columns})._maybe_dedup_names(hist_view.columns)

        fmt_nums = {
            "volume": "{:.2f}",
            "open_price": "{:.5f}",
            "close_price": "{:.5f}",
            "commission": "{:.2f}",
            "swap": "{:.2f}",
            "profit": "{:.2f}",
            "pnl_real": "{:.2f}",
            "equity_cum": "{:.2f}",
        }

        st.dataframe(
            hist_view.style.format(fmt_nums),
            use_container_width=True,
            height=380
        )

        csv = hist_view.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇️ Descargar historial completo (CSV)",
            data=csv,
            file_name=f"historial_{str(selected).replace(' ','_')}.csv",
            mime="text/csv"
        )

# ── Tab 3: Edge Analytics ─────────────────────────────────────────────────
with tab_edge:
    render_edge_tab(df_pos, tcol)