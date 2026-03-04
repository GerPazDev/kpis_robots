# app.py
import streamlit as st
import pandas as pd
import numpy as np
import re
from io import BytesIO
from bs4 import BeautifulSoup

st.set_page_config(page_title="KPIs por Robot — MT4 & MT5", layout="wide")
st.title("📊 KPIs por Robot 🦅 Aguila Trading (MT4 & MT5)")

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
    """Asegura columnas commission y swap, y crea real_profit = profit+commission+swap."""
    for c in ["commission", "swap", "profit"]:
        if c not in df.columns:
            df[c] = 0.0
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    df["real_profit"] = df["profit"] + df["commission"] + df["swap"]
    return df

def _merge_deals_to_trades(df: pd.DataFrame) -> pd.DataFrame:
    """Algoritmo FIFO para convertir historiales de 'Deals' (2 filas por trade) en Posiciones (1 fila)."""
    if "profit" not in df.columns or len(df) < 2:
        return df
        
    zeros = (df["profit"] == 0).sum()
    if zeros < len(df) * 0.3:
        # Si menos del 30% tiene profit 0, asumimos que ya son posiciones unificadas
        return df

    trades = []
    group_col = "Magic" if "Magic" in df.columns and df["Magic"].notna().any() else "Comment"
    open_deals = {}
    
    time_col = "close_time" if "close_time" in df.columns else "open_time"
    if time_col not in df.columns:
        return df
        
    df = df.sort_values(time_col)
    
    for idx, row in df.iterrows():
        # Ignorar depósitos y balances
        if "type" in df.columns and str(row["type"]).lower() in ["balance", "deposit", "withdrawal"]:
            continue
            
        grp = row.get(group_col, "UNKNOWN")
        sym = row.get("symbol", "UNKNOWN")
        key = (grp, sym)
        
        if key not in open_deals:
            open_deals[key] = []
            
        is_out = False
        
        # Un deal de salida (OUT) suele tener Profit o etiquetas de cierre en el Comment
        if float(row.get("profit", 0) or 0) != 0:
            is_out = True
        elif isinstance(row.get("Comment"), str) and any(x in str(row["Comment"]).lower() for x in ["[tp", "[sl", "close"]):
            is_out = True
        elif "direction" in df.columns and str(row["direction"]).lower() == "out":
            is_out = True
            
        # Si no lo detectamos pero ya hay un deal abierto del tipo opuesto, asumimos que es el cierre (FIFO)
        if not is_out and len(open_deals[key]) > 0:
            last_deal = open_deals[key][0]
            if "type" in row and "type" in last_deal:
                t1, t2 = str(last_deal["type"]).lower(), str(row["type"]).lower()
                if (t1 == "buy" and t2 == "sell") or (t1 == "sell" and t2 == "buy"):
                    is_out = True
                    
        if not is_out:
            # Guardamos el deal de entrada
            open_deals[key].append(row)
        else:
            if len(open_deals[key]) > 0:
                in_deal = open_deals[key].pop(0)
                
                # Fusionamos ambos en una sola operación
                trade = row.to_dict()
                trade["open_time"] = in_deal.get(time_col, row.get(time_col))
                trade["close_time"] = row.get(time_col)
                trade["open_price"] = in_deal.get("open_price", row.get("open_price"))
                trade["close_price"] = row.get("close_price", row.get("open_price"))
                
                # ¡Suma importante de comisiones de ambos deals!
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

    # POSITIONS
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

    # DEALS
    deals_headers = df_raw.iloc[deals_header].tolist()
    df_deals = df_raw.iloc[deals_header + 1 :].copy()
    df_deals.columns = deals_headers
    if "Order" in df_deals.columns: df_deals["Order"] = pd.to_numeric(df_deals["Order"], errors="coerce")
    if "Time" in df_deals.columns: df_deals["Time"] = pd.to_datetime(df_deals["Time"], errors="coerce")
    if "Volume" in df_deals.columns: df_deals["Volume"] = pd.to_numeric(df_deals["Volume"], errors="coerce")
    if "Symbol" in df_deals.columns: df_deals["Symbol"] = df_deals["Symbol"].astype(str)

    # 1) Map directo por Order (deal 'in') -> Comment / Magic
    if {"Order","Direction"}.issubset(df_deals.columns):
        df_in = df_deals[df_deals["Direction"].astype(str).str.lower().eq("in")].copy()
        df_in = df_in.dropna(subset=["Order"]).drop_duplicates("Order")
        
        if "Comment" in df_in.columns and "position" in df_pos.columns:
            robot_map_c = df_in.set_index("Order")["Comment"].to_dict()
            df_pos["Comment"] = df_pos["position"].map(robot_map_c)
            
        if "Magic" in df_in.columns and "position" in df_pos.columns:
            robot_map_m = df_in.set_index("Order")["Magic"].to_dict()
            df_pos["Magic"] = df_pos["position"].map(robot_map_m)

    # 2) Fallback: merge_asof por símbolo (hacia atrás)
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
def parse_csv(uploaded_csv: BytesIO) -> pd.DataFrame:
    file_bytes = uploaded_csv.read()
    
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
    
    # NUEVO: Convertimos los Deals en Trades enteros antes de procesar los KPIs
    df_pos = _merge_deals_to_trades(df_pos)
    
    return _ensure_fee_cols(df_pos)

# ========= Parsing MT4 (HTML) =========
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

# Selector de agrupación
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

# Procesamiento de Archivos
for uploaded in uploaded_files:
    suffix = uploaded.name.lower().split(".")[-1]
    try:
        if suffix == "xlsx":
            df_temp = parse_mt5_xlsx(uploaded.read())
        elif suffix == "csv":
            df_temp = parse_csv(uploaded)
        elif suffix in ("htm", "html"):
            df_temp = parse_mt4_html(uploaded.read())
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

# Unimos todos los DataFrames
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

df_pos = df_pos_full[df_pos_full["symbol"].isin(selected_symbols)].copy()

if df_pos.empty:
    st.warning("No hay operaciones para los símbolos seleccionados.")
    st.stop()

# ==========================================
# 📈 MAIN AREA: Resultados y Gráficos
# ==========================================

kpis_df, key_used, tcol = kpis_por_robot(df_pos)

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

# ====== Equity por TRADE ======
st.subheader("🔎 Curva de equity por trade (PnL real)")
robots = sorted(kpis_df['Robot ID'].astype(str).unique())
selected = st.selectbox("Seleccioná un robot para analizar a fondo:", robots)

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

    # =========================
    # 🧾 Historial de trades
    # =========================
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