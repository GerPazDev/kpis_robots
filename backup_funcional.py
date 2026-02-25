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

# ========= Parsing MT5 (XLSX/CSV) — Comment como robot =========
def parse_mt5_xlsx_use_comment(xlsx_bytes: bytes, time_tolerance="10min", vol_tol=1e-6) -> pd.DataFrame:
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
        "Time": "open_time",
        "Time_1": "close_time",
        "Position": "position",
        "Symbol": "symbol",
        "Type": "type",
        "Volume": "volume",
        "Price": "open_price",
        "Price_1": "close_price",
        "S / L": "sl",
        "T / P": "tp",
        "Commission": "commission",
        "Swap": "swap",
        "Profit": "profit",
    }
    df_pos = df_pos.rename(columns=rename_map)

    for c in ["open_time", "close_time"]:
        if c in df_pos.columns:
            df_pos[c] = pd.to_datetime(df_pos[c], errors="coerce")
    for c in ["position","volume","open_price","close_price","sl","tp","commission","swap","profit"]:
        if c in df_pos.columns:
            df_pos[c] = pd.to_numeric(df_pos[c], errors="coerce")
    if "symbol" in df_pos.columns:
        df_pos["symbol"] = df_pos["symbol"].astype(str)
    if "profit" in df_pos.columns:
        df_pos = df_pos[~df_pos["profit"].isna()].copy()

    # DEALS
    deals_headers = df_raw.iloc[deals_header].tolist()
    df_deals = df_raw.iloc[deals_header + 1 :].copy()
    df_deals.columns = deals_headers
    if "Order" in df_deals.columns:
        df_deals["Order"] = pd.to_numeric(df_deals["Order"], errors="coerce")
    if "Time" in df_deals.columns:
        df_deals["Time"] = pd.to_datetime(df_deals["Time"], errors="coerce")
    if "Volume" in df_deals.columns:
        df_deals["Volume"] = pd.to_numeric(df_deals["Volume"], errors="coerce")
    if "Symbol" in df_deals.columns:
        df_deals["Symbol"] = df_deals["Symbol"].astype(str)

    # 1) Map directo por Order→Comment (deal 'in')
    robot_map = {}
    if {"Order","Direction","Comment"}.issubset(df_deals.columns):
        df_in = df_deals[df_deals["Direction"].astype(str).str.lower().eq("in")].copy()
        df_in = df_in[["Order","Comment"]].dropna(subset=["Order"]).drop_duplicates("Order")
        robot_map = df_in.set_index("Order")["Comment"].to_dict()
    if "position" in df_pos.columns:
        df_pos["robot_id"] = df_pos["position"].map(robot_map)

    # 2) Fallback: merge_asof por símbolo (hacia atrás)
    need = df_pos["robot_id"].isna()
    if need.any() and {"Time","Symbol","Direction","Comment"}.issubset(df_deals.columns):
        df_in2 = df_deals[df_deals["Direction"].astype(str).str.lower().eq("in")].copy()
        df_in2 = df_in2[["Time","Symbol","Volume","Comment"]].dropna(subset=["Time","Symbol"])
        df_in2 = df_in2.sort_values("Time")

        df_pos_sorted = df_pos.sort_values("open_time")
        matched_robot = []
        for sym, grp_pos in df_pos_sorted[need].groupby("symbol"):
            grp_in = df_in2[df_in2["Symbol"] == str(sym)]
            if grp_in.empty:
                matched_robot.append(pd.Series(index=grp_pos.index, dtype=object))
                continue
            m = pd.merge_asof(
                grp_pos[["open_time","volume"]].sort_values("open_time"),
                grp_in[["Time","Volume","Comment"]].sort_values("Time"),
                left_on="open_time", right_on="Time",
                direction="backward", tolerance=pd.Timedelta(time_tolerance)
            )
            vol_match = np.isfinite(m["volume"]) & np.isfinite(m["Volume"])
            close_vol = (np.abs(m["volume"] - m["Volume"]) <= vol_tol) | (~vol_match)
            match_comment = m["Comment"].where(close_vol)
            match_comment.index = grp_pos.sort_values("open_time").index
            matched_robot.append(match_comment)

        if matched_robot:
            matched_robot = pd.concat(matched_robot).sort_index()
            df_pos.loc[matched_robot.index, "robot_id"] = df_pos.loc[matched_robot.index, "robot_id"].fillna(matched_robot)

    # Último recurso: separar por símbolo
    if df_pos["robot_id"].isna().any():
        df_pos.loc[df_pos["robot_id"].isna(), "robot_id"] = df_pos.loc[df_pos["robot_id"].isna(), "symbol"] + "_UNKNOWN"

    return _ensure_fee_cols(df_pos)

def parse_csv_use_comment(uploaded_csv: BytesIO) -> pd.DataFrame:
    df_pos = pd.read_csv(uploaded_csv)
    rename_try = {
        'Open Time':'open_time','Close Time':'close_time','Symbol':'symbol','Type':'type',
        'Volume':'volume','Open Price':'open_price','Close Price':'close_price',
        'Commission':'commission','Swap':'swap','Profit':'profit',
        'Comment':'robot_id'
    }
    df_pos = df_pos.rename(columns={k: v for k, v in rename_try.items() if k in df_pos.columns})
    for c in ['open_time','close_time']:
        if c in df_pos.columns:
            df_pos[c] = pd.to_datetime(df_pos[c], errors='coerce')
    for c in ['volume','open_price','close_price','commission','swap','profit']:
        if c in df_pos.columns:
            df_pos[c] = pd.to_numeric(df_pos[c], errors='coerce')
    if 'robot_id' not in df_pos.columns:
        df_pos['robot_id'] = "UNKNOWN"
    return _ensure_fee_cols(df_pos)

# ========= Parsing MT4 (HTML) — Comment como robot (limpia [tp]/[sl]/[s]) =========
def parse_mt4_html_use_comment(html_bytes: bytes) -> pd.DataFrame:
    soup = BeautifulSoup(html_bytes, "html.parser")
    rows = soup.find_all("tr", attrs={"align": "right"})
    data = []

    for tr in rows:
        tds = tr.find_all("td")
        if len(tds) < 14:
            continue

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
            try:
                return float(str(x).replace(",", ""))
            except Exception:
                return np.nan

        def to_time(x):
            return pd.to_datetime(x, errors="coerce")

        open_time = to_time(tds[1].get_text(strip=True))
        typ = tds[2].get_text(strip=True)
        volume = to_float(tds[3].get_text(strip=True))
        symbol = tds[4].get_text(strip=True)
        open_price = to_float(tds[5].get_text(strip=True))
        sl = to_float(tds[6].get_text(strip=True))
        tp = to_float(tds[7].get_text(strip=True))
        close_time = to_time(tds[8].get_text(strip=True))
        close_price = to_float(tds[9].get_text(strip=True))
        commission = to_float(tds[10].get_text(strip=True))
        swap = to_float(tds[12].get_text(strip=True))
        profit = profit_val

        data.append({
            "open_time": open_time,
            "close_time": close_time,
            "position": pd.to_numeric(ticket, errors="coerce"),
            "symbol": symbol,
            "type": typ,
            "volume": volume,
            "open_price": open_price,
            "sl": sl,
            "tp": tp,
            "close_price": close_price,
            "commission": commission,
            "swap": swap,
            "profit": profit,
            "robot_id": comment if comment else symbol + "_UNKNOWN"
        })

    df = pd.DataFrame(data)
    if not df.empty:
        for c in ["open_time", "close_time"]:
            df[c] = pd.to_datetime(df[c], errors="coerce")
        for c in ["position","volume","open_price","close_price","sl","tp","commission","swap","profit"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df["symbol"] = df["symbol"].astype(str)
        df["robot_id"] = df["robot_id"].astype(str)

    return _ensure_fee_cols(df)

# ========= KPIs (comunes; usan real_profit) =========
def kpis_por_robot(df_pos: pd.DataFrame):
    key = "robot_id"
    df = df_pos.copy()
    df[key] = df[key].astype(str)

    tcol = "close_time" if ("close_time" in df.columns and df["close_time"].notna().any()) else "open_time"
    df = df.sort_values(tcol)

    rows = []
    for rid, g in df.groupby(key):
        pnl_real = g["real_profit"].fillna(0.0)
        equity = pnl_real.cumsum()
        mdd = max_drawdown(equity)
        net_real = float(pnl_real.sum())
        ret_dd = float(net_real / mdd) if mdd > 0 else np.nan

        rows.append({
            "Robot (Comment)": rid,
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
        "comment": [g["robot_id"].iloc[0] if "robot_id" in g.columns else ""],
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

# ========= UI =========
uploaded = st.file_uploader("📥 Subí tu archivo (MT5: .xlsx/.csv | MT4: .htm/.html)", type=["xlsx","csv","htm","html"])
if not uploaded:
    st.info("Cargá un archivo de **MetaTrader 5 (.xlsx/.csv)** o **MetaTrader 4 (.htm/.html)** para comenzar.")
    st.stop()

suffix = uploaded.name.lower().split(".")[-1]
try:
    if suffix == "xlsx":
        df_pos = parse_mt5_xlsx_use_comment(uploaded.read())
    elif suffix == "csv":
        df_pos = parse_csv_use_comment(uploaded)
    elif suffix in ("htm", "html"):
        df_pos = parse_mt4_html_use_comment(uploaded.read())
    else:
        st.error("Extensión no soportada."); st.stop()

    if df_pos.empty:
        st.warning("No se encontraron operaciones cerradas con Profit numérico.")
        st.stop()

except Exception as e:
    st.error("No se pudo leer el archivo. Verificá el formato/export.")
    st.exception(e)
    st.stop()

# ====== Resultados ======
kpis_df, key_used, tcol = kpis_por_robot(df_pos)

# Tabla general (2 decimales; winrate en %)
st.subheader("📊 KPIs por Robot (agrupado por Comment)")
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

# ====== Equity por TRADE (reemplaza gráfico diario) ======
st.subheader("🔎 Curva de equity por trade (PnL real)")
robots = sorted(kpis_df['Robot (Comment)'].astype(str).unique())
selected = st.selectbox("Elegí un robot", robots)

if selected:
    g = df_pos.copy()
    g['robot_id'] = g['robot_id'].astype(str)
    sel = g[g['robot_id'] == str(selected)].copy()

    # Orden determinista de trades
    orden_cols = []
    if tcol in sel.columns:
        orden_cols.append(tcol)
    if "close_time" in sel.columns and "close_time" != tcol:
        orden_cols.append("close_time")
    if "open_time" in sel.columns and "open_time" != tcol:
        orden_cols.append("open_time")
    if "position" in sel.columns:
        orden_cols.append("position")

    sel = sel.sort_values(orden_cols if orden_cols else [tcol], kind="mergesort")

    # PnL por trade y equity acumulada por trade
    sel["pnl_real"] = sel["real_profit"].fillna(0.0)
    sel["#"] = np.arange(1, len(sel) + 1)
    sel["equity_trade"] = sel["pnl_real"].cumsum()

    # Gráfico por trade
    serie_equity_trade = pd.Series(sel["equity_trade"].values, index=sel["#"])
    st.line_chart(serie_equity_trade, height=280)
    st.caption("Equity acumulada por operación (secuencia de trades).")

    # KPIs del robot (2 decimales; winrate en %)
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
    st.markdown("### 🧾 Historial de trades del robot seleccionado")

    # Filtros rápidos
    with st.expander("Filtros", expanded=False):
        min_dt = pd.to_datetime(sel[tcol].min()) if (tcol in sel.columns and not sel.empty) else None
        max_dt = pd.to_datetime(sel[tcol].max()) if (tcol in sel.columns and not sel.empty) else None
        c1, c2, c3, c4 = st.columns([1,1,1,1])

        if (min_dt is not None) and (max_dt is not None):
            f_ini = c1.date_input("Desde", value=min_dt.date(), min_value=min_dt.date(), max_value=max_dt.date())
            f_fin = c2.date_input("Hasta", value=max_dt.date(), min_value=min_dt.date(), max_value=max_dt.date())
        else:
            f_ini, f_fin = None, None

        syms = sorted(sel["symbol"].dropna().astype(str).unique()) if "symbol" in sel.columns else []
        sym_pick = c3.multiselect("Símbolos", syms, default=syms)
        resultado = c4.selectbox("Resultado", ["Todos", "Ganadores", "Perdedores"])

    # Preparación del historial (partimos de 'sel' ya ordenado)
    hist = sel.copy()
    hist["pnl_real"] = hist["real_profit"].fillna(0.0)

    # Aplicar filtros
    if f_ini and f_fin and (tcol in hist.columns):
        mask_fecha = (hist[tcol].dt.date >= f_ini) & (hist[tcol].dt.date <= f_fin)
        hist = hist[mask_fecha]
    if sym_pick and "symbol" in hist.columns:
        hist = hist[hist["symbol"].astype(str).isin(sym_pick)]
    if resultado == "Ganadores":
        hist = hist[hist["pnl_real"] > 0]
    elif resultado == "Perdedores":
        hist = hist[hist["pnl_real"] < 0]

    # Equity acumulada (sobre el subconjunto filtrado)
    hist = hist.sort_values(orden_cols if orden_cols else [tcol], kind="mergesort")
    hist["equity_cum"] = hist["pnl_real"].cumsum()
    hist["#"] = np.arange(1, len(hist) + 1)

    # Columnas amigables para mostrar (sin duplicados)
    columnas = ["#"]
    if tcol in hist.columns:
        columnas.append(tcol)
    otra_time = "open_time" if tcol == "close_time" else "close_time"
    if otra_time in hist.columns and otra_time != tcol:
        columnas.append(otra_time)

    for col in ["symbol", "type", "volume", "open_price", "close_price",
                "commission", "swap", "profit", "pnl_real", "equity_cum",
                "position", "robot_id"]:
        if col in hist.columns:
            columnas.append(col)

    columnas = list(dict.fromkeys(columnas))
    hist_view = hist[columnas].copy()

    # Blindaje extra si algún export trae nombres repetidos
    if hist_view.columns.duplicated().any():
        from pandas.io.parsers import ParserBase
        hist_view.columns = ParserBase({'names': hist_view.columns})._maybe_dedup_names(hist_view.columns)

    # Formato numérico
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

    # Descargar CSV del historial filtrado
    csv = hist_view.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Descargar historial (CSV)",
        data=csv,
        file_name=f"historial_{str(selected).replace(' ','_')}.csv",
        mime="text/csv"
    )
