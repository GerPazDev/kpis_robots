# app.py
import streamlit as st
import pandas as pd
import numpy as np
import re
from io import BytesIO

st.set_page_config(page_title="KPIs por Robot - Aguila Trading", layout="wide")
st.title("KPIs por Robot - Aguila Trading 🦅")

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

# ========= Parsing XLSX (MT5) =========
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

def parse_mt5_xlsx_use_comment(xlsx_bytes: bytes, time_tolerance="10min", vol_tol=1e-6) -> pd.DataFrame:
    """
    Normaliza POSITIONS y etiqueta cada posición con robot_id = Comment (deal 'in').
    Si no hay match directo por Order/Position, usa merge_asof por símbolo y tiempo (hacia atrás).
    """
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

    # Último recurso: separar por símbolo (evitar UNASSIGNED global)
    if df_pos["robot_id"].isna().any():
        df_pos.loc[df_pos["robot_id"].isna(), "robot_id"] = df_pos.loc[df_pos["robot_id"].isna(), "symbol"] + "_UNKNOWN"

    return df_pos

# ========= KPIs =========
def kpis_por_robot(df_pos: pd.DataFrame):
    key = "robot_id"
    df = df_pos.copy()
    df[key] = df[key].astype(str)

    tcol = "close_time" if ("close_time" in df.columns and df["close_time"].notna().any()) else "open_time"
    df = df.sort_values(tcol)

    rows = []
    for rid, g in df.groupby(key):
        pnl = g["profit"].fillna(0.0)
        equity = pnl.cumsum()
        rows.append({
            "Robot (Comment)": rid,
            "Net profit": pnl.sum(),
            "# Trades": int((~pnl.isna()).sum()),
            "% Wins": float((pnl > 0).mean() * 100.0) if len(pnl) else 0.0,
            "PF": profit_factor(pnl),
            "Expectancy": expectancy(pnl),
            "Max DD": max_drawdown(equity),
            "Desde": g[tcol].min(),
            "Hasta": g[tcol].max(),
            "Símbolos": ", ".join(sorted(set(g["symbol"].dropna().astype(str)))) if "symbol" in g else "",
        })
    out = pd.DataFrame(rows).sort_values("Net profit", ascending=False)
    return out, key, tcol

def summary_kpis_robot(g: pd.DataFrame, tcol: str) -> pd.DataFrame:
    pnl = g["profit"].fillna(0.0)
    equity = pnl.cumsum()
    wins = int((pnl > 0).sum())
    losses = int((pnl < 0).sum())
    mdd = max_drawdown(equity)
    net = float(pnl.sum())
    ret_dd = float(net / mdd) if mdd > 0 else np.nan
    sharp = sharpe_per_trade(pnl)
    stab = stability_r2(equity)
    stagn = max_stagnation(g[tcol] if tcol in g.columns else None, equity)

    data = {
        "comment": [g["robot_id"].iloc[0] if "robot_id" in g.columns else ""],
        "trades": [int(len(pnl))],
        "net profit": [net],
        "max dd": [mdd],
        "ret/dd": [ret_dd],
        "winrate": [float((pnl > 0).mean() * 100.0) if len(pnl) else 0.0],
        "wins": [wins],
        "loss": [losses],
        "profit factor": [profit_factor(pnl)],
        "sharpe ratio": [sharp],
        "expectancy": [expectancy(pnl)],
        "stability": [stab],
        "stagnation": [stagn],
    }
    return pd.DataFrame(data)

# ========= UI =========
uploaded = st.file_uploader("📥 Subí tu archivo (.xlsx de MT5 o .csv)", type=["xlsx","csv"])
if not uploaded:
    st.info("Cargá un archivo de **MetaTrader (.xlsx)** o **CSV** para comenzar.")
    st.stop()

suffix = uploaded.name.lower().split(".")[-1]
try:
    if suffix == "xlsx":
        df_pos = parse_mt5_xlsx_use_comment(uploaded.read())
    else:
        df_pos = pd.read_csv(uploaded)
        # Normalización mínima para CSV: usar 'Comment' como robot_id
        rename_try = {
            'Open Time':'open_time','Close Time':'close_time','Symbol':'symbol','Type':'type',
            'Volume':'volume','Open Price':'open_price','Close Price':'close_price',
            'Commission':'commission','Swap':'swap','Profit':'profit',
            'Comment':'robot_id'
        }
        df_pos = df_pos.rename(columns={k:v for k,v in rename_try.items() if k in df_pos.columns})
        for c in ['open_time','close_time']:
            if c in df_pos.columns: df_pos[c] = pd.to_datetime(df_pos[c], errors='coerce')
        for c in ['volume','open_price','close_price','commission','swap','profit']:
            if c in df_pos.columns: df_pos[c] = pd.to_numeric(df_pos[c], errors='coerce')
        if 'robot_id' not in df_pos.columns:
            df_pos['robot_id'] = "UNKNOWN"
except Exception as e:
    st.error("No se pudo leer el archivo. Verificá el formato/export.")
    st.exception(e)
    st.stop()

# ====== Resultados ======
kpis_df, key_used, tcol = kpis_por_robot(df_pos)

# Tabla general con formatting (2 decimales, winrate en %)
st.subheader("📊 KPIs por Robot (agrupado por Comment)")
st.dataframe(
    kpis_df.style.format({
        "Net profit": "{:.2f}",
        "% Wins": "{:.2f}%",
        "PF": "{:.2f}",
        "Expectancy": "{:.2f}",
        "Max DD": "{:.2f}"
    }),
    use_container_width=True
)

# Curva de equity (diaria, consolidada al último valor por día)
st.subheader("🔎 Curva de equity por robot")
robots = kpis_df['Robot (Comment)'].astype(str).tolist()
selected = st.selectbox("Elegí un robot", robots)

if selected:
    g = df_pos.copy()
    g['robot_id'] = g['robot_id'].astype(str)
    sel = g[g['robot_id'] == str(selected)].sort_values(tcol)

    pnl = sel['profit'].fillna(0.0)
    equity_cum = pnl.cumsum()

    # Columna de tiempo y consolidación diaria (último valor por día)
    dates = sel[tcol].dt.floor("D") if tcol in sel.columns else None
    if dates is not None:
        equity_daily = pd.Series(equity_cum.values, index=dates).groupby(level=0).last()
    else:
        equity_daily = equity_cum  # fallback por si faltara fecha

    st.line_chart(equity_daily, height=280)
    st.caption("Equity acumulada (último valor por día) del robot seleccionado.")

    # KPIs del robot seleccionado (2 decimales, winrate en %)
    st.markdown("### 📐 KPIs del robot seleccionado")
    st.dataframe(
        summary_kpis_robot(sel, tcol).style.format({
            "net profit": "{:.2f}",
            "max dd": "{:.2f}",
            "ret/dd": "{:.2f}",
            "winrate": "{:.2f}%",
            "profit factor": "{:.2f}",
            "sharpe ratio": "{:.2f}",
            "expectancy": "{:.2f}",
            "stability": "{:.2f}"
        }),
        use_container_width=True
    )
