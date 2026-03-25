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

if "active_tab" not in st.session_state:
    st.session_state.active_tab = 0

# ========= Métricas =========
def max_drawdown(equity: pd.Series) -> float:
    if equity.empty:
        return 0.0
    roll_max = equity.cummax()
    dd = roll_max - equity
    return float(dd.max())

def current_drawdown_info(equity: pd.Series, account_balance: float | None = None) -> dict:
    """
    Calcula el drawdown en curso: distancia desde el último equity peak.
    Si se provee account_balance, calcula el DD como % del balance de la cuenta.
    """
    if equity.empty:
        return {
            "current_dd": 0.0, "dd_pct_account": 0.0 if account_balance else None,
            "max_dd": 0.0, "max_dd_pct_account": 0.0 if account_balance else None,
            "peak_value": 0.0, "current_value": 0.0,
            "trades_since_peak": 0, "dd_ratio": 0.0,
            "is_at_peak": True, "account_balance": account_balance,
        }

    e = equity.values.astype(float)
    peaks = np.maximum.accumulate(e)
    current_val = float(e[-1])
    peak_val = float(peaks[-1])
    curr_dd = float(peak_val - current_val)
    max_dd_val = float((peaks - e).max())

    peak_idx = np.where(e == peak_val)[0]
    last_peak_idx = int(peak_idx[-1]) if len(peak_idx) > 0 else 0
    trades_since = len(e) - 1 - last_peak_idx

    dd_ratio = (curr_dd / max_dd_val) if max_dd_val > 0 else 0.0
    is_at_peak = curr_dd < 1e-8

    dd_pct_account = None
    max_dd_pct_account = None
    if account_balance is not None and account_balance > 0:
        dd_pct_account = (curr_dd / account_balance) * 100.0
        max_dd_pct_account = (max_dd_val / account_balance) * 100.0

    return {
        "current_dd": curr_dd, "dd_pct_account": dd_pct_account,
        "max_dd": max_dd_val, "max_dd_pct_account": max_dd_pct_account,
        "peak_value": peak_val, "current_value": current_val,
        "trades_since_peak": trades_since, "dd_ratio": dd_ratio,
        "is_at_peak": is_at_peak, "account_balance": account_balance,
    }

def profit_factor(pnl: pd.Series) -> float:
    gains = pnl[pnl > 0].sum()
    losses = pnl[pnl < 0].sum()
    return float(gains / abs(losses)) if losses < 0 else (np.inf if gains > 0 else 0.0)

def expectancy(pnl: pd.Series) -> float:
    return float(pnl.mean()) if len(pnl) else 0.0

def sharpe_per_trade(pnl: pd.Series) -> float:
    if len(pnl) < 2: return 0.0
    std = pnl.std(ddof=1)
    if std == 0 or np.isnan(std): return 0.0
    return float(pnl.mean() / std * np.sqrt(len(pnl)))

def stability_r2(equity: pd.Series) -> float:
    y = equity.values.astype(float)
    if len(y) < 2 or np.allclose(y.std(), 0): return 0.0
    x = np.arange(len(y), dtype=float)
    x_mean, y_mean = x.mean(), y.mean()
    cov = ((x - x_mean) * (y - y_mean)).sum()
    var_x = ((x - x_mean) ** 2).sum()
    if var_x == 0: return 0.0
    beta = cov / var_x
    alpha = y_mean - beta * x_mean
    y_hat = alpha + beta * x
    sse = ((y - y_hat) ** 2).sum()
    sst = ((y - y_mean) ** 2).sum()
    return float(max(0.0, 1.0 - sse / sst)) if sst > 0 else 0.0

def max_stagnation(times: pd.Series | None, equity: pd.Series) -> str:
    e = equity.values.astype(float)
    if len(e) == 0: return "0"
    peaks = np.maximum.accumulate(e)
    uw = e < peaks
    if not uw.any(): return "0"
    longest = curr = 0
    start_idx = best_start = best_end = None
    for i, flag in enumerate(uw):
        if flag:
            curr += 1
            if curr == 1: start_idx = i
            if curr > longest:
                longest = curr
                best_start, best_end = start_idx, i
        else:
            curr = 0
    if times is not None and pd.api.types.is_datetime64_any_dtype(times):
        t0, t1 = times.iloc[best_start], times.iloc[best_end]
        if pd.isna(t0) or pd.isna(t1): return f"{longest} trades"
        return f"{(t1 - t0).days} d"
    return f"{longest} trades"

# ========= Helpers =========
def _rename_dupes(cols):
    seen, out = {}, []
    for c in cols:
        if c != c: out.append(None); continue
        if c not in seen: seen[c] = 0; out.append(c)
        else: seen[c] += 1; out.append(f"{c}_{seen[c]}")
    return out

def _ensure_fee_cols(df: pd.DataFrame) -> pd.DataFrame:
    for c in ["commission", "swap", "profit"]:
        if c not in df.columns: df[c] = 0.0
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    df["real_profit"] = df["profit"] + df["commission"] + df["swap"]
    return df

def _merge_deals_to_trades(df: pd.DataFrame) -> pd.DataFrame:
    if "profit" not in df.columns or len(df) < 2: return df
    zeros = (df["profit"] == 0).sum()
    if zeros < len(df) * 0.3: return df
    trades = []
    group_col = "Magic" if "Magic" in df.columns and df["Magic"].notna().any() else "Comment"
    open_deals = {}
    time_col = "close_time" if "close_time" in df.columns else "open_time"
    if time_col not in df.columns: return df
    df = df.sort_values(time_col)
    for idx, row in df.iterrows():
        if "type" in df.columns and str(row["type"]).lower() in ["balance", "deposit", "withdrawal"]: continue
        grp = row.get(group_col, "UNKNOWN")
        sym = row.get("symbol", "UNKNOWN")
        key = (grp, sym)
        if key not in open_deals: open_deals[key] = []
        is_out = False
        if float(row.get("profit", 0) or 0) != 0: is_out = True
        elif isinstance(row.get("Comment"), str) and any(x in str(row["Comment"]).lower() for x in ["[tp", "[sl", "close"]): is_out = True
        elif "direction" in df.columns and str(row["direction"]).lower() == "out": is_out = True
        if not is_out and len(open_deals[key]) > 0:
            last_deal = open_deals[key][0]
            if "type" in row and "type" in last_deal:
                t1, t2 = str(last_deal["type"]).lower(), str(row["type"]).lower()
                if (t1 == "buy" and t2 == "sell") or (t1 == "sell" and t2 == "buy"): is_out = True
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
    if not trades: return df
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
            df_pos["Comment"] = df_pos["position"].map(df_in.set_index("Order")["Comment"].to_dict())
        if "Magic" in df_in.columns and "position" in df_pos.columns:
            df_pos["Magic"] = df_pos["position"].map(df_in.set_index("Order")["Magic"].to_dict())
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
                direction="backward", tolerance=pd.Timedelta(time_tolerance))
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
    try: df_pos = pd.read_csv(BytesIO(file_bytes), sep=None, engine='python', encoding='utf-16')
    except UnicodeError:
        try: df_pos = pd.read_csv(BytesIO(file_bytes), sep=None, engine='python', encoding='utf-8-sig')
        except UnicodeError: df_pos = pd.read_csv(BytesIO(file_bytes), sep=None, engine='python', encoding='latin-1')
    rename_try = {
        'Open Time':'open_time', 'Close Time':'close_time', 'Time':'close_time', 'Date':'close_time',
        'Symbol':'symbol', 'Item':'symbol', 'Type':'type', 'Action':'type', 'Direction':'direction',
        'Volume':'volume', 'Size':'volume', 'Open Price':'open_price', 'Close Price':'close_price',
        'Price':'close_price', 'Commission':'commission', 'Swap':'swap', 'Profit':'profit',
        'Comment':'Comment', 'Magic':'Magic', 'MagicNumber':'Magic', 'Magic Number':'Magic',
        'Ticket':'position', 'Order':'position'
    }
    col_map = {}
    for col in df_pos.columns:
        col_lower = str(col).strip().lower()
        for k, v in rename_try.items():
            if col_lower == k.lower(): col_map[col] = v; break
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

# ========= Parsing MT5 Deals CSV =========
@st.cache_data(show_spinner=False)
def parse_mt5_deals_csv(file_bytes: bytes) -> pd.DataFrame:
    for enc in ["utf-16", "utf-8-sig", "utf-8", "latin-1"]:
        try: df_raw = pd.read_csv(BytesIO(file_bytes), sep=";", encoding=enc); break
        except Exception: continue
    else: raise ValueError("No se pudo leer el archivo CSV de Deals.")
    if "Type" in df_raw.columns:
        df_raw = df_raw[~df_raw["Type"].astype(str).str.lower().isin(["balance", "deposit", "withdrawal", "credit"])].copy()
    if "Time" in df_raw.columns: df_raw["Time"] = pd.to_datetime(df_raw["Time"], errors="coerce")
    for c in ["Volume", "Price", "Commission", "Fee", "Swap", "Profit", "Balance", "Magic", "Order"]:
        if c in df_raw.columns: df_raw[c] = pd.to_numeric(df_raw[c], errors="coerce")
    df_raw = df_raw.dropna(subset=["Time"]).sort_values("Time").reset_index(drop=True)
    trades, open_deals = [], {}
    for _, row in df_raw.iterrows():
        magic, sym = row.get("Magic", 0), str(row.get("Symbol", "UNKNOWN"))
        key = (magic, sym)
        if key not in open_deals: open_deals[key] = []
        profit_val = float(row.get("Profit", 0) or 0)
        comment = str(row.get("Comment", ""))
        is_out = (profit_val != 0) or comment.startswith("[")
        if not is_out:
            open_deals[key].append(row)
        else:
            if open_deals[key]:
                entry = open_deals[key].pop(0)
                trades.append({
                    "open_time": entry["Time"], "close_time": row["Time"], "symbol": sym,
                    "type": str(entry.get("Direction", "")),
                    "volume": float(entry.get("Volume", 0) or 0),
                    "open_price": float(entry.get("Price", 0) or 0),
                    "close_price": float(row.get("Price", 0) or 0),
                    "commission": float(entry.get("Commission", 0) or 0) + float(row.get("Commission", 0) or 0),
                    "swap": float(entry.get("Swap", 0) or 0) + float(row.get("Swap", 0) or 0),
                    "profit": profit_val, "Magic": magic,
                    "Comment": str(entry.get("Comment", "")),
                    "position": float(entry.get("Order", 0) or 0),
                })
            else:
                trades.append({
                    "open_time": row["Time"], "close_time": row["Time"], "symbol": sym,
                    "type": str(row.get("Direction", "")),
                    "volume": float(row.get("Volume", 0) or 0),
                    "open_price": float(row.get("Price", 0) or 0),
                    "close_price": float(row.get("Price", 0) or 0),
                    "commission": float(row.get("Commission", 0) or 0),
                    "swap": float(row.get("Swap", 0) or 0),
                    "profit": profit_val, "Magic": magic,
                    "Comment": str(row.get("Comment", "")),
                    "position": float(row.get("Order", 0) or 0),
                })
    if not trades: return pd.DataFrame()
    df = pd.DataFrame(trades)
    df["Comment"] = df["Comment"].replace("nan", np.nan)
    return _ensure_fee_cols(df)

def _is_deals_csv(file_bytes: bytes) -> bool:
    for enc in ["utf-16", "utf-8-sig", "utf-8", "latin-1"]:
        try:
            sample = file_bytes[:1000].decode(enc)
            first_line = sample.splitlines()[0] if sample.splitlines() else ""
            cols = [c.strip().lower() for c in first_line.split(";")]
            return "magic" in cols and "deal" in cols and "direction" in cols
        except Exception: continue
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
        try: profit_val = float(tds[13].get_text(strip=True).replace(",", ""))
        except ValueError: continue
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
            except: return np.nan
        def to_time(x): return pd.to_datetime(x, errors="coerce")
        data.append({
            "open_time": to_time(tds[1].get_text(strip=True)),
            "close_time": to_time(tds[8].get_text(strip=True)),
            "position": pd.to_numeric(ticket, errors="coerce"),
            "symbol": tds[4].get_text(strip=True), "type": tds[2].get_text(strip=True),
            "volume": to_float(tds[3].get_text(strip=True)),
            "open_price": to_float(tds[5].get_text(strip=True)),
            "sl": to_float(tds[6].get_text(strip=True)), "tp": to_float(tds[7].get_text(strip=True)),
            "close_price": to_float(tds[9].get_text(strip=True)),
            "commission": to_float(tds[10].get_text(strip=True)),
            "swap": to_float(tds[12].get_text(strip=True)),
            "profit": profit_val, "Comment": comment if comment else np.nan, "Magic": np.nan,
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
def kpis_por_robot(df_pos: pd.DataFrame, account_balance: float | None = None, risk_per_trade: float | None = None):
    key = "robot_id"
    df = df_pos.copy()
    df[key] = df[key].astype(str)
    if "close_time" in df.columns and df["close_time"].notna().any(): tcol = "close_time"
    elif "open_time" in df.columns and df["open_time"].notna().any(): tcol = "open_time"
    else:
        tcol = "close_time"
        if "close_time" not in df.columns: df["close_time"] = pd.NaT
    df = df.sort_values(tcol)
    rows = []
    for rid, g in df.groupby(key):
        pnl_real = g["real_profit"].fillna(0.0)
        equity = pnl_real.cumsum()
        mdd = max_drawdown(equity)
        net_real = float(pnl_real.sum())
        ret_dd = float(net_real / mdd) if mdd > 0 else np.nan
        dd_info = current_drawdown_info(equity, account_balance)
        
        # Calcular Calmar Ratio adaptado (si es menor a un año, toma Ret/DD, sino anualizado)
        calmar = np.nan
        if mdd > 0 and len(g) > 1 and pd.api.types.is_datetime64_any_dtype(g[tcol]):
            days = (g[tcol].max() - g[tcol].min()).days
            if days < 365.25:
                calmar = net_real / mdd
            elif days >= 1:
                ann_ret = net_real / (days / 365.25)
                calmar = ann_ret / mdd
                
        stability = stability_r2(equity)
        exp_val = expectancy(pnl_real)
        
        row_data = {
            "Robot ID": rid, 
            "Net profit (real)": net_real,
            "Stability": stability,
            "Calmar": calmar,
            "Expectancy": exp_val,
        }
        
        if risk_per_trade is not None and risk_per_trade > 0:
            row_data["Exp (% R)"] = exp_val / risk_per_trade
        
        row_data.update({
            "# Trades": int((~pnl_real.isna()).sum()),
            "% Wins": float((pnl_real > 0).mean() * 100.0) if len(pnl_real) else 0.0,
            "PF": profit_factor(pnl_real),
            "Ret/DD": ret_dd, 
            "Max DD": mdd,
            "DD Actual": dd_info["current_dd"],
            "Trades s/Peak": dd_info["trades_since_peak"],
            "En Peak": dd_info["is_at_peak"],
            "Desde": g[tcol].min(), "Hasta": g[tcol].max(),
            "Símbolos": ", ".join(sorted(set(g["symbol"].dropna().astype(str)))) if "symbol" in g else "",
        })
        if account_balance is not None and account_balance > 0:
            row_data["DD % Cuenta"] = dd_info["dd_pct_account"]
            row_data["Max DD % Cuenta"] = dd_info["max_dd_pct_account"]
        rows.append(row_data)
    out = pd.DataFrame(rows).sort_values("Net profit (real)", ascending=False)
    return out, key, tcol

def summary_kpis_robot(g: pd.DataFrame, tcol: str, account_balance: float | None = None, risk_per_trade: float | None = None) -> pd.DataFrame:
    pnl_real = g["real_profit"].fillna(0.0)
    equity = pnl_real.cumsum()
    wins, losses = int((pnl_real > 0).sum()), int((pnl_real < 0).sum())
    mdd = max_drawdown(equity)
    net_bruto, comm_total, swap_total = float(g["profit"].sum()), float(g["commission"].sum()), float(g["swap"].sum())
    net_real = float(pnl_real.sum())
    ret_dd = float(net_real / mdd) if mdd > 0 else np.nan
    dd_info = current_drawdown_info(equity, account_balance)
    
    calmar = np.nan
    if mdd > 0 and len(g) > 1 and pd.api.types.is_datetime64_any_dtype(g[tcol]):
        days = (g[tcol].max() - g[tcol].min()).days
        if days < 365.25:
            calmar = net_real / mdd
        elif days >= 1:
            ann_ret = net_real / (days / 365.25)
            calmar = ann_ret / mdd
            
    data = {
        "Robot ID": [g["robot_id"].iloc[0] if "robot_id" in g.columns else ""],
        "trades": [int(len(pnl_real))],
        "net profit (bruto)": [net_bruto], "commissions total": [comm_total],
        "swaps total": [swap_total], "net profit (real)": [net_real],
        "max dd ($)": [mdd], "dd actual ($)": [dd_info["current_dd"]],
        "trades s/peak": [dd_info["trades_since_peak"]],
    }
    if account_balance is not None and account_balance > 0:
        data["dd actual (% cuenta)"] = [dd_info["dd_pct_account"]]
        data["max dd (% cuenta)"] = [dd_info["max_dd_pct_account"]]
        
    data.update({
        "ret/dd": [ret_dd], "calmar ratio": [calmar],
        "winrate": [float((pnl_real > 0).mean() * 100.0) if len(pnl_real) else 0.0],
        "wins": [wins], "loss": [losses],
        "profit factor": [profit_factor(pnl_real)], "sharpe ratio": [sharpe_per_trade(pnl_real)],
        "expectancy": [expectancy(pnl_real)],
    })
    
    if risk_per_trade is not None and risk_per_trade > 0:
        data["exp (% r)"] = [expectancy(pnl_real) / risk_per_trade]
        
    data.update({
        "stability": [stability_r2(equity)],
        "stagnation": [max_stagnation(g[tcol] if tcol in g.columns else None, equity)],
    })
    return pd.DataFrame(data)

# ========= DD Render helpers =========
def dd_severity_label(dd_pct_account: float | None, is_at_peak: bool) -> str:
    if is_at_peak: return "🟢 En peak"
    if dd_pct_account is None: return "⚪ Sin balance"
    if dd_pct_account < 2: return "🟢 Leve"
    if dd_pct_account < 5: return "🟡 Moderado"
    if dd_pct_account < 10: return "🟠 Elevado"
    return "🔴 Severo"

def render_current_dd_panel(equity: pd.Series, account_balance: float | None = None, tcol_series: pd.Series | None = None):
    dd_info = current_drawdown_info(equity, account_balance)
    has_balance = account_balance is not None and account_balance > 0

    # Badge color
    if dd_info["is_at_peak"]:
        badge_color, badge_text, border_color = "#00d4aa", "EN PEAK ✓", "#00d4aa"
    elif has_balance:
        pct = dd_info["dd_pct_account"]
        if pct < 2: badge_color, badge_text, border_color = "#00d4aa", "DD LEVE", "#00d4aa"
        elif pct < 5: badge_color, badge_text, border_color = "#f0c040", "DD MODERADO", "#f0c040"
        elif pct < 10: badge_color, badge_text, border_color = "#f0a040", "DD ELEVADO", "#f0a040"
        else: badge_color, badge_text, border_color = "#ff4d6d", "DD SEVERO", "#ff4d6d"
    else:
        ratio = dd_info["dd_ratio"]
        if ratio < 0.3: badge_color, badge_text, border_color = "#00d4aa", "DD LEVE", "#00d4aa"
        elif ratio < 0.5: badge_color, badge_text, border_color = "#f0c040", "DD MODERADO", "#f0c040"
        elif ratio < 0.8: badge_color, badge_text, border_color = "#f0a040", "DD ELEVADO", "#f0a040"
        else: badge_color, badge_text, border_color = "#ff4d6d", "DD SEVERO", "#ff4d6d"

    st.markdown(
        f"<div style='background:{border_color}15; border-left:4px solid {border_color}; "
        f"padding:12px 16px; border-radius:8px; margin-bottom:12px;'>"
        f"<span style='font-size:0.75rem; color:#888; text-transform:uppercase; letter-spacing:1px;'>Drawdown en curso</span>"
        f"<span style='float:right; background:{badge_color}25; color:{badge_color}; "
        f"padding:2px 10px; border-radius:12px; font-size:0.75rem; font-weight:700;'>{badge_text}</span></div>",
        unsafe_allow_html=True)

    if has_balance:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("DD Actual ($)", f"${dd_info['current_dd']:.2f}")
        c2.metric("DD % Cuenta", f"{dd_info['dd_pct_account']:.2f}%")
        c3.metric("Max DD (% cuenta)", f"{dd_info['max_dd_pct_account']:.2f}%")
        c4.metric("Trades desde peak", f"{dd_info['trades_since_peak']}")

        if not dd_info["is_at_peak"]:
            pct_val = dd_info["dd_pct_account"]
            bar_max = max(pct_val * 1.5, 15.0)
            fill = min((pct_val / bar_max) * 100, 100)
            bc = "#00d4aa" if pct_val < 2 else "#f0c040" if pct_val < 5 else "#f0a040" if pct_val < 10 else "#ff4d6d"
            st.markdown(
                f"<div style='margin:4px 0 8px 0;'>"
                f"<div style='display:flex; justify-content:space-between; font-size:0.7rem; color:#888; margin-bottom:2px;'>"
                f"<span>0%</span><span>DD actual: <b style=\"color:{bc}\">{pct_val:.2f}%</b> de ${account_balance:,.0f}</span>"
                f"<span>{bar_max:.0f}%</span></div>"
                f"<div style='background:#1a1d2e; border-radius:6px; height:14px; overflow:hidden; position:relative;'>"
                f"<div style='background:{bc}; width:{fill:.1f}%; height:100%; border-radius:6px;'></div>"
                f"<div style='position:absolute;top:0;left:{min(2/bar_max*100,100):.1f}%;width:1px;height:100%;background:#00d4aa50'></div>"
                f"<div style='position:absolute;top:0;left:{min(5/bar_max*100,100):.1f}%;width:1px;height:100%;background:#f0c04080'></div>"
                f"<div style='position:absolute;top:0;left:{min(10/bar_max*100,100):.1f}%;width:1px;height:100%;background:#ff4d6d80'></div>"
                f"</div></div>", unsafe_allow_html=True)
    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("DD Actual ($)", f"${dd_info['current_dd']:.2f}")
        c2.metric("% del Max DD hist.", f"{dd_info['dd_ratio']*100:.1f}%")
        c3.metric("Max DD ($)", f"${dd_info['max_dd']:.2f}")
        c4.metric("Trades desde peak", f"{dd_info['trades_since_peak']}")
        st.caption("💡 Ingresá el balance de tu cuenta en la barra lateral para ver el DD como % de la cuenta.")

    if len(equity) > 1:
        e = equity.values.astype(float)
        peaks = np.maximum.accumulate(e)
        uw_df = pd.DataFrame({"Underwater ($)": e - peaks}, index=range(1, len(e) + 1))
        st.area_chart(uw_df, height=160, color="#ff4d6d")
        st.caption("Underwater equity: distancia al peak acumulado en cada punto.")

# ========= EDGE ANALYTICS =========
def compute_edge_score(expectancy_current: float, baseline: float) -> int:
    if baseline == 0:
        if expectancy_current > 0: return 3
        elif expectancy_current < 0: return -3
        return 0
    ratio = expectancy_current / abs(baseline)
    if expectancy_current < 0: return -3
    elif ratio >= 1.3: return 3
    elif ratio >= 1.1: return 2
    elif ratio >= 0.9: return 1
    elif ratio >= 0.7: return 0
    elif ratio >= 0.5: return -1
    elif ratio >= 0.2: return -2
    else: return -3

def edge_score_label(score: int) -> str:
    return {3:"🟢 Edge muy por encima del baseline", 2:"🟢 Edge por encima del baseline",
            1:"🟢 Edge estable", 0:"🟡 Edge levemente por debajo — monitorear",
            -1:"🟠 Edge en decay moderado", -2:"🔴 Edge en decay severo", -3:"🔴 Edge negativo"}.get(score, "❓ Sin datos")

def momentum_score(blocks: list[dict]) -> int | None:
    if len(blocks) < 2: return None
    recent = blocks[-3:] if len(blocks) >= 3 else blocks
    deltas = []
    for i in range(1, len(recent)):
        prev_e = recent[i-1]["expectancy"]
        if prev_e != 0: deltas.append(((recent[i]["expectancy"] - prev_e) / abs(prev_e)) * 100)
    if not deltas: return None
    avg = sum(deltas) / len(deltas)
    if avg >= 15: return 3
    elif avg >= 5: return 2
    elif avg >= -5: return 1
    elif avg >= -15: return 0
    elif avg >= -30: return -1
    elif avg >= -50: return -2
    else: return -3

def momentum_label(score: int | None) -> str:
    if score is None: return "❓ Sin datos suficientes"
    return {3:"🟢 Acelerando fuerte",2:"🟢 Mejorando",1:"🟢 Estable",0:"🟡 Desacelerando levemente",
            -1:"🟠 Decayendo",-2:"🔴 Decayendo rápido",-3:"🔴 Caída pronunciada"}.get(score,"❓")

def confluence_signal(vs_baseline, mom):
    if vs_baseline is None or mom is None: return "❓ Datos insuficientes", "#888"
    if vs_baseline >= 2 and mom >= 2: return "🌟 Excelente — edge sólido y en crecimiento", "#00d4aa"
    elif vs_baseline >= 1 and mom >= 1: return "✅ Positivo — edge sobre baseline y mejorando", "#00d4aa"
    elif vs_baseline <= -2 and mom <= -2: return "🚨 Atención — decay confirmado en ambas métricas", "#ff4d6d"
    elif vs_baseline <= -1 and mom <= -1: return "⚠️ Precaución — señales de debilitamiento", "#f0a040"
    elif (vs_baseline >= 1 and mom <= -1) or (vs_baseline <= -1 and mom >= 1): return "🔍 Investigar — señales contradictorias", "#f0c040"
    elif vs_baseline >= 1: return "🟡 Sobre baseline, momentum neutral", "#f0c040"
    elif mom >= 1: return "🟡 Mejorando, aún bajo baseline", "#f0c040"
    else: return "🟡 Neutral — monitorear evolución", "#f0c040"

def compute_rolling_expectancy(pnl, window): return pnl.rolling(window=window, min_periods=window).mean()

def compute_period_blocks(pnl, block_size):
    blocks, n = [], len(pnl)
    num_blocks = n // block_size
    if num_blocks == 0: return blocks
    for i in range(num_blocks):
        s, e = i * block_size, (i+1) * block_size
        chunk = pnl.iloc[s:e]
        blocks.append({"bloque":i+1,"desde_trade":s+1,"hasta_trade":e,"trades":block_size,
                        "expectancy":float(chunk.mean()),"winrate":float((chunk>0).mean()*100),"profit_factor":profit_factor(chunk)})
    rem = n % block_size
    if rem >= block_size // 2:
        chunk = pnl.iloc[num_blocks*block_size:]
        blocks.append({"bloque":num_blocks+1,"desde_trade":num_blocks*block_size+1,"hasta_trade":n,
                        "trades":rem,"expectancy":float(chunk.mean()),"winrate":float((chunk>0).mean()*100),"profit_factor":profit_factor(chunk)})
    return blocks

def render_edge_comparison_table(df_pos, tcol, robots, cp_size, mp_size, lp_size, baseline_trades, baseline_manual, account_balance=None):
    col_b_cp, col_b_mp, col_b_lp = f"B·CP ({cp_size})", f"B·MP ({mp_size})", f"B·LP ({lp_size})"
    col_m_cp, col_m_mp, col_m_lp = f"M·CP ({cp_size})", f"M·MP ({mp_size})", f"M·LP ({lp_size})"
    score_cols = [col_b_cp, col_b_mp, col_b_lp, col_m_cp, col_m_mp, col_m_lp]
    has_bal = account_balance is not None and account_balance > 0
    summary_rows = []
    for rid in robots:
        g = df_pos[df_pos["robot_id"].astype(str)==rid].copy()
        g = g.sort_values(tcol) if tcol in g.columns else g
        pnl = g["real_profit"].fillna(0.0).reset_index(drop=True)
        n = len(pnl)
        if n < cp_size: continue
        b_exp = float(baseline_manual) if baseline_manual is not None else float(pnl.iloc[:min(baseline_trades,n)].mean())
        def last_score(size):
            bl = compute_period_blocks(pnl, size)
            return (compute_edge_score(bl[-1]["expectancy"], b_exp), bl) if bl else (None, [])
        cp_s, cp_bl = last_score(cp_size); mp_s, mp_bl = last_score(mp_size); lp_s, lp_bl = last_score(lp_size)
        cp_m, mp_m, lp_m = momentum_score(cp_bl), momentum_score(mp_bl), momentum_score(lp_bl)
        ref_b = lp_s if lp_s is not None else (mp_s if mp_s is not None else cp_s)
        ref_m = lp_m if lp_m is not None else (mp_m if mp_m is not None else cp_m)
        conf_text, _ = confluence_signal(ref_b, ref_m)
        conf_short = conf_text.split("—")[-1].strip() if "—" in conf_text else conf_text.split(" ",1)[-1]
        equity = pnl.cumsum()
        dd_info = current_drawdown_info(equity, account_balance)
        dd_pct = dd_info["dd_pct_account"] if has_bal else None
        dd_sev = dd_severity_label(dd_pct, dd_info["is_at_peak"])
        nv = lambda s: s if s is not None else pd.NA
        row = {"Robot":rid,"Trades":n,col_b_cp:nv(cp_s),col_b_mp:nv(mp_s),col_b_lp:nv(lp_s),
               col_m_cp:nv(cp_m),col_m_mp:nv(mp_m),col_m_lp:nv(lp_m)}
        if has_bal: row["DD%Cta"] = dd_pct if dd_pct is not None else 0.0
        row.update({"DD Estado":dd_sev,"Confluencia":conf_short,"Exp.Base":round(b_exp,3)})
        summary_rows.append(row)
    if not summary_rows:
        st.info(f"Se necesitan al menos {cp_size} trades por robot."); return
    df_s = pd.DataFrame(summary_rows)
    for c in score_cols:
        if c in df_s.columns: df_s[c] = pd.array(df_s[c], dtype="Int64")
    def cs(val):
        try:
            v=int(val)
            if v>=2: return "color:#00d4aa;font-weight:bold"
            if v==1: return "color:#00d4aa"
            if v==0: return "color:#f0c040"
            if v==-1: return "color:#f0a040"
            return "color:#ff4d6d;font-weight:bold"
        except: return "color:#888"
    def cc(val):
        v=str(val).lower()
        if any(k in v for k in ["excelente","positivo","sólido"]): return "color:#00d4aa;font-weight:bold"
        if any(k in v for k in ["atención","decay","precaución"]): return "color:#ff4d6d;font-weight:bold"
        return ""
    def cdd(val):
        try:
            v=float(val)
            if v<0.01: return "color:#00d4aa;font-weight:bold"
            if v<2: return "color:#00d4aa"
            if v<5: return "color:#f0c040"
            if v<10: return "color:#f0a040;font-weight:bold"
            return "color:#ff4d6d;font-weight:bold"
        except: return ""
    def cde(val):
        v=str(val).lower()
        if "peak" in v: return "color:#00d4aa;font-weight:bold"
        if "leve" in v: return "color:#00d4aa"
        if "moderado" in v: return "color:#f0c040"
        if "elevado" in v: return "color:#f0a040;font-weight:bold"
        if "severo" in v: return "color:#ff4d6d;font-weight:bold"
        return ""
    ps = [c for c in score_cols if c in df_s.columns]
    fm = {c:(lambda v: f"+{int(v)}" if int(v)>0 else str(int(v)) if pd.notna(v) else "—") for c in ps}
    if has_bal and "DD%Cta" in df_s.columns: fm["DD%Cta"] = "{:.2f}%"
    sty = df_s.style.map(cs,subset=ps).map(cc,subset=["Confluencia"]).map(cde,subset=["DD Estado"]).format(fm,na_rep="—")
    if has_bal and "DD%Cta" in df_s.columns: sty = sty.map(cdd,subset=["DD%Cta"])
    co = ["Robot","Trades"]+ps
    if has_bal and "DD%Cta" in df_s.columns: co.append("DD%Cta")
    co += ["DD Estado","Confluencia","Exp.Base"]
    st.dataframe(sty, use_container_width=True, hide_index=True, column_order=co)

def render_edge_tab(df_pos, tcol, account_balance=None, risk_per_trade=None):
    st.markdown("""<style>
    .edge-card{background:#0f1117;border:1px solid #2a2d3e;border-radius:12px;padding:16px 20px;margin-bottom:12px}
    </style>""", unsafe_allow_html=True)
    st.subheader("🎯 Edge Analytics — Detección y Medición de Vida del Edge")
    st.caption("Monitorea si el edge de cada sistema está en crecimiento, estable o en decaimiento.")
    robots = sorted(df_pos["robot_id"].astype(str).unique())
    if not robots: st.warning("No hay robots disponibles."); return
    st.markdown("#### 🤖 Seleccionar Robot")
    robot_options = {f"🤖 {rid}  ({len(df_pos[df_pos['robot_id'].astype(str)==rid])} trades)": rid for rid in robots}
    selected_rid = robot_options[st.selectbox("Robot a analizar:", list(robot_options.keys()), key="edge_robot_selector")]
    st.markdown("---")
    st.markdown("#### ⚙️ Configuración de períodos")
    c1,c2,c3 = st.columns(3)
    cp_size = c1.number_input("Corto Plazo (trades)", min_value=5, max_value=50, value=10, step=5, key="edge_cp_size")
    mp_size = c2.number_input("Medio Plazo (trades)", min_value=10, max_value=100, value=20, step=5, key="edge_mp_size")
    lp_size = c3.number_input("Largo Plazo (trades)", min_value=20, max_value=200, value=50, step=10, key="edge_lp_size")
    st.markdown("#### 📐 Baseline de Expectancy")
    
    baseline_mode = st.radio("Fuente del baseline:",
        options=["📊 Automático (primeros N trades)", "✏️ Manual (desde backtesting)", "🎯 % del Riesgo (Normalizado)"],
        horizontal=True, key="edge_baseline_mode",
        index=2)
        
    if baseline_mode.startswith("📊"):
        bc1, bc2 = st.columns([1, 3])
        baseline_trades = bc1.number_input("Primeros N trades", min_value=10, max_value=200, value=20, step=10, key="edge_baseline_trades")
        baseline_manual = None
        _g = df_pos[df_pos["robot_id"].astype(str)==selected_rid]
        _pnl = _g["real_profit"].fillna(0.0).reset_index(drop=True)
        if len(_pnl) > 0:
            _bn = min(baseline_trades, len(_pnl))
            _bc = float(_pnl.iloc[:_bn].mean())
            _co = "#00d4aa" if _bc > 0 else "#f87171"
            bc2.markdown(f"<div style='background:{_co}18;border:1px solid {_co};border-radius:6px;padding:8px 14px;margin-top:4px'>"
                         f"<span style='font-size:.75rem;color:#8b949e'>Baseline ({_bn} de {len(_pnl)} trades)</span><br>"
                         f"<span style='font-size:1.3rem;font-weight:900;color:{_co}'>${_bc:.2f}</span>"
                         f"<span style='font-size:.8rem;color:#8b949e'> / trade</span></div>", unsafe_allow_html=True)
    elif baseline_mode.startswith("✏️"):
        bc1, bc2 = st.columns([1, 3])
        baseline_manual = bc1.number_input("Expectancy baseline ($/trade)", min_value=-1000.0, max_value=10000.0, value=0.0, step=0.01, format="%.2f", key="baseline_manual_input")
        baseline_trades = 20
        bc2.info(f"📌 Baseline manual: **${baseline_manual:.2f}** / trade")
    else: # 🎯 % del Riesgo
        bc1, bc2 = st.columns([1, 3])
        if risk_per_trade is None or risk_per_trade <= 0:
            st.warning("⚠️ Debes configurar el 'Riesgo por Trade' en la barra lateral para usar esta opción.")
            baseline_manual = 0.0
            baseline_trades = 20
        else:
            target_pct = bc1.number_input("Expectancy Objetivo (% del Riesgo)", min_value=0.0, max_value=500.0, value=9.0, step=1.0, key="edge_target_pct")
            baseline_manual = risk_per_trade * (target_pct / 100.0)
            baseline_trades = 20
            
            _co = "#00d4aa" if target_pct >= 20.0 else ("#f0c040" if target_pct >= 9.0 else "#f87171")
            bc2.markdown(f"<div style='background:{_co}18;border:1px solid {_co};border-radius:6px;padding:8px 14px;margin-top:4px'>"
                         f"<span style='font-size:.75rem;color:#8b949e'>Baseline Normalizado ({target_pct}% de ${risk_per_trade})</span><br>"
                         f"<span style='font-size:1.3rem;font-weight:900;color:{_co}'>${baseline_manual:.2f}</span>"
                         f"<span style='font-size:.8rem;color:#8b949e'> / trade</span></div>", unsafe_allow_html=True)
            
    st.markdown("---")

    rid = selected_rid
    g = df_pos[df_pos["robot_id"].astype(str)==rid].copy()
    g = g.sort_values(tcol) if tcol in g.columns else g
    pnl = g["real_profit"].fillna(0.0).reset_index(drop=True)
    n_trades = len(pnl)
    st.markdown(f"### 🤖 {rid} — {n_trades} trades totales")
    if n_trades < cp_size:
        st.info(f"⚠️ Mínimo {cp_size} trades necesarios. Actuales: {n_trades}")
    else:
        equity_for_dd = pnl.cumsum()
        baseline_exp = float(baseline_manual) if baseline_manual is not None else float(pnl.iloc[:min(baseline_trades,n_trades)].mean())
        if baseline_mode.startswith("🎯"): baseline_desc = f"Riesgo Normalizado — ${baseline_exp:.2f}/trade"
        elif baseline_mode.startswith("✏️"): baseline_desc = f"Manual — ${baseline_exp:.2f}/trade"
        else: baseline_desc = f"Auto ({min(baseline_trades,n_trades)} trades) — ${baseline_exp:.2f}/trade"
        
        def get_last_blocks(size, mx=3):
            bl = compute_period_blocks(pnl, size)
            return bl[-mx:] if bl else []
        cp_bl, mp_bl, lp_bl = get_last_blocks(cp_size), get_last_blocks(mp_size), get_last_blocks(lp_size)
        bts = lambda bl: [compute_edge_score(b["expectancy"], baseline_exp) for b in bl]
        cp_sc, mp_sc, lp_sc = bts(cp_bl), bts(mp_bl), bts(lp_bl)
        cp_now, mp_now, lp_now = (cp_sc[-1] if cp_sc else None), (mp_sc[-1] if mp_sc else None), (lp_sc[-1] if lp_sc else None)
        cp_mom, mp_mom, lp_mom = momentum_score(cp_bl), momentum_score(mp_bl), momentum_score(lp_bl)
        ref_b = lp_now if lp_now is not None else (mp_now if mp_now is not None else cp_now)
        ref_m = lp_mom if lp_mom is not None else (mp_mom if mp_mom is not None else cp_mom)
        conf_text, conf_color = confluence_signal(ref_b, ref_m)
        st.markdown(f"<div style='background:{conf_color}22;border-left:4px solid {conf_color};padding:12px 16px;border-radius:8px;margin-bottom:16px;'>"
                    f"<span style='font-size:1.05rem;font-weight:600;color:{conf_color};'>{conf_text}</span>"
                    f"<span style='color:#888;font-size:0.8rem;margin-left:12px;'>Señal de confluencia</span></div>", unsafe_allow_html=True)
        st.markdown("##### 📊 Score por horizonte")
        fs = lambda s: ("—" if s is None else (f"+{s}" if s > 0 else str(s)))
        score_data = {
            "Horizonte":[f"Corto ({cp_size})",f"Medio ({mp_size})",f"Largo ({lp_size})"],
            "vs Baseline":[fs(cp_now),fs(mp_now),fs(lp_now)],
            "Estado Baseline":[edge_score_label(x).split(" ",1)[-1] if x is not None else "—" for x in [cp_now,mp_now,lp_now]],
            "Momentum":[fs(cp_mom),fs(mp_mom),fs(lp_mom)],
            "Estado Momentum":[momentum_label(x).split(" ",1)[-1] if x is not None else "—" for x in [cp_mom,mp_mom,lp_mom]],
        }
        df_sc = pd.DataFrame(score_data)
        def csc(val):
            try:
                v=int(str(val).replace("+",""))
                if v>=2: return "color:#00d4aa;font-weight:bold"
                if v==1: return "color:#00d4aa"
                if v==0: return "color:#f0c040"
                if v==-1: return "color:#f0a040"
                return "color:#ff4d6d;font-weight:bold"
            except: return "color:#888"
        st.dataframe(df_sc.style.map(csc,subset=["vs Baseline","Momentum"]), use_container_width=True, hide_index=True)
        st.caption(f"Baseline: {baseline_desc}")
        st.markdown("---")
        st.markdown("##### 📋 Historial de bloques por período")
        tab_cp, tab_mp, tab_lp = st.tabs([f"CP ({cp_size})", f"MP ({mp_size})", f"LP ({lp_size})"])
        def render_blocks_table(tab, blocks, scores, bexp, rdp=-20.0):
            with tab:
                if not blocks: st.info("Insuficientes trades."); return
                rows, prev_s, prev_e, alerts = [], None, None, []
                for i,(b,s) in enumerate(zip(blocks,scores)):
                    ds = "—" if prev_s is None else f"{'↑' if s-prev_s>0 else '↓' if s-prev_s<0 else '→'} {s-prev_s:+d}"
                    if prev_e is None or prev_e==0: dp="—"
                    else:
                        pct=((b["expectancy"]-prev_e)/abs(prev_e))*100
                        dp=f"{'📈' if pct>0 else '📉'} {pct:+.1f}%"
                        if pct<=rdp: alerts.append(f"⚡ Decay rápido bloque #{b['bloque']}: {pct:.1f}%")
                    vb = f"{((b['expectancy']-bexp)/abs(bexp))*100:+.1f}%" if bexp!=0 else "—"
                    rows.append({"Bloque":f"#{b['bloque']}","Trades":f"{b['desde_trade']}–{b['hasta_trade']}","N":b["trades"],
                                 "Expectancy":round(b["expectancy"],3),"Δ% Exp":dp,"vs Baseline":vb,
                                 "Win Rate":f"{b['winrate']:.1f}%","PF":round(b["profit_factor"],2),
                                 "Score":f"{'+' if s>0 else ''}{s}","Δ Score":ds,"Estado":edge_score_label(s)})
                    prev_s, prev_e = s, b["expectancy"]
                dfb = pd.DataFrame(rows)
                def cs2(v):
                    try:
                        x=int(str(v).replace("+",""))
                        return "color:#00d4aa;font-weight:bold" if x>=1 else ("color:#ff4d6d;font-weight:bold" if x<=-1 else "color:#f0c040;font-weight:bold")
                    except: return ""
                def cdp(v):
                    if v=="—" or not isinstance(v,str): return ""
                    try:
                        n=float(v.replace("📈","").replace("📉","").replace("%","").strip())
                        return "color:#00d4aa" if n>=0 else ("color:#ff4d6d;font-weight:bold" if n<=rdp else "color:#f0a040")
                    except: return ""
                def cvb(v):
                    if v=="—": return ""
                    try:
                        n=float(str(v).replace("%","").strip())
                        return "color:#00d4aa" if n>=10 else ("color:#f0c040" if n>=-10 else "color:#ff4d6d")
                    except: return ""
                st.dataframe(dfb.style.map(cs2,subset=["Score"]).map(cdp,subset=["Δ% Exp"]).map(cvb,subset=["vs Baseline"]),
                             use_container_width=True, hide_index=True)
                for a in alerts: st.error(a)
                if len(blocks)>1:
                    cd=pd.DataFrame({"Bloque":[f"#{b['bloque']}" for b in blocks],"Expectancy":[b["expectancy"] for b in blocks],
                                     "Baseline":[bexp]*len(blocks)}).set_index("Bloque")
                    st.line_chart(cd, height=180); st.caption("Expectancy por bloque vs Baseline")
        render_blocks_table(tab_cp, cp_bl, cp_sc, baseline_exp)
        render_blocks_table(tab_mp, mp_bl, mp_sc, baseline_exp)
        render_blocks_table(tab_lp, lp_bl, lp_sc, baseline_exp)
        st.markdown("---")
        st.markdown("##### 📈 Expectancy rolling continua")
        roll_w = st.slider("Ventana rolling", min_value=5, max_value=min(100,n_trades), value=min(mp_size,n_trades), key="edge_roll_window")
        roll_exp = compute_rolling_expectancy(pnl, roll_w).dropna()
        if not roll_exp.empty:
            st.line_chart(pd.DataFrame({"Rolling":roll_exp.values,"Baseline":[baseline_exp]*len(roll_exp),"Cero":[0.0]*len(roll_exp)},
                                       index=range(1,len(roll_exp)+1)), height=220)
            st.caption(f"Rolling {roll_w} trades. Azul bajo naranja → decay.")
        else: st.info(f"Se necesitan {roll_w}+ trades.")
        st.markdown("##### 🔔 Estado del sistema")
        good, watch, alert = [], [], []
        dd_a = current_drawdown_info(equity_for_dd, account_balance)
        hb = account_balance is not None and account_balance > 0
        if dd_a["is_at_peak"]: good.append("🏔️ **Equity en máximo histórico**")
        elif hb:
            pa = dd_a["dd_pct_account"]
            if pa >= 10: alert.append(f"🚨 **DD severo**: ${dd_a['current_dd']:.2f} = **{pa:.2f}%** de la cuenta")
            elif pa >= 5: alert.append(f"⚠️ **DD elevado**: {pa:.2f}% de la cuenta")
            elif pa >= 2: watch.append(f"📉 **DD moderado**: {pa:.2f}% de la cuenta")
        else:
            r = dd_a["dd_ratio"]
            if r >= 0.8: alert.append(f"🚨 **DD severo**: {r*100:.0f}% del Max DD")
            elif r >= 0.5: watch.append(f"⚠️ **DD elevado**: {r*100:.0f}% del Max DD")
        if lp_now is not None and lp_now >= 2: good.append(f"🌟 **LP sobre baseline** (score {fs(lp_now)})")
        if lp_mom is not None and lp_mom >= 2: good.append("📈 **Momentum positivo LP**")
        if ref_b is not None and ref_b >= 2 and ref_m is not None and ref_m >= 2: good.append("🌟 **Confluencia positiva**")
        if cp_now is not None and cp_now <= -1 and (mp_now is None or mp_now >= 0): watch.append(f"🔍 **CP bajo baseline** ({fs(cp_now)})")
        if mp_mom is not None and mp_mom <= -1 and (lp_mom is None or lp_mom >= 0): watch.append("🔍 **Momentum MP desacelerando**")
        if mp_now is not None and mp_now <= -2: alert.append(f"⚠️ **MP decay severo** ({fs(mp_now)})")
        if lp_now is not None and lp_now <= -2: alert.append(f"🚨 **LP decay severo** ({fs(lp_now)})")
        if ref_b is not None and ref_b <= -1 and ref_m is not None and ref_m <= -1: alert.append("🚨 **Confluencia negativa**")
        def chk(bl,nm):
            r=[]
            for i in range(1,len(bl)):
                pe=bl[i-1]["expectancy"]
                if pe!=0:
                    p=((bl[i]["expectancy"]-pe)/abs(pe))*100
                    if p<=-20: r.append(f"⚡ **Caída en {nm} #{bl[i]['bloque']}**: {p:.1f}%")
            return r
        for r in chk(mp_bl,"MP"): alert.append(r)
        for r in chk(lp_bl,"LP"): alert.append(r)
        for s in good: st.success(s)
        for s in watch: st.warning(s)
        for s in alert: st.error(s)
        if not good and not watch and not alert: st.info("ℹ️ Sistema dentro de parámetros normales")
    st.markdown("---")
    st.markdown("#### 🏆 Comparativa de Edge — Todos los Robots")
    render_edge_comparison_table(df_pos, tcol, robots, cp_size, mp_size, lp_size, baseline_trades, baseline_manual, account_balance)
    with st.expander("📖 Leyenda"):
        st.markdown("""
### Scores
| Score | vs Baseline | Momentum |
|-------|-------------|----------|
| **+3** | ≥130% baseline | Acelerando ≥+15% |
| **+2** | ≥110% | Mejorando +5..15% |
| **+1** | 90-110% | Estable ±5% |
| **0** | 70-90% | Desacelerando -5..-15% |
| **-1** | 50-70% | Decayendo -15..-30% |
| **-2** | <50% | Rápido -30..-50% |
| **-3** | Negativa | Pronunciada <-50% |

### DD en curso (% cuenta)
| DD% | Severidad |
|-----|-----------|
| 0% | 🟢 En peak |
| <2% | 🟢 Leve |
| 2-5% | 🟡 Moderado |
| 5-10% | 🟠 Elevado |
| ≥10% | 🔴 Severo |
        """)

# ==========================================
# ⚙️ SIDEBAR
# ==========================================
st.sidebar.header("📁 Carga de Datos")
uploaded_files = st.sidebar.file_uploader("📥 Subí tus archivos", type=["xlsx","csv","htm","html"], accept_multiple_files=True)
if not uploaded_files:
    st.info("👈 Arrastrá tus historiales en la barra lateral para comenzar.")
    st.stop()

st.sidebar.markdown("---")
st.sidebar.header("⚙️ Configuración")

account_balance = st.sidebar.number_input(
    "💰 Balance de la cuenta ($)", min_value=0.0, max_value=100_000_000.0, value=0.0, step=100.0, format="%.2f",
    help="Balance total de la cuenta. Se usa para calcular DD como % del capital. Dejá en 0 para no usarlo.")
if account_balance <= 0:
    account_balance = None
    st.sidebar.caption("ℹ️ Sin balance → DD en $ absolutos y % del Max DD histórico.")
else:
    st.sidebar.caption(f"✅ Balance: **${account_balance:,.2f}**")

risk_per_trade = st.sidebar.number_input(
    "💸 Riesgo por Trade ($)", min_value=0.0, max_value=1_000_000.0, value=0.0, step=10.0, format="%.2f",
    help="Riesgo en dinero por operación. Usado para normalizar la Expectancy (% del Riesgo) y para Edge Analytics.")

st.sidebar.markdown("---")
agrupar_por = st.sidebar.radio("Agrupar robots por:", ["Comment", "Magic Number"])
fusionar_archivos = st.sidebar.checkbox("🔗 Fusionar historiales", value=False)

def clean_id(x):
    if pd.isna(x) or str(x).strip() in ["", "nan"]: return "UNKNOWN"
    sx = str(x).strip()
    return sx[:-2] if sx.endswith(".0") else sx

all_dfs = []
with st.spinner("⏳ Procesando..."):
  for uploaded in uploaded_files:
    suffix = uploaded.name.lower().split(".")[-1]
    try:
        file_bytes = uploaded.read()
        if suffix == "xlsx": df_temp = parse_mt5_xlsx(file_bytes)
        elif suffix == "csv":
            if _is_deals_csv(file_bytes):
                df_temp = parse_mt5_deals_csv(file_bytes)
                st.sidebar.success(f"✅ Deals CSV: {uploaded.name}")
            else: df_temp = parse_csv(BytesIO(file_bytes))
        elif suffix in ("htm","html"): df_temp = parse_mt4_html(file_bytes)
        else: st.sidebar.error(f"No soportado: {uploaded.name}"); continue
        
        if df_temp.empty: st.sidebar.warning(f"Sin operaciones: {uploaded.name}"); continue
        
        # --- Lógica común unificada para asignar Robot ID ---
        if "Comment" not in df_temp.columns: df_temp["Comment"] = np.nan
        if "Magic" not in df_temp.columns: df_temp["Magic"] = np.nan

        # Limpiamos comentarios y magics vacíos
        c_col = df_temp["Comment"].apply(lambda x: str(x).strip() if pd.notna(x) and str(x).strip() not in ["", "nan"] else "")
        m_col = df_temp["Magic"].apply(clean_id)

        if agrupar_por == "Magic Number":
            # Formato: Comment (Magic Number)
            df_temp["robot_id"] = np.where(c_col != "", c_col + " (" + m_col + ")", m_col)
        else:
            df_temp["robot_id"] = np.where(c_col != "", c_col, "UNKNOWN")

        # Fallback si quedó UNKNOWN
        mask = (df_temp["robot_id"] == "UNKNOWN") | (df_temp["robot_id"] == "")
        if mask.any() and "symbol" in df_temp.columns:
            df_temp.loc[mask, "robot_id"] = df_temp.loc[mask, "symbol"].astype(str) + "_UNKNOWN"

        # Añadimos el nombre de archivo si no se fusionan
        if not fusionar_archivos:
            df_temp["robot_id"] = df_temp["robot_id"].astype(str) + f" [{uploaded.name}]"
            
        df_temp["source_file"] = uploaded.name
        all_dfs.append(df_temp)

    except Exception as e:
        st.sidebar.error(f"Error: {uploaded.name}")
        st.sidebar.exception(e)

if not all_dfs: st.stop()
df_pos_full = pd.concat(all_dfs, ignore_index=True)

st.sidebar.markdown("---")
st.sidebar.header("🔎 Filtros")
all_symbols = sorted(df_pos_full["symbol"].dropna().astype(str).unique())
selected_symbols = st.sidebar.multiselect("Símbolo(s)", all_symbols, default=all_symbols)
if not selected_symbols: st.warning("⚠️ Seleccioná al menos un símbolo."); st.stop()

st.sidebar.markdown("---")
st.sidebar.header("🤖 Filtros de Robots")
min_trades_filter = st.sidebar.number_input("Mínimo trades/robot", min_value=1, max_value=500, value=1, step=1)
tcol_full = "close_time" if ("close_time" in df_pos_full.columns and df_pos_full["close_time"].notna().any()) else "open_time"
all_robot_dates = df_pos_full.groupby("robot_id")[tcol_full].max().dropna()
gmin = all_robot_dates.min().date() if not all_robot_dates.empty else None
gmax = all_robot_dates.max().date() if not all_robot_dates.empty else None
last_activity_after = st.sidebar.date_input("Último trade después de", value=gmin, min_value=gmin, max_value=gmax) if gmin and gmax and gmin<gmax else None

df_pos = df_pos_full[df_pos_full["symbol"].isin(selected_symbols)].copy()
if df_pos.empty: st.warning("Sin operaciones para esos símbolos."); st.stop()

rtc = df_pos.groupby("robot_id").size()
rla = df_pos.groupby("robot_id")[tcol_full].max() if tcol_full in df_pos.columns else None
robots_to_keep = set(rtc[rtc >= min_trades_filter].index)
if last_activity_after is not None and rla is not None:
    robots_to_keep &= set(rla[rla >= pd.Timestamp(last_activity_after)].index)
if robots_to_keep: df_pos = df_pos[df_pos["robot_id"].isin(robots_to_keep)].copy()
else: st.warning("⚠️ Ningún robot cumple los filtros."); st.stop()
nf = len(df_pos_full["robot_id"].unique()) - len(robots_to_keep)
if nf > 0: st.sidebar.caption(f"🔍 {nf} robot(s) ocultados.")

# ==========================================
# 📈 MAIN
# ==========================================
kpis_df, key_used, tcol = kpis_por_robot(df_pos, account_balance, risk_per_trade)

st.markdown("""<script>
(function(){function w(){document.querySelectorAll('[data-baseweb="tab"]').forEach(function(t,i){
t.addEventListener('click',function(){var k=['kpis','robot','edge'];var u=new URL(window.location.href);
u.searchParams.set('tab',k[i]||'kpis');window.history.replaceState({},'',u.toString())})})};
setTimeout(w,800);new MutationObserver(function(){setTimeout(w,300)}).observe(document.body,{childList:true,subtree:true})})();
</script>""", unsafe_allow_html=True)

tab_kpis, tab_robot, tab_dd, tab_edge = st.tabs(["📊 KPIs Globales", "🔎 Análisis por Robot", "📉 Drawdown", "🎯 Edge Analytics"])

with tab_kpis:
    st.subheader(f"📊 KPIs por Robot (Agrupado por {agrupar_por})")
    has_bal = account_balance is not None and account_balance > 0
    kd = kpis_df.copy()
    kd["DD Estado"] = kd.apply(lambda r: dd_severity_label(r.get("DD % Cuenta") if has_bal else None, r["En Peak"]), axis=1)
    
    # 📌 ORDEN DE COLUMNAS PRIORIZADO PARA CUANTITATIVO
    dc = ["Robot ID", "Net profit (real)", "Stability", "Calmar", "Expectancy"]
    if risk_per_trade and risk_per_trade > 0:
        dc.append("Exp (% R)")
    dc += ["PF", "% Wins", "# Trades", "Max DD", "DD Actual"]
    if has_bal: dc += ["DD % Cuenta", "Max DD % Cuenta"]
    dc += ["DD Estado", "Ret/DD", "Trades s/Peak", "Desde", "Hasta", "Símbolos"]
    dc = [c for c in dc if c in kd.columns]
    
    # ======== Estilos y Alertas =======
    def color_stability(val):
        if pd.isna(val): return ""
        if val >= 0.7: return "color:#00d4aa;font-weight:bold"
        if val < 0.6: return "color:#ff4d6d;font-weight:bold"
        return "color:#f0c040"
        
    def color_calmar(val):
        if pd.isna(val): return ""
        if val >= 1.0: return "color:#00d4aa;font-weight:bold"
        if val < 0.8: return "color:#ff4d6d;font-weight:bold"
        return "color:#f0c040"
        
    def color_exp_r(val):
        if pd.isna(val): return ""
        if val >= 0.20: return "color:#00d4aa;font-weight:bold"
        if val < 0.09: return "color:#ff4d6d;font-weight:bold"
        return "color:#f0c040"
        
    def cde(v):
        v2=str(v).lower()
        if "peak" in v2: return "color:#00d4aa;font-weight:bold"
        if "leve" in v2: return "color:#00d4aa"
        if "moderado" in v2: return "color:#f0c040"
        if "elevado" in v2: return "color:#f0a040;font-weight:bold"
        if "severo" in v2: return "color:#ff4d6d;font-weight:bold"
        return ""
        
    def cdp(v):
        try:
            x=float(v)
            if x<0.01: return "color:#00d4aa;font-weight:bold"
            if x<2: return "color:#00d4aa"
            if x<5: return "color:#f0c040"
            if x<10: return "color:#f0a040;font-weight:bold"
            return "color:#ff4d6d;font-weight:bold"
        except: return ""

    fm={"Net profit (real)":"{:.2f}","% Wins":"{:.2f}%","PF":"{:.2f}","Expectancy":"{:.2f}",
        "Stability":"{:.2f}","Calmar":"{:.2f}","Ret/DD":"{:.2f}","Max DD":"{:.2f}","DD Actual":"{:.2f}"}
    if "Exp (% R)" in dc: fm["Exp (% R)"] = "{:.2%}"
    if has_bal: fm["DD % Cuenta"]="{:.2f}%"; fm["Max DD % Cuenta"]="{:.2f}%"
    
    sty = kd[dc].style.format(fm, na_rep="—").map(cde, subset=["DD Estado"])
    
    if has_bal and "DD % Cuenta" in dc: sty = sty.map(cdp, subset=["DD % Cuenta"])
    if "Stability" in dc: sty = sty.map(color_stability, subset=["Stability"])
    if "Calmar" in dc: sty = sty.map(color_calmar, subset=["Calmar"])
    if "Exp (% R)" in dc: sty = sty.map(color_exp_r, subset=["Exp (% R)"])
    
    st.dataframe(sty, use_container_width=True)
    st.markdown("---")
    st.markdown("#### 🎯 Edge — Comparativa rápida")
    st.caption("Baseline auto (20 trades). Para ajustar → pestaña **Edge Analytics**.")
    render_edge_comparison_table(df_pos, tcol, sorted(kpis_df["Robot ID"].astype(str).unique()),
                                 10, 20, 50, 20, None, account_balance)

with tab_robot:
    st.subheader("🔎 Curva de equity por trade")
    robots = sorted(kpis_df['Robot ID'].astype(str).unique())
    selected = st.selectbox("Robot:", robots, key="tab2_robot_selector")
    if selected:
        g = df_pos.copy(); g['robot_id'] = g['robot_id'].astype(str)
        sel = g[g['robot_id']==str(selected)].copy()
        oc = []
        if tcol in sel.columns: oc.append(tcol)
        if "close_time" in sel.columns and "close_time"!=tcol: oc.append("close_time")
        if "open_time" in sel.columns and "open_time"!=tcol: oc.append("open_time")
        if "position" in sel.columns: oc.append("position")
        sel = sel.sort_values(oc if oc else [tcol], kind="mergesort")
        sel["pnl_real"] = sel["real_profit"].fillna(0.0)
        sel["#"] = np.arange(1, len(sel)+1)
        sel["equity_trade"] = sel["pnl_real"].cumsum()
        st.line_chart(pd.Series(sel["equity_trade"].values, index=sel["#"]), height=280)
        st.caption(f"Equity acumulada ({', '.join(selected_symbols)})")
        st.markdown("### 📐 KPIs")
        sdf = summary_kpis_robot(sel, tcol, account_balance, risk_per_trade)
        fms={"net profit (bruto)":"{:.2f}","commissions total":"{:.2f}","swaps total":"{:.2f}","net profit (real)":"{:.2f}",
             "max dd ($)":"{:.2f}","dd actual ($)":"{:.2f}","ret/dd":"{:.2f}","winrate":"{:.2f}%",
             "profit factor":"{:.2f}","sharpe ratio":"{:.2f}","expectancy":"{:.2f}","stability":"{:.2f}",
             "calmar ratio":"{:.2f}"}
        if "exp (% r)" in sdf.columns: fms["exp (% r)"] = "{:.2%}"
        if account_balance and account_balance > 0: fms["dd actual (% cuenta)"]="{:.2f}%"; fms["max dd (% cuenta)"]="{:.2f}%"
        st.dataframe(sdf.style.format(fms, na_rep="—"), use_container_width=True)
        st.markdown("### 🧾 Historial de trades")
        with st.expander("Filtros", expanded=False):
            mn = pd.to_datetime(sel[tcol].min()) if tcol in sel.columns and not sel.empty else None
            mx = pd.to_datetime(sel[tcol].max()) if tcol in sel.columns and not sel.empty else None
            c1,c2,c3 = st.columns(3)
            fi = c1.date_input("Desde", value=mn.date(), min_value=mn.date(), max_value=mx.date()) if mn and mx else None
            ff = c2.date_input("Hasta", value=mx.date(), min_value=mn.date(), max_value=mx.date()) if mn and mx else None
            res = c3.selectbox("Resultado", ["Todos","Ganadores","Perdedores"])
        hist = sel.copy(); hist["pnl_real"] = hist["real_profit"].fillna(0.0)
        if fi and ff and tcol in hist.columns: hist = hist[(hist[tcol].dt.date>=fi)&(hist[tcol].dt.date<=ff)]
        if res=="Ganadores": hist = hist[hist["pnl_real"]>0]
        elif res=="Perdedores": hist = hist[hist["pnl_real"]<0]
        hist = hist.sort_values(oc if oc else [tcol], kind="mergesort")
        hist["equity_cum"] = hist["pnl_real"].cumsum(); hist["#"] = np.arange(1,len(hist)+1)
        cols = ["#"]
        if tcol in hist.columns: cols.append(tcol)
        ot = "open_time" if tcol=="close_time" else "close_time"
        if ot in hist.columns and ot!=tcol: cols.append(ot)
        for c in ["symbol","type","volume","open_price","close_price","commission","swap","profit","pnl_real","equity_cum","position","robot_id","Comment","Magic","source_file"]:
            if c in hist.columns: cols.append(c)
        cols = list(dict.fromkeys(cols))
        hv = hist[cols].copy()
        st.dataframe(hv.style.format({"volume":"{:.2f}","open_price":"{:.5f}","close_price":"{:.5f}","commission":"{:.2f}","swap":"{:.2f}","profit":"{:.2f}","pnl_real":"{:.2f}","equity_cum":"{:.2f}"}),
                     use_container_width=True, height=380)
        st.download_button("⬇️ CSV", hv.to_csv(index=False).encode("utf-8"),
                           f"historial_{str(selected).replace(' ','_')}.csv", "text/csv")

with tab_dd:
    st.subheader("📉 Drawdown — Panel Centralizado")
    st.caption("Vista unificada del drawdown en curso para todos los robots, más análisis detallado por robot.")

    # ── Overview table ──────────────────────────────────────────────────────
    st.markdown("#### 🗺️ Estado DD — Todos los Robots")
    has_bal_dd = account_balance is not None and account_balance > 0
    robots_dd = sorted(kpis_df["Robot ID"].astype(str).unique())
    dd_rows = []
    for rid_dd in robots_dd:
        g_dd = df_pos[df_pos["robot_id"].astype(str) == rid_dd].copy()
        oc_dd = []
        if tcol in g_dd.columns: oc_dd.append(tcol)
        g_dd = g_dd.sort_values(oc_dd if oc_dd else [tcol], kind="mergesort") if oc_dd else g_dd
        pnl_dd = g_dd["real_profit"].fillna(0.0)
        eq_dd = pnl_dd.cumsum()
        ddi = current_drawdown_info(pd.Series(eq_dd.values), account_balance)
        sev = dd_severity_label(ddi["dd_pct_account"] if has_bal_dd else None, ddi["is_at_peak"])
        row_dd = {
            "Robot": rid_dd,
            "Trades": len(pnl_dd),
            "Equity actual ($)": round(float(eq_dd.iloc[-1]) if len(eq_dd) else 0.0, 2),
            "Peak ($)": round(ddi["peak_value"], 2),
            "DD Actual ($)": round(ddi["current_dd"], 2),
            "Max DD ($)": round(ddi["max_dd"], 2),
            "Trades s/Peak": ddi["trades_since_peak"],
            "Severidad": sev,
        }
        if has_bal_dd:
            row_dd["DD % Cta"] = round(ddi["dd_pct_account"], 2) if ddi["dd_pct_account"] is not None else 0.0
            row_dd["Max DD % Cta"] = round(ddi["max_dd_pct_account"], 2) if ddi["max_dd_pct_account"] is not None else 0.0
        dd_rows.append(row_dd)

    if dd_rows:
        df_dd_ov = pd.DataFrame(dd_rows)
        def _sev_color(v):
            s = str(v).lower()
            if "peak" in s: return "color:#00d4aa;font-weight:bold"
            if "leve" in s: return "color:#00d4aa"
            if "moderado" in s: return "color:#f0c040"
            if "elevado" in s: return "color:#f0a040;font-weight:bold"
            if "severo" in s: return "color:#ff4d6d;font-weight:bold"
            return ""
        def _pct_color(v):
            try:
                x = float(v)
                if x < 0.01: return "color:#00d4aa;font-weight:bold"
                if x < 2: return "color:#00d4aa"
                if x < 5: return "color:#f0c040"
                if x < 10: return "color:#f0a040;font-weight:bold"
                return "color:#ff4d6d;font-weight:bold"
            except: return ""
        ov_fmt = {"DD Actual ($)": "{:.2f}", "Max DD ($)": "{:.2f}", "Equity actual ($)": "{:.2f}", "Peak ($)": "{:.2f}"}
        if has_bal_dd:
            ov_fmt["DD % Cta"] = "{:.2f}%"
            ov_fmt["Max DD % Cta"] = "{:.2f}%"
        ov_sty = df_dd_ov.style.format(ov_fmt).map(_sev_color, subset=["Severidad"])
        if has_bal_dd and "DD % Cta" in df_dd_ov.columns:
            ov_sty = ov_sty.map(_pct_color, subset=["DD % Cta"])
        st.dataframe(ov_sty, use_container_width=True, hide_index=True)
    else:
        st.info("Sin datos de robots disponibles.")

    st.markdown("---")

    # ── Detail per robot ────────────────────────────────────────────────────
    st.markdown("#### 🔍 Análisis Detallado por Robot")
    selected_dd_robot = st.selectbox("Seleccioná un robot:", robots_dd, key="dd_tab_robot_selector")
    if selected_dd_robot:
        g_sel = df_pos[df_pos["robot_id"].astype(str) == selected_dd_robot].copy()
        oc_sel = []
        if tcol in g_sel.columns: oc_sel.append(tcol)
        g_sel = g_sel.sort_values(oc_sel if oc_sel else [tcol], kind="mergesort") if oc_sel else g_sel
        pnl_sel = g_sel["real_profit"].fillna(0.0)
        g_sel["#"] = np.arange(1, len(g_sel) + 1)
        eq_sel = pd.Series(pnl_sel.cumsum().values)
        render_current_dd_panel(eq_sel, account_balance, g_sel[tcol] if tcol in g_sel.columns else None)

        # Equity curve for context
        st.markdown("##### 📈 Curva de Equity")
        st.line_chart(pd.Series(eq_sel.values, index=range(1, len(eq_sel) + 1)), height=240)
        st.caption(f"Equity acumulada — {selected_dd_robot}")

with tab_edge:
    render_edge_tab(df_pos, tcol, account_balance, risk_per_trade)