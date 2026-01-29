# app.py
# Streamlit Cloud ready:
# - Load multi-sheet Excel (sheets 1..27)
# - Plot ALL models with ability to select which to show
# - Add uncertainty band using MIN/MAX across selected models (per DASARIAN)
# - Preserve DASARIAN order from Excel
# - Add horizontal threshold line (default 50 mm)

import io
import re
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go


TS_ROW_RE = re.compile(r".*\b(19|20)\d{2}\b.*")  # time-series rows contain a year


# -----------------------------
# Helpers
# -----------------------------
def read_all_sheets(excel_bytes: bytes) -> pd.DataFrame:
    xls = pd.ExcelFile(io.BytesIO(excel_bytes))
    dfs = []
    for sh in xls.sheet_names:
        df = pd.read_excel(io.BytesIO(excel_bytes), sheet_name=sh)
        if df is None or df.empty:
            continue
        df.columns = [str(c).strip() for c in df.columns]
        if "DASARIAN" not in df.columns:
            continue
        df["SHEET_ID"] = str(sh).strip()
        dfs.append(df)
    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True)


def split_ts_pmk(df_all: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    das = df_all["DASARIAN"].astype(str)
    mask_ts = das.apply(lambda x: bool(TS_ROW_RE.match(x)))
    ts = df_all[mask_ts].copy()
    pmk = df_all[~mask_ts].copy()
    return ts, pmk


def to_long(ts: pd.DataFrame) -> pd.DataFrame:
    id_cols = ["SHEET_ID", "DASARIAN"]
    value_cols = [c for c in ts.columns if c not in id_cols]

    ts_num = ts.copy()
    for c in value_cols:
        ts_num[c] = pd.to_numeric(ts_num[c], errors="coerce")

    return ts_num.melt(
        id_vars=id_cols,
        value_vars=value_cols,
        var_name="MODEL",
        value_name="RAINFALL",
    )


def get_dasarian_order(ts_long: pd.DataFrame, sheet_id: str) -> List[str]:
    das = ts_long.loc[ts_long["SHEET_ID"] == sheet_id, "DASARIAN"].dropna().tolist()
    return list(dict.fromkeys(das))  # dedupe, preserve order


def filter_dasarian_range(ts_long: pd.DataFrame, das_order: List[str], start: str, end: str) -> pd.DataFrame:
    idx = {d: i for i, d in enumerate(das_order)}
    if start not in idx or end not in idx:
        return ts_long.copy()

    i0, i1 = idx[start], idx[end]
    lo, hi = (i0, i1) if i0 <= i1 else (i1, i0)
    allowed = set(das_order[lo:hi + 1])
    return ts_long[ts_long["DASARIAN"].isin(allowed)].copy()


@st.cache_data(show_spinner=False)
def load_from_bytes(excel_bytes: bytes):
    df_all = read_all_sheets(excel_bytes)
    if df_all.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    ts, pmk = split_ts_pmk(df_all)
    ts_long = to_long(ts) if not ts.empty else pd.DataFrame()
    return df_all, ts, pmk, ts_long


def pivot_for_sheet(ts_long: pd.DataFrame, sheet: str) -> pd.DataFrame:
    """Wide table: index=DASARIAN, columns=MODEL, values=RAINFALL."""
    df = ts_long[ts_long["SHEET_ID"] == sheet].copy()
    pvt = df.pivot_table(index="DASARIAN", columns="MODEL", values="RAINFALL", aggfunc="mean")
    return pvt


def compute_uncertainty_band(pvt: pd.DataFrame, models: List[str]) -> pd.DataFrame:
    """Return df with min/max/mean across selected models, per DASARIAN."""
    sub = pvt[models].copy()
    out = pd.DataFrame(index=sub.index)
    out["min"] = sub.min(axis=1, skipna=True)
    out["max"] = sub.max(axis=1, skipna=True)
    out["mean"] = sub.mean(axis=1, skipna=True)
    return out


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Dasarian Multi-Model Dashboard", layout="wide")
st.title("Dasarian Multi Model Dashboard")

st.sidebar.header("Input")
uploaded = st.sidebar.file_uploader("Upload Excel (.xlsx)", type=["xlsx"])
if uploaded is None:
    st.info("Upload file Excel (.xlsx) untuk mulai.")
    st.stop()

df_all, ts, pmk, ts_long = load_from_bytes(uploaded.read())

if df_all.empty or ts_long.empty:
    st.error("Data tidak valid. Pastikan sheet punya kolom 'DASARIAN' dan kolom model numerik.")
    st.stop()

# -----------------------------
# Filters
# -----------------------------
st.sidebar.header("Filter")

sheet_ids = sorted(ts_long["SHEET_ID"].unique().tolist(), key=lambda x: (len(x), x))
sheet = st.sidebar.selectbox("Sheet", sheet_ids, index=0)

das_order_full = get_dasarian_order(ts_long, sheet)
if not das_order_full:
    st.error("Tidak ada DASARIAN untuk sheet ini.")
    st.stop()

start_das = st.sidebar.selectbox("Start DASARIAN", das_order_full, index=0)
end_das = st.sidebar.selectbox("End DASARIAN", das_order_full, index=len(das_order_full) - 1)

threshold_mm = st.sidebar.number_input("Threshold line (mm)", min_value=0.0, value=50.0, step=1.0)

# Preferred model ordering (your list)
preferred_models = [
    "CFSV2-RAW", "CFSV2-COR", "ECMWF-RAW", "ECMWF-COR", "BOBOT",
    "INAMME-I", "INAMME-II", "INARCM", "ARIMA", "WARIMA",
    "ANFIS", "WANFIS", "ENS_STA_1", "ENS_STA_2", "ML", "ML1",
    "ENCODER_TRANSFORMER", "ANALOGI_2009", "MEDIAN",
]

models_all = sorted(ts_long["MODEL"].dropna().unique().tolist())

# Order models by preferred list first, then the rest
models_ordered = [m for m in preferred_models if m in models_all] + [m for m in models_all if m not in preferred_models]

default_selection = [m for m in ["MEDIAN", "ENS_STA_1", "ENS_STA_2"] if m in models_ordered]
if not default_selection:
    default_selection = models_ordered[:3]

models_selected = st.sidebar.multiselect(
    "Models to show (lines)",
    options=models_ordered,
    default=default_selection
)

band_models = st.sidebar.multiselect(
    "Models for uncertainty band (min-max)",
    options=models_ordered,
    default=models_selected if models_selected else default_selection
)

show_mean_in_band = st.sidebar.toggle("Show band mean", value=True)

# -----------------------------
# Build data for selected sheet + range
# -----------------------------
ts_sheet = ts_long[ts_long["SHEET_ID"] == sheet].copy()
ts_sheet = filter_dasarian_range(ts_sheet, das_order_full, start_das, end_das)

# Keep DASARIAN order
das_order = list(dict.fromkeys(ts_sheet["DASARIAN"].tolist()))
ts_sheet["DASARIAN"] = pd.Categorical(ts_sheet["DASARIAN"], categories=das_order, ordered=True)
ts_sheet = ts_sheet.sort_values("DASARIAN")

# Pivot for band computation
pvt = ts_sheet.pivot_table(index="DASARIAN", columns="MODEL", values="RAINFALL", aggfunc="mean")

# Ensure selected lists exist
models_selected = [m for m in models_selected if m in pvt.columns]
band_models = [m for m in band_models if m in pvt.columns]

if not models_selected:
    st.warning("No models selected to plot.")
    st.stop()

# Uncertainty band
band_df = None
if band_models:
    band_df = compute_uncertainty_band(pvt, band_models)

# -----------------------------
# Plot
# -----------------------------
st.subheader(f"Multi Model Time Series — Sheet {sheet}")

fig = go.Figure()

# Add uncertainty band (min-max)
if band_df is not None and band_df.dropna(how="all").shape[0] > 0:
    x_vals = band_df.index.tolist()

    # Upper bound
    fig.add_trace(go.Scatter(
        x=x_vals,
        y=band_df["max"].tolist(),
        mode="lines",
        line=dict(width=0),
        name="Max (band)",
        showlegend=False,
        hoverinfo="skip",
    ))

    # Lower bound with fill to previous trace
    fig.add_trace(go.Scatter(
        x=x_vals,
        y=band_df["min"].tolist(),
        mode="lines",
        line=dict(width=0),
        fill="tonexty",
        name="Min-Max range",
        hovertemplate="Min: %{y:.2f}<extra></extra>",
    ))

    # Optional mean
    if show_mean_in_band:
        fig.add_trace(go.Scatter(
            x=x_vals,
            y=band_df["mean"].tolist(),
            mode="lines+markers",
            name="Mean (band)",
        ))

# Add model lines
for m in models_selected:
    y = pvt[m].tolist()
    fig.add_trace(go.Scatter(
        x=pvt.index.tolist(),
        y=y,
        mode="lines+markers",
        name=m,
    ))

# Threshold line
fig.add_hline(
    y=threshold_mm,
    line_dash="dash",
    line_color="red",
    line_width=2,
    annotation_text="50 mm threshold",
    annotation_position="top left",
)


fig.update_layout(
    xaxis_title="DASARIAN",
    yaxis_title="Curah Hujan (mm)",
    legend_title="Models",
    hovermode="x unified",
)

st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# Tables (optional)
# -----------------------------
with st.expander("Show wide table (DASARIAN × MODEL)"):
    st.dataframe(pvt.reset_index(), use_container_width=True, height=420)

# PMK block
with st.expander("Show PMK block"):
    pmk_sheet = pmk[pmk["SHEET_ID"] == sheet].copy()
    if pmk_sheet.empty:
        st.caption("PMK block tidak terdeteksi.")
    else:
        st.dataframe(pmk_sheet, use_container_width=True, height=420)

