import io
import re
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px


TS_ROW_RE = re.compile(r".*\b(19|20)\d{2}\b.*")  # rows containing a year are treated as time-series


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
    das_list = ts_long.loc[ts_long["SHEET_ID"] == sheet_id, "DASARIAN"].dropna().tolist()
    return list(dict.fromkeys(das_list))  # dedupe, preserve order


def filter_dasarian_range(ts_long: pd.DataFrame, das_order: List[str], start: str, end: str) -> pd.DataFrame:
    idx = {d: i for i, d in enumerate(das_order)}
    if start not in idx or end not in idx:
        return ts_long.copy()

    i0, i1 = idx[start], idx[end]
    lo, hi = (i0, i1) if i0 <= i1 else (i1, i0)
    allowed = set(das_order[lo:hi + 1])
    return ts_long[ts_long["DASARIAN"].isin(allowed)].copy()


def df_to_excel_bytes(dfs: Dict[str, pd.DataFrame]) -> bytes:
    out = io.BytesIO()
    with pd.ExcelWriter(out, engine="openpyxl") as writer:
        for name, df in dfs.items():
            df.to_excel(writer, sheet_name=name[:31], index=False)
    return out.getvalue()


@st.cache_data(show_spinner=False)
def load_from_bytes(excel_bytes: bytes):
    df_all = read_all_sheets(excel_bytes)
    if df_all.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    ts, pmk = split_ts_pmk(df_all)
    ts_long = to_long(ts) if not ts.empty else pd.DataFrame()
    return df_all, ts, pmk, ts_long


# -----------------------------
# Streamlit App
# -----------------------------
st.set_page_config(page_title="Dasarian Ensemble Dashboard", layout="wide")
st.title("Dasarian Ensemble Dashboard")

st.sidebar.header("Input")
uploaded = st.sidebar.file_uploader("Upload Excel (.xlsx)", type=["xlsx"])

if uploaded is None:
    st.info("Upload file Excel (.xlsx) untuk mulai.")
    st.stop()

excel_bytes = uploaded.read()
df_all, ts, pmk, ts_long = load_from_bytes(excel_bytes)

if df_all.empty or ts_long.empty:
    st.error("Data tidak valid. Pastikan semua sheet punya kolom 'DASARIAN' dan struktur seperti contoh.")
    st.stop()

# Sidebar filters
st.sidebar.header("Filter")
sheet_ids = sorted(ts_long["SHEET_ID"].unique().tolist(), key=lambda x: (len(x), x))
sheet = st.sidebar.selectbox("Sheet", sheet_ids, index=0)

das_order = get_dasarian_order(ts_long, sheet)
start_das = st.sidebar.selectbox("Start DASARIAN", das_order, index=0)
end_das = st.sidebar.selectbox("End DASARIAN", das_order, index=len(das_order) - 1)

models_all = sorted(ts_long["MODEL"].dropna().unique().tolist())
preferred_ens = [m for m in ["MEDIAN", "ENS_STA_1", "ENS_STA_2"] if m in models_all]
ens_options = preferred_ens if preferred_ens else models_all
ens_col = st.sidebar.selectbox("Ensemble column", ens_options, index=0)

threshold_mm = st.sidebar.number_input("Threshold line (mm)", min_value=0.0, value=50.0, step=1.0)

show_table = st.sidebar.toggle("Show table", value=True)
show_pmk = st.sidebar.toggle("Show PMK block", value=True)

# Data for plot
ens_df = ts_long[(ts_long["SHEET_ID"] == sheet) & (ts_long["MODEL"] == ens_col)].copy()
ens_df = filter_dasarian_range(ens_df, das_order, start_das, end_das)

# Preserve Excel order on x-axis
ens_df["DASARIAN"] = pd.Categorical(ens_df["DASARIAN"], categories=das_order, ordered=True)
ens_df = ens_df.sort_values("DASARIAN")

# Plot
st.subheader(f"Ensemble Time Series — {ens_col} (Sheet {sheet})")

if ens_df.empty:
    st.warning("Tidak ada data setelah filter.")
else:
    fig = px.line(ens_df, x="DASARIAN", y="RAINFALL", markers=True)
    fig.add_hline(y=threshold_mm, line_dash="dash")
    fig.update_layout(xaxis_title="DASARIAN", yaxis_title="Curah Hujan (mm)")
    st.plotly_chart(fig, use_container_width=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("N", int(ens_df["RAINFALL"].notna().sum()))
    c2.metric("Mean", float(np.nanmean(ens_df["RAINFALL"])))
    c3.metric("Max", float(np.nanmax(ens_df["RAINFALL"])))
    c4.metric("Min", float(np.nanmin(ens_df["RAINFALL"])))

if show_table:
    st.subheader("Filtered Data (Long)")
    st.dataframe(ens_df, use_container_width=True, height=320)

if show_pmk:
    st.subheader("PMK Block (Sheet)")
    pmk_sheet = pmk[pmk["SHEET_ID"] == sheet].copy()
    if pmk_sheet.empty:
        st.caption("PMK block tidak terdeteksi.")
    else:
        st.dataframe(pmk_sheet, use_container_width=True, height=320)

# Export
st.divider()
st.subheader("Export")

export_blocks = {"ensemble_filtered_long": ens_df}
if show_pmk:
    export_blocks["pmk_sheet"] = pmk_sheet if "pmk_sheet" in locals() else pd.DataFrame()

xlsx_bytes = df_to_excel_bytes(export_blocks)
st.download_button(
    "Download export (Excel)",
    data=xlsx_bytes,
    file_name=f"export_sheet_{sheet}_{ens_col}.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
)

csv_bytes = ens_df.to_csv(index=False).encode("utf-8")
st.download_button(
    "Download export (CSV)",
    data=csv_bytes,
    file_name=f"export_sheet_{sheet}_{ens_col}.csv",
    mime="text/csv",
)
