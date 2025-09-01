# app.py
import io
import json
import re
import zipfile
from datetime import datetime

import pandas as pd
import streamlit as st
import streamlit_authenticator as stauth

import pricing_logic as pl

# ──────────────────────────────────────────────────────────────────────────────
# Streamlit config MUST be first
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Co-op Monthly Pricing", layout="wide")

# ──────────────────────────────────────────────────────────────────────────────
# Authentication (replace hashes with your own when ready)
# ──────────────────────────────────────────────────────────────────────────────
credentials = {
    "usernames": {
        "admin": {
            "name": "Admin User",
            # bcrypt hash for "password123"
            "password": "$2b$12$8rxnD6N3F7tNwImAyt3VROvbnlz.YnOf8Ck2xnwbIX9jfojllHPHC"
        },
        "emily": {
            "name": "Emily Laird",
            # bcrypt hash for "test456"
            "password": "$2b$12$QudMjH4TT.4Y9oN9z/Hyd.L/2qB5puwmP9u9N4cQZ08uP67CEbewe"
        }
    }
}

authenticator = stauth.Authenticate(
    credentials,
    "coop_pricing_app",  # cookie name
    "abcdef",            # signature key
    cookie_expiry_days=30,
)

name, authentication_status, username = authenticator.login(
    fields={'Form name': 'Login', 'Username': 'Username', 'Password': 'Password', 'Login': 'Login'}
)

# ──────────────────────────────────────────────────────────────────────────────
# Small helpers (local)
# ──────────────────────────────────────────────────────────────────────────────
def _extract_dept_code(s):
    if pd.isna(s): return None
    m = re.match(r'^\s*(\d{3})\b', str(s))
    return m.group(1) if m else None

def _pretty_department(s):
    if pd.isna(s): return s
    s = re.sub(r'^\s*\d{1,3}\s*[-–—:]*\s*', '', str(s))
    return s.strip().title()

def _upc_check_digit(upc11: str) -> str:
    digits = [int(x) for x in upc11]
    odd_sum = sum(digits[0::2]); even_sum = sum(digits[1::2])
    total = odd_sum * 3 + even_sum
    return str((10 - (total % 10)) % 10)

def _normalize_upc(raw):
    if pd.isna(raw): return None
    try:
        s = str(int(raw)) if isinstance(raw, float) and float(raw).is_integer() else str(raw)
    except Exception:
        s = str(raw)
    digits = re.sub(r"\D+", "", s)
    if not digits: return None
    if len(digits) == 11: digits += _upc_check_digit(digits)
    if len(digits) == 13 and digits.startswith("0"): digits = digits[1:]
    elif len(digits) == 14 and digits.startswith("00"): digits = digits[2:]
    elif len(digits) > 12: digits = digits[-12:]
    return digits.zfill(12)

def _find_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    low = {str(c).strip().lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in low: return low[cand.lower()]
    for cand in candidates:
        for lc, orig in low.items():
            if cand.lower() in lc: return orig
    return None

# ──────────────────────────────────────────────────────────────────────────────
# Main app (only after successful login)
# ──────────────────────────────────────────────────────────────────────────────
if authentication_status:
    st.sidebar.success(f"Welcome {name}")
    try: authenticator.logout("Logout", "sidebar")
    except Exception: pass

    st.title("Co-op Monthly Pricing")
    st.caption("Upload inputs, optionally edit margins (built from Sales), click Process, then review and export change files.")

    # ------------------------------
    # Parsers & caching
    # ------------------------------
    def _is_excel(name: str) -> bool:
        return str(name or "").lower().endswith((".xlsx", ".xls"))

    @st.cache_data(show_spinner=False)
    def _excel_sheets(data: bytes) -> list[str]:
        try:
            with io.BytesIO(data) as bio:
                xl = pd.ExcelFile(bio)
                return xl.sheet_names
        except Exception:
            return []

    @st.cache_data(show_spinner=False)
    def _parse_bytes(name: str, data: bytes, sheet: str | int | None) -> pd.DataFrame:
        name = (name or "").lower()
        bio = io.BytesIO(data)
        try:
            if _is_excel(name):
                return pd.read_excel(bio, sheet_name=sheet if sheet is not None else 0)
            return pd.read_csv(bio)
        except Exception as e:
            raise RuntimeError(f"Failed to read {name or 'uploaded file'}: {e}")

    def _read_upload(upload, sheet=None) -> pd.DataFrame:
        if upload is None:
            return pd.DataFrame()
        return _parse_bytes(upload.name, upload.getvalue(), sheet)

    def _bytes_csv(df: pd.DataFrame) -> bytes:
        return df.to_csv(index=False).encode("utf-8")

    # ------------------------------
    # Display helpers (no Bucket anywhere)
    # ------------------------------
    def _prepare_display(df: pd.DataFrame):
        d = df.copy()
        if "department_name" in d.columns and "Department" not in d.columns:
            d["Department"] = d["department_name"].map(_pretty_department)
        if "pct_change" in d.columns: d["pct_change"] = d["pct_change"] * 100.0
        if "flag_large_change" in d.columns and "Flagged" not in d.columns:
            d["Flagged"] = d["flag_large_change"].map(lambda x: "Yes" if bool(x) else "")
        if "default_supplier" in d.columns and "Default Supplier" not in d.columns:
            d["Default Supplier"] = d["default_supplier"]
        elif "DefaultSupplier" in d.columns and "Default Supplier" not in d.columns:
            d["Default Supplier"] = d["DefaultSupplier"]

        # Attach raw POS supplier names (from Sales) if we have them
        pos_vendor_raw = st.session_state.get("pos_vendor_raw") or {}
        if "upc" in d.columns and pos_vendor_raw:
            d["POS Supplier (raw)"] = d["upc"].astype(str).map(pos_vendor_raw)

        order = [
            "upc","item_name","Department",
            "current_price","new_price","abs_change","pct_change",
            "cost_source","unfi_cost","kehe_cost",
            "Default Supplier","POS Supplier (raw)","core_price","rounding_rule","Flagged","reasons"
        ]
        cols = [c for c in order if c in d.columns]
        d = d[cols]

        colcfg = {
            "upc": st.column_config.TextColumn("UPC"),
            "item_name": st.column_config.TextColumn("Item"),
            "Department": st.column_config.TextColumn("Department"),
            "current_price": st.column_config.NumberColumn("Current Price", format="$%.2f"),
            "new_price": st.column_config.NumberColumn("New Price", format="$%.2f"),
            "abs_change": st.column_config.NumberColumn("$ Change", format="$%.2f"),
            "pct_change": st.column_config.NumberColumn("% Change", format="%.1f%%"),
            "cost_source": st.column_config.TextColumn("Cost Source"),
            "unfi_cost": st.column_config.NumberColumn("UNFI Cost", format="$%.2f"),
            "kehe_cost": st.column_config.NumberColumn("KeHE Cost", format="$%.2f"),
            "Default Supplier": st.column_config.TextColumn("Default Supplier (group)"),
            "POS Supplier (raw)": st.column_config.TextColumn("POS Supplier (actual)"),
            "core_price": st.column_config.NumberColumn("Core Ceiling", format="$%.2f"),
            "rounding_rule": st.column_config.TextColumn("Rounding"),
            "Flagged": st.column_config.TextColumn("Flagged"),
            "reasons": st.column_config.TextColumn("Reasons"),
        }
        return d, colcfg

    def render_table(df: pd.DataFrame):
        d, colcfg = _prepare_display(df)
        st.dataframe(d, use_container_width=True, hide_index=True, column_config=colcfg)

    # ------------------------------
    # Sidebar: Inputs (single submit form)
    # ------------------------------
    with st.sidebar.form("inputs"):
        st.header("Inputs")

        sales_file = st.file_uploader("Sales (CSV/XLSX)", type=["csv", "xlsx", "xls"], key="sales")
        sales_sheet = None
        if sales_file and _is_excel(sales_file.name):
            sh = _excel_sheets(sales_file.getvalue())
            sales_sheet = st.selectbox("Sales sheet", options=sh, index=0 if sh else 0)

        unfi_file  = st.file_uploader("UNFI Bid (CSV/XLSX)", type=["csv", "xlsx", "xls"], key="unfi")
        unfi_sheet = None
        if unfi_file and _is_excel(unfi_file.name):
            sh = _excel_sheets(unfi_file.getvalue())
            unfi_sheet = st.selectbox("UNFI sheet", options=sh, index=0 if sh else 0)

        kehe_file  = st.file_uploader("KeHE Price List (XLSX/CSV)", type=["xlsx", "xls", "csv"], key="kehe")
        kehe_sheet = None
        if kehe_file and _is_excel(kehe_file.name):
            sh = _excel_sheets(kehe_file.getvalue())
            kehe_sheet = st.selectbox("KeHE sheet", options=sh, index=0 if sh else 0)

        core_file  = st.file_uploader("Core Sets & Basics (XLSX/CSV)", type=["xlsx", "xls", "csv"], key="core")
        core_sheet = None
        if core_file and _is_excel(core_file.name):
            sh = _excel_sheets(core_file.getvalue())
            core_sheet = st.selectbox("Core/Basics sheet", options=sh, index=0 if sh else 0)

        # ── Targets override (no canonical bucket; infer departments from SALES)
        with st.expander("Optional: Department targets override"):
            st.caption("If enabled, build the margin table from departments found in the Sales file (no canonical bucket column).")
            allow_edit = st.checkbox("Edit margins in app", value=False)
            targets_override_df = None

            if allow_edit:
                # Build dept list from Sales
                sales_preview = pd.DataFrame()
                try:
                    if sales_file is not None:
                        sales_preview = _read_upload(sales_file, sales_sheet)
                except Exception:
                    sales_preview = pd.DataFrame()

                depts_df = pd.DataFrame()
                if not sales_preview.empty:
                    dept_col = _find_col(
                        sales_preview,
                        ["department","dept","department name","sales department","dep","deptraw"]
                    )
                    if dept_col:
                        tmp = pd.DataFrame({"_dept": sales_preview[dept_col].dropna().astype(str).unique()})
                        tmp["department_name"] = tmp["_dept"].map(_pretty_department)
                        tmp["department_code"] = tmp["_dept"].map(_extract_dept_code)
                        depts_df = tmp[["department_code","department_name"]].drop_duplicates().sort_values(["department_code","department_name"])

                if not depts_df.empty:
                    baked = pl.get_baked_targets()[["department_code","department_name","target_margin"]].copy()
                    targets_base = depts_df.merge(baked, on=["department_code","department_name"], how="left")
                else:
                    targets_base = pl.get_baked_targets()[["department_code","department_name","target_margin"]].copy()

                targets_override_df = st.data_editor(
                    targets_base,
                    num_rows="dynamic",
                    use_container_width=True,
                    column_config={
                        "department_code": st.column_config.TextColumn("Dept code"),
                        "department_name": st.column_config.TextColumn("Department"),
                        "target_margin": st.column_config.NumberColumn("Target margin", min_value=0.05, max_value=0.95, step=0.01, format="%.2f"),
                    },
                )
                st.session_state["targets_template_df"] = targets_base.copy()
            else:
                st.session_state["targets_template_df"] = pl.get_baked_targets()[["department_code","department_name","target_margin"]].copy()

        st.divider()
        epsilon = st.number_input(
            "Micro-change suppressor epsilon",
            min_value=0.0, max_value=1.0, value=0.02, step=0.01,
            help="Subtract this amount before rounding to avoid tiny changes."
        )
        flag_threshold = st.number_input(
            "Flag changes at or above this % (0 to disable)",
            min_value=0.0, max_value=1.0, value=0.0, step=0.05, format="%.2f",
            help="Example: 0.15 flags ±15% or larger changes."
        )

        submitted = st.form_submit_button("Process", type="primary", use_container_width=True)

    # ── Targets template download OUTSIDE the form
    if isinstance(st.session_state.get("targets_template_df"), pd.DataFrame) and not st.session_state["targets_template_df"].empty:
        st.sidebar.download_button(
            "Download current targets template CSV",
            _bytes_csv(st.session_state["targets_template_df"]),
            file_name="department_targets_template.csv",
            mime="text/csv",
            use_container_width=True,
        )

    # ------------------------------
    # Run once when you click Process
    # ------------------------------
    if submitted:
        try:
            with st.spinner("Reading files..."):
                sales_df = _read_upload(sales_file, sales_sheet)
                unfi_df  = _read_upload(unfi_file, unfi_sheet)
                kehe_df  = _read_upload(kehe_file, kehe_sheet)
                core_df  = _read_upload(core_file, core_sheet)

                # Build raw POS vendor map & list now that we have Sales parsed
                st.session_state['pos_vendor_raw'] = {}
                st.session_state['pos_vendor_list'] = []
                if not sales_df.empty:
                    upc_col = _find_col(sales_df, ["item id","upc_pos","upc","upc-a","upc a","catapult upc","sms upc","upc pos","upc code","upc number","item upc","u.p.c.","upc12"])
                    vendor_col = _find_col(sales_df, ["default supplier","defaultsupplier","supplier","primary supplier","primary vendor","vendor","default vendor"])
                    if upc_col and vendor_col:
                        m = (
                            sales_df[[upc_col, vendor_col]].dropna()
                            .assign(_upc=lambda df: df[upc_col].map(_normalize_upc),
                                    _v=lambda df: df[vendor_col].astype(str).str.strip())
                            .dropna(subset=["_upc"])
                            .drop_duplicates("_upc")
                        )
                        st.session_state['pos_vendor_raw'] = m.set_index("_upc")["_v"].to_dict()
                        st.session_state['pos_vendor_list'] = sorted({v for v in m["_v"] if v})

            if sales_df.empty or (unfi_df.empty and kehe_df.empty):
                st.error("Provide Sales and at least one vendor file (UNFI or KeHE).")
            else:
                with st.spinner("Computing price recommendations..."):
                    change, diag, summary, gm_summary, kehe_cheaper, kehe_default_unfi_available = pl.process_pricing(
                        sales_df=sales_df,
                        unfi_df=unfi_df,
                        kehe_df=kehe_df,
                        core_df=core_df,
                        targets_override_df=targets_override_df,   # optional
                        epsilon=epsilon,
                        flag_threshold=flag_threshold,
                    )
                st.session_state["results"] = {
                    "change": change,
                    "diag": diag,
                    "summary": summary,
                    "gm_summary": gm_summary,
                    "kehe_cheaper": kehe_cheaper,
                    "kehe_default_unfi_available": kehe_default_unfi_available,
                }
        except Exception as e:
            st.exception(e)

    # ------------------------------
    # Show results if present
    # ------------------------------
    res = st.session_state.get("results")
    if res:
        change = res["change"].copy()
        diag = res["diag"].copy()
        summary = res["summary"]
        gm_summary = res["gm_summary"]
        kehe_cheaper = res["kehe_cheaper"].copy()
        kehe_default_unfi_available = res["kehe_default_unfi_available"].copy()

        # ---- Summary
        st.subheader("Summary (all items)")
        a1, a2, a3 = st.columns(3)
        a1.metric("Items processed", f"{int(summary.get('items_processed', 0)):,}")
        a2.metric("Items needing change", f"{int(summary.get('items_needing_change', 0)):,}")
        a3.metric("GM$ delta (projected − current)", f"${float(summary.get('gm_total_delta', 0) or 0):,.2f}")

        # --------------------------
        # Filters
        # --------------------------
        st.subheader("Filter view")

        view = st.radio(
            "Show which items?",
            ["All changes", "Price decreases only", "Price increases only", "No change"],
            horizontal=True,
        )

        # Excel-like dropdown for POS Default Supplier (actual)
        pos_vendor_list = st.session_state.get("pos_vendor_list", [])
        if 'vendor_filter_df' not in st.session_state or set(st.session_state['vendor_filter_df']['Supplier']) != set(pos_vendor_list):
            # default: only UNFI / KEHE selected
            st.session_state['vendor_filter_df'] = pd.DataFrame({
                "Supplier": pos_vendor_list,
                "Include": [("UNFI" in str(v).upper()) or ("KEHE" in str(v).upper()) for v in pos_vendor_list],
            })

        with st.popover("Filter suppliers", use_container_width=True):
            st.caption("Select POS Default Suppliers to INCLUDE")
            # quick actions
            c1, c2, c3 = st.columns(3)
            if c1.button("Select All"):
                st.session_state['vendor_filter_df']["Include"] = True
            if c2.button("Select None"):
                st.session_state['vendor_filter_df']["Include"] = False
            if c3.button("UNFI + KeHE"):
                st.session_state['vendor_filter_df']["Include"] = st.session_state['vendor_filter_df']["Supplier"].str.upper().str.contains("UNFI|KEHE")

            st.session_state['vendor_filter_df'] = st.data_editor(
                st.session_state['vendor_filter_df'],
                hide_index=True,
                use_container_width=True,
                height=300,
                column_config={
                    "Supplier": st.column_config.TextColumn("Supplier"),
                    "Include": st.column_config.CheckboxColumn("Include", default=False),
                },
                key="vendor_editor",
            )

            sel_count = int(st.session_state['vendor_filter_df']["Include"].sum())
            st.caption(f"{sel_count} of {len(st.session_state['vendor_filter_df'])} selected")

        selected_suppliers = st.session_state['vendor_filter_df'].loc[
            st.session_state['vendor_filter_df']["Include"], "Supplier"
        ].tolist()

        def _dedup(df):
            keep_cols = [c for c in [
                "upc","item_name","department_name","current_price",
                "new_price","abs_change","pct_change","cost_source","unfi_cost","kehe_cost",
                "default_supplier","DefaultSupplier","core_price","rounding_rule","flag_large_change","reasons"
            ] if c in df.columns]
            return (df[keep_cols].sort_values(["upc"]).groupby("upc", as_index=False).first())

        def _include_actual_suppliers(df: pd.DataFrame, selected: list[str]) -> pd.DataFrame:
            if df is None or df.empty: return df
            if "upc" not in df.columns: return df
            raw_map = st.session_state.get("pos_vendor_raw") or {}
            if not raw_map: return df
            if not selected: return df.iloc[0:0]
            m = df["upc"].astype(str).map(raw_map)
            return df[m.isin(set(selected))]

        # Build the filtered view dataframe
        if view == "All changes":
            view_df = change.copy()
        elif view == "Price decreases only":
            view_df = change[change["abs_change"] < 0].copy()
        elif view == "Price increases only":
            view_df = change[change["abs_change"] > 0].copy()
        else:  # No change
            mask = (
                (~diag["excluded"].fillna(False)) &
                diag["CurrentPrice"].notna() & diag["new_price"].notna() &
                (diag["CurrentPrice"].round(2) == diag["new_price"].round(2))
            )
            view_df = _dedup(diag.loc[mask].copy())

        # Apply supplier INCLUDE filter
        view_df = _include_actual_suppliers(view_df, selected_suppliers)

        # ---- Dynamic KPIs (based on current filters)
        st.markdown("#### Filtered summary")
        sel_upcs = set(view_df["upc"].dropna().astype(str)) if not view_df.empty else set()
        items_in_view = len(sel_upcs)
        changes_in_view = items_in_view if view != "No change" else 0
        gm_delta_view = 0.0
        if sel_upcs and not diag.empty:
            gm_delta_view = float(
                diag[diag["upc"].astype(str).isin(sel_upcs) & (~diag["excluded"].fillna(False))]["gm_delta"].sum()
            )

        f1, f2, f3 = st.columns(3)
        f1.metric("Items in view", f"{items_in_view:,}")
        f2.metric("Changes in view", f"{changes_in_view:,}")
        f3.metric("GM$ delta (view)", f"${gm_delta_view:,.2f}")

        # ---- Filtered table + download
        render_table(view_df)
        st.download_button("Download filtered CSV", _bytes_csv(view_df), file_name="filtered_view.csv", mime="text/csv")

        # ---- Change file (dedup by UPC) — the export you use to change prices
        st.subheader("Change file (dedup by UPC) — Export this to update POS")
        change_filtered_for_display = _include_actual_suppliers(change.copy(), selected_suppliers)
        render_table(change_filtered_for_display)

        # ---- Diagnostics & reviews
        with st.expander("Diagnostics (all sales rows)"):
            st.dataframe(diag, use_container_width=True, hide_index=True)

        left, right = st.columns(2)
        with left:
            st.subheader("KeHE cheaper than UNFI (both costs present)")
            st.dataframe(kehe_cheaper.drop(columns=["canonical_bucket"], errors="ignore"),
                         use_container_width=True, hide_index=True)
        with right:
            st.subheader("KeHE default but UNFI available (FYI)")
            st.dataframe(kehe_default_unfi_available.drop(columns=["canonical_bucket"], errors="ignore"),
                         use_container_width=True, hide_index=True)

        st.subheader("GM$ delta by department")
        st.dataframe(gm_summary, use_container_width=True, hide_index=True)

        # ---- Downloads (full, unfiltered)
        st.subheader("Downloads")
        now_str = datetime.now().strftime("%Y-%m-%d_%H%M")
        base = f"pricing_outputs_{now_str}"
        change_name = "items_needing_price_change.csv"
        diag_name = "pricing_diagnostics_all.csv"
        gm_name = "gm_delta_by_department.csv"
        kehe_cheaper_name = "review_kehe_cheaper.csv"
        kehe_default_name = "review_kehe_default_unfi_available.csv"
        summary_name = "summary.json"

        st.download_button("Download change file (FULL)", _bytes_csv(change), file_name=change_name, mime="text/csv")
        st.download_button("Download diagnostics (all rows)", _bytes_csv(diag), file_name=diag_name, mime="text/csv")
        st.download_button("Download GM$ delta by department", _bytes_csv(gm_summary), file_name=gm_name, mime="text/csv")

        zbuf = io.BytesIO()
        with zipfile.ZipFile(zbuf, "w", compression=zipfile.ZIP_DEFLATED) as z:
            z.writestr(change_name, _bytes_csv(change))
            z.writestr(diag_name, _bytes_csv(diag))
            z.writestr(gm_name, _bytes_csv(gm_summary))
            z.writestr(kehe_cheaper_name, _bytes_csv(kehe_cheaper.drop(columns=["canonical_bucket"], errors="ignore")))
            z.writestr(kehe_default_name, _bytes_csv(kehe_default_unfi_available.drop(columns=["canonical_bucket"], errors="ignore")))
            z.writestr(summary_name, json.dumps(summary, indent=2))
        st.download_button("Download outputs.zip (FULL)", zbuf.getvalue(), file_name=f"{base}.zip", mime="application/zip")

    else:
        st.info("Choose files in the sidebar, pick sheets if applicable, optionally adjust department targets, then click **Process**.")

elif authentication_status is False:
    st.error("Username/password is incorrect")
elif authentication_status is None:
    st.warning("Please enter your username and password")
