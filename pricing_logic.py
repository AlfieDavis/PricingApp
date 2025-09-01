# pricing_logic.py
import pandas as pd
import numpy as np
import math, re
from datetime import datetime
from dateutil import parser as dateparser

# ------------------------------------------------------------------------------
# Baked-in targets & exclusions
# ------------------------------------------------------------------------------
_BAKED = [
    {"department_code": "110", "sales_department": "Packaged Grocery",     "canonical_bucket": "Packaged Grocery",  "target_margin": 0.40},
    {"department_code": "140", "sales_department": "Beer / Wine",          "canonical_bucket": "Beer / Wine",       "target_margin": 0.25},
    {"department_code": "150", "sales_department": "Refrigerated",         "canonical_bucket": "Refrigerated",      "target_margin": 0.32},
    {"department_code": "160", "sales_department": "Frozen",               "canonical_bucket": "Frozen",            "target_margin": 0.40},
    {"department_code": "170", "sales_department": "Bulk",                 "canonical_bucket": "Bulk",              "target_margin": 0.43},
    {"department_code": "180", "sales_department": "Vendor Packaged",      "canonical_bucket": "Packaged Bread",    "target_margin": 0.34},
    {"department_code": "190", "sales_department": "Meat / Fish",          "canonical_bucket": "Meat / Fish",       "target_margin": 0.30},
    {"department_code": "210", "sales_department": "Food Service",         "canonical_bucket": "Prepared Foods",    "target_margin": 0.65},
    {"department_code": "220", "sales_department": "Cheese & Specialty",   "canonical_bucket": "Cheese",            "target_margin": 0.38},
    {"department_code": "230", "sales_department": "Bakery",               "canonical_bucket": "In-House Bakery",   "target_margin": 0.70},
    {"department_code": "400", "sales_department": "Produce",              "canonical_bucket": "Produce",           "target_margin": 0.40},
    {"department_code": "510", "sales_department": "Body Care",            "canonical_bucket": "HBC",               "target_margin": 0.42},
    {"department_code": "520", "sales_department": "Supplements",          "canonical_bucket": "Supplements",       "target_margin": 0.45},
    {"department_code": "600", "sales_department": "Mercantile",           "canonical_bucket": "Gen Merch",         "target_margin": 0.42},
    # Exclusions
    {"department_code": "650", "sales_department": "Brand Merch",          "canonical_bucket": "EXCLUDE",           "target_margin": None},
    {"department_code": "800", "sales_department": "Deposits",             "canonical_bucket": "EXCLUDE",           "target_margin": None},
    {"department_code": "610", "sales_department": "Member Equity",        "canonical_bucket": "EXCLUDE",           "target_margin": None},
    {"department_code": "914", "sales_department": "Member Fee",           "canonical_bucket": "EXCLUDE",           "target_margin": None},
]

# ------------------------------------------------------------------------------
# Utilities
# ------------------------------------------------------------------------------
def _safe_parse_date(x):
    if pd.isna(x) or str(x).strip() == "":
        return None
    try:
        return dateparser.parse(str(x)).date()
    except Exception:
        return None

def _normalize_money(x):
    if pd.isna(x): return np.nan
    if isinstance(x, (int, float)): return float(x)
    s = str(x).replace("$", "").replace(",", "").strip()
    return float(s) if s else np.nan

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
    if len(digits) == 11:
        digits = digits + _upc_check_digit(digits)
    if len(digits) == 13 and digits.startswith("0"):
        digits = digits[1:]
    elif len(digits) == 14 and digits.startswith("00"):
        digits = digits[2:]
    elif len(digits) > 12:
        digits = digits[-12:]
    return digits.zfill(12)

def _norm_dept_name(s):
    if pd.isna(s): return None
    s = re.sub(r'^\s*\d{1,3}\s*[-–—:]*\s*', '', str(s))
    return re.sub(r'\s+', ' ', s).strip().lower()

def _keyize(s):
    if pd.isna(s): return None
    return re.sub(r'[^a-z0-9]+', ' ', str(s).lower()).strip()

def _extract_dept_code(s):
    if pd.isna(s): return None
    m = re.match(r'^\s*(\d{3})\b', str(s))
    return m.group(1) if m else None

SALES_ALIASES = {
    "UPC_POS": ["Item ID","UPC_POS","UPC","UPC-A","UPC A","Catapult UPC","SMS UPC","UPC POS",
                "UPC Code","UPC Number","Item UPC","U.P.C.","UPC12"],
    "ItemName": ["Receipt Alias","Item Name","ItemName","Description","Item Description","Product Name","Alias"],
    "DeptRaw": ["Department","Dept","DeptRaw","Department Name","Sales Department"],
    "CurrentPrice": ["Base Price","Price","Current Price","CurrentPrice","Retail","Current Retail","Base Retail"],
    "LocalCost": ["Last Cost","Average Cost","Cost","LocalCost","Avg Cost","LastCost"],
    "DefaultSupplier": ["Default Supplier","DefaultSupplier","Supplier","Primary Supplier","Primary Vendor","Vendor","Default Vendor"],
    "Store": ["Location","Store","Store Name"],
    "Units": ["Units","Qty","Movement","Units Sold","Quantity","QTY"],
}

def _map_first(df, alias_map, keys):
    out = {}
    for k in keys:
        found = None
        if k in alias_map:
            for cand in alias_map[k]:
                for col in df.columns:
                    if str(col).strip().lower() == str(cand).strip().lower():
                        found = col; break
                if found: break
        if not found:
            token = k.lower()
            for col in df.columns:
                if token in str(col).strip().lower():
                    found = col; break
        out[k] = found
    return out

def _vendor_from_text(s):
    if pd.isna(s): return None
    v = str(s).upper()
    if "UNFI" in v: return "UNFI"
    if "KEHE" in v or "KE-HE" in v or "KE HE" in v: return "KEHE"
    return "OTHER"

# ------------------------------------------------------------------------------
# Parsers
# ------------------------------------------------------------------------------
def _parse_sales(df: pd.DataFrame):
    if not isinstance(df, pd.DataFrame) or df.empty:
        return pd.DataFrame(columns=["upc"]), {}
    cols = _map_first(df, SALES_ALIASES, list(SALES_ALIASES.keys()))
    out = pd.DataFrame()
    out["upc"] = df[cols["UPC_POS"]].map(_normalize_upc) if cols["UPC_POS"] else None
    out["item_name"] = df[cols["ItemName"]] if cols["ItemName"] else None
    out["DeptRaw"] = df[cols["DeptRaw"]] if cols["DeptRaw"] else None
    out["CurrentPrice"] = df[cols["CurrentPrice"]].map(_normalize_money) if cols["CurrentPrice"] else np.nan
    out["LocalCost"] = df[cols["LocalCost"]].map(_normalize_money) if cols["LocalCost"] else np.nan
    out["DefaultSupplier"] = df[cols["DefaultSupplier"]].map(_vendor_from_text) if cols["DefaultSupplier"] else None
    out["Store"] = df[cols["Store"]] if cols["Store"] else None
    out["Units"] = pd.to_numeric(df[cols["Units"]], errors="coerce") if cols["Units"] else np.nan
    out = out.dropna(subset=["upc"])

    out["department_name"] = out["DeptRaw"]
    out["department_name_norm"] = out["department_name"].map(_norm_dept_name)
    out["department_key"] = out["department_name"].map(_keyize)
    out["department_code"] = out["DeptRaw"].map(_extract_dept_code)

    meta = {"sales_upc_col": cols.get("UPC_POS"), "sales_price_col": cols.get("CurrentPrice")}
    return out, meta

def _parse_unfi(df: pd.DataFrame):
    if not isinstance(df, pd.DataFrame) or df.empty:
        return pd.DataFrame(columns=["upc","unfi_cost"]), {"unfi_upc_col": None, "unfi_cost_col": None}
    upc_col = None
    for cand in ["UPC12","UPC","UPC-A","UPC A","Catapult UPC","SMS UPC","UPC Code","UPC Number","U.P.C.","Item UPC"]:
        for c in df.columns:
            if str(c).strip().lower() == cand.lower(): upc_col = c; break
        if upc_col: break
    reg_unit_col = None
    for c in df.columns:
        if str(c).strip().lower() == "reg unit": reg_unit_col = c; break
    if reg_unit_col is None:
        for c in df.columns:
            n = str(c).lower()
            if "reg" in n and "unit" in n: reg_unit_col = c; break
    out = pd.DataFrame()
    out["upc"] = df[upc_col].map(_normalize_upc) if upc_col else None
    out["unfi_cost"] = df[reg_unit_col].map(_normalize_money) if reg_unit_col else np.nan
    out = out.dropna(subset=["upc"])
    out = out[out["unfi_cost"].notna()]
    if out.empty:
        return pd.DataFrame(columns=["upc","unfi_cost"]), {"unfi_upc_col": upc_col, "unfi_cost_col": reg_unit_col}
    out = out.sort_values(["upc","unfi_cost"], ascending=[True,True])
    idx = out.groupby("upc")["unfi_cost"].idxmin()
    return out.loc[idx].reset_index(drop=True), {"unfi_upc_col": upc_col, "unfi_cost_col": reg_unit_col}

def _parse_kehe(df: pd.DataFrame):
    if not isinstance(df, pd.DataFrame) or df.empty:
        return pd.DataFrame(columns=["upc","kehe_cost"]), {"kehe_upc_col": None, "kehe_cost_col": None}
    upc_col = None
    for cand in ["UPC12","UPC 12","Catapult UPC","UPC","UPC-A","UPC A","SMS UPC","UPC Code","UPC Number","Item UPC","U.P.C."]:
        for c in df.columns:
            if str(c).strip().lower() == cand.lower(): upc_col = c; break
        if upc_col: break
    unit_price_col = None
    for c in df.columns:
        if str(c).strip().lower() == "unit price": unit_price_col = c; break
    out = pd.DataFrame()
    out["upc"] = df[upc_col].map(_normalize_upc) if upc_col else None
    out["kehe_cost"] = df[unit_price_col].map(_normalize_money) if unit_price_col else np.nan
    out = out.dropna(subset=["upc"])
    out = out.groupby("upc", as_index=False).agg({"kehe_cost":"min"})
    return out, {"kehe_upc_col": upc_col, "kehe_cost_col": unit_price_col}

def _parse_core(df: pd.DataFrame, active_only=False):
    if not isinstance(df, pd.DataFrame) or df.empty:
        return pd.DataFrame(columns=["upc","core_price"]), {"core_upc_col": None, "core_price_col": None}
    cat_col = next((c for c in df.columns if str(c).strip().lower() == "catapult upc"), None)
    price_col = next((c for c in df.columns if str(c).strip().lower() == "price ceiling"), None)
    start_col = next((c for c in df.columns if str(c).strip().lower() == "start"), None)
    end_col = next((c for c in df.columns if str(c).strip().lower() == "end"), None)
    out = pd.DataFrame()
    out["upc"] = df[cat_col].map(_normalize_upc) if cat_col else None
    if cat_col is None:
        return pd.DataFrame(columns=["upc","core_price"]), {"core_upc_col": None, "core_price_col": None}
    out["core_price"] = df[price_col].map(_normalize_money) if price_col else np.nan
    out["start"] = df[start_col].map(_safe_parse_date) if start_col else None
    out["end"] = df[end_col].map(_safe_parse_date) if end_col else None
    out = out.dropna(subset=["upc"])
    if active_only and len(out):
        today = datetime.utcnow().date()
        def active(r):
            s, e = r.get("start"), r.get("end")
            if s and s > today: return False
            if e and e < today: return False
            return True
        out = out[out.apply(active, axis=1)]
    out = out[pd.to_numeric(out["core_price"], errors="coerce").notna()].copy()
    if out.empty:
        return pd.DataFrame(columns=["upc","core_price"]), {"core_upc_col": cat_col, "core_price_col": price_col}
    out = out.sort_values(["upc","core_price"], ascending=[True,True])
    idx = out.groupby("upc")["core_price"].idxmin().dropna().astype(int)
    keep = out.loc[idx, ["upc","core_price"]].copy()
    return keep.reset_index(drop=True), {"core_upc_col": cat_col, "core_price_col": price_col}

# ------------------------------------------------------------------------------
# Rounding – exact <$10 grid + robust indexing; $10+ → .99
# ------------------------------------------------------------------------------
_UNDER10_MAP = {
    0: [0.09,0.19,0.29,0.39,0.49,0.59,0.69,0.79,0.89,0.99],
    1: [1.19,1.19,1.29,1.49,1.49,1.79,1.79,1.99,1.99,1.99],
    2: [2.19,2.19,2.29,2.49,2.49,2.79,2.79,2.99,2.99,2.99],
    3: [3.29,3.29,3.29,3.49,3.49,3.79,3.79,3.99,3.99,3.99],
    4: [4.29,4.29,4.29,4.49,4.49,4.79,4.79,4.99,4.99,4.99],
    5: [5.29,5.29,5.29,5.49,5.49,5.79,5.79,5.99,5.99,5.99],
    6: [6.49,6.49,6.49,6.49,6.49,6.99,6.99,6.99,6.99,6.99],
    7: [7.49,7.49,7.49,7.49,7.49,7.99,7.99,7.99,7.99,7.99],
    8: [8.49,8.49,8.49,8.49,8.49,8.99,8.99,8.99,8.99,8.99],
    9: [9.49,9.49,9.49,9.49,9.49,9.99,9.99,9.99,9.99,9.99],
}

def _ceil_to_99(x: float) -> float:
    n = math.floor(x)
    target = n + 0.99
    return round(target if x <= target + 1e-9 else (n + 1 + 0.99), 2)

def _round_under_10_by_grid(x: float):
    """
    Robust grid lookup:
      - row = clamp(floor(x), 0..9)
      - col = clamp(floor(10 * frac), 0..9)  # guard against float noise -> 10
      - if mapped price < x, bump to next row's first slot (always rounding UP)
      - if already at row 9 and still < x, return 10.99
    """
    x = float(max(0.01, x))
    if x >= 10.0:
        return _ceil_to_99(x), "UP to .99"

    row = int(math.floor(x))
    row = 0 if row < 0 else (9 if row > 9 else row)

    frac = x - row
    col = int(math.floor(frac * 10 + 1e-12))
    col = 0 if col < 0 else (9 if col > 9 else col)

    cand = _UNDER10_MAP[row][col]
    while cand + 1e-9 < x:
        if row < 9:
            row += 1
            cand = _UNDER10_MAP[row][0]
        else:
            return _ceil_to_99(x), "UP to .99"
    return round(cand, 2), f"UP to {cand:.2f}"

def _round_tiered_up(ideal_adj: float):
    if pd.isna(ideal_adj): return np.nan, "NO_ROUND"
    x = float(max(0.01, ideal_adj))
    if x >= 9.9995:  # treat float-near-10 as $10+
        return _ceil_to_99(x), "UP to .99"
    return _round_under_10_by_grid(x)

# ------------------------------------------------------------------------------
# Targets (baked-ins + optional overrides)
# ------------------------------------------------------------------------------
def _baked_targets_df():
    df = pd.DataFrame(_BAKED)
    df["department_code"] = df["department_code"].astype(str).str.zfill(3)
    df["department_name"] = df["sales_department"]
    df["department_name_norm"] = df["department_name"].map(_norm_dept_name)
    df["department_key"] = df["department_name"].map(_keyize)
    df["target_margin"] = pd.to_numeric(df["target_margin"], errors="coerce")
    df["exclude"] = (df["canonical_bucket"].str.upper() == "EXCLUDE") | df["target_margin"].isna()
    return df[["department_code","department_name","department_name_norm","department_key",
               "canonical_bucket","target_margin","exclude"]]

def get_baked_targets() -> pd.DataFrame:
    """For app editor: baked-in, editable view (exclusions removed)."""
    t = _baked_targets_df().copy()
    t = t[~t["exclude"]].copy()
    return t[["department_code","department_name","canonical_bucket","target_margin"]].reset_index(drop=True)

def _prepare_targets_overrides(targets_df: pd.DataFrame | None) -> pd.DataFrame:
    """Normalize overrides from an uploaded file or the in-app editor."""
    if not isinstance(targets_df, pd.DataFrame) or targets_df.empty:
        return pd.DataFrame(columns=[
            "department_code","department_name","department_name_norm",
            "department_key","canonical_bucket","target_margin","exclude"
        ])

    t = targets_df.copy()

    # Flexible column picking
    def pick(names: list[str]):
        for n in names:
            for c in t.columns:
                if str(c).strip().lower() == n:
                    return c
        return None

    code_c   = pick(["department_code","dept code","dept_code","code","dept"])
    name_c   = pick(["department_name","department","dept name","sales department"])
    bucket_c = pick(["canonical_bucket","bucket","canonical"])
    margin_c = pick(["target_margin","margin","target margin"])

    out = pd.DataFrame()
    out["department_code"] = (
        t[code_c].astype(str).str.extract(r'(\d{1,3})', expand=False).str.zfill(3)
        if code_c is not None else None
    )
    out["department_name"] = t[name_c].astype(str) if name_c is not None else None
    out["department_name_norm"] = out["department_name"].map(_norm_dept_name)
    out["department_key"] = out["department_name"].map(_keyize)
    out["canonical_bucket"] = t[bucket_c].astype(str) if bucket_c is not None else out["department_name"]
    out["target_margin"] = pd.to_numeric(t[margin_c], errors="coerce") if margin_c is not None else np.nan
    out["exclude"] = out["canonical_bucket"].astype(str).str.upper().eq("EXCLUDE") | out["target_margin"].isna()

    out = out.dropna(subset=["target_margin"], how="any")
    return out[["department_code","department_name","department_name_norm",
                "department_key","canonical_bucket","target_margin","exclude"]]

# ------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------
def process_pricing(
    sales_df: pd.DataFrame,
    unfi_df: pd.DataFrame | None = None,
    kehe_df: pd.DataFrame | None = None,
    core_df: pd.DataFrame | None = None,
    targets_override_df: pd.DataFrame | None = None,  # optional
    *,
    epsilon: float = 0.02,
    flag_threshold: float = 0.0,
):
    # Safe helper (avoid "truth value of a DataFrame is ambiguous")
    def _df_or_empty(x): return x if isinstance(x, pd.DataFrame) else pd.DataFrame()

    # Parse
    sales, _sales_meta = _parse_sales(_df_or_empty(sales_df))
    unfi,  _unfi_meta  = _parse_unfi(_df_or_empty(unfi_df))
    kehe,  _kehe_meta  = _parse_kehe(_df_or_empty(kehe_df))
    core,  _core_meta  = _parse_core(_df_or_empty(core_df), active_only=False)

    # UPC rollup
    upc = (
        sales.sort_values(["upc"])
        .groupby("upc", as_index=False)
        .agg(
            item_name=("item_name", "first"),
            department_name=("department_name", "first"),
            department_name_norm=("department_name_norm", "first"),
            department_key=("department_key", "first"),
            department_code=("department_code", "first"),
            current_price=("CurrentPrice", "first"),
            local_cost=("LocalCost", "first"),
            default_supplier=("DefaultSupplier", "first"),
            units_sum=("Units", "sum"),
        )
    )

    # Merge vendor costs and Core ceilings
    upc = upc.merge(unfi, on="upc", how="left").merge(kehe, on="upc", how="left").merge(core, on="upc", how="left")

    # Initialize fields
    for col, val in [("target_margin", np.nan), ("canonical_bucket", None), ("excluded", False), ("reasons", "")]:
        if col not in upc.columns:
            upc[col] = val

    # Targets: overrides first (by code, then name), then baked-ins
    t_override = _prepare_targets_overrides(_df_or_empty(targets_override_df))
    t_baked    = _baked_targets_df()

    if not t_override.empty:
        t_code = t_override.dropna(subset=["department_code"]).drop_duplicates("department_code", keep="first").set_index("department_code")
        t_name = t_override.dropna(subset=["department_key"]).drop_duplicates("department_key", keep="first").set_index("department_key")

        m = upc["department_code"].astype(str).str.zfill(3)
        hit = m.isin(t_code.index)
        upc.loc[hit, ["target_margin","canonical_bucket"]] = t_code.loc[m[hit], ["target_margin","canonical_bucket"]].to_numpy()
        upc.loc[hit, "excluded"] = upc.loc[hit, "excluded"] | t_code.loc[m[hit], "exclude"].to_numpy()

        m2 = upc["department_key"]
        hit2 = m2.isin(t_name.index) & upc["target_margin"].isna()
        upc.loc[hit2, ["target_margin","canonical_bucket"]] = t_name.loc[m2[hit2], ["target_margin","canonical_bucket"]].to_numpy()
        upc.loc[hit2, "excluded"] = upc.loc[hit2, "excluded"] | t_name.loc[m2[hit2], "exclude"].to_numpy()

    # baked-ins for any remaining
    t_code_b = t_baked.dropna(subset=["department_code"]).drop_duplicates("department_code", keep="first").set_index("department_code")
    t_name_b = t_baked.dropna(subset=["department_key"]).drop_duplicates("department_key", keep="first").set_index("department_key")

    m = upc["department_code"].astype(str).str.zfill(3)
    hit = m.isin(t_code_b.index) & upc["target_margin"].isna()
    upc.loc[hit, ["target_margin","canonical_bucket"]] = t_code_b.loc[m[hit], ["target_margin","canonical_bucket"]].to_numpy()
    upc.loc[hit, "excluded"] = upc.loc[hit, "excluded"] | t_code_b.loc[m[hit], "exclude"].to_numpy()

    m2 = upc["department_key"]
    hit2 = m2.isin(t_name_b.index) & upc["target_margin"].isna()
    upc.loc[hit2, ["target_margin","canonical_bucket"]] = t_name_b.loc[m2[hit2], ["target_margin","canonical_bucket"]].to_numpy()
    upc.loc[hit2, "excluded"] = upc.loc[hit2, "excluded"] | t_name_b.loc[m2[hit2], "exclude"].to_numpy()

    # NO fallback: still missing a target -> exclude
    missing_margin_mask = upc["target_margin"].isna() & (~upc["excluded"].fillna(False))
    if missing_margin_mask.any():
        upc.loc[missing_margin_mask, "excluded"] = True
        upc.loc[missing_margin_mask, "reasons"] = (
            upc.loc[missing_margin_mask, "reasons"].astype(str).str.rstrip(";") + ";NO_TARGET_MARGIN"
        ).str.strip(";")

    # Tag exclusions
    excl_mask = upc["excluded"].fillna(False)
    if excl_mask.any():
        tag = upc.loc[excl_mask, "canonical_bucket"].fillna("EXCLUDE")
        upc.loc[excl_mask, "reasons"] = (
            upc.loc[excl_mask, "reasons"].astype(str).str.rstrip(";") + ";EXCLUDE_DEPARTMENT:" + tag.astype(str)
        ).str.strip(";")

    # Cost choice: UNFI preferred, then KeHE
    upc["has_unfi"] = upc["unfi_cost"].notna()
    upc["has_kehe"] = upc["kehe_cost"].notna()
    def choose_cost(r):
        if pd.notna(r.get("unfi_cost")): return r["unfi_cost"], "UNFI"
        if pd.notna(r.get("kehe_cost")): return r["kehe_cost"], "KEHE"
        return np.nan, "NONE"
    upc["base_cost_used"], upc["cost_source"] = zip(*upc.apply(choose_cost, axis=1))

    # Ideal price and epsilon tweak (skip excluded)
    def ideal_from_margin(r):
        if r.get("excluded"): return np.nan
        tm, bc = r.get("target_margin"), r.get("base_cost_used")
        if pd.isna(tm) or pd.isna(bc) or tm <= 0 or tm >= 1 or bc <= 0: return np.nan
        return bc / (1.0 - tm)
    upc["ideal_price"] = upc.apply(ideal_from_margin, axis=1)
    upc["ideal_price_adj"] = upc["ideal_price"].map(lambda x: max(0.0, x - float(epsilon)) if pd.notna(x) else np.nan)

    # New price: Core ceiling wins; else grid rounding; never below cost
    new_price, rule, reason = [], [], []
    for _, r in upc.iterrows():
        if r.get("excluded"):
            new_price.append(np.nan); rule.append("EXCLUDED"); reason.append("EXCLUDED"); continue
        if pd.notna(r.get("core_price")):
            p = round(float(r["core_price"]), 2)
            new_price.append(p); rule.append("CORE_PRICE"); reason.append("CORE_PRICE"); continue
        if r.get("cost_source") not in ("UNFI","KEHE"):
            new_price.append(np.nan); rule.append("NO_VENDOR_COST"); reason.append(""); continue

        p, rr = _round_tiered_up(r.get("ideal_price_adj"))
        bc = r.get("base_cost_used")
        if pd.notna(p) and pd.notna(bc) and p + 1e-9 < bc:
            p, rr = _round_tiered_up(bc); rr = "BUMP_TO_COST"
        new_price.append(p); rule.append(rr); reason.append("MARGIN_TARGET" if rr != "BUMP_TO_COST" else rr)

    upc["new_price"] = new_price
    upc["rounding_rule"] = rule
    upc["reasons"] = (upc["reasons"].astype(str).str.rstrip(";") + ";" + pd.Series(reason, index=upc.index).astype(str)).str.strip(";")

    # Deltas / flags
    def pct_change(a, b):
        if pd.isna(a) or pd.isna(b) or a == 0: return np.nan
        return (b - a) / a
    upc["abs_change"] = upc["new_price"] - upc["current_price"]
    upc["pct_change"] = upc.apply(lambda r: pct_change(r["current_price"], r["new_price"]), axis=1)

    thr = float(flag_threshold or 0.0)
    upc["flag_large_change"] = upc["pct_change"].map(lambda p: bool(pd.notna(p) and abs(p) >= thr)) if thr > 0 else False
    if upc["flag_large_change"].any():
        mflag = upc["flag_large_change"].fillna(False)
        upc.loc[mflag, "reasons"] = (upc.loc[mflag, "reasons"].astype(str).str.rstrip(";") + ";FLAG_LARGE_CHANGE").str.strip(";")

    # KeHE default but UNFI available (FYI tag)
    mask_kdu = (upc["default_supplier"].eq("KEHE")) & (upc["has_unfi"])
    if mask_kdu.any():
        upc.loc[mask_kdu, "reasons"] = (
            upc.loc[mask_kdu, "reasons"].astype(str).str.rstrip(";") + ";KEHE_DEFAULT_BUT_UNFI_AVAILABLE"
        ).str.strip(";")

    # Needs change
    upc["needs_change"] = upc.apply(
        lambda r: ((not bool(r.get("excluded"))) and pd.notna(r.get("current_price")) and
                   pd.notna(r.get("new_price")) and round(r["current_price"], 2) != round(r["new_price"], 2)),
        axis=1,
    )

    # Change sheet (dedupe by UPC)
    change = (
        upc[upc["needs_change"] & (~upc["excluded"])]
        .copy()
        .assign(CORE=lambda df: np.where(df["core_price"].notna(), "★ Core", ""))
    )
    change = (
        change.groupby("upc", as_index=False)
        .agg({
            "item_name":"first","department_name":"first","canonical_bucket":"first",
            "current_price":"first","new_price":"first","abs_change":"first","pct_change":"first",
            "unfi_cost":"first","kehe_cost":"first","cost_source":"first","default_supplier":"first",
            "core_price":"first","rounding_rule":"first","CORE":"first",
            "flag_large_change":"max",
            "reasons": lambda s: ";".join(sorted(set(";".join(s.dropna().astype(str)).split(";")))).strip(";"),
        })
    )

    # Diagnostics (per sales row)
    sales_slim = sales[["upc","Store","DeptRaw","CurrentPrice","LocalCost","Units","DefaultSupplier"]].copy()
    diag = sales_slim.merge(upc.drop(columns=["local_cost"]), on="upc", how="left")
    diag["CORE"] = np.where(diag["core_price"].notna(), "★ Core", "")

    # GM$ summary (exclude excluded)
    def gm_now(r):
        if r.get("excluded"): return np.nan
        units, cp = r.get("Units"), r.get("CurrentPrice")
        cost = r.get("LocalCost") if pd.notna(r.get("LocalCost")) else r.get("base_cost_used")
        if pd.isna(units) or pd.isna(cp) or pd.isna(cost): return np.nan
        return (cp - cost) * units
    def gm_proj(r):
        if r.get("excluded"): return np.nan
        units, np_ = r.get("Units"), r.get("new_price")
        cost = r.get("base_cost_used")
        if pd.isna(units) or pd.isna(np_) or pd.isna(cost): return np.nan
        return (np_ - cost) * units
    diag["gm_current"] = diag.apply(gm_now, axis=1)
    diag["gm_projected"] = diag.apply(gm_proj, axis=1)
    diag["gm_delta"] = diag["gm_projected"] - diag["gm_current"]
    gm_summary = diag[~diag["excluded"].fillna(False)].groupby(["department_name"], dropna=False)["gm_delta"].sum().reset_index()

    # Review tables
    kehe_cheaper = upc[(upc["has_unfi"]) & (upc["has_kehe"]) & (upc["kehe_cost"] < upc["unfi_cost"]) & (~upc["excluded"])][
        ["upc","item_name","department_name","canonical_bucket","unfi_cost","kehe_cost"]
    ].copy()
    kehe_default_unfi_available = upc[(~upc["excluded"]) & mask_kdu][
        ["upc","item_name","department_name","canonical_bucket","default_supplier","unfi_cost","kehe_cost","cost_source"]
    ].copy()

    summary = {
        "items_processed": int(len(upc)),
        "% with UNFI costs": round(float(upc["has_unfi"].mean() * 100), 2) if len(upc) else 0.0,
        "% with KeHE costs": round(float(upc["has_kehe"].mean() * 100), 2) if len(upc) else 0.0,
        "% with ceilings": round(float(upc["core_price"].notna().mean() * 100), 2) if len(upc) else 0.0,
        "items_needing_change": int(change.shape[0]),
        "gm_total_delta": round(float(diag[~diag["excluded"].fillna(False)]["gm_delta"].sum()), 2),
        "excluded_count": int(upc["excluded"].sum()),
        "missing_margin_count": int(missing_margin_mask.sum()) if "missing_margin_mask" in locals() else 0,
        "detected_columns": {
            "unfi_upc_col": _unfi_meta.get("unfi_upc_col"),
            "unfi_cost_col": _unfi_meta.get("unfi_cost_col"),
            "kehe_upc_col": _kehe_meta.get("kehe_upc_col"),
            "kehe_cost_col": _kehe_meta.get("kehe_cost_col"),
            "core_upc_col": _core_meta.get("core_upc_col"),
            "core_price_col": _core_meta.get("core_price_col"),
            "sales_rows_after_parse": int(len(sales)),
            "unfi_rows_after_parse": int(len(unfi)),
            "kehe_rows_after_parse": int(len(kehe)),
            "core_rows_after_parse": int(len(core)),
        },
    }

    return change, diag, summary, gm_summary, kehe_cheaper, kehe_default_unfi_available
