"""COREX: sales and inventory performance dashboard.  Run:  streamlit run dashboard.py"""
import json
import io

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src import metrics
from src.clean_data import OUT

# ── colour palette (dark-theme friendly) ──────────────────────────────────────
ACCENT    = "#00D4AA"   # teal-green primary
ACCENT2   = "#7C5CFC"   # purple secondary
RED       = "#FF6B6B"   # loss / alert red
YELLOW    = "#FFD93D"   # warning yellow
CYAN      = "#4ECDC4"   # extra accent
GRID      = "#2a2d35"   # subtle grid
BG_CARD   = "#1a1d24"   # card background
BG_PLOT   = "#12141a"   # plot background
TEXT      = "#e0e0e0"   # text colour

GRADIENT = [ACCENT, CYAN, ACCENT2, YELLOW, "#FF8C42", RED, "#45B7D1", "#96CEB4",
            "#F7DC6F", "#BB8FCE", "#82E0AA", "#F0B27A", "#85C1E9", "#F1948A"]

MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

# ── continents (for region dropdown) ──────────────────────────────────────────
CONTINENTS = [
    "Africa", "Asia", "Europe", "North America", "South America",
    "Australia and Oceania", "Central America and the Caribbean",
    "Middle East and North Africa", "Sub-Saharan Africa",
]

# ── countries grouped by continent ────────────────────────────────────────────
COUNTRIES_BY_CONTINENT = {
    "Africa": ["Algeria", "Angola", "Benin", "Botswana", "Burkina Faso", "Burundi",
               "Cameroon", "Cape Verde", "Central African Republic", "Chad", "Comoros",
               "Democratic Republic of the Congo", "Republic of the Congo", "Djibouti",
               "Egypt", "Equatorial Guinea", "Eritrea", "Ethiopia", "Gabon", "Gambia",
               "Ghana", "Guinea", "Guinea-Bissau", "Ivory Coast", "Kenya", "Lesotho",
               "Liberia", "Libya", "Madagascar", "Malawi", "Mali", "Mauritania",
               "Mauritius", "Morocco", "Mozambique", "Namibia", "Niger", "Nigeria",
               "Rwanda", "Sao Tome and Principe", "Senegal", "Seychelles", "Sierra Leone",
               "Somalia", "South Africa", "South Sudan", "Sudan", "Swaziland", "Tanzania",
               "Togo", "Tunisia", "Uganda", "Zambia", "Zimbabwe"],
    "Asia": ["Afghanistan", "Armenia", "Azerbaijan", "Bahrain", "Bangladesh", "Bhutan",
             "Brunei", "Cambodia", "China", "Cyprus", "Georgia", "India", "Indonesia",
             "Iran", "Iraq", "Israel", "Japan", "Jordan", "Kazakhstan", "Kuwait",
             "Kyrgyzstan", "Laos", "Lebanon", "Malaysia", "Maldives", "Mongolia",
             "Myanmar", "Nepal", "North Korea", "Oman", "Pakistan", "Palestine",
             "Philippines", "Qatar", "Saudi Arabia", "Singapore", "South Korea",
             "Sri Lanka", "Syria", "Taiwan", "Tajikistan", "Thailand", "Timor-Leste",
             "Turkey", "Turkmenistan", "UAE", "Uzbekistan", "Vietnam", "Yemen"],
    "Europe": ["Albania", "Andorra", "Austria", "Belarus", "Belgium", "Bosnia and Herzegovina",
               "Bulgaria", "Croatia", "Czech Republic", "Denmark", "Estonia", "Finland",
               "France", "Germany", "Greece", "Hungary", "Iceland", "Ireland", "Italy",
               "Kosovo", "Latvia", "Liechtenstein", "Lithuania", "Luxembourg", "Malta",
               "Moldova", "Monaco", "Montenegro", "Netherlands", "North Macedonia",
               "Norway", "Poland", "Portugal", "Romania", "Russia", "San Marino",
               "Serbia", "Slovakia", "Slovenia", "Spain", "Sweden", "Switzerland",
               "Ukraine", "United Kingdom", "Vatican City"],
    "North America": ["Canada", "Mexico", "United States"],
    "South America": ["Argentina", "Bolivia", "Brazil", "Chile", "Colombia", "Ecuador",
                      "Guyana", "Paraguay", "Peru", "Suriname", "Uruguay", "Venezuela"],
    "Australia and Oceania": ["Australia", "Fiji", "Kiribati", "Marshall Islands",
                              "Micronesia", "Nauru", "New Zealand", "Palau",
                              "Papua New Guinea", "Samoa", "Solomon Islands", "Tonga",
                              "Tuvalu", "Vanuatu"],
    "Central America and the Caribbean": ["Antigua and Barbuda", "Bahamas", "Barbados",
                                           "Belize", "Costa Rica", "Cuba", "Dominica",
                                           "Dominican Republic", "El Salvador", "Grenada",
                                           "Guatemala", "Haiti", "Honduras", "Jamaica",
                                           "Nicaragua", "Panama", "Saint Kitts and Nevis",
                                           "Saint Lucia", "Saint Vincent and the Grenadines",
                                           "Trinidad and Tobago"],
    "Middle East and North Africa": ["Algeria", "Bahrain", "Egypt", "Iran", "Iraq",
                                      "Israel", "Jordan", "Kuwait", "Lebanon", "Libya",
                                      "Morocco", "Oman", "Palestine", "Qatar",
                                      "Saudi Arabia", "Syria", "Tunisia", "Turkey",
                                      "UAE", "Yemen"],
}

ALL_COUNTRIES = sorted(set(c for cs in COUNTRIES_BY_CONTINENT.values() for c in cs))

ALL_CATEGORIES = [
    "Baby Food", "Beverages", "Cereal", "Clothes", "Cosmetics", "Fruits",
    "Household", "Meat", "Office Supplies", "Personal Care", "Snacks", "Vegetables",
    # extra categories for user data
    "Electronics", "Appliances", "Furniture", "Fashion", "Grocery",
    "Beauty & Personal Care", "Sports & Outdoors", "Books & Stationery",
    "Automotive", "Health & Wellness", "Toys & Games", "Home & Garden",
    "Jewellery & Watches", "Pet Supplies", "Musical Instruments", "Tools & Hardware",
]

st.set_page_config(page_title="COREX", page_icon="📦", layout="wide")

# ── inject custom CSS for dark glass look ─────────────────────────────────────
st.markdown("""
<style>
    /* ---- Overall background ---- */
    .stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
        background: linear-gradient(135deg, #0a0c10 0%, #12141a 40%, #0d1117 100%) !important;
    }
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d1017 0%, #141821 100%) !important;
        border-right: 1px solid #1e2130 !important;
    }

    /* ---- Metric cards ---- */
    [data-testid="stMetric"] {
        background: linear-gradient(135deg, #1a1d24 0%, #1e2130 100%);
        border: 1px solid #2a2d3a;
        border-radius: 16px;
        padding: 18px 22px;
        box-shadow: 0 4px 24px rgba(0,0,0,0.3), inset 0 1px 0 rgba(255,255,255,0.03);
        transition: transform 0.2s, box-shadow 0.2s;
    }
    [data-testid="stMetric"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 32px rgba(0,212,170,0.12), inset 0 1px 0 rgba(255,255,255,0.05);
        border-color: #00D4AA40;
    }
    [data-testid="stMetricLabel"] {
        color: #8b8fa3 !important;
        font-weight: 600 !important;
        letter-spacing: 0.5px;
        text-transform: uppercase;
        font-size: 0.75rem !important;
    }
    [data-testid="stMetricValue"] {
        color: #ffffff !important;
        font-weight: 700 !important;
        font-size: 1.6rem !important;
    }

    /* ---- Tabs ---- */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        background: #1a1d24;
        border-radius: 12px;
        padding: 4px;
        border: 1px solid #2a2d3a;
    }
    .stTabs [data-baseweb="tab"] {
        color: #8b8fa3 !important;
        border-radius: 8px !important;
        padding: 10px 20px !important;
        font-weight: 600 !important;
        transition: all 0.2s;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #00D4AA20, #7C5CFC20) !important;
        color: #00D4AA !important;
        border-bottom: none !important;
    }
    .stTabs [data-baseweb="tab-highlight"] { display:none; }
    .stTabs [data-baseweb="tab-border"] { display:none; }

    /* ---- File uploader glow ---- */
    [data-testid="stFileUploader"] {
        border: 2px dashed #00D4AA60 !important;
        border-radius: 12px !important;
        background: #00D4AA08 !important;
        padding: 10px !important;
    }

    /* ---- Download button ---- */
    .stDownloadButton button {
        background: linear-gradient(135deg, #00D4AA, #00B894) !important;
        color: #0a0c10 !important;
        border: none !important;
        font-weight: 700 !important;
        border-radius: 8px !important;
    }

    /* ---- Title styling ---- */
    h1 {
        background: linear-gradient(90deg, #00D4AA, #7C5CFC);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800 !important;
    }

    /* ---- Hide default streamlit branding ---- */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    @keyframes glow {
        from { box-shadow: 0 4px 20px rgba(0,212,170,0.3); }
        to   { box-shadow: 0 4px 30px rgba(124,92,252,0.4); }
    }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data
def load_default():
    """Load the built-in example data."""
    if not (OUT / "sales.csv.gz").exists():
        from src import clean_data, generate_data
        generate_data.main()
        clean_data.main()
    sales = pd.read_csv(OUT / "sales.csv.gz", parse_dates=["order_date"])
    inv = pd.read_csv(OUT / "inventory.csv.gz", parse_dates=["date"])
    report = json.loads((OUT / "quality_report.json").read_text())
    return sales, inv, report


def normalise_uploaded_sales(raw: pd.DataFrame) -> pd.DataFrame:
    """Map ANY common CSV layout to the internal column schema.

    The mapper is case-insensitive, strips whitespace, and handles common aliases
    including the popular '100 Sales Records' Kaggle dataset format:
        Region, Country, Item Type, Sales Channel, Order Priority,
        Order Date, Order ID, Ship Date, Units Sold, Unit Price,
        Unit Cost, Total Revenue, Total Cost, Total Profit
    """
    # Build a lowercase lookup for original column names
    lower_map = {c.lower().strip(): c for c in raw.columns}

    # (internal_name -> list of candidate names to look for, in priority order)
    FIELD_MAP = {
        "order_date":     ["order date", "order_date", "date", "sale_date", "sale date",
                           "transaction_date", "invoice date", "invoice_date"],
        "order_id":       ["order id", "order_id", "id", "transaction_id", "invoice_id",
                           "invoice no", "invoice"],
        "region":         ["region", "market", "continent", "area", "zone", "territory"],
        "country":        ["country", "nation", "country_name", "country name"],
        "city":           ["city", "town", "location"],
        "category":       ["item type", "item_type", "category", "product_category",
                           "product category", "segment", "department"],
        "sub_category":   ["sub_category", "subcategory", "sub category",
                           "product_sub_category", "sub-category"],
        "product_name":   ["product_name", "product name", "product", "item",
                           "item_name", "item name", "description"],
        "product_id":     ["product_id", "product id", "sku", "item_id", "item id",
                           "stock code", "stockcode"],
        "quantity":       ["units sold", "units_sold", "quantity", "qty", "units",
                           "amount", "order quantity"],
        "unit_price":     ["unit price", "unit_price", "price", "selling_price",
                           "sale_price", "sale price"],
        "unit_cost":      ["unit cost", "unit_cost", "cost", "cost_price", "cost price",
                           "cogs per unit"],
        "discount":       ["discount", "discount_pct", "disc", "discount %"],
        "revenue":        ["total revenue", "total_revenue", "revenue", "sales",
                           "total_sales", "total sales", "net_sales", "net sales",
                           "total", "amount"],
        "profit":         ["total profit", "total_profit", "profit", "net_profit",
                           "net profit", "earnings", "income"],
        "shipping_cost":  ["shipping_cost", "shipping cost", "shipping", "delivery_cost",
                           "freight"],
        "gross_sales":    ["gross_sales", "gross sales", "gross", "gross_revenue"],
        "total_cost":     ["total cost", "total_cost"],
        "sales_channel":  ["sales channel", "sales_channel", "channel"],
    }

    renames = {}
    used = set()
    for internal, candidates in FIELD_MAP.items():
        for cand in candidates:
            if cand in lower_map and lower_map[cand] not in used:
                renames[lower_map[cand]] = internal
                used.add(lower_map[cand])
                break

    df = raw.rename(columns=renames)

    # --- parse dates ---
    if "order_date" in df.columns:
        df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce", format="mixed")

    # --- coerce numeric columns (handles $, ₹, commas) ---
    for num_col in ["quantity", "unit_price", "unit_cost", "discount", "revenue",
                    "profit", "shipping_cost", "gross_sales", "total_cost"]:
        if num_col in df.columns and not pd.api.types.is_numeric_dtype(df[num_col]):
            cleaned = df[num_col].astype(str).str.replace(r"[$₹,%\s]", "", regex=True)
            df[num_col] = pd.to_numeric(cleaned, errors="coerce")

    # --- compute missing financial columns ---
    if "revenue" not in df.columns:
        if "unit_price" in df.columns and "quantity" in df.columns:
            disc = df["discount"] if "discount" in df.columns else 0
            df["revenue"] = df["quantity"] * df["unit_price"] * (1 - disc)

    if "profit" not in df.columns:
        if "revenue" in df.columns and "total_cost" in df.columns:
            df["profit"] = df["revenue"] - df["total_cost"]
        elif "revenue" in df.columns and "unit_cost" in df.columns and "quantity" in df.columns:
            ship = df["shipping_cost"] if "shipping_cost" in df.columns else 0
            df["profit"] = df["revenue"] - df["quantity"] * df["unit_cost"] - ship
        elif "revenue" in df.columns:
            df["profit"] = df["revenue"] * 0.15

    if "gross_sales" not in df.columns:
        if "unit_price" in df.columns and "quantity" in df.columns:
            df["gross_sales"] = df["quantity"] * df["unit_price"]
        elif "revenue" in df.columns:
            df["gross_sales"] = df["revenue"]

    # --- fill defaults for missing columns ---
    for col, default in [
        ("discount", 0), ("shipping_cost", 0), ("region", "Unknown"),
        ("country", ""), ("city", ""), ("sub_category", None),
        ("product_id", None), ("product_name", None), ("order_id", None),
        ("quantity", 0),
    ]:
        if col not in df.columns:
            if default is None:
                if col == "sub_category":
                    df[col] = df.get("category", "General")
                elif col == "product_id":
                    df[col] = [f"P{i:04d}" for i in range(len(df))]
                elif col == "product_name":
                    if "category" in df.columns:
                        df[col] = df["category"]
                    else:
                        df[col] = df["product_id"]
                elif col == "order_id":
                    df[col] = [f"ORD{i:06d}" for i in range(len(df))]
            else:
                df[col] = default

    if "category" not in df.columns:
        df["category"] = "General"

    # margin column
    if "revenue" in df.columns:
        df["margin"] = df["profit"] / df["revenue"].replace(0, np.nan)

    return df


def build_inventory_from_sales(sales: pd.DataFrame) -> pd.DataFrame:
    """Build a minimal inventory proxy from sales when no inventory CSV exists."""
    if sales.empty:
        return pd.DataFrame()

    # aggregate daily by product+region
    grp_cols = ["order_date"]
    if "product_id" in sales.columns:
        grp_cols.append("product_id")
    if "product_name" in sales.columns:
        grp_cols.append("product_name")
    if "region" in sales.columns:
        grp_cols.append("region")
    if "category" in sales.columns:
        grp_cols.append("category")

    agg = {}
    if "quantity" in sales.columns:
        agg["quantity"] = "sum"
    if "unit_price" in sales.columns:
        agg["unit_price"] = "first"
    if "unit_cost" in sales.columns:
        agg["unit_cost"] = "first"

    if not agg:
        return pd.DataFrame()

    daily = sales.groupby(grp_cols, as_index=False).agg(agg)

    inv = pd.DataFrame()
    inv["date"] = daily["order_date"]
    for c in ["product_id", "product_name", "region", "category"]:
        if c in daily.columns:
            inv[c] = daily[c]
    inv["units_sold"] = daily.get("quantity", 0)
    inv["lost_units"] = 0
    inv["closing_stock"] = 100
    inv["opening_stock"] = 100
    inv["reorder_point"] = 10
    inv["unit_price"] = daily.get("unit_price", 0)
    inv["unit_cost"] = daily.get("unit_cost", daily.get("unit_price", pd.Series([0]*len(daily))) * 0.7)
    inv["lead_time_days"] = 7
    return inv


# ══════════════════════════════════════════════════════════════════════════════
#  METRICS FOR UPLOADED DATA (self-contained, doesn't need clean_data schema)
# ══════════════════════════════════════════════════════════════════════════════

def custom_kpis(sales: pd.DataFrame, inv: pd.DataFrame | None = None) -> dict:
    """Compute KPIs from the normalised sales dataframe."""
    revenue = float(sales["revenue"].sum()) if "revenue" in sales.columns else 0
    profit = float(sales["profit"].sum()) if "profit" in sales.columns else 0
    margin = profit / revenue if revenue else 0
    orders = int(sales["order_id"].nunique()) if "order_id" in sales.columns else len(sales)

    stockout_rate = 0.0
    lost_revenue = 0.0
    fill_rate = 1.0
    if inv is not None and not inv.empty and "closing_stock" in inv.columns:
        stockout_rate = float((inv["closing_stock"] == 0).mean())
        if "lost_units" in inv.columns and "unit_price" in inv.columns:
            lost_revenue = float((inv["lost_units"] * inv["unit_price"]).sum())
        demand = inv.get("units_sold", pd.Series(dtype=float)).sum() + inv.get("lost_units", pd.Series(dtype=float)).sum()
        if demand > 0 and "units_sold" in inv.columns:
            fill_rate = float(inv["units_sold"].sum() / demand)

    return {
        "revenue": revenue, "profit": profit, "margin": margin,
        "orders": orders, "stockout_rate": stockout_rate,
        "fill_rate": fill_rate, "lost_revenue": lost_revenue,
    }


def custom_monthly(sales: pd.DataFrame) -> pd.DataFrame:
    """Monthly aggregation for any normalised sales frame."""
    if "order_date" not in sales.columns:
        return pd.DataFrame()
    df = sales.assign(month=sales["order_date"].dt.to_period("M").dt.to_timestamp())
    g = df.groupby("month", as_index=False).agg(
        revenue=("revenue", "sum"),
        profit=("profit", "sum"),
        quantity=("quantity", "sum") if "quantity" in df.columns else ("revenue", "count"),
    )
    g["margin"] = g["profit"] / g["revenue"].replace(0, np.nan)
    return g


def custom_summarise(sales: pd.DataFrame, by) -> pd.DataFrame:
    """Group sales by one or more columns and compute totals + margin."""
    if isinstance(by, str):
        by = [by]
    valid_by = [c for c in by if c in sales.columns]
    if not valid_by:
        return pd.DataFrame()
    agg_cols = {c: "sum" for c in ["revenue", "profit", "quantity"] if c in sales.columns}
    if not agg_cols:
        return pd.DataFrame()
    g = sales.groupby(valid_by, as_index=False).agg(agg_cols)
    if "revenue" in g.columns and "profit" in g.columns:
        g["margin"] = g["profit"] / g["revenue"].replace(0, np.nan)
    return g


def custom_seasonality(sales: pd.DataFrame) -> pd.DataFrame | None:
    """Seasonality index by category (revenue-based)."""
    if "order_date" not in sales.columns or "category" not in sales.columns:
        return None
    df = sales.copy()
    df["month_num"] = df["order_date"].dt.month
    if df["month_num"].isna().all():
        return None
    pivot = df.groupby(["category", "month_num"])["revenue"].mean().unstack(fill_value=0)
    row_mean = pivot.mean(axis=1).replace(0, np.nan)
    idx = pivot.div(row_mean, axis=0).fillna(0)
    idx = idx.reindex(columns=range(1, 13), fill_value=0)
    return idx


def custom_discount_bands(sales: pd.DataFrame) -> pd.DataFrame:
    """Margin by discount band."""
    if "discount" not in sales.columns:
        return pd.DataFrame()
    df = sales.copy()
    df["band"] = pd.cut(df["discount"], [-0.01, 0.05, 0.10, 0.15, 0.20, 1],
                        labels=["0-5%", "6-10%", "11-15%", "16-20%", "Over 20%"])
    g = df.groupby("band", observed=True, as_index=False).agg(
        revenue=("revenue", "sum"), profit=("profit", "sum"), quantity=("quantity", "sum"))
    g["margin"] = g["profit"] / g["revenue"].replace(0, np.nan)
    return g.dropna(subset=["margin"])


def custom_recommendations(sales: pd.DataFrame) -> list[dict]:
    """Simple rule-based recommendations for uploaded data."""
    recs = []

    # 1. High-discount items
    if "discount" in sales.columns:
        deep = sales[sales["discount"] > 0.20]
        if len(deep) > 0 and "revenue" in deep.columns:
            avg_margin = deep["profit"].sum() / deep["revenue"].sum() if deep["revenue"].sum() else 0
            gain = float(deep["revenue"].sum() * 0.05)  # rough 5% recovery
            recs.append({
                "title": "Cap Discounts at 20%",
                "finding": f"{len(deep):,} orders were discounted above 20% at {avg_margin:.1%} average margin.",
                "action": "Require approval for promotions above 20% and A/B test shallower offers.",
                "impact": gain,
                "basis": "Rough estimate: 5% revenue recovery on deep-discount orders.",
            })

    # 2. Weak categories
    if "category" in sales.columns:
        cat = custom_summarise(sales, "category").sort_values("margin")
        if len(cat) > 2:
            weak = cat.head(3)
            recs.append({
                "title": "Reprice Low-Margin Categories",
                "finding": "Lowest margins: " + ", ".join(
                    f"{r.category} ({r.margin:.1%})" for r in weak.itertuples()) + ".",
                "action": "Test a 3% price rise or negotiate supplier costs; bundle with high-margin items.",
                "impact": float(weak["revenue"].sum() * 0.03),
                "basis": "Assumes 3% higher net price and unchanged volume.",
            })

    # 3. Top-region shipping (if available)
    if "shipping_cost" in sales.columns and "region" in sales.columns:
        ship = sales.groupby("region").agg(cost=("shipping_cost", "mean"), n=("order_id", "size"))
        if len(ship) > 1:
            worst = ship["cost"].idxmax()
            others = ship.drop(worst)
            if others["n"].sum() > 0:
                typical = (others["cost"] * others["n"]).sum() / others["n"].sum()
                if ship.loc[worst, "cost"] > typical * 1.1:
                    recs.append({
                        "title": f"Cut Delivery Cost in {worst}",
                        "finding": f"Average shipping is ${ship.loc[worst, 'cost']:.0f} vs ${typical:.0f} elsewhere.",
                        "action": "Compare carriers, consolidate shipments or add a regional hub.",
                        "impact": float((ship.loc[worst, "cost"] - typical) * ship.loc[worst, "n"] * 0.5),
                        "basis": "Assumes half of the cost gap can be closed.",
                    })

    # 4. Best-performing products for investment
    if "product_name" in sales.columns:
        prods = custom_summarise(sales, "product_name").sort_values("revenue", ascending=False)
        if len(prods) >= 5:
            top5 = prods.head(5)
            recs.append({
                "title": "Double Down on Top Products",
                "finding": "Top 5 products by revenue: " + ", ".join(top5["product_name"].tolist()) + ".",
                "action": "Increase marketing spend and stock levels for these high-performers.",
                "impact": float(top5["revenue"].sum() * 0.10),
                "basis": "Assumes 10% revenue uplift from increased investment.",
            })

    # 5. Category diversification
    if "category" in sales.columns:
        cat_rev = custom_summarise(sales, "category")
        if len(cat_rev) > 0:
            total = cat_rev["revenue"].sum()
            top_cat = cat_rev.sort_values("revenue", ascending=False).iloc[0]
            concentration = top_cat["revenue"] / total if total else 0
            if concentration > 0.3:
                recs.append({
                    "title": "Diversify Revenue Streams",
                    "finding": f"{top_cat['category']} accounts for {concentration:.0%} of total revenue.",
                    "action": "Invest in marketing for underrepresented categories to reduce concentration risk.",
                    "impact": float(total * 0.05),
                    "basis": "Assumes 5% total revenue uplift from better category balance.",
                })

    return sorted(recs, key=lambda r: r["impact"], reverse=True)


# ── prediction helpers ────────────────────────────────────────────────────────
def predict_future(monthly_df: pd.DataFrame, target_year: int) -> pd.DataFrame:
    """Simple linear-trend + seasonal forecast for monthly revenue/profit."""
    df = monthly_df.copy()
    if len(df) < 3:
        return pd.DataFrame()
    df["ordinal"] = np.arange(len(df))
    df["month_num"] = df["month"].dt.month

    result_frames = []
    for col in ["revenue", "profit"]:
        if col not in df.columns or df[col].isna().all():
            continue
        x = df["ordinal"].values.astype(float)
        y = df[col].values.astype(float)
        coeffs = np.polyfit(x, y, 1)
        trend = np.poly1d(coeffs)

        seasonal = df.groupby("month_num")[col].mean()
        overall_mean = df[col].mean()
        factors = seasonal / overall_mean if overall_mean != 0 else seasonal * 0 + 1

        rows = []
        last_ord = x[-1]
        for m in range(1, 13):
            last_ord += 1
            base = trend(last_ord)
            factor = factors.get(m, 1.0)
            rows.append({
                "month": pd.Timestamp(year=target_year, month=m, day=1),
                col: max(0, base * factor),
            })
        result_frames.append(pd.DataFrame(rows))

    if not result_frames:
        return pd.DataFrame()

    future = result_frames[0]
    for extra in result_frames[1:]:
        future = future.merge(extra, on="month", how="outer")

    if "revenue" in future.columns and "profit" in future.columns:
        future["margin"] = future["profit"] / future["revenue"].replace(0, np.nan)
    future["quantity"] = 0
    future["is_forecast"] = True
    return future


# ── chart helpers (dark theme) ────────────────────────────────────────────────
def money(v: float) -> str:
    if abs(v) >= 1e9:
        return f"${v / 1e9:.2f}B"
    if abs(v) >= 1e6:
        return f"${v / 1e6:.2f}M"
    if abs(v) >= 1e3:
        return f"${v / 1e3:.1f}K"
    return f"${v:,.0f}"


def dark_layout(fig, title, height=380):
    fig.update_layout(
        title=dict(text=title, font=dict(color=TEXT, size=16, family="Inter, sans-serif"), x=0.02),
        height=height,
        margin=dict(l=10, r=10, t=50, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
        font=dict(color="#8b8fa3", size=12),
        hoverlabel=dict(bgcolor="#1e2130", font_color="#e0e0e0", bordercolor="#2a2d3a"),
    )
    fig.update_xaxes(gridcolor="#1e2130", title=None, tickfont=dict(color="#8b8fa3"),
                     linecolor="#2a2d3a", zeroline=False)
    fig.update_yaxes(gridcolor="#1e2130", title=None, tickfont=dict(color="#8b8fa3"),
                     linecolor="#2a2d3a", zeroline=False)
    st.plotly_chart(fig, width="stretch")


def line_chart(df, y, title, fmt, forecast_df=None):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["month"], y=df[y], mode="lines+markers",
        line=dict(color=ACCENT, width=3),
        marker=dict(size=8, color=ACCENT, line=dict(width=2, color="#0a0c10")),
        hovertemplate="%{x|%b %Y}: %{y:" + fmt + "}<extra>Actual</extra>",
        fill="tozeroy", fillcolor="rgba(0,212,170,0.08)",
        name="Actual",
    ))
    if forecast_df is not None and not forecast_df.empty and y in forecast_df.columns:
        connect_df = pd.concat([df.tail(1), forecast_df], ignore_index=True)
        fig.add_trace(go.Scatter(
            x=connect_df["month"], y=connect_df[y], mode="lines+markers",
            line=dict(color=ACCENT2, width=3, dash="dot"),
            marker=dict(size=8, color=ACCENT2, symbol="diamond",
                        line=dict(width=2, color="#0a0c10")),
            hovertemplate="%{x|%b %Y}: %{y:" + fmt + "}<extra>Forecast</extra>",
            fill="tozeroy", fillcolor="rgba(124,92,252,0.06)",
            name="Forecast",
        ))
        fig.update_layout(showlegend=True, legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
            font=dict(color="#8b8fa3"), bgcolor="rgba(0,0,0,0)"))

    if y == "margin":
        fig.update_yaxes(tickformat=".0%")
    else:
        fig.update_yaxes(rangemode="tozero")
    dark_layout(fig, title)


def bar_chart(df, x, y, title, percent=False, colors=None):
    if df.empty:
        return
    if colors is None:
        n = len(df)
        colors = [GRADIENT[i % len(GRADIENT)] for i in range(n)]
    fmt = ".1%" if percent else ",.0f"
    fig = go.Figure(go.Bar(
        x=df[x], y=df[y], orientation="h",
        marker=dict(color=colors, cornerradius=6, line=dict(width=0)),
        hovertemplate="%{y}: %{x:" + fmt + "}<extra></extra>",
    ))
    fig.update_layout(bargap=0.35)
    fig.update_yaxes(autorange="reversed")
    if percent:
        fig.update_xaxes(tickformat=".0%")
    dark_layout(fig, title)


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE LAYOUT
# ══════════════════════════════════════════════════════════════════════════════

# ---- Header ----
header_l, header_r = st.columns([3, 1])
with header_l:
    st.title("⬡ COREX")
    st.caption("Sales & Inventory Performance Dashboard")

with header_r:
    st.markdown("")
    load_own = st.toggle("📂 **LOAD YOUR DATA**", value=False,
                         help="Upload your own CSV files to analyse instead of the demo dataset")

# ---- Data source ----
using_custom = False

if load_own:
    st.markdown("""
    <div style='background: linear-gradient(135deg, #00D4AA15, #7C5CFC15);
                border: 1px solid #00D4AA40; border-radius: 14px; padding: 20px; margin-bottom: 20px;'>
        <h4 style='color: #00D4AA; margin: 0 0 8px 0;'>📂 Upload Your Data</h4>
        <p style='color: #8b8fa3; margin: 0; font-size: 0.9rem;'>
            Upload a <b>Sales / Orders CSV</b> (required). Common formats are auto-detected:<br>
            <code>Region, Country, Item Type, Order Date, Units Sold, Unit Price, Unit Cost,
            Total Revenue, Total Profit</code> … and many more column names.
            <br>Optionally upload an <b>Inventory CSV</b>.
        </p>
    </div>
    """, unsafe_allow_html=True)

    up1, up2 = st.columns(2)
    with up1:
        sales_file = st.file_uploader("📊 Sales / Orders CSV", type=["csv"], key="sales_up")
    with up2:
        inv_file = st.file_uploader("📦 Inventory CSV (optional)", type=["csv"], key="inv_up")

    if sales_file is not None:
        raw_sales = pd.read_csv(sales_file)
        sales = normalise_uploaded_sales(raw_sales)
        if "order_date" not in sales.columns:
            st.error("❌ Could not find a date column (e.g. `Order Date`, `date`, `order_date`) in the uploaded CSV.")
            st.stop()
        sales = sales.dropna(subset=["order_date"])
        if sales.empty:
            st.error("❌ No valid dates found in the uploaded CSV's date column.")
            st.stop()

        if inv_file is not None:
            inv = pd.read_csv(inv_file)
            for c in inv.columns:
                if "date" in c.lower():
                    inv[c] = pd.to_datetime(inv[c], errors="coerce", format="mixed")
        else:
            inv = build_inventory_from_sales(sales)

        report = None   # no cleaning report for custom data
        using_custom = True
        st.success(f"✅ Loaded **{len(sales):,}** sales rows" +
                   (f" and **{len(inv):,}** inventory rows" if inv_file else "") + ".")
    else:
        st.info("⬆️ Upload at least a Sales CSV to get started. Meanwhile the demo dataset is shown below.")
        sales, inv, report = load_default()
else:
    sales, inv, report = load_default()


# ══════════════════════════════════════════════════════════════════════════════
#  FILTERS
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("---")
f1, f2, f3, f4 = st.columns([1, 1, 2, 2])

# -- Year (with future prediction years) --
existing_years = sorted(sales["order_date"].dt.year.dropna().unique())
max_data_year = int(max(existing_years)) if existing_years else 2025
future_years = list(range(max_data_year + 1, max_data_year + 6))
year_options = ["All years"] + [int(y) for y in existing_years[::-1]] + \
               [f"📈 {y} (Predict)" for y in future_years]

with f1:
    year_choice = st.selectbox("📅 Year", year_options)

prediction_mode = False
predict_year = None
if isinstance(year_choice, str) and "Predict" in str(year_choice):
    prediction_mode = True
    predict_year = int(str(year_choice).split()[1])
    year = "All years"
elif year_choice == "All years":
    year = "All years"
else:
    year = int(year_choice)

# -- Region = Continents,  Country = Countries --
data_regions = sorted(sales["region"].dropna().unique()) if "region" in sales.columns else []
all_region_options = sorted(set(data_regions + CONTINENTS))

data_countries = sorted(sales["country"].dropna().unique()) if "country" in sales.columns else []
all_country_options = sorted(set(data_countries + ALL_COUNTRIES))

with f2:
    geo_level = st.radio("🌍 Geo level", ["Region (Continent)", "Country"], horizontal=True)

with f3:
    if geo_level == "Country":
        geo_filter = st.multiselect("🌐 Country", all_country_options, placeholder="All countries")
    else:
        geo_filter = st.multiselect("🗺️ Region", all_region_options, placeholder="All regions")

# -- Category --
data_cats = sorted(sales["category"].dropna().unique()) if "category" in sales.columns else []
all_cat_options = sorted(set(data_cats + ALL_CATEGORIES))

with f4:
    cats = st.multiselect("🏷️ Category", all_cat_options, placeholder="All categories")


# ---- Apply filters ----
s = sales.copy()
i = inv.copy() if inv is not None else pd.DataFrame()

if geo_filter:
    if geo_level == "Country" and "country" in s.columns:
        s = s[s["country"].isin(geo_filter)]
        if "country" in i.columns:
            i = i[i["country"].isin(geo_filter)]
    elif "region" in s.columns:
        s = s[s["region"].isin(geo_filter)]
        if "region" in i.columns:
            i = i[i["region"].isin(geo_filter)]

if cats and "category" in s.columns:
    s = s[s["category"].isin(cats)]
    if "category" in i.columns:
        i = i[i["category"].isin(cats)]

previous = None
if year != "All years":
    if (year - 1) in [int(y) for y in existing_years]:
        prev_s = s[s["order_date"].dt.year == year - 1]
        prev_i = i[i["date"].dt.year == year - 1] if "date" in i.columns and not i.empty else pd.DataFrame()
        if not prev_s.empty:
            if using_custom:
                previous = custom_kpis(prev_s, prev_i if not prev_i.empty else None)
            else:
                previous = metrics.kpis(prev_s, prev_i)
    s = s[s["order_date"].dt.year == year]
    if "date" in i.columns and not i.empty:
        i = i[i["date"].dt.year == year]

if s.empty and not prediction_mode:
    st.warning("⚠️ No orders match these filters. Try adjusting your selections.")
    st.stop()


# ---- Forecast ----
forecast_df = None
if prediction_mode:
    if using_custom:
        m_all = custom_monthly(s) if not s.empty else pd.DataFrame()
    else:
        m_all = metrics.monthly(s) if not s.empty else pd.DataFrame()
    if not m_all.empty:
        forecast_df = predict_future(m_all, predict_year)


# ---- KPIs ----
if not s.empty:
    if using_custom:
        k = custom_kpis(s, i if not i.empty else None)
    else:
        k = metrics.kpis(s, i)
elif prediction_mode and forecast_df is not None and not forecast_df.empty:
    k = {
        "revenue": float(forecast_df["revenue"].sum()),
        "profit": float(forecast_df["profit"].sum()),
        "margin": float(forecast_df["profit"].sum() / forecast_df["revenue"].sum()) if forecast_df["revenue"].sum() else 0,
        "orders": 0, "stockout_rate": 0, "fill_rate": 1, "lost_revenue": 0,
    }
else:
    st.warning("⚠️ Not enough data for predictions.")
    st.stop()


def change(key, points=False):
    if not previous:
        return None
    if points:
        return f"{(k[key] - previous[key]) * 100:+.1f} pts vs prior year"
    pv = previous.get(key, 0)
    if pv == 0:
        return None
    return f"{k[key] / pv - 1:+.1%} vs prior year"


# Prediction banner
if prediction_mode:
    st.markdown(f"""
    <div style='background: linear-gradient(135deg, #7C5CFC20, #00D4AA10);
                border: 1px solid #7C5CFC60; border-radius: 12px; padding: 16px; margin-bottom: 16px;
                display: flex; align-items: center; gap: 12px;'>
        <span style='font-size: 2rem;'>🔮</span>
        <div>
            <span style='color: #7C5CFC; font-weight: 700; font-size: 1.1rem;'>
                Prediction Mode — {predict_year}</span><br>
            <span style='color: #8b8fa3; font-size: 0.85rem;'>
                Forecasts use linear trend + seasonal patterns from historical data. Actual results may vary.
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)

# KPI row
c = st.columns(5)
c[0].metric("💰 Revenue", money(k["revenue"]), change("revenue"))
c[1].metric("📈 Profit", money(k["profit"]), change("profit"))
c[2].metric("📊 Profit Margin", f"{k['margin']:.1%}", change("margin", True))
c[3].metric("🚨 Stockout Rate", f"{k['stockout_rate']:.1%}", change("stockout_rate", True), delta_color="inverse")
c[4].metric("💸 Unmet Demand", money(k["lost_revenue"]), change("lost_revenue"), delta_color="inverse")


# ══════════════════════════════════════════════════════════════════════════════
#  TABS
# ══════════════════════════════════════════════════════════════════════════════
overview, profit_tab, season_tab, stock_tab, ideas_tab, quality_tab = st.tabs(
    ["📊 Overview", "💎 Profitability", "🌡️ Seasonality", "📦 Stockouts", "💡 Recommendations", "🔍 Data Quality"])

# ───────────── OVERVIEW ─────────────
with overview:
    if using_custom:
        m = custom_monthly(s) if not s.empty else pd.DataFrame()
    else:
        m = metrics.monthly(s) if not s.empty else pd.DataFrame()

    # Revenue + Margin lines
    a, b = st.columns(2)
    with a:
        if not m.empty:
            line_chart(m, "revenue",
                       f"📈 Monthly Revenue" + (f" + {predict_year} Forecast" if prediction_mode else ""),
                       ",.0f", forecast_df=forecast_df if prediction_mode else None)
    with b:
        if not m.empty:
            line_chart(m, "margin",
                       f"📊 Profit Margin" + (f" + {predict_year} Forecast" if prediction_mode else ""),
                       ".1%", forecast_df=forecast_df if prediction_mode else None)

    # Bar charts
    if not s.empty:
        a, b = st.columns(2)
        with a:
            reg_data = custom_summarise(s, "region").sort_values("revenue", ascending=False)
            if not reg_data.empty:
                bar_chart(reg_data, "revenue", "region", "🗺️ Revenue by Region")
        with b:
            cat_data = custom_summarise(s, "category").sort_values("revenue", ascending=False)
            if not cat_data.empty:
                bar_chart(cat_data, "revenue", "category", "🏷️ Revenue by Category")

        # Donut + Top products
        a, b = st.columns(2)
        with a:
            cat_rev = custom_summarise(s, "category").sort_values("revenue", ascending=False)
            if not cat_rev.empty:
                fig = go.Figure(go.Pie(
                    labels=cat_rev["category"], values=cat_rev["revenue"],
                    hole=0.6,
                    marker=dict(colors=GRADIENT[:len(cat_rev)],
                                line=dict(color="#0a0c10", width=2)),
                    textinfo="label+percent",
                    textfont=dict(color="#e0e0e0", size=11),
                    hovertemplate="%{label}<br>$%{value:,.0f}<br>%{percent}<extra></extra>",
                ))
                fig.update_layout(annotations=[dict(text="Revenue<br>Split", x=0.5, y=0.5,
                                                    font=dict(size=14, color=ACCENT), showarrow=False)])
                dark_layout(fig, "🍩 Revenue Distribution", height=380)
        with b:
            top_prods = custom_summarise(s, "product_name").sort_values("revenue", ascending=False).head(10)
            if not top_prods.empty:
                bar_chart(top_prods, "revenue", "product_name", "🏆 Top 10 Products")

        # Country breakdown (if available)
        if "country" in s.columns and s["country"].notna().any() and s["country"].str.len().sum() > 0:
            st.subheader("🌍 Revenue by Country")
            country_data = custom_summarise(s, "country").sort_values("revenue", ascending=False).head(15)
            if not country_data.empty:
                bar_chart(country_data, "revenue", "country", "🌍 Top 15 Countries by Revenue")


# ───────────── PROFITABILITY ─────────────
with profit_tab:
    if s.empty:
        st.info("No data for profitability analysis with current filters.")
    else:
        a, b = st.columns(2)
        with a:
            sub = custom_summarise(s, "sub_category").sort_values("margin") if "sub_category" in s.columns else pd.DataFrame()
            if not sub.empty and "margin" in sub.columns:
                colors = [RED if v < 0 else ACCENT if v > 0.15 else CYAN for v in sub["margin"]]
                bar_chart(sub, "margin", "sub_category", "📊 Margin by Sub-Category", percent=True, colors=colors)
            else:
                cat_m = custom_summarise(s, "category").sort_values("margin")
                if not cat_m.empty and "margin" in cat_m.columns:
                    colors = [RED if v < 0 else ACCENT if v > 0.15 else CYAN for v in cat_m["margin"]]
                    bar_chart(cat_m, "margin", "category", "📊 Margin by Category", percent=True, colors=colors)

        with b:
            db = custom_discount_bands(s)
            if not db.empty:
                fig = go.Figure(go.Bar(
                    x=db["band"].astype(str), y=db["margin"],
                    marker=dict(cornerradius=6,
                                color=[RED if v < 0 else ACCENT for v in db["margin"]]),
                    hovertemplate="%{x} discount: %{y:.1%} margin<extra></extra>",
                ))
                fig.update_yaxes(tickformat=".0%")
                dark_layout(fig, "📉 Margin by Discount Band")
            else:
                st.info("No discount data available to show discount bands.")

        # Product table
        prod_by = ["product_name"]
        if "category" in s.columns:
            prod_by.append("category")
        products = custom_summarise(s, prod_by).sort_values("profit")
        if not products.empty:
            st.subheader("🏷️ Product Performance")
            display = products.copy()
            if "margin" in display.columns:
                display["margin"] = display["margin"] * 100
            col_config = {"product_name": "Product"}
            if "category" in display.columns:
                col_config["category"] = "Category"
            col_config.update({
                "revenue": st.column_config.NumberColumn("Revenue ($)", format="%.0f"),
                "profit": st.column_config.NumberColumn("Profit ($)", format="%.0f"),
            })
            if "quantity" in display.columns:
                col_config["quantity"] = "Units"
            if "margin" in display.columns:
                col_config["margin"] = st.column_config.NumberColumn("Margin", format="%.1f%%")
            st.dataframe(display, hide_index=True, width="stretch", column_config=col_config)


# ───────────── SEASONALITY ─────────────
with season_tab:
    if s.empty:
        st.info("No data for seasonality analysis with current filters.")
    else:
        if using_custom:
            idx = custom_seasonality(s)
        else:
            idx = metrics.seasonality_index(s)

        if idx is not None and not idx.empty:
            display_idx = idx.copy()
            display_idx.columns = [MONTHS[int(m) - 1] for m in display_idx.columns]
            fig = px.imshow(
                display_idx,
                color_continuous_scale=["#00D4AA", "#1a1d24", "#FF6B6B"],
                color_continuous_midpoint=1.0,
                text_auto=".2f", aspect="auto",
            )
            fig.update_traces(xgap=3, ygap=3,
                              hovertemplate="%{y}, %{x}: %{z:.2f}x average<extra></extra>",
                              textfont=dict(color="#e0e0e0"))
            fig.update_coloraxes(showscale=True, colorbar=dict(
                title=dict(text="Index", font=dict(color="#8b8fa3")),
                tickfont=dict(color="#8b8fa3"),
                outlinecolor="#2a2d3a"))
            dark_layout(fig, "🌡️ Seasonality Index (1.00 = average month)", 420)
            st.caption("🔴 Red months run above average, 🟢 green months below. "
                       "Plan stock one lead time before red months.")
        else:
            st.info("Not enough data to compute seasonality. Need multiple months of data with categories.")


# ───────────── STOCKOUTS ─────────────
with stock_tab:
    if using_custom:
        # For custom data: show what we can
        has_inv_data = (not i.empty and "date" in i.columns
                        and "closing_stock" in i.columns
                        and "lost_units" in i.columns)
        if has_inv_data:
            try:
                table = metrics.stockout_table(i)
                a, b = st.columns(2)
                with a:
                    if "unit_price" in i.columns:
                        monthly_lost = i.assign(
                            month=i["date"].dt.to_period("M").dt.to_timestamp(),
                            lost=i["lost_units"] * i["unit_price"],
                        ).groupby("month")["lost"].sum().reset_index()
                        line_chart(monthly_lost, "lost", "💸 Unmet Demand by Month", ",.0f")
                with b:
                    if "product_name" in table.columns:
                        worst = table.groupby("product_name")["lost_revenue"].sum().nlargest(10).reset_index()
                        colors = [RED if ii < 3 else YELLOW if ii < 6 else CYAN for ii in range(len(worst))]
                        bar_chart(worst, "lost_revenue", "product_name",
                                  "🚨 Top 10 by Unmet Demand", colors=colors)
            except Exception:
                st.info("Could not compute stockout metrics from the inventory data. "
                        "Ensure columns: date, product_id, region, units_sold, lost_units, closing_stock.")
        else:
            st.info("📦 No inventory data with stockout info. Upload an inventory CSV with columns: "
                    "date, closing_stock, lost_units, units_sold.")

            # Show a summary from sales instead
            if not s.empty and "quantity" in s.columns:
                st.subheader("📊 Sales Volume Trend (from sales data)")
                vol = s.assign(month=s["order_date"].dt.to_period("M").dt.to_timestamp()
                               ).groupby("month", as_index=False)["quantity"].sum()
                vol.columns = ["month", "revenue"]  # reuse line_chart's y column
                line_chart(vol, "revenue", "📦 Monthly Units Sold", ",.0f")
    else:
        # Default data path
        try:
            table = metrics.stockout_table(i)
            a, b = st.columns(2)
            with a:
                monthly_lost = i.assign(
                    month=i["date"].dt.to_period("M").dt.to_timestamp(),
                    lost=i["lost_units"] * i["unit_price"],
                ).groupby("month")["lost"].sum().reset_index()
                line_chart(monthly_lost, "lost", "💸 Unmet Demand by Month", ",.0f")
            with b:
                worst = table.groupby("product_name")["lost_revenue"].sum().nlargest(10).reset_index()
                colors = [RED if ii < 3 else YELLOW if ii < 6 else CYAN for ii in range(len(worst))]
                bar_chart(worst, "lost_revenue", "product_name",
                          "🚨 Top 10 by Unmet Demand", colors=colors)

            st.subheader("📋 Reorder Plan")
            plan = table[["product_name", "region", "stockout_days", "lost_units", "lost_revenue",
                          "lead_time", "current_rop", "peak_rop", "days_of_cover"]]
            st.dataframe(plan, hide_index=True, width="stretch", column_config={
                "product_name": "Product", "region": "Region",
                "stockout_days": "Stockout Days", "lost_units": "Lost Units",
                "lost_revenue": st.column_config.NumberColumn("Unmet Demand ($)", format="%.0f"),
                "lead_time": "Lead Time (days)", "current_rop": "Current ROP",
                "peak_rop": st.column_config.NumberColumn("Suggested Peak ROP", format="%.0f"),
                "days_of_cover": st.column_config.NumberColumn("Days of Cover", format="%.0f"),
            })
            st.download_button("⬇️ Download Reorder Plan (CSV)", plan.to_csv(index=False),
                               "corex_reorder_plan.csv", "text/csv")
        except Exception as e:
            st.error(f"Error computing stockout metrics: {e}")


# ───────────── RECOMMENDATIONS ─────────────
with ideas_tab:
    if s.empty:
        st.info("No data for recommendations with current filters.")
    else:
        st.caption("Rule-based suggestions for the current filters, largest first. "
                   "Impacts are rough estimates and should not be added together.")

        if using_custom:
            recs = custom_recommendations(s)
        else:
            recs = metrics.recommendations(s, i)

        if not recs:
            st.info("No strong recommendations for the current selection.")
        for idx_r, r in enumerate(recs):
            icon = ["🎯", "📦", "💰", "🚚", "📊", "🏆"][idx_r % 6]
            with st.container(border=True):
                a, b = st.columns([4, 1])
                a.subheader(f"{icon} {r['title']}")
                a.write(r["finding"])
                a.write(f"**Action:** {r['action']}")
                a.caption(r["basis"])
                b.metric("Est. Impact", money(r["impact"]))


# ───────────── DATA QUALITY ─────────────
with quality_tab:
    if using_custom:
        st.markdown("### 📂 Uploaded Data Summary")
        q1, q2, q3 = st.columns(3)
        q1.metric("Total Rows", f"{len(sales):,}")
        q2.metric("Date Range", f"{sales['order_date'].min():%Y-%m-%d} → {sales['order_date'].max():%Y-%m-%d}")
        q3.metric("Columns", f"{len(sales.columns)}")

        st.markdown("---")
        a, b = st.columns(2)
        with a:
            st.markdown("**📊 Regions in Data:**")
            for r in sorted(sales["region"].dropna().unique()):
                st.write(f"  • {r}")
        with b:
            st.markdown("**🏷️ Categories in Data:**")
            for c in sorted(sales["category"].dropna().unique()):
                st.write(f"  • {c}")

        if "country" in sales.columns and sales["country"].notna().any():
            st.markdown("**🌍 Countries in Data:**")
            countries = sorted(sales["country"].dropna().unique())
            # show in columns
            cols = st.columns(4)
            for idx_c, c in enumerate(countries):
                cols[idx_c % 4].write(f"  • {c}")

        st.markdown("---")
        st.markdown("**🔍 Column Types:**")
        type_df = pd.DataFrame({
            "Column": sales.columns,
            "Type": [str(sales[c].dtype) for c in sales.columns],
            "Non-Null": [int(sales[c].notna().sum()) for c in sales.columns],
            "Null": [int(sales[c].isna().sum()) for c in sales.columns],
        })
        st.dataframe(type_df, hide_index=True, width="stretch")
    else:
        st.write("What the Python cleaning pipeline did to the raw order file (whole dataset, ignores filters).")
        st.dataframe(pd.DataFrame({
            "Check": [x.replace("_", " ").capitalize() for x in report],
            "Rows": list(report.values()),
        }), hide_index=True)


# ---- Footer ----
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #4a4e5a; padding: 10px;'>
    <span style='font-size: 0.8rem;'>⬡ COREX Dashboard • Built with Streamlit & Plotly •
    Powered by data-driven insights</span>
</div>
""", unsafe_allow_html=True)
