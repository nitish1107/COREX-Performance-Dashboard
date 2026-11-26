"""Clean raw CSVs, join them and write analysis-ready tables plus a quality report."""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW, OUT = ROOT / "data" / "raw", ROOT / "data" / "processed"


def clean_products(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["category"] = df["category"].str.strip().str.title()
    # recover missing categories from other products in the same sub-category
    lookup = df.dropna(subset=["category"]).drop_duplicates("sub_category").set_index("sub_category")["category"]
    df["category"] = df["category"].fillna(df["sub_category"].map(lookup))
    return df


def clean_orders(df: pd.DataFrame, regions: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    report = {"raw_rows": len(df)}
    df = df.copy()
    df["product_id"] = df["product_id"].str.strip().str.upper()
    df["city"] = df["city"].str.strip().str.title()

    report["duplicates_removed"] = int(df.duplicated("order_id").sum())
    df = df.drop_duplicates("order_id")

    iso = pd.to_datetime(df["order_date"], format="%Y-%m-%d", errors="coerce")
    dmy = pd.to_datetime(df["order_date"], format="%d/%m/%Y", errors="coerce")
    report["dates_reformatted"] = int(dmy.notna().sum())
    df["order_date"] = iso.fillna(dmy)

    price = df["unit_price"].astype(str)
    report["prices_parsed_from_text"] = int(price.str.contains("[₹,]").sum())
    df["unit_price"] = pd.to_numeric(price.str.replace("[₹,]", "", regex=True), errors="coerce")

    # city is the reliable key for region
    standard = df["city"].map(regions.set_index("city")["region"])
    report["regions_fixed"] = int((df["region"] != standard).sum())
    df["region"] = standard

    report["missing_discounts_set_to_zero"] = int(df["discount"].isna().sum())
    df["discount"] = df["discount"].fillna(0.0)

    valid = (df["quantity"] > 0) & df[["order_date", "unit_price", "region"]].notna().all(axis=1)
    report["invalid_rows_removed"] = int((~valid).sum())
    df = df[valid]
    report["clean_rows"] = len(df)
    return df, report


def add_financials(orders: pd.DataFrame, products: pd.DataFrame) -> pd.DataFrame:
    df = orders.merge(products.drop(columns="unit_price"), on="product_id", how="left", validate="many_to_one")
    if df["unit_cost"].isna().any():
        raise ValueError("Orders reference unknown product_id values")
    df["gross_sales"] = df["quantity"] * df["unit_price"]
    df["revenue"] = df["gross_sales"] * (1 - df["discount"])
    df["cogs"] = df["quantity"] * df["unit_cost"]
    df["profit"] = df["revenue"] - df["cogs"] - df["shipping_cost"]
    return df.sort_values(["order_date", "order_id"]).reset_index(drop=True)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    products = clean_products(pd.read_csv(RAW / "products.csv"))
    regions = pd.read_csv(RAW / "regions.csv")
    orders, report = clean_orders(pd.read_csv(RAW / "orders.csv"), regions)
    sales = add_financials(orders, products)
    inventory = pd.read_csv(RAW / "inventory.csv", parse_dates=["date"]).merge(
        products[["product_id", "product_name", "category", "unit_price", "unit_cost", "lead_time_days"]],
        on="product_id", validate="many_to_one")
    sales.to_csv(OUT / "sales.csv.gz", index=False)
    inventory.to_csv(OUT / "inventory.csv.gz", index=False)
    (OUT / "quality_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
