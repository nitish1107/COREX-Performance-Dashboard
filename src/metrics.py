"""KPI calculations and rule-based recommendations (no UI code here)."""
import numpy as np
import pandas as pd


def kpis(sales: pd.DataFrame, inv: pd.DataFrame) -> dict:
    revenue, profit = sales["revenue"].sum(), sales["profit"].sum()
    demand = inv["units_sold"].sum() + inv["lost_units"].sum()
    return {
        "revenue": revenue,
        "profit": profit,
        "margin": profit / revenue if revenue else 0.0,
        "orders": sales["order_id"].nunique(),
        "stockout_rate": float((inv["closing_stock"] == 0).mean()) if len(inv) else 0.0,
        "fill_rate": inv["units_sold"].sum() / demand if demand else 1.0,
        "lost_revenue": float((inv["lost_units"] * inv["unit_price"]).sum()),
    }


def summarise(sales: pd.DataFrame, by) -> pd.DataFrame:
    g = sales.groupby(by, observed=True)[["revenue", "profit", "quantity"]].sum().reset_index()
    g["margin"] = g["profit"] / g["revenue"]
    return g


def monthly(sales: pd.DataFrame) -> pd.DataFrame:
    return summarise(sales.assign(month=sales["order_date"].dt.to_period("M").dt.to_timestamp()), "month")


def seasonality_index(sales: pd.DataFrame) -> pd.DataFrame:
    """Average daily revenue of each calendar month / average daily revenue overall, by category (1.0 = normal)."""
    daily = sales.groupby(["category", "order_date"])["revenue"].sum().unstack(fill_value=0)
    by_month = daily.T.groupby(daily.columns.month).mean().T
    return by_month.div(daily.mean(axis=1), axis=0)


def discount_bands(sales: pd.DataFrame) -> pd.DataFrame:
    bands = pd.cut(sales["discount"], [-0.01, 0.05, 0.10, 0.15, 0.20, 1],
                   labels=["0-5%", "6-10%", "11-15%", "16-20%", "Over 20%"])
    return summarise(sales.assign(band=bands), "band")


def stockout_table(inv: pd.DataFrame) -> pd.DataFrame:
    """Per product and region: lost sales and a suggested reorder point.

    Suggested ROP = mean daily demand x lead time + 1.65 x std x sqrt(lead time)
    (about 95% cycle service level), computed on the busiest 25% of months.
    """
    d = inv.assign(demand=inv["units_sold"] + inv["lost_units"],
                   lost_revenue=inv["lost_units"] * inv["unit_price"],
                   stockout=inv["closing_stock"] == 0,
                   period=inv["date"].dt.to_period("M"))
    keys = ["product_id", "product_name", "region"]
    out = d.groupby(keys).agg(stockout_days=("stockout", "sum"), lost_units=("lost_units", "sum"),
                              lost_revenue=("lost_revenue", "sum"), current_rop=("reorder_point", "last"),
                              lead_time=("lead_time_days", "first"), avg_demand=("demand", "mean"),
                              avg_stock=("closing_stock", "mean"), unit_cost=("unit_cost", "first")).reset_index()

    month = d.groupby(keys + ["period"])["demand"].agg(["mean", "std"]).reset_index()
    peak = month[month["mean"] >= month.groupby(keys)["mean"].transform("quantile", 0.75)]
    peak = peak.groupby(keys)[["mean", "std"]].mean().reset_index()
    out = out.merge(peak, on=keys, how="left")
    out["peak_rop"] = np.ceil(out["mean"] * out["lead_time"] + 1.65 * out["std"].fillna(0) * np.sqrt(out["lead_time"]))
    out["days_of_cover"] = out["avg_stock"] / out["avg_demand"].replace(0, np.nan)
    out["stock_value"] = out["avg_stock"] * out["unit_cost"]
    return out.drop(columns=["mean", "std"]).sort_values("lost_revenue", ascending=False)


def recommendations(sales: pd.DataFrame, inv: pd.DataFrame) -> list[dict]:
    recs = []

    # 1. deep discounts: what changes if discounts were capped at 20%, with the same volume
    deep = sales[sales["discount"] > 0.20]
    if len(deep):
        gain = float((deep["gross_sales"] * (deep["discount"] - 0.20)).sum())
        recs.append({"title": "Cap discounts at 20%",
                     "finding": f"{len(deep):,} orders were discounted above 20%, at a {deep['profit'].sum() / deep['revenue'].sum():.1%} margin.",
                     "action": "Require approval for promotions above 20% and A/B test shallower offers.",
                     "impact": gain, "basis": "Upper bound: extra profit if all those orders had still happened at 20% off."})

    # 2. stockouts
    table = stockout_table(inv)
    top = table.head(10)
    if top["lost_revenue"].sum() > 0:
        lost = inv.assign(v=inv["lost_units"] * (inv["unit_price"] - inv["unit_cost"]))
        keys = set(zip(top["product_id"], top["region"]))
        mask = [k in keys for k in zip(lost["product_id"], lost["region"])]
        recs.append({"title": "Raise reorder points before peak months",
                     "finding": f"The 10 worst product-region pairs account for ₹{top['lost_revenue'].sum():,.0f} of unmet demand (list price).",
                     "action": "Use the season-adjusted reorder points in the Stockouts tab and order one lead time ahead of the peak.",
                     "impact": float(lost.loc[mask, "v"].sum()),
                     "basis": "Upper bound: list-price gross margin on units lost in those 10 pairs. Ignores holding cost."})

    # 3. weak sub-categories: +3% price
    sub = summarise(sales, ["sub_category"]).sort_values("margin")
    if len(sub) > 1:
        weak = sub.head(3)
        recs.append({"title": "Reprice or renegotiate low-margin lines",
                     "finding": "Lowest margins: " + ", ".join(f"{r.sub_category} ({r.margin:.1%})" for r in weak.itertuples()) + ".",
                     "action": "Test a 3% price rise or negotiate supplier cost; bundle with high-margin items.",
                     "impact": float(weak["revenue"].sum() * 0.03), "basis": "Assumes 3% higher net price and unchanged volume."})

    # 4. costly shipping region
    ship = sales.groupby("region").agg(cost=("shipping_cost", "mean"), n=("order_id", "size"))
    if len(ship) > 1:
        worst = ship["cost"].idxmax()
        others = ship.drop(worst)
        typical = (others["cost"] * others["n"]).sum() / others["n"].sum()
        recs.append({"title": f"Cut delivery cost in the {worst} region",
                     "finding": f"Average shipping is ₹{ship.loc[worst, 'cost']:.0f} per order against ₹{typical:.0f} elsewhere.",
                     "action": "Compare carriers, consolidate shipments or add a regional stocking point.",
                     "impact": float((ship.loc[worst, "cost"] - typical) * ship.loc[worst, "n"] * 0.5),
                     "basis": "Assumes half of the cost gap can be closed."})

    # 5. overstock
    over = table[table["days_of_cover"] > 45]
    if len(over):
        excess = float((over["stock_value"] * (1 - 45 / over["days_of_cover"])).sum())
        recs.append({"title": "Release cash from overstocked items",
                     "finding": f"{len(over)} product-region pairs hold more than 45 days of cover.",
                     "action": "Reduce order quantities and clear excess with targeted offers.",
                     "impact": excess * 0.20, "basis": f"₹{excess:,.0f} excess stock at an assumed 20% yearly holding cost."})

    return sorted(recs, key=lambda r: r["impact"], reverse=True)
