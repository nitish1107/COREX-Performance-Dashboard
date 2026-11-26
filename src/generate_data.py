"""Generate a SYNTHETIC, deliberately messy retail dataset (India, 2024-2025)."""
from pathlib import Path

import numpy as np
import pandas as pd

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"

CITIES = {"North": ["Delhi", "Jaipur", "Chandigarh"], "South": ["Bengaluru", "Chennai", "Hyderabad"],
          "East": ["Kolkata", "Patna", "Bhubaneswar"], "West": ["Mumbai", "Pune", "Ahmedabad"]}
REGION_SHARE = {"North": 0.27, "South": 0.31, "East": 0.16, "West": 0.26}
SHIP_PER_ORDER = {"North": 70, "South": 60, "East": 130, "West": 65}

# category: (sub-category, price range, cost as share of price, mean discount)
CATALOG = {
    "Electronics": (["Headphones", "Speakers", "Smartwatches"], (1200, 7000), (0.76, 0.84), 0.13),
    "Appliances": (["Fans", "Air Coolers", "Mixers"], (1500, 9000), (0.68, 0.76), 0.07),
    "Furniture": (["Chairs", "Desks", "Shelves"], (1800, 9500), (0.64, 0.74), 0.09),
    "Fashion": (["Shirts", "Shoes", "Jackets"], (500, 3500), (0.48, 0.60), 0.12),
    "Grocery": (["Snacks", "Beverages", "Staples"], (80, 600), (0.74, 0.84), 0.03),
}

SINGULAR = {"Headphones": "Headphone", "Speakers": "Speaker", "Smartwatches": "Smartwatch", "Fans": "Fan",
            "Air Coolers": "Air Cooler", "Mixers": "Mixer", "Chairs": "Chair", "Desks": "Desk", "Shelves": "Shelf",
            "Shirts": "Shirt", "Shoes": "Shoe", "Jackets": "Jacket", "Snacks": "Snack", "Beverages": "Beverage",
            "Staples": "Staple"}


def season(month: int, category: str, sub: str) -> float:
    m = 1.0
    if month in (10, 11):  # festive season
        m *= {"Electronics": 1.9, "Fashion": 1.7, "Appliances": 1.35, "Furniture": 1.35, "Grocery": 1.2}[category]
    if sub in ("Fans", "Air Coolers"):
        m *= {3: 1.4, 4: 2.2, 5: 2.5, 6: 1.8, 11: 0.5, 12: 0.4, 1: 0.4}.get(month, 1.0)
    if sub == "Jackets":
        m *= {11: 1.6, 12: 2.2, 1: 2.0, 4: 0.4, 5: 0.3, 6: 0.3}.get(month, 1.0)
    return m


def generate(seed: int = 42) -> dict:
    rng = np.random.default_rng(seed)

    products = []
    for category, (subs, price, cost, disc) in CATALOG.items():
        for i in range(8):
            sub = subs[i % 3]
            p = round(float(rng.uniform(*price)), -1)
            products.append({"product_id": f"P{len(products) + 1:03d}", "product_name": f"{SINGULAR.get(sub, sub)} {chr(65 + i)}",
                             "category": category, "sub_category": sub, "unit_price": p,
                             "unit_cost": round(p * float(rng.uniform(*cost)), 2),
                             "lead_time_days": int(rng.integers(4, 15))})
    products = pd.DataFrame(products)

    dates = pd.date_range("2024-01-01", "2025-12-31")
    orders, inventory = [], []
    order_no = 100000
    for _, pr in products.iterrows():
        popularity = float(rng.lognormal(0, 0.5))
        base_disc = CATALOG[pr.category][3]
        overstocked = rng.random() < 0.15
        for region, share in REGION_SHARE.items():
            base = 4.2 * popularity * share  # mean units per day
            # static replenishment policy: ignores seasonality, which causes stockouts
            reorder_point = int(np.ceil(base * pr.lead_time_days * 1.25))
            order_qty = int(np.ceil(base * (120 if overstocked else 24)))
            stock, pipeline = reorder_point + order_qty, []
            for day_no, date in enumerate(dates):
                received = sum(q for d, q in pipeline if d == day_no)
                pipeline = [(d, q) for d, q in pipeline if d > day_no]
                opening = stock + received
                growth = 1 + 0.15 * day_no / 730
                demand = int(rng.poisson(base * growth * season(date.month, pr.category, pr.sub_category)))
                sold = min(demand, opening)
                stock = opening - sold
                if stock <= reorder_point and not pipeline:
                    pipeline.append((day_no + pr.lead_time_days, order_qty))
                inventory.append((date, pr.product_id, region, opening, sold, demand - sold, stock, reorder_point))

                left = sold
                while left > 0:
                    qty = min(left, int(rng.integers(1, 4)))
                    left -= qty
                    extra = 0.06 if date.month in (10, 11) else 0.0
                    if pr.category == "Electronics" and rng.random() < 0.22:
                        discount = float(rng.choice([0.25, 0.30]))  # aggressive promotions
                    else:
                        discount = float(np.clip(rng.normal(base_disc + extra, 0.04), 0, 0.22))
                    order_no += 1
                    orders.append((f"ORD{order_no}", date, pr.product_id, region, str(rng.choice(CITIES[region])), qty,
                                   pr.unit_price, round(discount, 2),
                                   round(SHIP_PER_ORDER[region] * (0.35 if pr.category == "Grocery" else 1) * float(rng.uniform(0.7, 1.3)), 2)))

    orders = pd.DataFrame(orders, columns=["order_id", "order_date", "product_id", "region", "city", "quantity",
                                           "unit_price", "discount", "shipping_cost"])
    inventory = pd.DataFrame(inventory, columns=["date", "product_id", "region", "opening_stock", "units_sold",
                                                 "lost_units", "closing_stock", "reorder_point"])
    regions = pd.DataFrame([(c, r) for r, cs in CITIES.items() for c in cs], columns=["city", "region"])
    return {"orders": _mess_up(orders, rng), "products": _mess_products(products), "regions": regions, "inventory": inventory}


def _mess_up(df: pd.DataFrame, rng) -> pd.DataFrame:
    df = df.sort_values(["order_date", "order_id"]).reset_index(drop=True)
    pick = lambda frac: rng.choice(len(df), int(len(df) * frac), replace=False)
    dates = df.order_date.dt.strftime("%Y-%m-%d")
    alt = pick(0.15)
    dates.iloc[alt] = df.order_date.iloc[alt].dt.strftime("%d/%m/%Y")
    df["order_date"] = dates
    for col in ("region", "city", "unit_price"):
        df[col] = df[col].astype(object)
    c = df.columns.get_loc
    for i in pick(0.08):
        v = df.iat[i, c("region")]
        df.iat[i, c("region")] = str(rng.choice([v.upper(), v.lower(), f" {v} "]))
    df.iloc[pick(0.02), c("region")] = np.nan
    for i in pick(0.04):
        df.iat[i, c("city")] = f"  {df.iat[i, c('city')].upper()} "
    for i in pick(0.05):
        df.iat[i, c("unit_price")] = f"₹{df.iat[i, c('unit_price')]:,.0f}"
    for i in pick(0.03):
        df.iat[i, c("product_id")] = f" {df.iat[i, c('product_id')].lower()}"
    df.iloc[pick(0.03), c("discount")] = np.nan
    df.iloc[pick(0.004), c("quantity")] *= -1  # invalid rows
    dupes = df.iloc[pick(0.015)]
    return pd.concat([df, dupes]).sample(frac=1, random_state=7).reset_index(drop=True)


def _mess_products(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.loc[df.index[::7], "category"] = df.loc[df.index[::7], "category"].str.upper()
    df.loc[df.index[3::11], "category"] = np.nan
    return df


def main() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    for name, frame in generate().items():
        frame.to_csv(RAW / f"{name}.csv", index=False, date_format="%Y-%m-%d")
        print(f"{name}: {len(frame):,} rows")


if __name__ == "__main__":
    main()
