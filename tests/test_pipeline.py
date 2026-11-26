import pandas as pd
import pytest

from src.clean_data import add_financials, clean_orders

REGIONS = pd.DataFrame({"city": ["Delhi"], "region": ["North"]})


def raw(**overrides):
    row = {"order_id": "O1", "order_date": "05/03/2024", "product_id": " p001", "region": None, "city": " DELHI ",
           "quantity": 2, "unit_price": "₹1,000", "discount": None, "shipping_cost": 50.0}
    return {**row, **overrides}


def test_cleaning_fixes_messy_values():
    df, report = clean_orders(pd.DataFrame([raw(), raw(), raw(order_id="O2", quantity=-1)]), REGIONS)
    assert len(df) == 1 and report["duplicates_removed"] == 1 and report["invalid_rows_removed"] == 1
    r = df.iloc[0]
    assert (r.product_id, r.region, r.unit_price, r.discount) == ("P001", "North", 1000, 0)
    assert r.order_date == pd.Timestamp("2024-03-05")


def test_profit_formula():
    orders, _ = clean_orders(pd.DataFrame([raw(discount=0.1)]), REGIONS)
    products = pd.DataFrame({"product_id": ["P001"], "unit_price": [1000], "unit_cost": [600]})
    r = add_financials(orders, products).iloc[0]
    assert r.revenue == pytest.approx(1800)
    assert r.profit == pytest.approx(1800 - 1200 - 50)
