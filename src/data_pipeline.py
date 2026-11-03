"""Part A: turn the intentionally messy synthetic exports into validated tables."""
from pathlib import Path
import sqlite3
import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
RAW_DIR, CLEAN_DIR = PROJECT / "data/raw", PROJECT / "data/clean"
DATABASE = PROJECT / "data/instamart.db"


def _read_export(stem):
    # keep_default_na=False preserves intentional blank strings for the audit.
    return pd.read_csv(RAW_DIR / f"{stem}.csv", keep_default_na=False)


def audit():
    for stem in ("customers", "products", "orders"):
        frame = _read_export(stem)
        # CSV exports often encode missing values as empty strings, so include blanks in this audit.
        missing_view = frame.replace(r"^\s*$", np.nan, regex=True)
        nulls = pd.DataFrame({"missing_count": missing_view.isna().sum(),
                              "missing_pct": (100 * missing_view.isna().mean()).round(2)})
        print(f"\n{stem}: shape={frame.shape}")
        print(frame.dtypes)
        print(nulls)
    raw_orders, raw_customers = _read_export("orders"), _read_export("customers")
    numeric_amount = pd.to_numeric(raw_orders["order_amount"], errors="coerce")
    numeric_delivery = pd.to_numeric(raw_orders["delivery_minutes"], errors="coerce")
    print("\nOrders audit:", {
        "rows": len(raw_orders), "exact_duplicates": int(raw_orders.duplicated().sum()),
        "pending": int(raw_orders.status.eq("pending").sum()),
        "distinct_order_id": int(raw_orders.order_id.nunique()),
        "bad_amount": int(numeric_amount.isna().sum()),
        "negative_delivery": int(numeric_delivery.lt(0).sum()),
        "blank_delivery": int(raw_orders.delivery_minutes.astype(str).str.strip().eq("").sum()),
    })
    age = pd.to_numeric(raw_customers.age, errors="coerce")
    print("Customer audit:", {
        "blank_age": int(raw_customers.age.astype(str).str.strip().eq("").sum()),
        "age_outside_10_100": int(((age < 10) | (age > 100)).sum()),
        "raw_city_spellings": int(raw_customers.city.nunique()),
        "raw_gender_spellings": int(raw_customers.loc[raw_customers.gender.astype(str).str.strip().ne(""), "gender"].nunique()),
        "blank_gender": int(raw_customers.gender.astype(str).str.strip().eq("").sum()),
    })


def clean_data():
    CLEAN_DIR.mkdir(parents=True, exist_ok=True)
    customers, products, orders = (_read_export(name) for name in ("customers", "products", "orders"))

    # Standardise customer descriptors before filling city-specific age gaps.
    customers["city"] = customers["city"].astype(str).str.strip().str.title()
    gender_map = {"male": "male", "m": "male", "female": "female", "f": "female"}
    customers["gender"] = customers["gender"].astype(str).str.strip().str.lower().map(gender_map).fillna("unknown")
    customers["age"] = pd.to_numeric(customers["age"], errors="coerce")
    customers.loc[~customers["age"].between(10, 100), "age"] = np.nan
    city_medians = customers.groupby("city")["age"].transform("median")
    customers["age"] = customers["age"].fillna(city_medians).round().astype(int)
    customers["signup_date"] = pd.to_datetime(customers["signup_date"]).dt.strftime("%Y-%m-%d")

    # Keep the final/latest record for each order ID after removing exact duplicates.
    orders["order_ts"] = pd.to_datetime(orders["order_ts"])
    before_exact = len(orders)
    orders = orders.drop_duplicates()
    exact_removed = before_exact - len(orders)
    orders = orders.sort_values(["order_id", "order_ts"], kind="mergesort")
    before_key = len(orders)
    orders = orders.drop_duplicates(subset="order_id", keep="last").copy()
    repeated_order_rows_removed = before_key - len(orders)

    for column in ("quantity", "discount_pct", "order_amount", "delivery_minutes"):
        orders[column] = pd.to_numeric(orders[column], errors="coerce")
    orders["discount_pct"] = orders["discount_pct"].fillna(0)
    orders.loc[orders["delivery_minutes"] < 0, "delivery_minutes"] = np.nan
    catalogue_price = products.set_index("product_id")["price_inr"]
    missing_amount = orders["order_amount"].isna()
    calculated = (orders.loc[missing_amount, "product_id"].map(catalogue_price)
                  * orders.loc[missing_amount, "quantity"]
                  * (1 - orders.loc[missing_amount, "discount_pct"] / 100))
    orders.loc[missing_amount, "order_amount"] = calculated.round(2)

    category_lookup = products.set_index("product_id")["category"]
    order_category = orders["product_id"].map(category_lookup)
    is_completed = orders["status"].eq("completed")
    category_medians = orders.loc[is_completed].assign(_category=order_category[is_completed]).groupby("_category")["delivery_minutes"].median()
    fill_delivery = is_completed & orders["delivery_minutes"].isna()
    orders.loc[fill_delivery, "delivery_minutes"] = order_category[fill_delivery].map(category_medians)
    orders["delivery_minutes"] = orders["delivery_minutes"].round().astype("Int64")
    orders["order_ts"] = orders["order_ts"].dt.strftime("%Y-%m-%d %H:%M:%S")

    # Persist the same trusted tables as inspectable CSVs and SQLite tables.
    for name, frame in (("customers", customers), ("products", products), ("orders", orders)):
        frame.to_csv(CLEAN_DIR / f"{name}.csv", index=False)
    with sqlite3.connect(DATABASE) as connection:
        for name, frame in (("customers", customers), ("products", products), ("orders", orders)):
            frame.to_sql(name, connection, if_exists="replace", index=False)
    print("Cleaning changes:", {"exact_duplicate_rows_removed": exact_removed,
                                "extra_same_order_rows_removed": repeated_order_rows_removed,
                                "amounts_reconstructed": int(missing_amount.sum()),
                                "negative_delivery_values_set_missing": int(pd.to_numeric(_read_export("orders")["delivery_minutes"], errors="coerce").lt(0).sum()),
                                "completed_delivery_values_filled": int(fill_delivery.sum())})
    return {"customers": customers, "products": products, "orders": orders}


def validate():
    customers = pd.read_csv(CLEAN_DIR / "customers.csv")
    orders = pd.read_csv(CLEAN_DIR / "orders.csv", keep_default_na=False)
    products = pd.read_csv(CLEAN_DIR / "products.csv")
    completed = orders[orders.status.eq("completed")]
    cancelled = orders[orders.status.eq("cancelled")]
    amount = pd.to_numeric(orders.order_amount, errors="coerce")
    done_delivery = pd.to_numeric(completed.delivery_minutes, errors="coerce")
    cancelled_delivery = pd.to_numeric(cancelled.delivery_minutes, errors="coerce")
    assert len(customers) == 1200 and len(products) == 40
    assert customers.city.nunique() == 6 and set(customers.gender) <= {"male", "female", "unknown"}
    assert customers.age.notna().all() and customers.age.between(18, 60).all()
    assert orders.order_id.is_unique and len(orders) == 12309
    assert not orders.status.eq("pending").any()
    assert amount.notna().all() and abs(float(completed.order_amount.astype(float).sum()) - 2998668.80) < 0.01
    assert len(completed) == 11110 and len(cancelled) == 1199
    assert done_delivery.notna().all() and cancelled_delivery.isna().all()
    with sqlite3.connect(DATABASE) as connection:
        table_names = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert {"customers", "products", "orders"}.issubset(table_names)
    print("ALL CHECKPOINTS PASSED")


if __name__ == "__main__":
    audit(); clean_data(); validate()
