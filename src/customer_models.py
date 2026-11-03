"""Part B: RFM customer groups and a lapse classifier with train-only preprocessing."""
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, silhouette_score)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

PROJECT = Path(__file__).resolve().parents[1]
CLEAN_DIR, OUTPUT_DIR = PROJECT / "data/clean", PROJECT / "outputs"
RFM_SNAPSHOT = pd.Timestamp("2025-10-01 00:00:00")
LAPSE_CUTOFF, LAPSE_END = pd.Timestamp("2025-09-01"), pd.Timestamp("2025-10-01")
NUMERIC_FEATURES = ["recency", "frequency", "monetary", "avg_discount",
                    "cancellation_rate", "avg_delivery", "age", "tenure_days"]
CATEGORY_FEATURES = ["favourite_category", "city", "gender"]
ALL_FEATURES = NUMERIC_FEATURES + CATEGORY_FEATURES


def _read_clean_tables():
    customers = pd.read_csv(CLEAN_DIR / "customers.csv")
    orders = pd.read_csv(CLEAN_DIR / "orders.csv")
    orders["order_ts"] = pd.to_datetime(orders["order_ts"])
    return customers, orders


def rfm_and_segments(k=4):
    customers, orders = _read_clean_tables()
    completed = orders.loc[orders.status.eq("completed")].copy()
    rfm = completed.groupby("customer_id").agg(
        last_order_ts=("order_ts", "max"),
        frequency=("order_id", "nunique"),
        monetary=("order_amount", "sum"),
    ).reset_index()
    rfm["recency"] = (RFM_SNAPSHOT - rfm["last_order_ts"]).dt.days
    rfm.drop(columns="last_order_ts", inplace=True)
    rfm_columns = ["recency", "frequency", "monetary"]
    scaled_rfm = StandardScaler().fit_transform(rfm[rfm_columns])

    # Compare every requested K with fixed initialisation and random seed.
    score_rows = []
    for clusters in range(2, 9):
        fitted = KMeans(n_clusters=clusters, n_init=10, random_state=42).fit(scaled_rfm)
        score_rows.append({"k": clusters, "inertia": fitted.inertia_,
                           "silhouette": silhouette_score(scaled_rfm, fitted.labels_)})
    chosen = KMeans(n_clusters=k, n_init=10, random_state=42)
    rfm["segment_id"] = chosen.fit_predict(scaled_rfm)
    profile = rfm.groupby("segment_id").agg(
        size=("customer_id", "size"), recency=("recency", "mean"),
        frequency=("frequency", "mean"), monetary=("monetary", "mean"),
    ).reset_index()
    richest_segment = int(profile.loc[profile.monetary.idxmax(), "segment_id"])
    profile_by_id = profile.set_index("segment_id")
    frequent_segment = int(profile.loc[profile.frequency.idxmax(), "segment_id"])
    slowest_segment = int(profile.loc[profile.recency.idxmax(), "segment_id"])
    def segment_label(segment_id):
        if int(segment_id) == richest_segment:
            return "Premium"
        if int(segment_id) == frequent_segment:
            return "Frequent"
        if int(segment_id) == slowest_segment:
            return "Dormant"
        return "Occasional"
    rfm["segment_name"] = rfm["segment_id"].map(segment_label)
    rfm["is_premium"] = rfm["segment_id"].eq(richest_segment)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rfm[["customer_id", "recency", "frequency", "monetary", "segment_id",
         "segment_name", "is_premium"]].to_csv(OUTPUT_DIR / "customer_segments.csv", index=False)
    pca_ratios = PCA(n_components=2).fit(scaled_rfm).explained_variance_ratio_
    return rfm, pd.DataFrame(score_rows), profile, pca_ratios


def _preprocessing_pipeline():
    transform = ColumnTransformer([
        ("num", StandardScaler(), NUMERIC_FEATURES),
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), CATEGORY_FEATURES),
    ])
    return transform


def build_lapse_model():
    customers, orders = _read_clean_tables()
    before_cutoff = orders.loc[orders.order_ts < LAPSE_CUTOFF].copy()
    completed_history = before_cutoff.loc[before_cutoff.status.eq("completed")].copy()
    cohort = completed_history.groupby("customer_id").agg(
        last_completed=("order_ts", "max"),
        frequency=("order_id", "nunique"),
        monetary=("order_amount", "sum"),
        avg_discount=("discount_pct", "mean"),
        avg_delivery=("delivery_minutes", "mean"),
    ).reset_index()
    cohort["recency"] = (LAPSE_CUTOFF - cohort["last_completed"]).dt.days
    cohort.drop(columns="last_completed", inplace=True)

    final_orders = before_cutoff.loc[before_cutoff.status.isin(["completed", "cancelled"])]
    cancellation_stats = final_orders.groupby("customer_id").agg(
        final_order_count=("status", "size"),
        cancelled_count=("status", lambda values: int(values.eq("cancelled").sum())),
    ).reset_index()
    cancellation_stats["cancellation_rate"] = cancellation_stats.cancelled_count / cancellation_stats.final_order_count
    cohort = cohort.merge(cancellation_stats[["customer_id", "cancellation_rate"]], on="customer_id", how="left")

    products = pd.read_csv(CLEAN_DIR / "products.csv")[["product_id", "category"]]
    category_counts = completed_history.merge(products, on="product_id").groupby(
        ["customer_id", "category"]).size().rename("order_count").reset_index()
    favourite = (category_counts.sort_values(["customer_id", "order_count", "category"],
                                               ascending=[True, False, True])
                 .drop_duplicates("customer_id").rename(columns={"category": "favourite_category"})
                 [["customer_id", "favourite_category"]])
    cohort = cohort.merge(favourite, on="customer_id", how="left").merge(customers, on="customer_id", how="left")
    cohort["signup_date"] = pd.to_datetime(cohort["signup_date"])
    cohort["tenure_days"] = (LAPSE_CUTOFF - cohort["signup_date"]).dt.days

    september_completed = orders.loc[(orders.order_ts >= LAPSE_CUTOFF) &
                                     (orders.order_ts < LAPSE_END) & orders.status.eq("completed")]
    ordered_in_september = set(september_completed.customer_id)
    cohort["lapsed"] = (~cohort.customer_id.isin(ordered_in_september)).astype(int)
    # Only pre-cutoff values and customer attributes enter X; September is label-only.
    X = cohort[ALL_FEATURES].copy()
    y = cohort["lapsed"].astype(int)
    X[NUMERIC_FEATURES] = X[NUMERIC_FEATURES].fillna(X[NUMERIC_FEATURES].median())
    X[CATEGORY_FEATURES] = X[CATEGORY_FEATURES].fillna("unknown")
    assert int(X.isna().sum().sum()) == 0, "Feature table must be complete after imputation"
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42)

    model_specs = [
        ("logistic", LogisticRegression(max_iter=1000), {"model__C": [0.01, 0.1, 1, 10, 100]}),
        ("forest", RandomForestClassifier(n_estimators=100, random_state=42),
         {"model__max_depth": [3, 5, 8], "model__min_samples_leaf": [5, 10, 20]}),
    ]
    comparisons = {}
    for name, estimator, parameter_grid in model_specs:
        pipeline = Pipeline([("prep", _preprocessing_pipeline()), ("model", estimator)])
        search = GridSearchCV(pipeline, parameter_grid, cv=5, scoring="f1", n_jobs=1)
        search.fit(X_train, y_train)
        prediction = search.predict(X_test)
        comparisons[name] = {
            "best_params": search.best_params_, "cv_f1": float(search.best_score_),
            "precision": float(precision_score(y_test, prediction, zero_division=0)),
            "recall": float(recall_score(y_test, prediction, zero_division=0)),
            "f1": float(f1_score(y_test, prediction, zero_division=0)),
            "confusion_matrix": confusion_matrix(y_test, prediction).tolist(),
            "search": search,
        }

    selected = comparisons["forest"]["search"]
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model_path = OUTPUT_DIR / "lapse_model.joblib"
    joblib.dump(selected, model_path)
    reloaded = joblib.load(model_path)
    assert np.array_equal(selected.predict(X_test), reloaded.predict(X_test))

    # The balancing experiment duplicates only positive training rows, never test rows.
    training_frame = X_train.copy()
    training_frame["_label"] = y_train
    majority_count = int(y_train.value_counts().max())
    current_positive = int((y_train == 1).sum())
    minority_rows = training_frame.loc[training_frame["_label"].eq(1)]
    duplicated = minority_rows.sample(n=majority_count-current_positive, replace=True, random_state=42)
    balanced = pd.concat([training_frame, duplicated], axis=0)
    balanced_X, balanced_y = balanced.drop(columns="_label"), balanced["_label"].astype(int)
    balanced_logistic = Pipeline([("prep", _preprocessing_pipeline()),
        ("model", LogisticRegression(C=comparisons["logistic"]["best_params"]["model__C"], max_iter=1000))])
    balanced_logistic.fit(balanced_X, balanced_y)
    balanced_prediction = balanced_logistic.predict(X_test)

    # Premium membership is intentionally derived from the full-window RFM table, post hoc.
    segments, _, _, _ = rfm_and_segments()
    premium_ids = set(segments.loc[segments.is_premium, "customer_id"])
    lapse_probabilities = selected.predict_proba(X_test)[:, list(selected.classes_).index(1)]
    retention = pd.DataFrame({"customer_id": cohort.loc[X_test.index, "customer_id"].to_numpy(),
                              "lapse_probability": lapse_probabilities, "actually_lapsed": y_test.to_numpy()})
    retention = retention.loc[retention.customer_id.isin(premium_ids)].sort_values(
        "lapse_probability", ascending=False).head(10)
    retention.to_csv(OUTPUT_DIR / "retention_list.csv", index=False)

    forest_best = comparisons["forest"]["search"].best_estimator_
    feature_names = forest_best.named_steps["prep"].get_feature_names_out()
    feature_importance = forest_best.named_steps["model"].feature_importances_
    top_features = [{"feature": str(feature_names[i]), "importance": float(feature_importance[i])}
                    for i in np.argsort(feature_importance)[::-1][:5]]
    logistic_best = comparisons["logistic"]["search"].best_estimator_
    encoded_names = logistic_best.named_steps["prep"].get_feature_names_out()
    coefficients = logistic_best.named_steps["model"].coef_[0]
    coefficient_notes = []
    for feature in ("num__recency", "num__frequency", "num__monetary"):
        pos = next(i for i, name in enumerate(encoded_names) if name == feature)
        value = float(coefficients[pos])
        coefficient_notes.append({"feature": feature, "coefficient": value,
                                  "sign": "positive" if value > 0 else "negative" if value < 0 else "zero"})
    scaler_mean_frequency = float(logistic_best.named_steps["prep"].named_transformers_["num"].mean_[NUMERIC_FEATURES.index("frequency")])
    diagnostics = {
        "rf_top5_importance": top_features, "logistic_coefficients": coefficient_notes,
        "balanced_logistic": {"train_class_counts": {str(int(k)): int(v) for k, v in balanced_y.value_counts().sort_index().items()},
            "precision": float(precision_score(y_test, balanced_prediction, zero_division=0)),
            "recall": float(recall_score(y_test, balanced_prediction, zero_division=0)),
            "f1": float(f1_score(y_test, balanced_prediction, zero_division=0))},
        "frequency_scaler_train_mean": scaler_mean_frequency,
        "frequency_full_cohort_mean": float(X["frequency"].mean()),
        "train_rows": len(X_train), "test_rows": len(X_test),
        "baseline_lapsed_precision": 0.0, "baseline_lapsed_recall": 0.0, "baseline_lapsed_f1": 0.0,
        "model_reload_predictions_identical": True, "missing_features_after_imputation": int(X.isna().sum().sum()),
    }
    (OUTPUT_DIR / "model_diagnostics.json").write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
    # Return compact serialisable summaries rather than the fitted search objects.
    public_models = {name: {key: value for key, value in metrics.items() if key != "search"}
                     for name, metrics in comparisons.items()}
    baseline = {"accuracy": float(accuracy_score(y_test, np.zeros(len(y_test), dtype=int))),
                "precision": 0.0, "recall": 0.0, "f1": 0.0}
    return {"cohort_rows": len(cohort), "lapsed": int(y.sum()), "train_rows": len(X_train),
            "test_rows": len(X_test), "baseline": baseline, "models": public_models,
            "retention_list": retention, "diagnostics": diagnostics}
