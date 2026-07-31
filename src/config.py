"""Shared variables (from notebook 04)."""
 
HORIZON = 42
N_TEST_ORIGINS = 6
NUM_BOOST_ROUND = 600
 
PARAMS = dict(
    objective="regression",
    metric="l2",
    learning_rate=0.05,
    num_leaves=127,
    min_data_in_leaf=100,
    feature_fraction=0.8,
    bagging_fraction=0.8,
    bagging_freq=1,
    verbosity=-1,
    n_jobs=-1,
    seed=42,
)
 
CATEGORICAL = ["Store", "StoreType", "Assortment", "StateHoliday", "DayOfWeek"]
 
FEATURES = [
    # Last known state at origin
    "asof_mean_7d", "asof_mean_28d", "asof_mean_91d", "asof_std_28d",
    "asof_ratio_7_28", "asof_ratio_28_91", "asof_cv_28", "days_since_snap",
    # horizon
    "h", "h_week",
    # same day the previous year
    "sales_ly",
    # date and promotion related features (known in advance)
    "DayOfWeek", "Promo", "SchoolHoliday", "StateHoliday", "month", "day", "weekofyear",
    # store-related features
    "Store", "StoreType", "Assortment", "CompetitionDistance", "Promo2",
]
 
# two variants usable via command line
VARIANTS = {
    "full": FEATURES,
    "no_ly": [f for f in FEATURES if f != "sales_ly"],
}
 
TRACKING_URI = "sqlite:///mlflow.db"
EXPERIMENT_NAME = "rossmann_04_production"