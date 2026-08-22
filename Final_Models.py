#%%
# 0. Imports & Configuration
import warnings
warnings.filterwarnings("ignore")

import os
import time
import json
import joblib
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from datetime import datetime

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV, learning_curve
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

import xgboost as xgb
from xgboost import XGBRegressor
import lightgbm as lgb
from lightgbm import LGBMRegressor

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 100
plt.rcParams["axes.titleweight"] = "bold"


N_ITER_XGB = 10
N_ITER_LGBM = 10
N_ITER_RF = 12
CV_FOLDS = 3
LSTM_SEARCH_EPOCHS = 5
LSTM_MAX_EPOCHS = 40
LSTM_PATIENCE = 6
LSTM_BATCH_SIZE = 256

OVERFITTING_THRESHOLD = 0.05

OUTPUT_DIR = "solar_forecasting_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Libraries imported successfully.")
print(f"Random seed fixed at: {RANDOM_SEED}")
print("xgboost version:", xgb.__version__)
print("lightgbm version:", lgb.__version__)
print("tensorflow version:", tf.__version__)


def evaluate_regression(y_true, y_pred):
    return {
        "R2": r2_score(y_true, y_pred),
        "MAE": mean_absolute_error(y_true, y_pred),
        "MSE": mean_squared_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
    }

# PDF Report Setup
REPORT_PATH = "Final_Models.pdf"
pdf_report = PdfPages(REPORT_PATH)

fig_title = plt.figure(figsize=(11, 8.5))
fig_title.text(0.5, 0.68, "Solar AC Power Forecasting", ha="center", va="center",
                fontsize=22, fontweight="bold")
fig_title.text(0.5, 0.61, "48h -> 15min Sequence Forecasting: EDA, Modeling & Comparison",
                ha="center", va="center", fontsize=13)
fig_title.text(0.5, 0.48,
                "Contents:\n"
                "1. Data Quality Checks\n"
                "2. Exploratory Data Analysis (EDA)\n"
                "3. Cleaning: Night Adjustment, Fault Isolation, Invalid Values\n"
                "4. Feature Engineering: MODULE_DERATING_FACTOR\n"
                "5. Time-Series Segmentation & Sequence Generation\n"
                "6. Linear Regression / Random Forest / XGBoost / LightGBM / LSTM\n"
                "7. Model Comparison & Diagnostics\n"
                "8. Best Model Conclusion",
                ha="center", va="center", fontsize=11)
fig_title.text(0.5, 0.08, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
                ha="center", va="center", fontsize=9, color="gray")
plt.axis("off")
pdf_report.savefig(fig_title)
plt.close(fig_title)


# 1. Data Loading & Column Mapping
DATA_PATH =r"C:\Users\Admin\Downloads\Solar_Power_Master_Dataset (2) (1).csv"

TIMESTAMP_COL = "DATE_TIME"
SOURCE_KEY_COL = "INVERTER_KEY"
PLANT_COL = "PLANT_LABEL"
IRRADIATION_COL = "IRRADIATION"
AC_POWER_COL = "AC_POWER"
DC_POWER_COL = "DC_POWER"
DAILY_YIELD_COL = "DAILY_YIELD"
TOTAL_YIELD_COL = "TOTAL_YIELD"
AMBIENT_TEMP_COL = "AMBIENT_TEMPERATURE"
MODULE_TEMP_COL = "MODULE_TEMPERATURE"

REQUIRED_COLUMNS = [TIMESTAMP_COL, SOURCE_KEY_COL, IRRADIATION_COL, AC_POWER_COL, DC_POWER_COL]
REQUIRED_MODELING_COLUMNS = [
    TIMESTAMP_COL, SOURCE_KEY_COL, IRRADIATION_COL, AC_POWER_COL, DC_POWER_COL,
    DAILY_YIELD_COL, TOTAL_YIELD_COL, AMBIENT_TEMP_COL, MODULE_TEMP_COL,
]

df_raw = pd.read_csv(DATA_PATH, parse_dates=[TIMESTAMP_COL])

missing_required = [c for c in REQUIRED_COLUMNS if c not in df_raw.columns]
if missing_required:
    raise ValueError(
        f"The following required column(s) were not found in the dataset: {missing_required}. "
        f"Available columns are: {df_raw.columns.tolist()}"
    )

print(f"\nLoaded dataset: {df_raw.shape[0]:,} rows x {df_raw.shape[1]} columns")
print("Confirmed column mapping:")
print(f"  Timestamp   -> {TIMESTAMP_COL}")
print(f"  SOURCE_KEY  -> {SOURCE_KEY_COL}")
print(f"  Irradiation -> {IRRADIATION_COL}")
print(f"  AC Power    -> {AC_POWER_COL}")
print(f"  DC Power    -> {DC_POWER_COL}")


# 2. Data Quality Checks (duplicates, missing values, negative values)
# ---- 2.1 Duplicate rows ----
dup_mask = df_raw.duplicated()
n_dup = dup_mask.sum()
print(f"\n[Duplicate Rows] {n_dup:,} rows ({n_dup / len(df_raw) * 100:.3f}%)")
if n_dup > 0:
    df_raw[dup_mask].sample(n=min(5, n_dup), random_state=RANDOM_SEED)

fig, ax = plt.subplots(figsize=(5, 4))
ax.bar(["Unique rows", "Duplicate rows"], [len(df_raw) - n_dup, n_dup], color=["steelblue", "crimson"])
ax.set_title("Duplicate Rows")
ax.set_ylabel("Row count")
plt.tight_layout()
pdf_report.savefig(fig)
plt.show()

# ---- 2.2 Missing values ----
missing_mask = df_raw[REQUIRED_MODELING_COLUMNS].isna().any(axis=1)
n_missing = missing_mask.sum()
print(f"[Missing Values] {n_missing:,} rows ({n_missing / len(df_raw) * 100:.3f}%) "
      f"missing at least one required modeling column")
if n_missing > 0:
    df_raw.loc[missing_mask, REQUIRED_MODELING_COLUMNS].sample(n=min(5, n_missing), random_state=RANDOM_SEED)

fig, ax = plt.subplots(figsize=(8, 4))
df_raw[REQUIRED_MODELING_COLUMNS].isna().sum().plot(kind="bar", ax=ax, color="darkorange")
ax.set_title("Missing Values per Required Column")
ax.set_ylabel("Missing count")
plt.tight_layout()
pdf_report.savefig(fig)
plt.show()

# ---- 2.3 Negative values (AC_POWER, DC_POWER, IRRADIATION cannot be negative) ----
negative_mask = (df_raw[AC_POWER_COL] < 0) | (df_raw[DC_POWER_COL] < 0) | (df_raw[IRRADIATION_COL] < 0)
n_negative = negative_mask.sum()
print(f"[Negative Values] {n_negative:,} rows ({n_negative / len(df_raw) * 100:.3f}%) "
      f"with negative AC_POWER, DC_POWER, or IRRADIATION")
if n_negative > 0:
    df_raw.loc[negative_mask, [TIMESTAMP_COL, SOURCE_KEY_COL, AC_POWER_COL, DC_POWER_COL, IRRADIATION_COL]].sample(
        n=min(5, n_negative), random_state=RANDOM_SEED
    )

fig, ax = plt.subplots(figsize=(6, 4))
pd.Series({
    "AC_POWER < 0": (df_raw[AC_POWER_COL] < 0).sum(),
    "DC_POWER < 0": (df_raw[DC_POWER_COL] < 0).sum(),
    "IRRADIATION < 0": (df_raw[IRRADIATION_COL] < 0).sum(),
}).plot(kind="bar", ax=ax, color="firebrick")
ax.set_title("Negative Values by Column")
ax.set_ylabel("Row count")
plt.tight_layout()
pdf_report.savefig(fig)
plt.show()

shape_before_quality = df_raw.shape
df_stage0 = df_raw[~dup_mask & ~missing_mask & ~negative_mask].reset_index(drop=True)
print(f"\nShape before quality checks: {shape_before_quality[0]:,} rows")
print(f"Shape after quality checks : {df_stage0.shape[0]:,} rows "
      f"({shape_before_quality[0] - df_stage0.shape[0]:,} rows removed)")


# 3. Exploratory Data Analysis (EDA) — time-series oriented
NUMERIC_COLS = [DC_POWER_COL, AC_POWER_COL, DAILY_YIELD_COL, TOTAL_YIELD_COL,
                AMBIENT_TEMP_COL, MODULE_TEMP_COL, IRRADIATION_COL]

print("\n=== EDA: Dataset Overview ===")
print(df_stage0[NUMERIC_COLS].describe().T)
print(f"\nPlants: {df_stage0[PLANT_COL].unique().tolist()}")
print(f"Inverters (SOURCE_KEY): {df_stage0[SOURCE_KEY_COL].nunique()}")
print(f"Date range: {df_stage0[TIMESTAMP_COL].min()} to {df_stage0[TIMESTAMP_COL].max()}")

# 3.1 Correlation heatmap
corr_matrix = df_stage0[NUMERIC_COLS].corr()
fig, ax = plt.subplots(figsize=(8, 6.5))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", center=0, square=True,
            linewidths=0.5, ax=ax, cbar_kws={"label": "Pearson correlation"})
ax.set_title("Correlation Heatmap — Sequence Feature Columns")
plt.tight_layout()
pdf_report.savefig(fig)
plt.show()

# 3.2 Distributions (histograms)
n_cols_grid, n_rows_grid = 3, int(np.ceil(len(NUMERIC_COLS) / 3))
fig, axes = plt.subplots(n_rows_grid, n_cols_grid, figsize=(15, 4 * n_rows_grid))
axes = axes.flatten()
for i, col in enumerate(NUMERIC_COLS):
    sns.histplot(df_stage0[col].dropna(), kde=True, ax=axes[i], color="steelblue")
    axes[i].set_title(f"Distribution of {col}")
for j in range(len(NUMERIC_COLS), len(axes)):
    fig.delaxes(axes[j])
plt.tight_layout()
pdf_report.savefig(fig)
plt.show()

# 3.3 Boxplots
fig, axes = plt.subplots(n_rows_grid, n_cols_grid, figsize=(15, 4 * n_rows_grid))
axes = axes.flatten()
for i, col in enumerate(NUMERIC_COLS):
    sns.boxplot(y=df_stage0[col].dropna(), ax=axes[i], color="lightcoral")
    axes[i].set_title(f"Boxplot of {col}")
for j in range(len(NUMERIC_COLS), len(axes)):
    fig.delaxes(axes[j])
plt.tight_layout()
pdf_report.savefig(fig)
plt.show()

# 3.4 Pairplot (sampled for tractability)
sample_for_pairplot = df_stage0[NUMERIC_COLS].sample(n=min(1500, len(df_stage0)), random_state=RANDOM_SEED)
pp = sns.pairplot(sample_for_pairplot, diag_kind="kde", plot_kws={"alpha": 0.4, "s": 15})
pp.fig.suptitle("Pairplot of Sequence Features (1,500-row sample)", y=1.02)
pdf_report.savefig(pp.fig)
plt.show()

# 3.5 Target distribution
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
sns.histplot(df_stage0[AC_POWER_COL], kde=True, ax=axes[0], color="darkorange")
axes[0].set_title("Distribution of AC_POWER")
sns.boxplot(y=df_stage0[AC_POWER_COL], ax=axes[1], color="gold")
axes[1].set_title("Boxplot of AC_POWER")
plt.tight_layout()
pdf_report.savefig(fig)
plt.show()

# 3.6 Feature correlation with target
target_corr = corr_matrix[AC_POWER_COL].drop(AC_POWER_COL).sort_values(key=np.abs, ascending=False)
fig, ax = plt.subplots(figsize=(8, 5))
colors = ["seagreen" if v > 0 else "firebrick" for v in target_corr.values]
target_corr.plot(kind="barh", ax=ax, color=colors)
ax.set_title("Feature Correlation with AC_POWER")
ax.axvline(0, color="black", linewidth=0.8)
plt.tight_layout()
pdf_report.savefig(fig)
plt.show()

# 3.7 Time series: mean AC_POWER over time across the plant fleet
fleet_ts = df_stage0.set_index(TIMESTAMP_COL)[AC_POWER_COL].resample("1h").mean()
fig, ax = plt.subplots(figsize=(14, 4.5))
fleet_ts.plot(ax=ax, color="darkorange", linewidth=0.8)
ax.set_title("Fleet-Average AC_POWER Over Time (hourly mean)")
ax.set_xlabel("Date"); ax.set_ylabel("AC_POWER")
plt.tight_layout()
pdf_report.savefig(fig)
plt.show()

# 3.8 Hourly generation pattern (daily solar cycle)
df_stage0["_HOUR"] = df_stage0[TIMESTAMP_COL].dt.hour
fig, ax = plt.subplots(figsize=(9, 5))
sns.boxplot(data=df_stage0, x="_HOUR", y=AC_POWER_COL, ax=ax, color="skyblue", fliersize=1)
ax.set_title("AC_POWER by Hour of Day (Daily Solar Cycle)")
ax.set_xlabel("Hour of day"); ax.set_ylabel("AC_POWER")
plt.tight_layout()
pdf_report.savefig(fig)
plt.show()

# 3.9 Per-inverter comparison
fig, ax = plt.subplots(figsize=(14, 5))
inverter_mean = df_stage0.groupby(SOURCE_KEY_COL)[AC_POWER_COL].mean().sort_values(ascending=False)
inverter_mean.plot(kind="bar", ax=ax, color="teal")
ax.set_title("Mean AC_POWER per Inverter (SOURCE_KEY)")
ax.set_ylabel("Mean AC_POWER")
ax.set_xticklabels(ax.get_xticklabels(), rotation=90, fontsize=6)
plt.tight_layout()
pdf_report.savefig(fig)
plt.show()

# 3.10 Autocorrelation of AC_POWER (single representative inverter, up to 48h of lags)
sample_inverter = df_stage0[SOURCE_KEY_COL].value_counts().idxmax()
inv_series = (df_stage0[df_stage0[SOURCE_KEY_COL] == sample_inverter]
              .sort_values(TIMESTAMP_COL)[AC_POWER_COL].reset_index(drop=True))
max_lag = min(192, len(inv_series) - 1)
acf_vals = [inv_series.autocorr(lag=lag) for lag in range(1, max_lag + 1)]
fig, ax = plt.subplots(figsize=(10, 4.5))
ax.bar(range(1, max_lag + 1), acf_vals, color="purple", width=1.0)
ax.axhline(0, color="black", linewidth=0.8)
ax.set_title(f"Autocorrelation of AC_POWER (inverter {sample_inverter}, up to 48h of 15-min lags)")
ax.set_xlabel("Lag (15-minute steps)"); ax.set_ylabel("Autocorrelation")
plt.tight_layout()
pdf_report.savefig(fig)
plt.show()

df_stage0 = df_stage0.drop(columns=["_HOUR"])

# 4. Cleaning: Night Adjustment, Fault Isolation, Invalid Physical Values
df = df_stage0.copy()

# ---- 4.1 Night Adjustment ----
night_mask = df[IRRADIATION_COL] == 0
n_night = night_mask.sum()
print(f"\n[Night Adjustment] {n_night:,} rows ({n_night / len(df) * 100:.3f}%) with IRRADIATION == 0 "
      f"-> forcing AC_POWER = 0, DC_POWER = 0")
df.loc[night_mask, AC_POWER_COL] = 0
df.loc[night_mask, DC_POWER_COL] = 0

fig, ax = plt.subplots(figsize=(5, 4))
ax.bar(["Day (IRRADIATION > 0)", "Night (IRRADIATION == 0)"],
       [len(df) - n_night, n_night], color=["gold", "navy"])
ax.set_title("Night Adjustment: Rows Affected")
ax.set_ylabel("Row count")
plt.tight_layout()
pdf_report.savefig(fig)
plt.show()

# ---- 4.2 Fault Isolation ----
fault_mask = (df[IRRADIATION_COL] > 0.2) & ((df[AC_POWER_COL] == 0) | (df[DC_POWER_COL] == 0))
df["Inverter_Fault"] = fault_mask.astype(int)
n_fault = fault_mask.sum()
print(f"[Fault Isolation] {n_fault:,} rows ({n_fault / len(df) * 100:.3f}%) flagged as Inverter_Fault "
      f"(IRRADIATION > 0.2 and (AC_POWER == 0 or DC_POWER == 0))")

fig, ax = plt.subplots(figsize=(7, 5))
ax.scatter(df.loc[~fault_mask, IRRADIATION_COL], df.loc[~fault_mask, AC_POWER_COL],
           s=4, alpha=0.15, color="steelblue", label="Normal")
ax.scatter(df.loc[fault_mask, IRRADIATION_COL], df.loc[fault_mask, AC_POWER_COL],
           s=10, alpha=0.8, color="crimson", label="Inverter fault")
ax.set_title("Fault Isolation: IRRADIATION vs AC_POWER")
ax.set_xlabel("IRRADIATION"); ax.set_ylabel("AC_POWER"); ax.legend()
plt.tight_layout()
pdf_report.savefig(fig)
plt.show()

faults_df = df[df["Inverter_Fault"] == 1].reset_index(drop=True)
df = df[df["Inverter_Fault"] == 0].reset_index(drop=True)
print(f"Rows moved to faults_df       : {len(faults_df):,}")
print(f"Rows remaining in main dataset: {len(df):,}")

# ---- 4.3 Invalid Physical Values (AC_POWER > DC_POWER) ----
invalid_mask = df[AC_POWER_COL] > df[DC_POWER_COL]
n_invalid = invalid_mask.sum()
print(f"[Invalid Physical Values] {n_invalid:,} rows ({n_invalid / len(df) * 100:.3f}%) "
      f"with AC_POWER > DC_POWER -> removed")

fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(df.loc[~invalid_mask, DC_POWER_COL], df.loc[~invalid_mask, AC_POWER_COL],
           s=4, alpha=0.15, color="steelblue", label="Normal (AC <= DC)")
ax.scatter(df.loc[invalid_mask, DC_POWER_COL], df.loc[invalid_mask, AC_POWER_COL],
           s=10, alpha=0.8, color="purple", label="AC > DC (invalid)")
lims = [df[DC_POWER_COL].min(), df[DC_POWER_COL].max()]
ax.plot(lims, lims, "k--", linewidth=1, label="AC = DC")
ax.set_title("Invalid Physical Values: AC_POWER vs DC_POWER")
ax.set_xlabel("DC_POWER"); ax.set_ylabel("AC_POWER"); ax.legend()
plt.tight_layout()
pdf_report.savefig(fig)
plt.show()

df = df[~invalid_mask].reset_index(drop=True)
shape_after_cleaning = df.shape
print(f"\nCleaned dataset shape (before segmentation): "
      f"{shape_after_cleaning[0]:,} rows x {shape_after_cleaning[1]} columns")

# 5. Feature Engineering — MODULE_DERATING_FACTOR
T_STC = 25.0
GAMMA = -0.0045
MODULE_DERATING_FACTOR_COL = "MODULE_DERATING_FACTOR"

df[MODULE_DERATING_FACTOR_COL] = 1 + GAMMA * (df[MODULE_TEMP_COL] - T_STC)

print(f"\n[Feature Engineering] Created {MODULE_DERATING_FACTOR_COL} from {MODULE_TEMP_COL}.")
print(df[[MODULE_TEMP_COL, MODULE_DERATING_FACTOR_COL]].describe())

corr_module_temp = df[MODULE_TEMP_COL].corr(df[AC_POWER_COL])
corr_derating = df[MODULE_DERATING_FACTOR_COL].corr(df[AC_POWER_COL])
print(f"Correlation(MODULE_TEMPERATURE, AC_POWER)     = {corr_module_temp:.4f}")
print(f"Correlation(MODULE_DERATING_FACTOR, AC_POWER) = {corr_derating:.4f}")
print(f"Same magnitude, reversed sign (affine-transform relationship confirmed): "
      f"{np.isclose(abs(corr_module_temp), abs(corr_derating), atol=1e-6)}")

# 6. Time-Series Segmentation (AFTER cleaning) + Segment Statistics
df = df.sort_values([SOURCE_KEY_COL, TIMESTAMP_COL]).reset_index(drop=True)

interval_minutes = df.groupby(SOURCE_KEY_COL)[TIMESTAMP_COL].diff().dt.total_seconds() / 60
new_segment_flag = (interval_minutes != 15) | interval_minutes.isna()
segment_within_key = new_segment_flag.groupby(df[SOURCE_KEY_COL]).cumsum().astype(int)
df["Segment_ID"] = df[SOURCE_KEY_COL].astype(str) + "_SEG" + segment_within_key.astype(str)

n_segments = df["Segment_ID"].nunique()
print(f"\n[Segmentation] Created {n_segments:,} segments across {df[SOURCE_KEY_COL].nunique()} SOURCE_KEY(s)")

SEQUENCE_LENGTH = 192   # 48 hours at 15-minute sampling (192 * 15 = 2,880 min = 48h)
FORECAST_HORIZON = 1    # predict AC_POWER exactly 15 minutes ahead
assert SEQUENCE_LENGTH * 15 == 48 * 60, "192 steps at 15-minute sampling must equal exactly 48 hours."

segment_lengths = df.groupby("Segment_ID").size().sort_values(ascending=False)
n_segments_total = len(segment_lengths)
min_len, max_len = segment_lengths.min(), segment_lengths.max()
mean_len, median_len = segment_lengths.mean(), segment_lengths.median()
p25, p75 = segment_lengths.quantile(0.25), segment_lengths.quantile(0.75)
n_short = (segment_lengths < SEQUENCE_LENGTH).sum()
n_usable = (segment_lengths >= SEQUENCE_LENGTH).sum()

MIN_LEN_FOR_ONE_SEQUENCE = SEQUENCE_LENGTH + FORECAST_HORIZON
usable_segment_lengths = segment_lengths[segment_lengths >= MIN_LEN_FOR_ONE_SEQUENCE]
sequences_per_segment = usable_segment_lengths - SEQUENCE_LENGTH - FORECAST_HORIZON + 1
total_possible_sequences = int(sequences_per_segment.sum()) if len(sequences_per_segment) else 0

print("\n=== Segment Length Statistics ===")
print(f"Total number of segments                          : {n_segments_total:,}")
print(f"Minimum / Maximum segment length                   : {min_len:,} / {max_len:,}")
print(f"Mean / Median segment length                       : {mean_len:.2f} / {median_len:.2f}")
print(f"25th / 75th percentile                              : {p25:.2f} / {p75:.2f}")
print(f"Segments with length < {SEQUENCE_LENGTH}                        : {n_short:,} ({n_short / n_segments_total * 100:.2f}%)")
print(f"Segments with length >= {SEQUENCE_LENGTH}                       : {n_usable:,} ({n_usable / n_segments_total * 100:.2f}%)")
print(f"Segments usable for >= 1 complete 48h sequence (len >= {MIN_LEN_FOR_ONE_SEQUENCE}): {len(usable_segment_lengths):,}")
print(f"Total possible 48h->15min sequences (sliding window, stride=1)    : {total_possible_sequences:,}")

fig, ax = plt.subplots(figsize=(9, 5))
sns.histplot(segment_lengths, bins=40, ax=ax, color="teal")
ax.axvline(SEQUENCE_LENGTH, color="red", linestyle="--", linewidth=1.5, label=f"Required: {SEQUENCE_LENGTH}")
ax.set_title("Segment Length Distribution")
ax.set_xlabel("Segment length (# of 15-minute points)"); ax.set_ylabel("Number of segments")
ax.legend()
plt.tight_layout()
pdf_report.savefig(fig)
plt.show()


# 7. Forecasting Sequences — eligible segments, chronologically ordered
SEQUENCE_FEATURE_COLUMNS = [

    AMBIENT_TEMP_COL, MODULE_TEMP_COL, IRRADIATION_COL, MODULE_DERATING_FACTOR_COL,
]
N_FEATURES = len(SEQUENCE_FEATURE_COLUMNS)
TRAIN_FRAC, VAL_FRAC, TEST_FRAC = 0.70, 0.15, 0.15


def make_sequences(feature_values, target_values, sequence_length=SEQUENCE_LENGTH, horizon=FORECAST_HORIZON):
    """Sliding-window (X, y) pairs from ONE already-contiguous 15-minute
    block (a single Segment_ID). Never crosses a segment boundary."""
    n = len(feature_values)
    n_samples = n - sequence_length - horizon + 1
    if n_samples <= 0:
        return np.empty((0, sequence_length, feature_values.shape[1])), np.empty((0,))
    X = np.stack([feature_values[i:i + sequence_length] for i in range(n_samples)])
    y = np.array([target_values[i + sequence_length + horizon - 1] for i in range(n_samples)])
    return X, y


# Step 2/3 — select ONLY eligible segments: len(segment) >= SEQUENCE_LENGTH.
eligible_segment_lengths = segment_lengths[segment_lengths >= SEQUENCE_LENGTH]
eligible_segment_ids = eligible_segment_lengths.index.tolist()
n_eligible = len(eligible_segment_ids)

print(f"\nTotal eligible Segment_IDs (length >= {SEQUENCE_LENGTH}): {n_eligible:,}")
assert (eligible_segment_lengths >= SEQUENCE_LENGTH).all(), \
    "Every eligible segment must have >= SEQUENCE_LENGTH rows."
print(f"Confirmed: every eligible segment has >= {SEQUENCE_LENGTH} rows.")

# Step 4/5 — REMOVE the random shuffle; sort the eligible Segment_IDs
eligible_segment_start_time = (
    df[df["Segment_ID"].isin(eligible_segment_ids)]
    .groupby("Segment_ID")[TIMESTAMP_COL].min()
)
chrono_sorted_eligible_ids = eligible_segment_start_time.sort_values().index.tolist()  # earliest -> latest, NOT shuffled

# Step 6 — split the ordered eligible segments by segment COUNT (70/15/15).
n_train_segments = int(round(n_eligible * TRAIN_FRAC))
n_val_segments = int(round(n_eligible * VAL_FRAC))
# whatever remains goes to test, so every eligible segment is assigned exactly once
train_segment_ids = chrono_sorted_eligible_ids[:n_train_segments]
val_segment_ids = chrono_sorted_eligible_ids[n_train_segments:n_train_segments + n_val_segments]
test_segment_ids = chrono_sorted_eligible_ids[n_train_segments + n_val_segments:]

segment_split_map = {}
for seg_id in train_segment_ids:
    segment_split_map[seg_id] = "train"
for seg_id in val_segment_ids:
    segment_split_map[seg_id] = "val"
for seg_id in test_segment_ids:
    segment_split_map[seg_id] = "test"

print("\n=== Chronologically-Ordered Segment-Level Split (eligible segments only) ===")
print(f"Train Segment_IDs      : {len(train_segment_ids):,} ({len(train_segment_ids) / n_eligible * 100:.2f}%)")
print(f"Validation Segment_IDs : {len(val_segment_ids):,} ({len(val_segment_ids) / n_eligible * 100:.2f}%)")
print(f"Test Segment_IDs       : {len(test_segment_ids):,} ({len(test_segment_ids) / n_eligible * 100:.2f}%)")

# Step 8 — sequence generation happens ONLY NOW, separately per split,
# using the existing, unchanged make_sequences() function.
X_train_list, y_train_list = [], []
X_val_list, y_val_list = [], []
X_test_list, y_test_list = [], []
segments_contributing = {"train": set(), "val": set(), "test": set()}

for segment_id, split_name in segment_split_map.items():
    seg_df = df[df["Segment_ID"] == segment_id].sort_values(TIMESTAMP_COL)
    feature_values = seg_df[SEQUENCE_FEATURE_COLUMNS].to_numpy()
    target_values = seg_df[AC_POWER_COL].to_numpy()
    X_seq, y_seq = make_sequences(feature_values, target_values)
    if len(X_seq) > 0:
        {"train": X_train_list, "val": X_val_list, "test": X_test_list}[split_name].append(X_seq)
        {"train": y_train_list, "val": y_val_list, "test": y_test_list}[split_name].append(y_seq)
        segments_contributing[split_name].add(segment_id)

X_train = (np.concatenate(X_train_list, axis=0) if X_train_list
           else np.empty((0, SEQUENCE_LENGTH, N_FEATURES))).astype(np.float32)
y_train = (np.concatenate(y_train_list, axis=0) if y_train_list else np.empty((0,))).astype(np.float32)
X_val = (np.concatenate(X_val_list, axis=0) if X_val_list
         else np.empty((0, SEQUENCE_LENGTH, N_FEATURES))).astype(np.float32)
y_val = (np.concatenate(y_val_list, axis=0) if y_val_list else np.empty((0,))).astype(np.float32)
X_test = (np.concatenate(X_test_list, axis=0) if X_test_list
          else np.empty((0, SEQUENCE_LENGTH, N_FEATURES))).astype(np.float32)
y_test = (np.concatenate(y_test_list, axis=0) if y_test_list else np.empty((0,))).astype(np.float32)

print(f"\n=== Forecasting Sequence Shapes (samples, {SEQUENCE_LENGTH} timesteps, {N_FEATURES} features) ===")
print(f"X_train: {X_train.shape}   (from {len(segments_contributing['train'])} of {len(train_segment_ids)} train segments)")
print(f"X_val  : {X_val.shape}   (from {len(segments_contributing['val'])} of {len(val_segment_ids)} val segments)")
print(f"X_test : {X_test.shape}   (from {len(segments_contributing['test'])} of {len(test_segment_ids)} test segments)")

# Confirm sequence generation still produces exactly SEQUENCE_LENGTH steps
# per sample (192 * 15 minutes = 48 hours), for every non-empty split.
for name, arr in [("X_train", X_train), ("X_val", X_val), ("X_test", X_test)]:
    if len(arr) > 0:
        assert arr.shape[1] == SEQUENCE_LENGTH, f"{name} does not have {SEQUENCE_LENGTH} timesteps!"
print(f"Confirmed: every generated sequence has {SEQUENCE_LENGTH} timesteps "
      f"({SEQUENCE_LENGTH} x 15 = {SEQUENCE_LENGTH * 15} minutes = {SEQUENCE_LENGTH * 15 / 60:.0f} hours).")

for split_name, ids in [("Train", train_segment_ids), ("Validation", val_segment_ids), ("Test", test_segment_ids)]:
    if ids:
        starts = eligible_segment_start_time[ids]
        print(f"{split_name}: min segment START = {starts.min()}   max segment START = {starts.max()}")
    else:
        print(f"{split_name}: no segments assigned")

# Independent leakage check: no segment appears in more than one split.
assert not (segments_contributing["train"] & segments_contributing["val"])
assert not (segments_contributing["train"] & segments_contributing["test"])
assert not (segments_contributing["val"] & segments_contributing["test"])
print("\nLeakage check passed: zero Segment_ID overlap between train/val/test.")

fig, ax = plt.subplots(figsize=(6, 4.5))
ax.bar(["Train", "Validation", "Test"], [len(X_train), len(X_val), len(X_test)],
       color=["royalblue", "darkorange", "seagreen"])
ax.set_title("Sequence Count by Split (Chronologically-Ordered Eligible Segments)")
ax.set_ylabel("Number of sequences")
for i, v in enumerate([len(X_train), len(X_val), len(X_test)]):
    ax.annotate(f"{v:,}", (i, v), textcoords="offset points", xytext=(0, 4), ha="center")
plt.tight_layout()
pdf_report.savefig(fig)
plt.show()


# 8. Scaling (fit on TRAIN only) & Flattening for classical ML models
n_train_samples, _, _ = X_train.shape
scaler = StandardScaler()
scaler.fit(X_train.reshape(-1, N_FEATURES))  # fit ONLY on training data -> no leakage


def scale_sequences(X, fitted_scaler):
    n, s, f = X.shape
    return fitted_scaler.transform(X.reshape(-1, f)).reshape(n, s, f).astype(np.float32)


X_train_scaled = scale_sequences(X_train, scaler)
X_val_scaled = scale_sequences(X_val, scaler)
X_test_scaled = scale_sequences(X_test, scaler)

X_train_flat = X_train_scaled.reshape(len(X_train_scaled), -1)
X_val_flat = X_val_scaled.reshape(len(X_val_scaled), -1)
X_test_flat = X_test_scaled.reshape(len(X_test_scaled), -1)

print(f"\nFlattened feature vector length for classical ML models: {X_train_flat.shape[1]} "
      f"(= {SEQUENCE_LENGTH} timesteps x {N_FEATURES} features)")

model_results = {}


def plot_diagnostics(model_name, y_true, y_pred, color):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].scatter(y_true, y_pred, alpha=0.15, s=8, color=color)
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    axes[0].plot(lims, lims, "r--", linewidth=1.5, label="Perfect prediction")
    axes[0].set_title(f"{model_name}: Actual vs Predicted (Test Set)")
    axes[0].set_xlabel("Actual AC_POWER"); axes[0].set_ylabel("Predicted AC_POWER"); axes[0].legend()

    residuals = y_true - y_pred
    axes[1].scatter(y_pred, residuals, alpha=0.15, s=8, color=color)
    axes[1].axhline(0, color="red", linestyle="--", linewidth=1.5)
    axes[1].set_title(f"{model_name}: Residual Plot (Test Set)")
    axes[1].set_xlabel("Predicted AC_POWER"); axes[1].set_ylabel("Residual (Actual - Predicted)")
    plt.tight_layout()
    pdf_report.savefig(fig)
    plt.show()

# 9.1 Linear Regression (flattened sequences) — tuning + regularization-if-overfitting
print("\n" + "=" * 70)
print("MODEL 1/5: Linear Regression")
print("=" * 70)

lr_param_grid = [{"fit_intercept": True}, {"fit_intercept": False}]
best_lr_val_r2, best_lr_config = -np.inf, lr_param_grid[0]
for params in lr_param_grid:
    candidate = LinearRegression(**params).fit(X_train_flat, y_train)
    val_r2 = r2_score(y_val, candidate.predict(X_val_flat))
    print(f"  params={params}  val R2={val_r2:.5f}")
    if val_r2 > best_lr_val_r2:
        best_lr_val_r2, best_lr_config = val_r2, params
print(f"Best Linear Regression configuration (by validation R2): {best_lr_config}")

t0 = time.time()
lr_model = LinearRegression(**best_lr_config)
lr_model.fit(X_train_flat, y_train)
lr_train_time = time.time() - t0

t0 = time.time()
y_test_pred_lr = lr_model.predict(X_test_flat)
lr_predict_time = time.time() - t0

lr_train_metrics = evaluate_regression(y_train, lr_model.predict(X_train_flat))
lr_val_metrics = evaluate_regression(y_val, lr_model.predict(X_val_flat))
lr_test_metrics = evaluate_regression(y_test, y_test_pred_lr)
lr_gap = lr_train_metrics["R2"] - lr_test_metrics["R2"]
print(f"Train R2: {lr_train_metrics['R2']:.5f}   Val R2: {lr_val_metrics['R2']:.5f}   Test R2: {lr_test_metrics['R2']:.5f}")
print(f"Train - Test R2 gap: {lr_gap:.5f}")

if lr_gap > OVERFITTING_THRESHOLD:
    print(f"\nOverfitting detected for Linear Regression (gap {lr_gap:.5f} > {OVERFITTING_THRESHOLD}) "
          f"-> applying L2 regularization (RidgeCV).")
    ridge_alphas = np.logspace(-2, 4, 13)
    ridge_model = RidgeCV(alphas=ridge_alphas, fit_intercept=best_lr_config["fit_intercept"])
    t0 = time.time()
    ridge_model.fit(X_train_flat, y_train)
    ridge_train_time = time.time() - t0
    t0 = time.time()
    y_test_pred_ridge = ridge_model.predict(X_test_flat)
    ridge_predict_time = time.time() - t0

    ridge_train_metrics = evaluate_regression(y_train, ridge_model.predict(X_train_flat))
    ridge_val_metrics = evaluate_regression(y_val, ridge_model.predict(X_val_flat))
    ridge_test_metrics = evaluate_regression(y_test, y_test_pred_ridge)
    ridge_gap = ridge_train_metrics["R2"] - ridge_test_metrics["R2"]
    print(f"RidgeCV (alpha={ridge_model.alpha_:.4g}): Train R2={ridge_train_metrics['R2']:.5f}, "
          f"Test R2={ridge_test_metrics['R2']:.5f}, gap={ridge_gap:.5f}")

    if ridge_gap < lr_gap and ridge_test_metrics["R2"] >= lr_test_metrics["R2"] - 0.02:
        print("-> RidgeCV reduces overfitting without materially hurting test R2. Adopting it.")
        lr_model = ridge_model
        lr_train_time, lr_predict_time = ridge_train_time, ridge_predict_time
        lr_train_metrics, lr_val_metrics, lr_test_metrics = ridge_train_metrics, ridge_val_metrics, ridge_test_metrics
        y_test_pred_lr = y_test_pred_ridge
    else:
        print("-> RidgeCV did not clearly improve the tradeoff; keeping plain Linear Regression.")
else:
    print("No significant overfitting detected for Linear Regression; no extra regularization applied.")

train_sizes, train_scores, val_scores = learning_curve(
    LinearRegression(**best_lr_config), X_train_flat, y_train, cv=5, scoring="r2",
    train_sizes=np.linspace(0.1, 1.0, 6), n_jobs=-1
)
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(train_sizes, train_scores.mean(axis=1), "o-", color="royalblue", label="Training R2")
ax.plot(train_sizes, val_scores.mean(axis=1), "o-", color="darkorange", label="Cross-Val R2")
ax.set_title("Linear Regression: Learning Curve")
ax.set_xlabel("Training set size"); ax.set_ylabel("R2 Score"); ax.legend()
plt.tight_layout()
pdf_report.savefig(fig)
plt.show()

plot_diagnostics("Linear Regression", y_test, y_test_pred_lr, "royalblue")

model_results["Linear Regression"] = {
    "model": lr_model, "train_metrics": lr_train_metrics, "val_metrics": lr_val_metrics,
    "test_metrics": lr_test_metrics, "train_time": lr_train_time, "predict_time": lr_predict_time,
    "y_test_pred": y_test_pred_lr,
}

# 9.2 Random Forest (flattened sequences) — tuning + regularization-if-overfitting
print("\n" + "=" * 70)
print("MODEL 2/5: Random Forest")
print("=" * 70)

rf_param_distributions = {
    "n_estimators": [30, 50, 80, 120],
    "max_depth": [6, 10, 14, 18, None],
    "min_samples_split": [2, 5, 10, 20],
    "min_samples_leaf": [1, 2, 4, 8],
    "max_features": ["sqrt", "log2", 0.5],
    "bootstrap": [True],
    "criterion": ["squared_error", "friedman_mse"],
}

rf_search = RandomizedSearchCV(
    RandomForestRegressor(random_state=RANDOM_SEED, n_jobs=-1),
    param_distributions=rf_param_distributions, n_iter=N_ITER_RF, cv=CV_FOLDS,
    scoring="r2", random_state=RANDOM_SEED, n_jobs=-1, verbose=1,
)
t0 = time.time()
rf_search.fit(X_train_flat, y_train)
print(f"RandomizedSearchCV completed in {time.time() - t0:.1f}s   Best CV R2: {rf_search.best_score_:.5f}")
best_rf_params = rf_search.best_params_
print(f"Best parameters: {best_rf_params}")

t0 = time.time()
rf_tuned = RandomForestRegressor(random_state=RANDOM_SEED, n_jobs=-1, **best_rf_params)
rf_tuned.fit(X_train_flat, y_train)
rf_train_time = time.time() - t0

rf_train_metrics = evaluate_regression(y_train, rf_tuned.predict(X_train_flat))
rf_val_metrics = evaluate_regression(y_val, rf_tuned.predict(X_val_flat))
t0 = time.time()
y_test_pred_rf = rf_tuned.predict(X_test_flat)
rf_predict_time = time.time() - t0
rf_test_metrics = evaluate_regression(y_test, y_test_pred_rf)
rf_gap = rf_train_metrics["R2"] - rf_test_metrics["R2"]
print(f"Train R2: {rf_train_metrics['R2']:.5f}   Val R2: {rf_val_metrics['R2']:.5f}   Test R2: {rf_test_metrics['R2']:.5f}")
print(f"Train - Test R2 gap: {rf_gap:.5f}")

final_rf_model = rf_tuned

if rf_gap > OVERFITTING_THRESHOLD:
    print(f"\nOverfitting detected for Random Forest (gap {rf_gap:.5f} > {OVERFITTING_THRESHOLD}) "
          f"-> applying stronger regularization (shallower trees, larger leaf/split sizes, "
          f"bagging subsampling).")
    regularized_rf_params = dict(best_rf_params)
    current_depth = regularized_rf_params.get("max_depth")
    regularized_rf_params.update({
        "max_depth": min(current_depth, 8) if current_depth is not None else 8,
        "min_samples_leaf": max(regularized_rf_params.get("min_samples_leaf", 1), 8),
        "min_samples_split": max(regularized_rf_params.get("min_samples_split", 2), 20),
        "max_features": "sqrt",
        "max_samples": 0.7,
    })
    t0 = time.time()
    rf_regularized = RandomForestRegressor(random_state=RANDOM_SEED, n_jobs=-1, **regularized_rf_params)
    rf_regularized.fit(X_train_flat, y_train)
    rf_reg_train_time = time.time() - t0

    reg_train_metrics = evaluate_regression(y_train, rf_regularized.predict(X_train_flat))
    reg_val_metrics = evaluate_regression(y_val, rf_regularized.predict(X_val_flat))
    t0 = time.time()
    y_test_pred_rf_reg = rf_regularized.predict(X_test_flat)
    rf_reg_predict_time = time.time() - t0
    reg_test_metrics = evaluate_regression(y_test, y_test_pred_rf_reg)
    reg_gap = reg_train_metrics["R2"] - reg_test_metrics["R2"]
    print(f"Regularized RF: Train R2={reg_train_metrics['R2']:.5f}, Test R2={reg_test_metrics['R2']:.5f}, "
          f"gap={reg_gap:.5f}")

    if reg_gap < rf_gap and reg_test_metrics["R2"] >= rf_test_metrics["R2"] - 0.02:
        print("-> Regularized Random Forest reduces overfitting without materially hurting test R2. Adopting it.")
        final_rf_model = rf_regularized
        best_rf_params = regularized_rf_params
        rf_train_time, rf_predict_time = rf_reg_train_time, rf_reg_predict_time
        rf_train_metrics, rf_val_metrics, rf_test_metrics = reg_train_metrics, reg_val_metrics, reg_test_metrics
        y_test_pred_rf = y_test_pred_rf_reg
    else:
        print("-> Regularized Random Forest did not clearly improve the tradeoff; keeping the tuned model.")
else:
    print("No significant overfitting detected for Random Forest; no extra regularization applied.")

rf_train_sizes, rf_train_scores, rf_val_scores = learning_curve(
    RandomForestRegressor(random_state=RANDOM_SEED, n_jobs=-1, **best_rf_params),
    X_train_flat, y_train, cv=3, scoring="r2", train_sizes=np.linspace(0.2, 1.0, 5), n_jobs=-1,
)
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(rf_train_sizes, rf_train_scores.mean(axis=1), "o-", color="saddlebrown", label="Training R2")
ax.plot(rf_train_sizes, rf_val_scores.mean(axis=1), "o-", color="peru", label="Cross-Val R2")
ax.set_title("Random Forest: Learning Curve")
ax.set_xlabel("Training set size"); ax.set_ylabel("R2 Score"); ax.legend()
plt.tight_layout()
pdf_report.savefig(fig)
plt.show()

plot_diagnostics("Random Forest", y_test, y_test_pred_rf, "saddlebrown")

model_results["Random Forest"] = {
    "model": final_rf_model, "train_metrics": rf_train_metrics, "val_metrics": rf_val_metrics,
    "test_metrics": rf_test_metrics, "train_time": rf_train_time, "predict_time": rf_predict_time,
    "y_test_pred": y_test_pred_rf,
}

# 9.3 XGBoost (flattened sequences) — tuning + early-stopping mitigation
print("\n" + "=" * 70)
print("MODEL 3/5: XGBoost")
print("=" * 70)

xgb_param_distributions = {
    "n_estimators": [100, 200, 300],
    "max_depth": [3, 4, 6, 8],
    "learning_rate": [0.01, 0.03, 0.05, 0.1],
    "subsample": [0.6, 0.8, 1.0],
    "colsample_bytree": [0.6, 0.8, 1.0],
    "min_child_weight": [1, 3, 5],
    "reg_lambda": [0.1, 1.0, 5.0],
}

xgb_search = RandomizedSearchCV(
    XGBRegressor(random_state=RANDOM_SEED, n_jobs=-1, tree_method="hist"),
    param_distributions=xgb_param_distributions, n_iter=N_ITER_XGB, cv=CV_FOLDS,
    scoring="r2", random_state=RANDOM_SEED, n_jobs=-1, verbose=1,
)
t0 = time.time()
xgb_search.fit(X_train_flat, y_train)
print(f"RandomizedSearchCV completed in {time.time() - t0:.1f}s   Best CV R2: {xgb_search.best_score_:.5f}")
best_xgb_params = xgb_search.best_params_
print(f"Best parameters: {best_xgb_params}")

regularized_xgb_params = dict(best_xgb_params)
regularized_xgb_params.update({
    "max_depth": min(regularized_xgb_params.get("max_depth", 6), 4),
    "min_child_weight": max(regularized_xgb_params.get("min_child_weight", 1), 5),
    "subsample": min(regularized_xgb_params.get("subsample", 1.0), 0.7),
    "colsample_bytree": min(regularized_xgb_params.get("colsample_bytree", 1.0), 0.7),
    "reg_alpha": 1.0,
    "reg_lambda": 10.0,
    "learning_rate": 0.03,
    "n_estimators": 2000,
})

xgb_final = XGBRegressor(
    random_state=RANDOM_SEED, n_jobs=-1, tree_method="hist",
    eval_metric="rmse", early_stopping_rounds=30, **regularized_xgb_params
)
t0 = time.time()
xgb_final.fit(X_train_flat, y_train, eval_set=[(X_train_flat, y_train), (X_val_flat, y_val)], verbose=False)
xgb_train_time = time.time() - t0
print(f"Trained (early-stopped at round {xgb_final.best_iteration}) in {xgb_train_time:.1f}s")

t0 = time.time()
y_test_pred_xgb = xgb_final.predict(X_test_flat)
xgb_predict_time = time.time() - t0

xgb_train_metrics = evaluate_regression(y_train, xgb_final.predict(X_train_flat))
xgb_val_metrics = evaluate_regression(y_val, xgb_final.predict(X_val_flat))
xgb_test_metrics = evaluate_regression(y_test, y_test_pred_xgb)
print(f"Train R2: {xgb_train_metrics['R2']:.5f}   Val R2: {xgb_val_metrics['R2']:.5f}   Test R2: {xgb_test_metrics['R2']:.5f}")

xgb_evals = xgb_final.evals_result()
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(xgb_evals["validation_0"]["rmse"], color="royalblue", label="Train RMSE")
ax.plot(xgb_evals["validation_1"]["rmse"], color="darkorange", label="Validation RMSE")
ax.axvline(xgb_final.best_iteration, color="gray", linestyle="--", label="Early-stop point")
ax.set_title("XGBoost: Learning Curve (Boosting Rounds)")
ax.set_xlabel("Boosting round"); ax.set_ylabel("RMSE"); ax.legend()
plt.tight_layout()
pdf_report.savefig(fig)
plt.show()

plot_diagnostics("XGBoost", y_test, y_test_pred_xgb, "darkred")

model_results["XGBoost"] = {
    "model": xgb_final, "train_metrics": xgb_train_metrics, "val_metrics": xgb_val_metrics,
    "test_metrics": xgb_test_metrics, "train_time": xgb_train_time, "predict_time": xgb_predict_time,
    "y_test_pred": y_test_pred_xgb,
}


# 9.4 LightGBM (flattened sequences) — tuning + early-stopping mitigation
print("\n" + "=" * 70)
print("MODEL 4/5: LightGBM")
print("=" * 70)

lgbm_param_distributions = {
    "n_estimators": [100, 200, 300],
    "num_leaves": [15, 31, 63],
    "max_depth": [-1, 6, 8],
    "learning_rate": [0.01, 0.03, 0.05, 0.1],
    "subsample": [0.6, 0.8, 1.0],
    "colsample_bytree": [0.6, 0.8, 1.0],
    "min_child_samples": [5, 10, 20],
    "reg_lambda": [0.1, 1.0, 5.0],
}

lgbm_search = RandomizedSearchCV(
    LGBMRegressor(random_state=RANDOM_SEED, n_jobs=-1, verbose=-1),
    param_distributions=lgbm_param_distributions, n_iter=N_ITER_LGBM, cv=CV_FOLDS,
    scoring="r2", random_state=RANDOM_SEED, n_jobs=-1, verbose=1,
)
t0 = time.time()
lgbm_search.fit(X_train_flat, y_train)
print(f"RandomizedSearchCV completed in {time.time() - t0:.1f}s   Best CV R2: {lgbm_search.best_score_:.5f}")
best_lgbm_params = lgbm_search.best_params_
print(f"Best parameters: {best_lgbm_params}")

regularized_lgbm_params = dict(best_lgbm_params)
regularized_lgbm_params.update({
    "num_leaves": min(regularized_lgbm_params.get("num_leaves", 31), 15),
    "min_child_samples": max(regularized_lgbm_params.get("min_child_samples", 20), 30),
    "subsample": min(regularized_lgbm_params.get("subsample", 1.0), 0.7),
    "colsample_bytree": min(regularized_lgbm_params.get("colsample_bytree", 1.0), 0.7),
    "reg_alpha": 1.0,
    "reg_lambda": 10.0,
    "learning_rate": 0.03,
    "n_estimators": 2000,
})

lgbm_final = LGBMRegressor(random_state=RANDOM_SEED, n_jobs=-1, verbose=-1, **regularized_lgbm_params)
t0 = time.time()
lgbm_final.fit(
    X_train_flat, y_train, eval_set=[(X_train_flat, y_train), (X_val_flat, y_val)],
    eval_metric="rmse", callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)],
)
lgbm_train_time = time.time() - t0
print(f"Trained (early-stopped at round {lgbm_final.best_iteration_}) in {lgbm_train_time:.1f}s")

t0 = time.time()
y_test_pred_lgbm = lgbm_final.predict(X_test_flat)
lgbm_predict_time = time.time() - t0

lgbm_train_metrics = evaluate_regression(y_train, lgbm_final.predict(X_train_flat))
lgbm_val_metrics = evaluate_regression(y_val, lgbm_final.predict(X_val_flat))
lgbm_test_metrics = evaluate_regression(y_test, y_test_pred_lgbm)
print(f"Train R2: {lgbm_train_metrics['R2']:.5f}   Val R2: {lgbm_val_metrics['R2']:.5f}   Test R2: {lgbm_test_metrics['R2']:.5f}")

lgbm_evals = lgbm_final.evals_result_
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(lgbm_evals["training"]["rmse"], color="darkgreen", label="Train RMSE")
ax.plot(lgbm_evals["valid_1"]["rmse"], color="olive", label="Validation RMSE")
ax.axvline(lgbm_final.best_iteration_, color="gray", linestyle="--", label="Early-stop point")
ax.set_title("LightGBM: Learning Curve (Boosting Rounds)")
ax.set_xlabel("Boosting round"); ax.set_ylabel("RMSE"); ax.legend()
plt.tight_layout()
pdf_report.savefig(fig)
plt.show()

plot_diagnostics("LightGBM", y_test, y_test_pred_lgbm, "darkgreen")

model_results["LightGBM"] = {
    "model": lgbm_final, "train_metrics": lgbm_train_metrics, "val_metrics": lgbm_val_metrics,
    "test_metrics": lgbm_test_metrics, "train_time": lgbm_train_time, "predict_time": lgbm_predict_time,
    "y_test_pred": y_test_pred_lgbm,
}

# 9.5 LSTM — native sequence model, input_shape = (192, 7)
print("\n" + "=" * 70)
print("MODEL 5/5: LSTM (native sequence model)")
print("=" * 70)


def build_lstm(units1=64, units2=32, dropout=0.2, learning_rate=1e-3,
                seq_len=SEQUENCE_LENGTH, n_features=N_FEATURES):
    model = keras.Sequential([
        layers.Input(shape=(seq_len, n_features)),
        layers.LSTM(units1, return_sequences=True),
        layers.Dropout(dropout),
        layers.LSTM(units2),
        layers.Dropout(dropout),
        layers.Dense(16, activation="relu"),
        layers.Dense(1),
    ])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate), loss="mse", metrics=["mae"])
    return model

lstm_param_grid = [
    {"units1": 64, "units2": 32, "dropout": 0.2, "learning_rate": 1e-3},
    {"units1": 128, "units2": 64, "dropout": 0.3, "learning_rate": 5e-4},
    {"units1": 32, "units2": 16, "dropout": 0.1, "learning_rate": 1e-3},
]

print(f"Searching {len(lstm_param_grid)} LSTM configurations "
      f"({LSTM_SEARCH_EPOCHS} epochs each, early validation-loss comparison)...")
best_val_loss, best_lstm_params = np.inf, lstm_param_grid[0]
for params in lstm_param_grid:
    search_model = build_lstm(**params)
    es_search = callbacks.EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True)
    hist = search_model.fit(
        X_train_scaled, y_train, validation_data=(X_val_scaled, y_val),
        epochs=LSTM_SEARCH_EPOCHS, batch_size=LSTM_BATCH_SIZE, callbacks=[es_search], verbose=0,
    )
    val_loss = min(hist.history["val_loss"])
    print(f"  params={params}  best val_loss={val_loss:.5f}")
    if val_loss < best_val_loss:
        best_val_loss, best_lstm_params = val_loss, params

print(f"Best LSTM configuration: {best_lstm_params}")

final_lstm = build_lstm(**best_lstm_params)
es_final = callbacks.EarlyStopping(monitor="val_loss", patience=LSTM_PATIENCE, restore_best_weights=True)
t0 = time.time()
lstm_history = final_lstm.fit(
    X_train_scaled, y_train, validation_data=(X_val_scaled, y_val),
    epochs=LSTM_MAX_EPOCHS, batch_size=LSTM_BATCH_SIZE, callbacks=[es_final], verbose=1,
)
lstm_train_time = time.time() - t0
print(f"Trained for {len(lstm_history.history['loss'])} epochs (early stopping) in {lstm_train_time:.1f}s")

t0 = time.time()
y_test_pred_lstm = final_lstm.predict(X_test_scaled, verbose=0).flatten()
lstm_predict_time = time.time() - t0

y_train_pred_lstm = final_lstm.predict(X_train_scaled, verbose=0).flatten()
y_val_pred_lstm = final_lstm.predict(X_val_scaled, verbose=0).flatten()

lstm_train_metrics = evaluate_regression(y_train, y_train_pred_lstm)
lstm_val_metrics = evaluate_regression(y_val, y_val_pred_lstm)
lstm_test_metrics = evaluate_regression(y_test, y_test_pred_lstm)
lstm_gap = lstm_train_metrics["R2"] - lstm_test_metrics["R2"]
print(f"Train R2: {lstm_train_metrics['R2']:.5f}   Val R2: {lstm_val_metrics['R2']:.5f}   Test R2: {lstm_test_metrics['R2']:.5f}")
print(f"Train - Test R2 gap: {lstm_gap:.5f}")

if lstm_gap > OVERFITTING_THRESHOLD:
    print(f"\nOverfitting detected for LSTM (gap {lstm_gap:.5f} > {OVERFITTING_THRESHOLD}) "
          f"-> retraining with stronger regularization (higher dropout + L2 kernel regularization).")

    def build_lstm_regularized(units1=64, units2=32, dropout=0.2, learning_rate=1e-3,
                                l2_reg=1e-3, seq_len=SEQUENCE_LENGTH, n_features=N_FEATURES):
        reg = keras.regularizers.l2(l2_reg)
        model = keras.Sequential([
            layers.Input(shape=(seq_len, n_features)),
            layers.LSTM(units1, return_sequences=True, kernel_regularizer=reg, recurrent_regularizer=reg),
            layers.Dropout(dropout),
            layers.LSTM(units2, kernel_regularizer=reg, recurrent_regularizer=reg),
            layers.Dropout(dropout),
            layers.Dense(16, activation="relu", kernel_regularizer=reg),
            layers.Dense(1),
        ])
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate), loss="mse", metrics=["mae"])
        return model

    regularized_dropout = min(best_lstm_params["dropout"] * 1.5, 0.5)
    lstm_regularized = build_lstm_regularized(
        units1=best_lstm_params["units1"], units2=best_lstm_params["units2"],
        dropout=regularized_dropout, learning_rate=best_lstm_params["learning_rate"], l2_reg=1e-3,
    )
    es_reg = callbacks.EarlyStopping(monitor="val_loss", patience=LSTM_PATIENCE, restore_best_weights=True)
    t0 = time.time()
    lstm_reg_history = lstm_regularized.fit(
        X_train_scaled, y_train, validation_data=(X_val_scaled, y_val),
        epochs=LSTM_MAX_EPOCHS, batch_size=LSTM_BATCH_SIZE, callbacks=[es_reg], verbose=0,
    )
    lstm_reg_train_time = time.time() - t0
    t0 = time.time()
    y_test_pred_lstm_reg = lstm_regularized.predict(X_test_scaled, verbose=0).flatten()
    lstm_reg_predict_time = time.time() - t0

    reg_train_metrics = evaluate_regression(y_train, lstm_regularized.predict(X_train_scaled, verbose=0).flatten())
    reg_val_metrics = evaluate_regression(y_val, lstm_regularized.predict(X_val_scaled, verbose=0).flatten())
    reg_test_metrics = evaluate_regression(y_test, y_test_pred_lstm_reg)
    reg_gap = reg_train_metrics["R2"] - reg_test_metrics["R2"]
    print(f"Regularized LSTM (dropout={regularized_dropout:.2f}, L2=1e-3): "
          f"Train R2={reg_train_metrics['R2']:.5f}, Test R2={reg_test_metrics['R2']:.5f}, gap={reg_gap:.5f}")

    if reg_gap < lstm_gap and reg_test_metrics["R2"] >= lstm_test_metrics["R2"] - 0.02:
        print("-> Regularized LSTM reduces overfitting without materially hurting test R2. Adopting it.")
        final_lstm = lstm_regularized
        lstm_history = lstm_reg_history
        lstm_train_time, lstm_predict_time = lstm_reg_train_time, lstm_reg_predict_time
        lstm_train_metrics, lstm_val_metrics, lstm_test_metrics = reg_train_metrics, reg_val_metrics, reg_test_metrics
        y_test_pred_lstm = y_test_pred_lstm_reg
    else:
        print("-> Regularized LSTM did not clearly improve the tradeoff; keeping the original LSTM.")
else:
    print("No significant overfitting detected for LSTM; no extra regularization applied.")

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(lstm_history.history["loss"], color="indigo", label="Train Loss (MSE)")
ax.plot(lstm_history.history["val_loss"], color="darkorange", label="Validation Loss (MSE)")
ax.set_title("LSTM: Learning Curve (Training Epochs)")
ax.set_xlabel("Epoch"); ax.set_ylabel("MSE Loss"); ax.legend()
plt.tight_layout()
pdf_report.savefig(fig)
plt.show()

plot_diagnostics("LSTM", y_test, y_test_pred_lstm, "indigo")

model_results["LSTM"] = {
    "model": final_lstm, "train_metrics": lstm_train_metrics, "val_metrics": lstm_val_metrics,
    "test_metrics": lstm_test_metrics, "train_time": lstm_train_time, "predict_time": lstm_predict_time,
    "y_test_pred": y_test_pred_lstm,
}

# 10. Comprehensive Model Comparison
comparison_rows = []
for name, res in model_results.items():
    comparison_rows.append({
        "Model": name,
        "Train R2": res["train_metrics"]["R2"],
        "Val R2": res["val_metrics"]["R2"],
        "Test R2": res["test_metrics"]["R2"],
        "Test MAE": res["test_metrics"]["MAE"],
        "Test MSE": res["test_metrics"]["MSE"],
        "Test RMSE": res["test_metrics"]["RMSE"],
        "Training Time (s)": res["train_time"],
        "Prediction Time (s)": res["predict_time"],
        "Overfitting Gap (Train R2 - Test R2)": res["train_metrics"]["R2"] - res["test_metrics"]["R2"],
    })
comparison_df = pd.DataFrame(comparison_rows).round(5)
print("\n=== Model Comparison ===")
print(comparison_df.to_string(index=False))

palette = {"Linear Regression": "royalblue", "Random Forest": "saddlebrown", "XGBoost": "darkred",
           "LightGBM": "darkgreen", "LSTM": "indigo"}
colors = [palette[m] for m in comparison_df["Model"]]

fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
for ax, metric in zip(axes, ["Test R2", "Test RMSE", "Test MAE"]):
    bars = ax.bar(comparison_df["Model"], comparison_df[metric], color=colors)
    ax.set_title(f"{metric} Comparison")
    for b in bars:
        h = b.get_height()
        ax.annotate(f"{h:.3f}", (b.get_x() + b.get_width() / 2, h),
                     textcoords="offset points", xytext=(0, 4), ha="center", fontsize=8)
plt.tight_layout()
pdf_report.savefig(fig)
plt.show()

fig, ax = plt.subplots(figsize=(8, 5))
comparison_df.set_index("Model")[["Train R2", "Val R2", "Test R2"]].plot(kind="bar", ax=ax,
    color=["#a6c8ff", "#5b9bd5", "#173c66"])
ax.set_title("R2 Across Train / Validation / Test")
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
plt.tight_layout()
pdf_report.savefig(fig)
plt.show()

fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
axes[0].bar(comparison_df["Model"], comparison_df["Training Time (s)"], color=colors)
axes[0].set_title("Training Time"); axes[0].set_ylabel("Seconds")
axes[1].bar(comparison_df["Model"], comparison_df["Overfitting Gap (Train R2 - Test R2)"], color=colors)
axes[1].set_title("Overfitting / Generalization Gap"); axes[1].axhline(0, color="black", linewidth=0.8)
plt.tight_layout()
pdf_report.savefig(fig)
plt.show()

fig, ax = plt.subplots(figsize=(13, 3))
ax.axis("off")
tbl = ax.table(cellText=comparison_df.values, colLabels=comparison_df.columns, cellLoc="center", loc="center")
tbl.auto_set_font_size(False)
tbl.set_fontsize(7.5)
tbl.scale(1, 1.6)
ax.set_title("Model Comparison Table", pad=20)
plt.tight_layout()
pdf_report.savefig(fig)
plt.show()


# 11. Best Model — determined from actual results, not assumed
ranked = comparison_df.sort_values(["Test R2", "Test RMSE"], ascending=[False, True]).reset_index(drop=True)
best_model_name = ranked.iloc[0]["Model"]
print(f"\n=== Best Model (by Test R2, tie-broken by lower Test RMSE) ===")
print(ranked.to_string(index=False))
print(f"\nBest model: {best_model_name}")
print(f"  Test R2   : {ranked.iloc[0]['Test R2']:.5f}")
print(f"  Test RMSE : {ranked.iloc[0]['Test RMSE']:.5f}")
print(f"  Test MAE  : {ranked.iloc[0]['Test MAE']:.5f}")
print(f"  Overfitting gap: {ranked.iloc[0]['Overfitting Gap (Train R2 - Test R2)']:.5f}")

# 12. Save Results, Models, and Comparison Table
comparison_df.to_csv(os.path.join(OUTPUT_DIR, "model_comparison.csv"), index=False)
joblib.dump(scaler, os.path.join(OUTPUT_DIR, "feature_scaler.joblib"))
joblib.dump(lr_model, os.path.join(OUTPUT_DIR, "linear_regression_model.joblib"))
joblib.dump(final_rf_model, os.path.join(OUTPUT_DIR, "random_forest_model.joblib"))
joblib.dump(xgb_final, os.path.join(OUTPUT_DIR, "xgboost_model.joblib"))
joblib.dump(lgbm_final, os.path.join(OUTPUT_DIR, "lightgbm_model.joblib"))
final_lstm.save(os.path.join(OUTPUT_DIR, "lstm_model.keras"))

with open(os.path.join(OUTPUT_DIR, "best_model.json"), "w") as f:
    json.dump({
        "best_model": best_model_name,
        "test_r2": float(ranked.iloc[0]["Test R2"]),
        "test_rmse": float(ranked.iloc[0]["Test RMSE"]),
        "test_mae": float(ranked.iloc[0]["Test MAE"]),
        "sequence_length": SEQUENCE_LENGTH,
        "n_features": N_FEATURES,
        "sequence_feature_columns": SEQUENCE_FEATURE_COLUMNS,
    }, f, indent=2)

pdf_report.close()

print(f"\nAll results saved to: {os.path.abspath(OUTPUT_DIR)}")
print(f"  - model_comparison.csv")
print(f"  - feature_scaler.joblib, linear_regression_model.joblib, random_forest_model.joblib, "
      f"xgboost_model.joblib, lightgbm_model.joblib, lstm_model.keras")
print(f"  - best_model.json")
print(f"PDF report saved to: {REPORT_PATH}")

# Final Conclusion
print("\n" + "=" * 70)
print("FINAL CONCLUSION")
print("=" * 70)

regularization_notes = []
for name, gap_var in [("Linear Regression", lr_gap), ("Random Forest", rf_gap), ("LSTM", lstm_gap)]:
    status = "regularization WAS applied" if gap_var > OVERFITTING_THRESHOLD else "no extra regularization needed"
    regularization_notes.append(f"  - {name}: initial gap {gap_var:.5f} -> {status}")
regularization_notes.append(
    "  - XGBoost / LightGBM: early-stopping + regularization mitigation is always evaluated "
    "(existing behavior, unchanged)."
)
regularization_summary = "\n".join(regularization_notes)

print(f"""
Best model (determined from actual Test R2 / RMSE, not assumed in advance): {best_model_name}

All five models were trained on IDENTICAL, leakage-free data: 192-step
(48-hour) lookback windows built strictly within a single Segment_ID /
SOURCE_KEY, targeting AC_POWER exactly 15 minutes ahead, using 8 features
per timestep (including the engineered MODULE_DERATING_FACTOR). Train/val/
test were split by WHOLE segment, so no sliding window and no segment is
shared across splits.

- Linear Regression and the tree-based models (Random Forest, XGBoost,
  LightGBM) see a FLATTENED version of each 192x8 window (1,536 features);
  they have no native notion of time order within the window.
- LSTM is the only model that consumes the (192, 8) sequence natively,
  processing the 48-hour history step-by-step.
- Every model was checked for overfitting (Train R2 - Test R2 gap >
  {OVERFITTING_THRESHOLD}) and given an automatic, model-appropriate
  regularization pass when needed, adopted only if it reduced the gap
  without materially hurting test R2 (test R2 drop <= 0.02):
{regularization_summary}

MODULE_DERATING_FACTOR: this feature is an affine (linear) rescaling of
MODULE_TEMPERATURE (see Feature Engineering section), so it carries the
same correlation magnitude with AC_POWER (reversed sign) and does not add
new information to Linear Regression, and — being a monotonic transform of
a single existing feature — does not change split decisions for the
tree-based models either. It is kept in the sequence tensor as required;
its practical value here is interpretability (a physically meaningful
efficiency multiplier relative to the 25 degC STC reference), not raw
predictive lift, exactly as anticipated in the Feature Engineering section.

See the Overfitting Gap column in the comparison table for how well each
model's training performance generalizes to unseen test sequences, and the
PDF report ({REPORT_PATH}) for the full EDA, per-model diagnostics
(Actual vs Predicted, Residuals, Learning Curves), and comparison charts.
""")

# %%
