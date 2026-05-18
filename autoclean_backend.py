import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import IsolationForest
from rapidfuzz import fuzz


def detect_column_types(df):

    num_cols = []
    cat_cols = []
    text_cols = []

    for col in df.columns:

        if pd.api.types.is_numeric_dtype(df[col]):
            num_cols.append(col)

        elif df[col].nunique() < 25:
            cat_cols.append(col)

        else:
            text_cols.append(col)

    return num_cols, cat_cols, text_cols


def clean_text(series):

    series = series.astype(str)

    series = series.str.lower()
    series = series.str.strip()

    series = series.str.replace(r"[^a-zA-Z0-9\s]", "", regex=True)

    series = series.str.replace(r"\s+", " ", regex=True)

    return series


def clean_text_columns(df, text_cols):

    for col in text_cols:

        df[col] = clean_text(df[col])

    return df


def handle_missing(df, num_cols, cat_cols):

    missing_before = df.isnull().sum().sum()

    if num_cols:

        imputer = SimpleImputer(strategy="mean")

        df[num_cols] = imputer.fit_transform(df[num_cols])

    for col in cat_cols:

        df[col] = df[col].fillna(df[col].mode()[0])

    missing_after = df.isnull().sum().sum()

    return df, missing_before, missing_after


def remove_exact_duplicates(df):

    before = df.shape[0]

    df = df.drop_duplicates()

    after = df.shape[0]

    removed = before - after

    return df, removed


def remove_fuzzy_duplicates(df, text_cols, threshold=90):

    if not text_cols:
        return df, 0

    # Skip fuzzy check if dataset is large
    if len(df) > 10000:
        return df, 0

    col = text_cols[0]

    seen = []
    drop_idx = []

    for i, val in enumerate(df[col]):

        for s in seen:

            if fuzz.ratio(str(val), str(s)) > threshold:

                drop_idx.append(i)
                break

        else:
            seen.append(val)

    df = df.drop(index=drop_idx)

    return df, len(drop_idx)


def remove_outliers(df, num_cols):

    if not num_cols:
        return df, 0

    iso = IsolationForest(contamination=0.05, random_state=42)

    preds = iso.fit_predict(df[num_cols])

    before = df.shape[0]

    df = df[preds == 1]

    after = df.shape[0]

    removed = before - after

    return df, removed


def encode_and_scale(df, num_cols, cat_cols):

    for col in cat_cols:

        df[col] = LabelEncoder().fit_transform(df[col].astype(str))

    if num_cols:

        scaler = StandardScaler()

        df[num_cols] = scaler.fit_transform(df[num_cols])

    return df


def calculate_quality(missing_before, missing_after, dup_removed, out_removed, original_rows):

    missing_score = (missing_before - missing_after) / (missing_before + 1) * 100

    dup_score = dup_removed / (original_rows + 1) * 100

    out_score = out_removed / (original_rows + 1) * 100

    quality = (missing_score + dup_score + out_score) / 3

    return round(quality, 2)