import streamlit as st
import pandas as pd
import numpy as np

from autoclean_backend import (
    detect_column_types,
    clean_text_columns,
    handle_missing,
    remove_exact_duplicates,
    remove_fuzzy_duplicates,
    remove_outliers,
    encode_and_scale,
    calculate_quality
)

st.set_page_config(page_title="AI Data Cleaning System", layout="wide")

# Custom CSS
st.markdown("""
<style>

/* Change dataframe selected row border color */
[data-testid="stDataFrame"] div[role="gridcell"]:focus {
    border: 2px solid blue !important;
}

/* Selected row highlight */
[data-testid="stDataFrame"] div[role="gridcell"]:focus-within {
    border: 2px solid blue !important;
}

</style>
""", unsafe_allow_html=True)

st.title("🧹 Intelligent Data Cleaning System")

st.write("Upload a dataset and automatically clean it using machine learning techniques.")

uploaded_file = st.file_uploader(
    "Upload CSV or Excel file",
    type=["csv", "xlsx"]
)


@st.cache_data
def run_cleaning(df):

    original_rows = len(df)

    num_cols, cat_cols, text_cols = detect_column_types(df)

    df = clean_text_columns(df, text_cols)

    df, m_before, m_after = handle_missing(df, num_cols, cat_cols)

    df, dup1 = remove_exact_duplicates(df)

    df, dup2 = remove_fuzzy_duplicates(df, text_cols)

    df, out_removed = remove_outliers(df, num_cols)

    df = encode_and_scale(df, num_cols, cat_cols)

    quality = calculate_quality(
        m_before,
        m_after,
        dup1 + dup2,
        out_removed,
        original_rows
    )

    return (
        df,
        m_before,
        m_after,
        dup1,
        dup2,
        out_removed,
        quality,
        original_rows
    )


if uploaded_file:

    # Read file
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    # Dataset Overview
    st.subheader("📊 Dataset Overview")

    col1, col2 = st.columns(2)

    col1.metric("Rows", df.shape[0])
    col2.metric("Columns", df.shape[1])

    # Dataset Preview
    st.subheader("📋 Dataset Preview")

    if st.checkbox("Show Full Dataset (May be slow)"):
        st.dataframe(df)
    else:
        st.dataframe(df.head(50))

    # Missing Values Section
    st.subheader("❌ Missing Values Before Cleaning")

    missing_cols = df.columns[df.isnull().any()].tolist()

    if missing_cols:
        missing_rows = df[df[missing_cols].isnull().any(axis=1)]
        st.dataframe(missing_rows)
    else:
        st.write("No missing values found.")

    # Duplicate Rows Section
    st.subheader("🔁 Duplicate Rows Before Cleaning")

    duplicates = df[df.duplicated()]

    if not duplicates.empty:
        st.dataframe(duplicates)
    else:
        st.write("No duplicate rows found.")

    # Cleaning Button
    if st.button("Run Auto Cleaning"):

        with st.spinner("Cleaning dataset..."):

            (
                df_clean,
                m_before,
                m_after,
                dup1,
                dup2,
                out_removed,
                quality,
                original_rows
            ) = run_cleaning(df)

        st.success("✅ Cleaning Completed Successfully")

        # Cleaning Results
        st.subheader("📈 Cleaning Results")

        c1, c2, c3 = st.columns(3)

        c1.metric("Missing Values Fixed", m_before - m_after)
        c2.metric("Duplicates Removed", dup1 + dup2)
        c3.metric("Outliers Removed", out_removed)

        # Quality Score
        st.subheader("⭐ Data Quality Score")

        st.progress(int(quality))

        st.write(f"Quality Score: **{quality}%**")

        # Graph
        st.subheader("📊 Cleaning Impact Graph")

        graph_data = pd.DataFrame({

            "Category": [
                "Original Rows",
                "Duplicates Removed",
                "Outliers Removed",
                "Final Rows"
            ],

            "Count": [
                original_rows,
                dup1 + dup2,
                out_removed,
                len(df_clean)
            ]
        })

        st.bar_chart(graph_data.set_index("Category"))

        # Cleaned Dataset Preview
        st.subheader("🧾 Cleaned Dataset Preview")

        if st.checkbox("Show Full Cleaned Dataset"):
            st.dataframe(df_clean)
        else:
            st.dataframe(df_clean.head(50))

        # Download Button
        csv = df_clean.to_csv(index=False).encode("utf-8")

        st.download_button(
            "⬇ Download Cleaned Dataset",
            csv,
            "cleaned_dataset.csv",
            "text/csv"
        )