from numpy import False_
import streamlit as st
import pandas as pd
from psycopmlutils.model_comparison.model_comparison import ModelComparison
from psycopmlutils.model_comparison.model_plotter import ModelPlotter


uploaded_file = st.file_uploader("Select file")
if uploaded_file is not None:
    df = pd.read_json(uploaded_file, orient="records", lines=True)
    columns = df.columns
    metadata_cols = [col for col in columns if col != "value"]

    selectors = {}
    for col in metadata_cols:
        selectors[col] = st.sidebar.expander(col).multiselect(
            f"Use the following values",
            df[col].unique().tolist(),
            df[col].unique().tolist(),
            key=f"{col}_select",
        )

    st.write(selectors["class"])

    # Subset dataframe based on selected values

    # Show head
    with st.expander("Show data"):
        st.table(df)

    # Basic plotting
