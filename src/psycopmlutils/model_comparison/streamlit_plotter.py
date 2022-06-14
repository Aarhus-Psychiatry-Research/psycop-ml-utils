from numpy import False_
import streamlit as st
import pandas as pd
from psycopmlutils.model_comparison.model_performance import ModelPerformance
from psycopmlutils.model_comparison.model_plotter import ModelPlotter, plotly_plotter


from typing import Optional

import plotly.express as px


## TODO sort plots by descending value

st.set_page_config(layout="wide", page_title="Model Comparison")

if "df" not in st.session_state:
    st.session_state["df"] = "NA"

uploaded_file = st.file_uploader("Select file")

if uploaded_file is not None:
    df = pd.read_json(uploaded_file, orient="records", lines=True)
    df = df.reset_index(drop=True)
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

    # Subset data
    df_sub = df.copy()
    for col, vals in selectors.items():
        df_sub = df_sub[df_sub[col].isin(vals)]

    # Subset dataframe based on selected values

    # Show data
    st.subheader("Show data")
    with st.expander("Show filtered data"):
        st.table(df_sub)

    st.subheader("Choose variables to plot")
    col1, col2 = st.columns(2)
    plots = {}
    possible_vars = df_sub.columns.tolist()
    possible_vars.append(None)
    buttons = {}

    for i, var in enumerate(["x", "y", "color", "facet_col", "facet_row"]):
        if i % 2 == 0:
            buttons[var] = col1.selectbox(
                var, possible_vars, index=(len(possible_vars) - 1), key=f"{var}_select"
            )
        else:
            buttons[var] = col2.selectbox(
                var, possible_vars, index=(len(possible_vars) - 1), key=f"{var}_select"
            )

    st.subheader("Choose plot type")
    with st.expander("Suggestions for scatter plot"):
        st.write(
            "Subset to only include 1 level (or facet by level). Value on x-axis, model_name on y-axis, color by class."
        )
    with st.expander("Suggestions for line plot"):
        st.write("class on x, value on y, color by model_name, facet_col by score_type")
    plotting_funs = ["scatter", "line", "strip"]
    chosen_plot = st.selectbox("Plot type", plotting_funs)

    make_title = st.checkbox("Autogenerate title?")
    if make_title:
        score_types = " ".join(df_sub["score_type"].unique())
        classes = " ".join(df_sub["class"].unique())
        extra_cols = [
            col
            for col in metadata_cols
            if col not in ["score_type", "class", "model_name"]
        ]
        extra_cols = {col: str(df_sub[col].unique().tolist()) for col in extra_cols}
        title = f"{score_types} for {classes} with {extra_cols}"
    else:
        title = None
    if chosen_plot:
        st.plotly_chart(
            plotly_plotter(
                df=df_sub,
                type=chosen_plot,
                x=buttons["x"],
                y=buttons["y"],
                color=buttons["color"],
                facet_col=buttons["facet_col"],
                facet_row=buttons["facet_row"],
                title=title,
            ),
            use_container_width=True,
        )

    st.subheader("Create and export table")
    with st.expander("Table"):

        pivot = st.checkbox("Pivot data?")
        col1, col2, col3 = st.columns(3)
        pivot_opts = {}

        export_df = df_sub.copy()

        if pivot:
            pivot_opts["index"] = col1.multiselect("Index", possible_vars, key="index")
            pivot_opts["columns"] = col2.multiselect(
                "Columns", possible_vars, key="columns"
            )
            pivot_opts["values"] = col3.multiselect(
                "Values", possible_vars, key="values"
            )
            # flatten to one list to only keep the relevant columns (necesary for proper pivoting)
            keep_cols = [item for sublist in pivot_opts.values() for item in sublist]

            export_df = export_df[keep_cols]

            export_df = export_df.pivot(
                index=pivot_opts["index"],
                columns=pivot_opts["columns"],
                values=pivot_opts["values"],
            )
        st.write(export_df)
        st.subheader("LaTeX code")
        st.write(export_df.to_latex())

        # selection = aggrid_interactive_table(df_sub)
        # st.write(selection)
        # table = pd.DataFrame(selection["selected_rows"])
        # st.write(table)
    # Boxplot
    # Make pretty tables (wide format) -> latex export?
