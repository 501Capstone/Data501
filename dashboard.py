import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.impute import SimpleImputer

st.set_page_config(layout="wide", page_title="Oil Sands Lake Analysis Dashboard")

st.title("Oil Sands Lake Analysis Dashboard")
st.caption("Interactive dashboard for comparing lake chemistry across Near, Mid, and Far distance groups.")

# =========================
# LOAD DATA
# =========================
@st.cache_data
def load_data():
    df = pd.read_csv("oilsands_master_clean_FINAL.csv")
    df.columns = df.columns.str.lower().str.strip()

    if "sampling_timestamp" in df.columns:
        df["sampling_timestamp"] = pd.to_datetime(df["sampling_timestamp"], errors="coerce")

    return df

df = load_data()

# =========================
# FILTERS
# =========================
col1, col2 = st.columns(2)

with col1:
    selected_group = st.selectbox(
        "Distance Group",
        ["All"] + sorted(df["distance_group"].dropna().unique().tolist())
    )

with col2:
    selected_lake = st.selectbox(
        "Lake",
        ["All"] + sorted(df["lake"].dropna().unique().tolist())
    )

filtered_df = df.copy()

if selected_group != "All":
    filtered_df = filtered_df[filtered_df["distance_group"] == selected_group]

if selected_lake != "All":
    filtered_df = filtered_df[filtered_df["lake"] == selected_lake]

# =========================
# TABS
# =========================
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Overview", "Distributions", "PCA", "ANOVA", "Random Forest"])
# =========================
# PCA
# =========================
with tab3:
    st.subheader("PCA Analysis")

    # Only chemistry variables
    chemical_params = [
        "depth",
        "aluminum_dissolved",
        "aluminum_total_recoverable",
        "calcium_dissolved",
        "calcium_total_recoverable",
        "copper_dissolved",
        "copper_total_recoverable",
        "lead_dissolved",
        "lead_total_recoverable",
        "nickel_dissolved",
        "nickel_total_recoverable",
        "nitrogen_dissolved",
        "nitrogen_dissolved_nitrate",
        "nitrogen_dissolved_nitrite",
        "nitrogen_kjeldahl_dissolved",
        "oxygen_biochemical_demand",
        "oxygen_dissolved_field_meter",
        "oxygen_dissolved_percent_saturation",
        "oxygen_total_cod",
        "ph",
        "phosphorus_total",
        "phosphorus_total_dissolved",
        "turbidity",
        "vanadium_dissolved",
        "vanadium_total_recoverable"
    ]

    available_pca_vars = [col for col in chemical_params if col in filtered_df.columns]

    st.markdown("### Variables Included")
    st.write(available_pca_vars)

    if len(available_pca_vars) < 2:
        st.warning("Not enough numeric chemistry variables are available to run PCA.")
    else:
        # Build PCA dataset
        X = filtered_df[available_pca_vars].copy()
        X = X.apply(pd.to_numeric, errors="coerce")

        # Drop fully empty columns
        X = X.dropna(axis=1, how="all")

        # Drop zero-variance columns
        X = X.loc[:, X.nunique() > 1]

        # Fill missing values
        X = X.fillna(X.median())

        if X.shape[1] < 2:
            st.warning("After cleaning, there are not enough variables left to compute PCA.")
        else:
            # Standardize
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            # Run PCA
            pca = PCA(n_components=2)
            pcs = pca.fit_transform(X_scaled)

            # PCA scores
            pca_df = pd.DataFrame(pcs, columns=["PC1", "PC2"])
            pca_df["distance_group"] = filtered_df["distance_group"].values
            pca_df["lake"] = filtered_df["lake"].values

            # Explained variance
            pc1_var = pca.explained_variance_ratio_[0] * 100
            pc2_var = pca.explained_variance_ratio_[1] * 100

            col1, col2 = st.columns([2, 1])

            with col1:
                fig_pca = px.scatter(
                    pca_df,
                    x="PC1",
                    y="PC2",
                    color="distance_group",
                    hover_data=["lake"],
                    title=f"PCA Scatter Plot (PC1: {pc1_var:.1f}%, PC2: {pc2_var:.1f}%)",
                    color_discrete_map={
                        "Near": "#111111",
                        "Mid": "#2563eb",
                        "Far": "#dc2626"
                    }
                )
                fig_pca.update_layout(
                    xaxis_title=f"PC1 ({pc1_var:.1f}%)",
                    yaxis_title=f"PC2 ({pc2_var:.1f}%)",
                    legend_title="Distance Group"
                )
                st.plotly_chart(fig_pca, width="stretch")

            with col2:
                st.markdown("### Explained Variance")
                variance_df = pd.DataFrame({
                    "Component": ["PC1", "PC2"],
                    "Explained Variance (%)": [pc1_var, pc2_var]
                })
                st.dataframe(variance_df, width="stretch")

            # Loadings
            loadings = pd.DataFrame(
                pca.components_.T,
                columns=["PC1", "PC2"],
                index=X.columns
            ).reset_index()
            loadings.columns = ["variable", "PC1", "PC2"]

            st.markdown("### PCA Loadings")

            selected_pc = st.selectbox(
                "Select principal component for loading plot",
                ["PC1", "PC2"],
                key="pca_loading_select"
            )

            loadings_sorted = loadings.sort_values(selected_pc, key=lambda s: s.abs(), ascending=False)

            fig_loadings = px.bar(
                loadings_sorted.head(15),
                x=selected_pc,
                y="variable",
                orientation="h",
                title=f"Top Variable Loadings for {selected_pc}"
            )
            fig_loadings.update_layout(yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(fig_loadings, width="stretch")

            st.markdown("### Full Loadings Table")
            st.dataframe(loadings.sort_values("PC1", key=lambda s: s.abs(), ascending=False), width="stretch")
            
# =========================
# ANOVA
# =========================
from scipy.stats import f_oneway
import math

with tab4:
    st.subheader("ANOVA Analysis")

    anova_vars = [
        "depth",
        "aluminum_dissolved",
        "aluminum_total_recoverable",
        "calcium_dissolved",
        "calcium_total_recoverable",
        "copper_dissolved",
        "copper_total_recoverable",
        "lead_dissolved",
        "lead_total_recoverable",
        "nickel_dissolved",
        "nickel_total_recoverable",
        "nitrogen_dissolved",
        "nitrogen_dissolved_nitrate",
        "nitrogen_dissolved_nitrite",
        "nitrogen_kjeldahl_dissolved",
        "oxygen_biochemical_demand",
        "oxygen_dissolved_field_meter",
        "oxygen_dissolved_percent_saturation",
        "oxygen_total_cod",
        "ph",
        "phosphorus_total",
        "phosphorus_total_dissolved",
        "turbidity",
        "vanadium_dissolved",
        "vanadium_total_recoverable"
    ]

    available_anova_vars = [col for col in anova_vars if col in filtered_df.columns]

    # Decide grouping variable
    if filtered_df["distance_group"].nunique() > 1:
        group_col = "distance_group"
    elif filtered_df["lake"].nunique() > 1:
        group_col = "lake"
    else:
        group_col = None

    st.markdown("### Grouping Variable")
    if group_col is None:
        st.warning("ANOVA requires at least two groups. Change the filters to include more than one lake or more than one distance group.")
    else:
        st.write(group_col)

        st.markdown("### Variables Included")
        st.write(available_anova_vars)

        anova_results = []

        for col in available_anova_vars:
            temp = filtered_df[[group_col, col]].copy()
            temp[col] = pd.to_numeric(temp[col], errors="coerce")

            grouped = []
            group_names = []

            for g in sorted(temp[group_col].dropna().unique()):
                vals = temp[temp[group_col] == g][col].dropna()
                if len(vals) > 1:
                    grouped.append(vals)
                    group_names.append(g)

            if len(grouped) >= 2:
                f_stat, p_val = f_oneway(*grouped)
                anova_results.append({
                    "variable": col,
                    "f_statistic": f_stat,
                    "p_value": p_val,
                    "num_groups": len(grouped)
                })

        if not anova_results:
            st.warning("No valid ANOVA results could be computed with the current filter and variable set.")
        else:
            anova_df = pd.DataFrame(anova_results).sort_values("p_value")

            st.markdown("### Ranked ANOVA Results")
            st.dataframe(anova_df, width="stretch")

            # Significance threshold
            alpha = st.slider("Significance threshold (alpha)", 0.01, 0.10, 0.05, 0.01)

            significant_vars = anova_df[anova_df["p_value"] < alpha]["variable"].tolist()

            st.markdown("### Significant Variables")
            if significant_vars:
                st.write(significant_vars)
            else:
                st.info("No variables met the selected significance threshold.")

            # Boxplot grid
            max_plots = st.slider("Maximum number of boxplots", 4, 16, 8, 2)

            vars_to_plot = significant_vars[:max_plots] if significant_vars else anova_df["variable"].head(max_plots).tolist()

            if vars_to_plot:
                st.markdown("### Boxplot Grid")

                n_cols = 2
                rows = math.ceil(len(vars_to_plot) / n_cols)

                for r in range(rows):
                    cols = st.columns(n_cols)
                    for c in range(n_cols):
                        idx = r * n_cols + c
                        if idx < len(vars_to_plot):
                            var = vars_to_plot[idx]

                            plot_df = filtered_df[[group_col, var]].copy()
                            plot_df[var] = pd.to_numeric(plot_df[var], errors="coerce")

                            fig_box = px.box(
                                plot_df,
                                x=group_col,
                                y=var,
                                color=group_col,
                                title=f"{var} (p = {anova_df.loc[anova_df['variable'] == var, 'p_value'].values[0]:.4f})"
                            )
                            cols[c].plotly_chart(fig_box, width="stretch")

            # Optional download-ready summary
            st.markdown("### Summary")
            st.write(
                f"ANOVA was computed across **{group_col}** for "
                f"**{len(available_anova_vars)}** candidate variables. "
                f"**{len(significant_vars)}** variables met alpha = {alpha:.2f}."
            )
# =========================
# RANDOM FOREST
# =========================
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.impute import SimpleImputer
import plotly.figure_factory as ff

with tab5:
    st.subheader("Random Forest Analysis")

    rf_vars = [
        "depth",
        "aluminum_dissolved",
        "aluminum_total_recoverable",
        "calcium_dissolved",
        "calcium_total_recoverable",
        "copper_dissolved",
        "copper_total_recoverable",
        "lead_dissolved",
        "lead_total_recoverable",
        "nickel_dissolved",
        "nickel_total_recoverable",
        "nitrogen_dissolved",
        "nitrogen_dissolved_nitrate",
        "nitrogen_dissolved_nitrite",
        "nitrogen_kjeldahl_dissolved",
        "oxygen_biochemical_demand",
        "oxygen_dissolved_field_meter",
        "oxygen_dissolved_percent_saturation",
        "oxygen_total_cod",
        "ph",
        "phosphorus_total",
        "phosphorus_total_dissolved",
        "turbidity",
        "vanadium_dissolved",
        "vanadium_total_recoverable"
    ]

    available_rf_vars = [col for col in rf_vars if col in filtered_df.columns]

    # Decide prediction target automatically
    if filtered_df["distance_group"].nunique() > 1:
        target_col = "distance_group"
    elif filtered_df["lake"].nunique() > 1:
        target_col = "lake"
    else:
        target_col = None

    st.markdown("### Prediction Target")
    if target_col is None:
        st.warning("Random Forest requires at least two target classes. Change the filters to include more than one distance group or more than one lake.")
    else:
        st.write(target_col)

        st.markdown("### Variables Included")
        st.write(available_rf_vars)

        if len(available_rf_vars) < 2:
            st.warning("Not enough chemistry variables are available to run Random Forest.")
        else:
            # Prepare X and y
            X = filtered_df[available_rf_vars].copy()
            X = X.apply(pd.to_numeric, errors="coerce")
            X = X.dropna(axis=1, how="all")
            X = X.loc[:, X.nunique() > 1]

            y_rf = filtered_df[target_col].copy()

            if X.shape[1] < 2 or y_rf.nunique() < 2:
                st.warning("Not enough usable variables or target classes remain after cleaning.")
            else:
                # Impute missing values
                imputer = SimpleImputer(strategy="median")
                X_imputed = imputer.fit_transform(X)

                # Controls
                col1, col2, col3 = st.columns(3)
                with col1:
                    n_estimators = st.slider("Number of trees", 100, 500, 300, 50)
                with col2:
                    max_depth = st.selectbox("Max depth", [None, 3, 5, 10, 20], index=0)
                with col3:
                    test_size = st.slider("Test size", 0.1, 0.4, 0.2, 0.05)

                # Train/test split
                X_train, X_test, y_train, y_test = train_test_split(
                    X_imputed,
                    y_rf,
                    test_size=test_size,
                    random_state=42,
                    stratify=y_rf
                )

                # Fit model
                rf = RandomForestClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    random_state=42,
                    class_weight="balanced"
                )
                rf.fit(X_train, y_train)
                y_pred = rf.predict(X_test)

                # Metrics
                acc = accuracy_score(y_test, y_pred)

                col1, col2 = st.columns(2)
                col1.metric("Accuracy", f"{acc:.3f}")
                col2.metric("Classes Predicted", y_rf.nunique())

                st.markdown("### Classification Report")
                report_df = pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)).transpose()
                st.dataframe(report_df, width="stretch")

                # Confusion matrix
                st.markdown("### Confusion Matrix")
                labels = sorted(y_rf.unique().tolist())
                cm = confusion_matrix(y_test, y_pred, labels=labels)

                fig_cm = ff.create_annotated_heatmap(
                    z=cm,
                    x=labels,
                    y=labels,
                    colorscale="Blues",
                    showscale=True
                )
                fig_cm.update_layout(
                    title="Confusion Matrix",
                    xaxis_title="Predicted",
                    yaxis_title="Actual"
                )
                st.plotly_chart(fig_cm, width="stretch")

                # Feature importance
                st.markdown("### Feature Importance")

                importance_df = pd.DataFrame({
                    "feature": X.columns,
                    "importance": rf.feature_importances_
                }).sort_values("importance", ascending=False)

                top_n = st.slider("Top features to display", 5, min(20, len(importance_df)), min(10, len(importance_df)))

                fig_imp = px.bar(
                    importance_df.head(top_n),
                    x="importance",
                    y="feature",
                    orientation="h",
                    title=f"Top {top_n} Variable Drivers"
                )
                fig_imp.update_layout(yaxis={"categoryorder": "total ascending"})
                st.plotly_chart(fig_imp, width="stretch")

                st.markdown("### Full Feature Importance Table")
                st.dataframe(importance_df, width="stretch")

                st.markdown("### Summary")
                st.write(
                    f"Random Forest was trained to predict **{target_col}** using "
                    f"**{X.shape[1]}** chemistry variables. "
                    f"The model achieved an accuracy of **{acc:.3f}** on the held-out test set."
                )