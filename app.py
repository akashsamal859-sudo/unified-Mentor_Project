"""
Nassau Candy Distributor
Factory Reallocation & Shipping Optimization Dashboard
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings("ignore")

from analysis.core import (
    load_and_prepare, build_features, train_models,
    simulate_all_factories, generate_recommendations,
    factory_distance_profile, FACTORIES, PRODUCT_FACTORY,
)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Nassau Candy — Factory Optimizer",
    page_icon="🍬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  [data-testid="stMetricValue"] { font-size: 1.6rem; font-weight: 700; }
  .block-container { padding-top: 1rem; }
  .stTabs [data-baseweb="tab"] { font-size: 0.95rem; font-weight: 600; }
  div[data-testid="metric-container"] {
    background: #1e1e2e; border-radius: 10px; padding: 10px 16px;
  }
</style>
""", unsafe_allow_html=True)

# ── Data loading (cached) ─────────────────────────────────────────────────────
@st.cache_data
def get_data():
    return load_and_prepare("Nassau_Candy_Distributor.csv")

@st.cache_data
def get_models(data):
    X, y, encoders, cols = build_features(data)
    fitted, results, best_name = train_models(X, y)
    return fitted, results, best_name

@st.cache_data
def get_recommendations(data):
    return generate_recommendations(data)

df = get_data()
fitted, model_results, best_name = get_models(df)
recs_df = get_recommendations(df)
fdp = factory_distance_profile(df)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/5/5b/Willy_Wonka_%26_the_Chocolate_Factory_logo.svg/320px-Willy_Wonka_%26_the_Chocolate_Factory_logo.svg.png",
             use_column_width=True)
    st.title("🍬 Nassau Candy")
    st.caption("Factory Reallocation & Shipping Optimizer")
    st.divider()

    st.subheader("🔧 Filters")
    selected_division = st.multiselect(
        "Division", options=sorted(df["Division"].unique()), default=list(df["Division"].unique())
    )
    selected_region = st.multiselect(
        "Region", options=sorted(df["Region"].unique()), default=list(df["Region"].unique())
    )
    selected_ship_mode = st.selectbox(
        "Ship Mode", options=["All"] + sorted(df["Ship Mode"].unique())
    )
    st.divider()
    st.subheader("⚙️ Optimization Priority")
    opt_weight = st.slider("Speed vs Profit", 0, 100, 70,
                           help="0 = maximize profit, 100 = minimize lead time")
    st.caption(f"{'Speed-focused' if opt_weight > 60 else 'Balanced' if opt_weight > 40 else 'Profit-focused'}")

# ── Filter master dataframe ────────────────────────────────────────────────────
fdf = df[
    df["Division"].isin(selected_division) &
    df["Region"].isin(selected_region)
]
if selected_ship_mode != "All":
    fdf = fdf[fdf["Ship Mode"] == selected_ship_mode]

# ── Top header ────────────────────────────────────────────────────────────────
st.title("🍬 Nassau Candy — Factory Reallocation & Shipping Optimizer")
st.caption(f"Dataset: {len(df):,} orders | {df['Product Name'].nunique()} products | 5 factories | Filtered view: {len(fdf):,} orders")
st.divider()

# ── KPI row ───────────────────────────────────────────────────────────────────
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Total Revenue",    f"${fdf['Sales'].sum():,.0f}")
k2.metric("Total Gross Profit", f"${fdf['Gross Profit'].sum():,.0f}")
k3.metric("Avg Lead Time",    f"{fdf['Lead Time'].mean():.0f} days")
k4.metric("Avg Margin",       f"{fdf['Margin_pct'].mean():.1f}%")
k5.metric("Products to Reassign", f"{recs_df[recs_df['Reassignment Needed']].shape[0]} / {len(PRODUCT_FACTORY)}")

st.divider()

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 EDA & Overview",
    "🏭 Factory Simulator",
    "🔄 What-If Scenarios",
    "🎯 Recommendations",
    "🤖 ML Model Performance",
])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — EDA & OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("Exploratory Data Analysis")

    c1, c2 = st.columns(2)

    with c1:
        # Revenue by division
        rev_div = fdf.groupby("Division")[["Sales","Gross Profit"]].sum().reset_index()
        fig = px.bar(rev_div, x="Division", y=["Sales","Gross Profit"],
                     barmode="group", title="Revenue & Gross Profit by Division",
                     color_discrete_sequence=["#6C63FF","#48CAE4"])
        fig.update_layout(legend_title="Metric")
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        # Lead time distribution
        fig2 = px.histogram(fdf, x="Lead Time", color="Ship Mode", nbins=40,
                            title="Lead Time Distribution by Ship Mode",
                            color_discrete_sequence=px.colors.qualitative.Pastel)
        st.plotly_chart(fig2, use_container_width=True)

    c3, c4 = st.columns(2)

    with c3:
        # Orders by region
        region_cnt = fdf.groupby("Region")["Order ID"].count().reset_index(name="Orders")
        fig3 = px.pie(region_cnt, values="Orders", names="Region",
                      title="Orders by Region",
                      color_discrete_sequence=px.colors.qualitative.Set3)
        st.plotly_chart(fig3, use_container_width=True)

    with c4:
        # Avg distance by factory
        fact_dist = fdf.groupby("Factory")["Distance_km"].mean().reset_index()
        fact_dist = fact_dist.sort_values("Distance_km")
        fig4 = px.bar(fact_dist, x="Factory", y="Distance_km",
                      title="Avg Shipping Distance by Factory (km)",
                      color="Distance_km",
                      color_continuous_scale="RdYlGn_r",
                      text_auto=".0f")
        fig4.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig4, use_container_width=True)

    c5, c6 = st.columns(2)

    with c5:
        # Monthly sales trend
        monthly = fdf.groupby(["Order Year","Order Month"])["Sales"].sum().reset_index()
        monthly["Period"] = monthly["Order Year"].astype(str) + "-" + monthly["Order Month"].astype(str).str.zfill(2)
        fig5 = px.line(monthly, x="Period", y="Sales", title="Monthly Sales Trend",
                       markers=True, color_discrete_sequence=["#6C63FF"])
        fig5.update_xaxes(tickangle=45)
        st.plotly_chart(fig5, use_container_width=True)

    with c6:
        # Margin by product
        prod_margin = fdf.groupby("Product Name")["Margin_pct"].mean().reset_index().sort_values("Margin_pct", ascending=False)
        fig6 = px.bar(prod_margin, x="Margin_pct", y="Product Name", orientation="h",
                      title="Average Profit Margin by Product (%)",
                      color="Margin_pct", color_continuous_scale="Greens",
                      text_auto=".1f")
        fig6.update_layout(yaxis_title="", coloraxis_showscale=False, height=420)
        st.plotly_chart(fig6, use_container_width=True)

    # Factory distance heatmap
    st.subheader("Factory–Region Distance Heatmap (km)")
    pivot = fdp.pivot(index="Factory", columns="Region", values="Distance_km")
    fig7 = px.imshow(pivot, text_auto=True, aspect="auto",
                     color_continuous_scale="RdYlGn_r",
                     title="Distance from each Factory to each Region centroid (km)")
    st.plotly_chart(fig7, use_container_width=True)

    # Raw summary table
    with st.expander("📋 Summary statistics"):
        st.dataframe(fdf.describe(include="number").T.round(2), use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — FACTORY SIMULATOR
# ═══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("🏭 Factory Optimization Simulator")
    st.caption("Select a product to see how it performs across all five factories.")

    col_a, col_b = st.columns([1, 2])

    with col_a:
        sim_product = st.selectbox("Select Product", sorted(PRODUCT_FACTORY.keys()), key="sim_prod")
        sim_mode    = st.selectbox("Filter by Ship Mode", ["All"] + sorted(df["Ship Mode"].unique()), key="sim_mode")

    sim_result = simulate_all_factories(
        df, sim_product,
        ship_mode_filter=None if sim_mode == "All" else sim_mode
    )

    if not sim_result.empty:
        with col_a:
            curr_row = sim_result[sim_result["Is Current"] == True]
            if not curr_row.empty:
                st.metric("Current Factory", curr_row["Factory"].values[0])
                st.metric("Current Avg Distance", f"{curr_row['Avg Distance (km)'].values[0]:,.0f} km")
                best_row = sim_result.iloc[0]
                if best_row["Factory"] != curr_row["Factory"].values[0]:
                    st.metric("Optimal Factory", best_row["Factory"],
                              delta=f"{best_row['Avg Distance (km)'] - curr_row['Avg Distance (km)'].values[0]:+.0f} km")

        with col_b:
            # Bar chart
            colors = ["#6C63FF" if r else "#48CAE4" for r in sim_result["Is Current"]]
            fig_sim = go.Figure()
            fig_sim.add_trace(go.Bar(
                x=sim_result["Factory"],
                y=sim_result["Avg Distance (km)"],
                marker_color=colors,
                text=sim_result["Avg Distance (km)"].astype(int),
                textposition="outside",
                name="Avg Distance (km)",
            ))
            fig_sim.update_layout(
                title=f"Avg Shipping Distance — {sim_product}",
                yaxis_title="Distance (km)",
                showlegend=False,
                annotations=[dict(x=0.5, y=1.05, xref="paper", yref="paper",
                                  text="🟣 Current  🔵 Alternative", showarrow=False)]
            )
            st.plotly_chart(fig_sim, use_container_width=True)

        # Lead time comparison
        fig_lt = px.bar(
            sim_result, x="Factory", y="Estimated Lead Time (days)",
            color="Is Current",
            color_discrete_map={True: "#6C63FF", False: "#48CAE4"},
            title=f"Estimated Lead Time by Factory — {sim_product}",
            text_auto=".0f",
        )
        fig_lt.update_layout(showlegend=False)
        st.plotly_chart(fig_lt, use_container_width=True)

        st.dataframe(
            sim_result.drop(columns=["Is Current"]).style.highlight_min(
                subset=["Avg Distance (km)","Estimated Lead Time (days)"], color="#d4edda"
            ).highlight_max(subset=["Avg Gross Profit ($)"], color="#d4edda"),
            use_container_width=True,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — WHAT-IF SCENARIO ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("🔄 What-If Scenario Analysis")
    st.caption("Compare current configuration vs a hypothetical factory reassignment.")

    wa, wb, wc = st.columns(3)
    with wa:
        wi_product  = st.selectbox("Product", sorted(PRODUCT_FACTORY.keys()), key="wi_prod")
    with wb:
        wi_alt_factory = st.selectbox(
            "Reassign to Factory",
            [f for f in FACTORIES if f != PRODUCT_FACTORY.get(wi_product)],
            key="wi_fact",
        )
    with wc:
        wi_region = st.selectbox("Region Focus", ["All"] + sorted(df["Region"].unique()), key="wi_reg")

    curr_factory = PRODUCT_FACTORY[wi_product]

    sub_curr = df[df["Product Name"] == wi_product].copy()
    sub_alt  = df[df["Product Name"] == wi_product].copy()

    if wi_region != "All":
        sub_curr = sub_curr[sub_curr["Region"] == wi_region]
        sub_alt  = sub_alt[sub_alt["Region"]  == wi_region]

    from analysis.core import haversine, FACTORIES
    def add_dist(row, factory):
        fc = FACTORIES[factory]
        return haversine(fc["lat"], fc["lon"], row["State Lat"], row["State Lon"])

    sub_curr["Sim_Distance"] = sub_curr.apply(lambda r: add_dist(r, curr_factory), axis=1)
    sub_alt["Sim_Distance"]  = sub_alt.apply(lambda r: add_dist(r, wi_alt_factory), axis=1)

    # Compute delta
    curr_dist_avg = sub_curr["Sim_Distance"].mean()
    alt_dist_avg  = sub_alt["Sim_Distance"].mean()
    curr_lt_avg   = sub_curr["Lead Time"].mean()
    dist_delta    = alt_dist_avg - curr_dist_avg
    lt_est_delta  = dist_delta * 0.05  # heuristic

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Current Factory",      curr_factory)
    m2.metric("Proposed Factory",     wi_alt_factory)
    m3.metric("Distance Change",      f"{dist_delta:+,.0f} km",
              delta_color="inverse" if dist_delta > 0 else "normal")
    m4.metric("Est. Lead Time Change",f"{lt_est_delta:+.1f} days",
              delta_color="inverse" if lt_est_delta > 0 else "normal")

    st.divider()
    c1, c2 = st.columns(2)

    with c1:
        compare_df = pd.DataFrame({
            "Scenario":     ["Current", "Proposed"],
            "Factory":      [curr_factory, wi_alt_factory],
            "Avg Distance (km)": [round(curr_dist_avg, 0), round(alt_dist_avg, 0)],
            "Est. Lead Time (days)": [round(curr_lt_avg, 1), round(curr_lt_avg + lt_est_delta, 1)],
            "Avg Gross Profit ($)": [
                round(sub_curr["Gross Profit"].mean(), 2),
                round(sub_alt["Gross Profit"].mean(), 2),
            ],
        })
        fig_comp = go.Figure()
        for metric, color in [("Avg Distance (km)","#6C63FF"),("Est. Lead Time (days)","#FF6B6B")]:
            fig_comp.add_trace(go.Bar(
                name=metric,
                x=compare_df["Scenario"],
                y=compare_df[metric],
                marker_color=color,
                text=compare_df[metric],
                textposition="outside",
            ))
        fig_comp.update_layout(barmode="group", title="Current vs Proposed Scenario",
                               yaxis_title="Value")
        st.plotly_chart(fig_comp, use_container_width=True)

    with c2:
        # Distribution of per-order distances
        all_dists = pd.DataFrame({
            "Distance_km": list(sub_curr["Sim_Distance"]) + list(sub_alt["Sim_Distance"]),
            "Scenario":    ["Current"]*len(sub_curr) + ["Proposed"]*len(sub_alt),
        })
        fig_hist = px.histogram(all_dists, x="Distance_km", color="Scenario",
                                barmode="overlay", nbins=40, opacity=0.7,
                                title=f"Distance Distribution — {wi_product}",
                                color_discrete_map={"Current":"#6C63FF","Proposed":"#48CAE4"})
        st.plotly_chart(fig_hist, use_container_width=True)

    # Per-region breakdown
    st.subheader("Region-wise Impact")
    region_compare = []
    for region in df["Region"].unique():
        s_c = df[(df["Product Name"]==wi_product) & (df["Region"]==region)]
        s_a = df[(df["Product Name"]==wi_product) & (df["Region"]==region)]
        if s_c.empty:
            continue
        d_c = s_c.apply(lambda r: add_dist(r, curr_factory), axis=1).mean()
        d_a = s_a.apply(lambda r: add_dist(r, wi_alt_factory), axis=1).mean()
        region_compare.append({
            "Region":          region,
            "Current Dist (km)": round(d_c, 0),
            "Proposed Dist (km)": round(d_a, 0),
            "Delta (km)":      round(d_a - d_c, 0),
            "Better?":         "✅ Yes" if d_a < d_c else "❌ No",
        })
    if region_compare:
        st.dataframe(pd.DataFrame(region_compare), use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — RECOMMENDATIONS
# ═══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.subheader("🎯 Factory Reassignment Recommendations")

    reassign_df = recs_df[recs_df["Reassignment Needed"]].copy()
    no_change   = recs_df[~recs_df["Reassignment Needed"]].copy()

    st.markdown(f"""
    | Status | Count |
    |--------|-------|
    | ✅ Reassignment Recommended | **{len(reassign_df)}** |
    | 🟢 Optimal (no change needed) | **{len(no_change)}** |
    """)

    st.divider()

    if not reassign_df.empty:
        # Blend score (speed + profit)
        spd_w  = opt_weight / 100
        prof_w = 1 - spd_w
        reassign_df["Composite Score"] = (
            spd_w  * (reassign_df["Distance Reduction (%)"] / reassign_df["Distance Reduction (%)"].max()) +
            prof_w * (reassign_df["Avg Gross Profit ($)"] / reassign_df["Avg Gross Profit ($)"].max())
        ).round(3)
        reassign_df = reassign_df.sort_values("Composite Score", ascending=False).reset_index(drop=True)

        c1, c2 = st.columns(2)

        with c1:
            fig_r1 = px.bar(
                reassign_df, x="Distance Reduction (%)", y="Product",
                orientation="h", color="Distance Reduction (%)",
                color_continuous_scale="Greens",
                title="Distance Reduction by Reassignment (%)",
                text_auto=".1f",
            )
            fig_r1.update_layout(yaxis_title="", coloraxis_showscale=False)
            st.plotly_chart(fig_r1, use_container_width=True)

        with c2:
            fig_r2 = px.scatter(
                reassign_df,
                x="Distance Reduction (%)", y="Avg Gross Profit ($)",
                size="Total Orders", color="Division",
                hover_name="Product",
                title="Distance Reduction vs Profitability",
                text="Product",
            )
            fig_r2.update_traces(textposition="top center")
            st.plotly_chart(fig_r2, use_container_width=True)

        # Sankey diagram
        st.subheader("Factory Reassignment Flow")
        src, tgt, val, lbl = [], [], [], list(set(
            list(reassign_df["Current Factory"]) +
            list(reassign_df["Recommended Factory"])
        ))
        lbl_idx = {l: i for i, l in enumerate(lbl)}
        for _, row in reassign_df.iterrows():
            src.append(lbl_idx[row["Current Factory"]])
            tgt.append(lbl_idx[row["Recommended Factory"]])
            val.append(row["Total Orders"])
        fig_sankey = go.Figure(go.Sankey(
            node=dict(label=lbl, pad=15, thickness=20,
                      color=["#6C63FF","#48CAE4","#FF6B6B","#FFD93D","#6BCB77"][:len(lbl)]),
            link=dict(source=src, target=tgt, value=val),
        ))
        fig_sankey.update_layout(title_text="Order Flow: Current → Recommended Factory")
        st.plotly_chart(fig_sankey, use_container_width=True)

        # Detail table
        st.subheader("Recommendation Details")
        display_cols = [
            "Product","Division","Current Factory","Recommended Factory",
            "Distance Reduction (km)","Distance Reduction (%)",
            "Est. Lead Time Improvement (days)","Avg Gross Profit ($)","Composite Score"
        ]
        st.dataframe(
            reassign_df[display_cols].style
                .background_gradient(subset=["Distance Reduction (%)","Composite Score"], cmap="Greens")
                .format({"Distance Reduction (%)": "{:.1f}%",
                         "Est. Lead Time Improvement (days)": "{:.1f}",
                         "Avg Gross Profit ($)": "${:.2f}",
                         "Composite Score": "{:.3f}"}),
            use_container_width=True,
        )

    st.subheader("🟢 Products Already Optimally Assigned")
    if not no_change.empty:
        st.dataframe(no_change[["Product","Division","Current Factory",
                                "Current Avg Distance (km)","Avg Gross Profit ($)"]],
                     use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 5 — ML MODEL PERFORMANCE
# ═══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.subheader("🤖 Machine Learning — Lead Time Prediction")
    st.caption(f"Best model selected: **{best_name}**")

    # Model metrics table
    metrics_df = pd.DataFrame(model_results).T.reset_index().rename(columns={"index": "Model"})
    st.dataframe(
        metrics_df.style.highlight_max(subset=["R²"], color="#d4edda")
                        .highlight_min(subset=["RMSE","MAE"], color="#d4edda"),
        use_container_width=True,
    )

    st.info("""
    **Note on R² values:** The lead times in this dataset show high variance driven primarily by
    order year (2024 vs 2025 cohorts ship to different horizons), which limits the predictive power
    of operational features alone. The distance-based simulation engine uses haversine distances
    as the primary optimization signal, which is more actionable and interpretable.
    """)

    c1, c2 = st.columns(2)

    with c1:
        # Feature importance (Random Forest)
        rf_model = fitted["Random Forest"][0]
        feat_names = ["Ship Mode","Region","Division","Factory","Distance_km",
                      "Units","Sales","Cost","Order Month"]
        fi_df = pd.DataFrame({
            "Feature": feat_names,
            "Importance": rf_model.feature_importances_,
        }).sort_values("Importance", ascending=False)

        fig_fi = px.bar(fi_df, x="Importance", y="Feature", orientation="h",
                        title="Feature Importances (Random Forest)",
                        color="Importance", color_continuous_scale="Purples",
                        text_auto=".3f")
        fig_fi.update_layout(yaxis_title="", coloraxis_showscale=False)
        st.plotly_chart(fig_fi, use_container_width=True)

    with c2:
        # Actual vs predicted scatter
        from sklearn.model_selection import train_test_split
        X_all, y_all, _, _ = build_features(df)
        X_tr, X_te, y_tr, y_te = train_test_split(X_all, y_all, test_size=0.2, random_state=42)
        best_m, best_sc = fitted[best_name]
        X_pred = best_sc.transform(X_te) if best_sc else X_te
        y_hat  = best_m.predict(X_pred)

        scatter_df = pd.DataFrame({"Actual": y_te[:500], "Predicted": y_hat[:500]})
        fig_ap = px.scatter(scatter_df, x="Actual", y="Predicted",
                            title=f"Actual vs Predicted Lead Time ({best_name})",
                            opacity=0.4, color_discrete_sequence=["#6C63FF"])
        fig_ap.add_shape(type="line",
                         x0=scatter_df["Actual"].min(), y0=scatter_df["Actual"].min(),
                         x1=scatter_df["Actual"].max(), y1=scatter_df["Actual"].max(),
                         line=dict(color="red", dash="dash"))
        st.plotly_chart(fig_ap, use_container_width=True)

    # Model comparison radar
    metrics_norm = metrics_df.set_index("Model").copy()
    metrics_norm["RMSE_inv"] = 1 / metrics_norm["RMSE"]
    metrics_norm["MAE_inv"]  = 1 / metrics_norm["MAE"]

    fig_bar_compare = px.bar(
        metrics_df.melt(id_vars="Model", value_vars=["RMSE","MAE","R²"]),
        x="Model", y="value", color="variable", barmode="group",
        title="Model Metric Comparison",
        color_discrete_sequence=["#FF6B6B","#FFD93D","#6BCB77"],
    )
    st.plotly_chart(fig_bar_compare, use_container_width=True)

    with st.expander("📖 Methodology Notes"):
        st.markdown("""
        **Data Preparation**
        - Parsed order/ship dates; computed lead time in days
        - Mapped products to factories using the provided correlation table
        - Computed haversine distance from each factory to customer state centroid
        - Encoded categorical features (Ship Mode, Region, Division, Factory)

        **Models Trained**
        | Model | Description |
        |-------|-------------|
        | Linear Regression | Baseline; scaled features via StandardScaler |
        | Random Forest | 200 estimators; captures non-linear interactions |
        | Gradient Boosting | 200 estimators; sequential error correction |

        **Optimization Logic**
        - Primary signal: haversine distance (factory → customer state centroid)
        - Lead time delta estimated at 0.05 days/km deviation from current
        - Composite score = (speed_weight × distance_reduction%) + (profit_weight × normalized_profit)
        - Priority slider in sidebar controls speed vs profit weighting
        """)

