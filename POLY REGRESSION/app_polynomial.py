import streamlit as st
import pandas as pd
import numpy as np
import joblib
import io

import plotly.express as px
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

st.set_page_config(
    page_title="CO2 Emissions Predictor (Polynomial Regression)",
    layout="wide",
)

FEATURES = ["Energy_Consumption", "Renewable_Percentage", "GDP"]
TARGET = "CO2_Emissions"

st.title("🌍 CO2 Emissions Predictor — Polynomial Regression Dashboard")
st.write(
    "Predicts **CO2 Emissions** from **Energy Consumption**, "
    "**Renewable Energy %**, and **GDP** using polynomial regression."
)

# ===========================================================
# Sidebar — data & model controls
# ===========================================================
st.sidebar.header("⚙️ Settings")

uploaded_file = st.sidebar.file_uploader("Upload sustainability_data.csv", type=["csv"])
DEFAULT_PATH = "sustainability_data.csv"

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
else:
    try:
        data = pd.read_csv(DEFAULT_PATH)
        st.sidebar.info(f"Using bundled `{DEFAULT_PATH}` found alongside the app.")
    except FileNotFoundError:
        st.warning(
            "No dataset loaded yet. Please upload `sustainability_data.csv` "
            "using the sidebar to continue."
        )
        st.stop()

missing_cols = [c for c in FEATURES + [TARGET] if c not in data.columns]
if missing_cols:
    st.error(f"Dataset is missing required column(s): {missing_cols}")
    st.stop()

missing_before = data.isnull().sum().sum()
data = data.dropna()

st.sidebar.markdown("---")
degree = st.sidebar.slider("Polynomial degree", 1, 5, 2)
test_size = st.sidebar.slider("Test set size", 0.1, 0.5, 0.2, 0.05)
random_state = st.sidebar.number_input("Random state", value=42, step=1)

# ===========================================================
# Train model (cached on inputs that affect it)
# ===========================================================
@st.cache_resource(show_spinner="Training model...")
def train_model(df, degree, test_size, random_state):
    X = df[FEATURES]
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=int(random_state)
    )
    poly = PolynomialFeatures(degree=degree)
    X_poly_train = poly.fit_transform(X_train)
    X_poly_test = poly.transform(X_test)

    model = LinearRegression()
    model.fit(X_poly_train, y_train)
    y_pred = model.predict(X_poly_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    return model, poly, X_train, X_test, y_train, y_test, y_pred, mse, r2


model, poly, X_train, X_test, y_train, y_test, y_pred, mse, r2 = train_model(
    data, degree, test_size, random_state
)

# ===========================================================
# Tabs
# ===========================================================
tab_data, tab_eda, tab_perf, tab_predict, tab_download = st.tabs(
    ["📄 Data", "🔎 Explore", "📊 Model Performance", "🎛️ Predict", "💾 Download"]
)

# -----------------------------------------------------------
# Tab: Data
# -----------------------------------------------------------
with tab_data:
    st.subheader("Dataset preview")
    st.dataframe(data.head(20), use_container_width=True)
    if missing_before > 0:
        st.caption(f"Dropped rows with missing values ({missing_before} missing cells found).")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", len(data))
    c2.metric("Features", len(FEATURES))
    c3.metric("Train rows", len(X_train))
    c4.metric("Test rows", len(X_test))

    st.subheader("Summary statistics")
    st.dataframe(data[FEATURES + [TARGET]].describe(), use_container_width=True)

# -----------------------------------------------------------
# Tab: Explore (EDA)
# -----------------------------------------------------------
with tab_eda:
    st.subheader("Correlation heatmap")
    corr = data[FEATURES + [TARGET]].corr()
    fig_corr = px.imshow(
        corr,
        text_auto=".2f",
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        aspect="auto",
        title="Feature correlation matrix",
    )
    st.plotly_chart(fig_corr, use_container_width=True)

    st.subheader("Feature vs. Target")
    feat_choice = st.selectbox("Choose a feature to plot against CO2 Emissions", FEATURES)
    fig_scatter = px.scatter(
        data,
        x=feat_choice,
        y=TARGET,
        trendline="ols",
        opacity=0.7,
        title=f"{feat_choice} vs {TARGET}",
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

    st.subheader("Distributions")
    dist_col = st.selectbox("Choose a column to view its distribution", FEATURES + [TARGET], index=len(FEATURES))
    fig_hist = px.histogram(data, x=dist_col, nbins=30, marginal="box", title=f"Distribution of {dist_col}")
    st.plotly_chart(fig_hist, use_container_width=True)

# -----------------------------------------------------------
# Tab: Model Performance
# -----------------------------------------------------------
with tab_perf:
    st.subheader("Performance metrics")
    c1, c2, c3 = st.columns(3)
    c1.metric("Mean Squared Error", f"{mse:,.2f}")
    c2.metric("R² Score", f"{r2:.4f}")
    c3.metric("Polynomial Degree", degree)

    st.subheader("Actual vs. Predicted")
    perf_df = pd.DataFrame({"Actual": y_test.values, "Predicted": y_pred})
    fig_pred = px.scatter(
        perf_df,
        x="Actual",
        y="Predicted",
        opacity=0.7,
        title="Polynomial Regression: Predictions vs. Actual Values",
    )
    min_v, max_v = perf_df["Actual"].min(), perf_df["Actual"].max()
    fig_pred.add_trace(
        go.Scatter(x=[min_v, max_v], y=[min_v, max_v], mode="lines", name="Perfect prediction", line=dict(color="red"))
    )
    st.plotly_chart(fig_pred, use_container_width=True)

    st.subheader("Residuals")
    residuals = perf_df["Actual"] - perf_df["Predicted"]
    fig_res = px.scatter(
        x=perf_df["Predicted"],
        y=residuals,
        labels={"x": "Predicted", "y": "Residual"},
        opacity=0.7,
        title="Residual plot",
    )
    fig_res.add_hline(y=0, line_dash="dash", line_color="red")
    st.plotly_chart(fig_res, use_container_width=True)

    st.subheader("Residual distribution")
    fig_res_hist = px.histogram(residuals, nbins=30, title="Residuals distribution")
    fig_res_hist.update_layout(showlegend=False, xaxis_title="Residual")
    st.plotly_chart(fig_res_hist, use_container_width=True)

# -----------------------------------------------------------
# Tab: Predict
# -----------------------------------------------------------
with tab_predict:
    st.subheader("Try a prediction")
    st.write("Adjust the inputs below to predict CO2 Emissions.")

    col1, col2, col3 = st.columns(3)
    with col1:
        energy_in = st.slider(
            "Energy Consumption",
            float(data["Energy_Consumption"].min()),
            float(data["Energy_Consumption"].max()),
            float(data["Energy_Consumption"].mean()),
        )
    with col2:
        renew_in = st.slider(
            "Renewable Percentage",
            float(data["Renewable_Percentage"].min()),
            float(data["Renewable_Percentage"].max()),
            float(data["Renewable_Percentage"].mean()),
        )
    with col3:
        gdp_in = st.slider(
            "GDP",
            float(data["GDP"].min()),
            float(data["GDP"].max()),
            float(data["GDP"].mean()),
        )

    input_df = pd.DataFrame(
        {"Energy_Consumption": [energy_in], "Renewable_Percentage": [renew_in], "GDP": [gdp_in]}
    )
    input_poly = poly.transform(input_df)
    prediction = model.predict(input_poly)[0]

    st.success(f"Predicted CO2 Emissions: **{prediction:,.2f}**")

    st.markdown("---")
    st.subheader("Sensitivity: how prediction changes with one feature")
    sweep_feature = st.selectbox("Feature to sweep", FEATURES, key="sweep")
    sweep_range = np.linspace(data[sweep_feature].min(), data[sweep_feature].max(), 50)

    # Build sweep dataframe using current slider values as the baseline for other features
    baseline = {"Energy_Consumption": energy_in, "Renewable_Percentage": renew_in, "GDP": gdp_in}
    sweep_df = pd.DataFrame([baseline] * 50)
    sweep_df[sweep_feature] = sweep_range
    sweep_pred = model.predict(poly.transform(sweep_df))

    fig_sweep = px.line(
        x=sweep_range,
        y=sweep_pred,
        labels={"x": sweep_feature, "y": "Predicted CO2 Emissions"},
        title=f"Predicted CO2 Emissions as {sweep_feature} varies (other features held constant)",
    )
    fig_sweep.add_vline(x=baseline[sweep_feature], line_dash="dash", line_color="green")
    st.plotly_chart(fig_sweep, use_container_width=True)

# -----------------------------------------------------------
# Tab: Download
# -----------------------------------------------------------
with tab_download:
    st.subheader("Download trained model")
    model_buffer = io.BytesIO()
    joblib.dump({"model": model, "poly": poly, "features": FEATURES}, model_buffer)
    model_buffer.seek(0)
    st.download_button(
        label="Download model (.pkl)",
        data=model_buffer,
        file_name="polynomialRegModel.pkl",
        mime="application/octet-stream",
    )

    st.subheader("Download predictions on test set")
    result_df = X_test.copy()
    result_df["Actual"] = y_test.values
    result_df["Predicted"] = y_pred
    csv_buffer = result_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download test predictions (.csv)",
        data=csv_buffer,
        file_name="test_predictions.csv",
        mime="text/csv",
    )
