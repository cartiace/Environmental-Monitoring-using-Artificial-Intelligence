import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

# App Config
st.set_page_config(page_title="Environmental Monitoring Using AI", layout="wide")
st.title("🌍 Environmental Monitoring Using AI")
st.markdown("This dashboard presents AI-powered insights for **Air Quality Monitoring** and **Deforestation Tracking** across Himachal Pradesh.")

# Tabs: Aligned with project objectives
tab1, tab2 = st.tabs(["🌫️ Air Quality Monitoring", "🌲 Deforestation Monitoring"])

# -----------------------
# Tab 1: Air Quality Monitoring
# -----------------------
with tab1:
    st.header("📊 Forecasted vs Actual Air Quality Levels")

    # Load prediction data
    df = pd.read_csv("air_quality_predictions.csv", parse_dates=["Hour"])
    pollutants = ['PM2.5 (ug/m3)', 'CO (mg/m3)', 'Ozone (ug/m3)', 'NOx (ug/m3)']

    selected_pollutant = st.selectbox("Choose a Pollutant", pollutants)

    # Line chart
    df_plot = df[["Hour", "Stage", selected_pollutant]]
    fig, ax = plt.subplots(figsize=(12, 5))
    for stage, group in df_plot.groupby("Stage"):
        ax.plot(group["Hour"], group[selected_pollutant],
                label=stage,
                marker='o' if stage != "Input" else '.',
                linestyle='-')

    ax.set_title(f"{selected_pollutant} Forecast")
    ax.set_xlabel("Date & Time")
    ax.set_ylabel(selected_pollutant)
    ax.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig)

    with st.expander("🔍 View Raw Data"):
        st.subheader("Past Inputs")
        st.write(df[df["Stage"] == "Input"][["Hour", selected_pollutant]])
        st.subheader("Ground Truth (Labels)")
        st.write(df[df["Stage"] == "Label"][["Hour", selected_pollutant]])
        st.subheader("Model Predictions")
        st.write(df[df["Stage"] == "Prediction"][["Hour", selected_pollutant]])

    with st.expander("⬇️ Download Dataset"):
        st.download_button("Download CSV", df.to_csv(index=False), file_name="air_quality_predictions.csv")


# -----------------------
# Tab 2: Deforestation Monitoring
# -----------------------
with tab2:
    st.header("🌲 Deforestation Prediction Models Overview")

    model_data = {
        "Model": ["2D CNN", "3D CNN"],
        "Accuracy": ["91.4%", "94.2%"],
        "F1 Score": ["0.88", "0.91"],
        "IoU": ["0.81", "0.85"]
    }
    st.table(pd.DataFrame(model_data))

    st.subheader("📍 Regional Evaluation (2019–2020)")
    full_data = {
        "Train Region": ["Kullu", "Kinnaur", "Shimla", "Hamirpur", "Kangra"],
        "2019 AUC": [0.990, 0.879, 0.997, 0.984, 0.806],
        "2019 F1": [0.903, 0.663, 0.904, 0.865, 0.532],
        "2020 AUC": [0.839, 0.861, 0.849, 0.876, 0.803],
        "2020 F1": [0.580, 0.593, 0.589, 0.623, 0.527]
    }
    st.dataframe(pd.DataFrame(full_data))

    st.markdown("---")
    st.header("🗺️ Deforestation Heatmaps")

    st.subheader("Simulated Interpolated Deforestation Map")
    heatmap1 = Image.open("deforestation_heatmap_hp.png")
    resized1 = heatmap1.resize((700, int(700 * heatmap1.height / heatmap1.width)))
    st.image(resized1)


