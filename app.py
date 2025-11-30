import streamlit as st
import pandas as pd
import numpy as np
import joblib
import torch
from torch import nn
import matplotlib.pyplot as plt
import json


# ================== LOAD METADATA ==================
with open("saved_model/feature_columns.json", "r") as f:
    FEATURE_COLS = json.load(f)

with open("saved_model/target_columns.json", "r") as f:
    TARGET_COLS = json.load(f)

scaler = joblib.load("saved_model/scaler1.pkl")

# ================= MODEL ARCHITECTURE ==================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2)*(-np.log(10000.0)/d_model))
        pe[:, 0::2], pe[:, 1::2] = torch.sin(position * div_term), torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class TimeSeriesTransformer(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_layers, dim_feedforward, dropout, num_targets, seq_len):
        super().__init__()

        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len=seq_len+5)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(d_model, num_targets)

    def forward(self, x):
        x = self.input_proj(x)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        return self.fc_out(x.mean(dim=1))


# ================== LOAD MODEL ==================
SEQ_LEN = 24
model = TimeSeriesTransformer(
    input_dim=len(FEATURE_COLS),
    d_model=128,
    nhead=4,
    num_layers=3,
    dim_feedforward=256,
    dropout=0.1,
    num_targets=len(TARGET_COLS),
    seq_len=SEQ_LEN
)

model.load_state_dict(torch.load("saved_model/weather_transformer.pth", map_location="cpu"))
model.eval()


# ================== FORECAST FUNCTION ==================
def forecast_next_hour(df_input):

    #  Ensure all scaler columns exist
    missing_cols = [c for c in FEATURE_COLS if c not in df_input.columns]

    for col in missing_cols:
        df_input[col] = 0   # filler for missing features

    #  Remove any extra features not part of model input
    df_input = df_input[FEATURE_COLS]

    # Transform safely
    scaled = scaler.transform(df_input)

    seq = torch.tensor(scaled, dtype=torch.float32).unsqueeze(0)
    pred = model(seq).detach().numpy()[0]

    return pred



# ================== STREAMLIT UI ==================
st.title("🌦 Next-Hour Weather Prediction Dashboard")
st.write("No uploads — running directly on latest weather data from file path.")


# LOAD LAST 24 HOURS DIRECTLY
df = pd.read_csv("processed_data/weather_processed.csv", index_col="Formatted Date", parse_dates=True)
df_last = df.tail(24)

y_pred = forecast_next_hour(df_last)


# ================== WEATHER INTERPRETATION ==================
st.subheader("📊 Next-Hour Forecast Summary")

hum_scaled = y_pred[TARGET_COLS.index("Humidity")]
hum = np.clip(hum_scaled * 100, 0, 100)   # Convert + Clamp to physical range
temp = y_pred[TARGET_COLS.index("Temperature (C)")] * 40
feel = y_pred[TARGET_COLS.index("Apparent Temperature (C)")] * 40
press = y_pred[TARGET_COLS.index("Pressure (millibars)")] * 40 + 950
wind = y_pred[TARGET_COLS.index("Wind Speed (km/h)")] * 60
rain_prob = np.clip((hum - 50) * 1.3, 0, 100)
dew_point_spread = temp - feel
if dew_point_spread < 2 and press < 1005 and hum > 85:
    rain_prob += 20

col1, col2 = st.columns(2)
with col1:
    st.metric("🌧 Rain Chance", f"{rain_prob:.1f}%")
    st.metric("💧 Humidity", f"{hum:.1f}%")
    st.metric("🔥 Feels Like", f"{feel:.1f} °C")
with col2:
    st.metric("🌡 Temperature", f"{temp:.1f} °C")
    st.metric("🍃 Wind Speed", f"{wind:.1f} km/h")
    st.metric("🔽 Pressure", f"{press:.1f} mb")


# ================== GRAPH: PAST 24H + PREDICTION ==================
st.write("### 📈 Past 24 Hours + Predicted Next Hour (Humidity)")

past_humidity = df_last["Humidity"].values * 100

fig, ax = plt.subplots(figsize=(8,3))
ax.plot(range(-23,1), past_humidity, label="Past Humidity", color="skyblue")
ax.scatter(1, hum, color="red", label="Predicted Next Hour", s=90)
ax.axvline(0, ls="--", color="white")

ax.set_ylabel("Humidity (%)")
ax.set_xlabel("Hours Timeline")
ax.legend()
st.pyplot(fig)


# ================== NATURAL LANGUAGE FORECAST ==================
st.write("### 🔍 Automated Forecast Interpretation")

if rain_prob>70: st.write("→ **High chance of rain. Carry umbrella.**")
elif rain_prob>40: st.write("→ **Moderate chance of showers expected.**")
else: st.write("→ **Low rain chance. Sky mostly clear.**")

if feel-temp > 2: st.write("→ Feels warmer than actual temp due to humidity.")
elif temp-feel > 2: st.write("→ Feels cooler because of wind factor.")

if press<1005: st.write("→ Low pressure — unstable weather risk.")
elif press>1015: st.write("→ High pressure — clearer, calm sky.")


st.success("Prediction Complete ✔ Ready for deployment")
