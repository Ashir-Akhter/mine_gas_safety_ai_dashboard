import streamlit as st
import torch
import numpy as np
import time
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import base64
from pathlib import Path

from simulation import create_base_layout, init_state, to_torch, step_gpu
from train_model import GasCNN


# =====================================================
# PAGE
# =====================================================

st.set_page_config(layout="wide")

H, W = 128, 128
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# =====================================================
# 🔊 SAME audio_test.py METHOD
# =====================================================

def audio_html(path: str):
    audio_bytes = Path(path).read_bytes()
    b64 = base64.b64encode(audio_bytes).decode()

    return f"""
    <audio autoplay loop>
        <source src="data:audio/wav;base64,{b64}" type="audio/wav">
    </audio>
    """


# =====================================================
# SESSION STATE
# =====================================================

if "running" not in st.session_state:
    st.session_state.running = False
if "alarm_enabled" not in st.session_state:
    st.session_state.alarm_enabled = False
if "history" not in st.session_state:
    st.session_state.history = []

S = st.session_state


# =====================================================
# LOAD MODELS
# =====================================================

@st.cache_resource
def load_models():
    models = {}
    for name, file in [
        ("Best", "gas_model_best.pt"),
        ("Last", "gas_model_last.pt")
    ]:
        try:
            m = GasCNN().to(DEVICE)
            m.load_state_dict(torch.load(file, map_location=DEVICE))
            m.eval()
            models[name] = m
        except:
            pass
    return models


MODELS = load_models()


# =====================================================
# RESET
# =====================================================

def reset():
    layout = create_base_layout(H, W)
    gas, fans, leaks = init_state(layout)

    for _ in range(5):
        i, j = np.random.randint(0, H), np.random.randint(0, W)
        if layout[i, j] == 1:
            leaks[i, j] = 1.5

    layout_t, gas_t, fans_t, leaks_t = to_torch(layout, gas, fans, leaks, DEVICE)

    S.layout = layout
    S.layout_t = layout_t
    S.gas_t = gas_t
    S.fans_t = fans_t
    S.leaks_t = leaks_t
    S.history = []
    S.running = False


if "layout" not in S:
    reset()


# =====================================================
# UI
# =====================================================

st.title("🚨 Mine Gas Safety AI Dashboard")

left, right = st.columns([3, 1])

model_name = right.selectbox("Model", list(MODELS.keys()))
threshold = right.slider("Risk Threshold", 0.0, 1.0, 0.5, 0.01)
speed = right.slider("Speed", 1, 10, 3)

if right.button("🔊 Enable Alarm"):
    S.alarm_enabled = True
    st.success("Audio enabled")


b1, b2, b3 = right.columns(3)

if b1.button("Start"):
    S.running = True

if b2.button("Stop"):
    S.running = False

if b3.button("Reset"):
    reset()


plot_area = left.empty()
prob_area = right.empty()
status_area = right.empty()
history_area = right.empty()

# 🔊 ONE persistent audio placeholder
audio_slot = right.empty()

model = MODELS[model_name]


# =====================================================
# SIMULATION LOOP
# =====================================================

while S.running:

    for _ in range(speed):
        S.gas_t = step_gpu(
            S.layout_t,
            S.gas_t,
            S.fans_t,
            S.leaks_t,
            leak_strength=0.6,
            fan_strength=0.04,
            diffusion=0.8
        )

    gas_np = S.gas_t.cpu().numpy()
    fans_np = S.fans_t.cpu().numpy()
    layout_np = np.asarray(S.layout)

    gas_norm = gas_np / (gas_np.max() + 1e-6)

    x = np.stack([gas_norm, fans_np, layout_np], axis=0)
    x = torch.from_numpy(x).float().unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        prob = torch.sigmoid(model(x)).item()

    label = "UNSAFE" if prob > threshold else "SAFE"

    S.history.append(prob)
    S.history = S.history[-150:]


    # =====================================================
    # 🔊 TRUE PLAY/STOP (THIS FIXES YOUR BUG)
    # =====================================================

    if S.alarm_enabled and label == "UNSAFE":
        audio_slot.markdown(audio_html("beep.mp3"), unsafe_allow_html=True)
    else:
        audio_slot.empty()   # <-- removes element → stops instantly


    # =====================================================
    # PLOT
    # =====================================================

    fig, ax = plt.subplots(figsize=(3.8, 3.8))
    ax.imshow(layout_np, cmap="gray", alpha=0.25, origin="lower")
    ax.imshow(gas_np, cmap="inferno", norm=LogNorm(vmin=1e-4, vmax=1), origin="lower")

    color = "red" if label == "UNSAFE" else "green"
    ax.set_title(f"{label} | Risk={prob:.2f}", color=color)
    ax.axis("off")

    plot_area.pyplot(fig)
    plt.close(fig)

    prob_area.progress(prob)
    status_area.metric("Status", label)
    history_area.line_chart(S.history)

    time.sleep(0.03)