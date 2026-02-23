import streamlit as st
import base64
from pathlib import Path

st.set_page_config(page_title="Alarm Beep Test")

st.title("🚨 Continuous Alarm Beep (Chrome-Safe)")

# ---------- helper: embed audio as base64 ----------
def audio_html(path: str):
    audio_bytes = Path(path).read_bytes()
    b64 = base64.b64encode(audio_bytes).decode()

    return f"""
    <audio id="alarm" autoplay loop>
        <source src="data:audio/wav;base64,{b64}" type="audio/wav">
    </audio>

    <script>
        var a = document.getElementById("alarm");
        a.play();
    </script>
    """


# ---------- UI ----------
st.write("Chrome requires a click before audio can play.")

if st.button("🔊 Start Alarm"):
    st.markdown(audio_html("beep.mp3"), unsafe_allow_html=True)

if st.button("🛑 Stop Alarm"):
    st.markdown(
        """
        <script>
        var a = document.getElementById("alarm");
        if (a) { a.pause(); a.currentTime = 0; }
        </script>
        """,
        unsafe_allow_html=True,
    )