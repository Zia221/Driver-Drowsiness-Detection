import time
import threading

try:
    import winsound
except ImportError:
    winsound = None

import av
import streamlit as st
from streamlit_webrtc import webrtc_streamer

from src.drowsiness_core import DrowsinessDetector


# ---------------------------------------------------------
# PAGE
# ---------------------------------------------------------

st.set_page_config(
    page_title="DrowsyGuard AI",
    page_icon="🚗",
    layout="centered"
)

st.markdown(
    """
    <style>
    .main {
        max-width: 900px;
        margin: auto;
    }

    h1 {
        text-align: center;
        margin-bottom: 5px;
    }

    .subtitle {
        text-align: center;
        color: #888;
        margin-bottom: 25px;
    }

    .status {
        text-align: center;
        padding: 20px;
        border-radius: 12px;
        background: #162116;
        margin: 20px 0;
    }

    .status-title {
        font-size: 14px;
        color: #999;
    }

    .status-value {
        font-size: 32px;
        font-weight: bold;
        color: #35d07f;
    }

    .metric {
        text-align: center;
        padding: 15px;
        border-radius: 10px;
        background: #151b24;
    }

    .metric-title {
        color: #999;
        font-size: 13px;
    }

    .metric-value {
        font-size: 24px;
        font-weight: bold;
    }

    .footer {
        text-align: center;
        color: #777;
        margin-top: 30px;
        font-size: 12px;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# ---------------------------------------------------------
# TITLE
# ---------------------------------------------------------

st.title("🚗 DrowsyGuard AI")

st.markdown(
    '<div class="subtitle">Driver Drowsiness Detection</div>',
    unsafe_allow_html=True
)


# ---------------------------------------------------------
# DETECTOR
# ---------------------------------------------------------

@st.cache_resource
def load_detector():
    return DrowsinessDetector()


detector = load_detector()


# ---------------------------------------------------------
# SHARED STATE
# ---------------------------------------------------------

lock = threading.Lock()

state = {
    "status": "WAITING",
    "ear": 0.0,
    "left_prediction": 0.0,
    "right_prediction": 0.0,
    "closed_frames": 0
}


# ---------------------------------------------------------
# ALARM
# ---------------------------------------------------------

alarm_active = False

def alarm():

    while alarm_active:

        if winsound is not None:
            winsound.Beep(
                1000,
                500
            )

        time.sleep(0.1)


def start_alarm():

    global alarm_active

    if not alarm_active:

        alarm_active = True

        threading.Thread(
            target=alarm,
            daemon=True
        ).start()


def stop_alarm():

    global alarm_active

    alarm_active = False


# ---------------------------------------------------------
# CAMERA CALLBACK
# ---------------------------------------------------------

def video_frame_callback(frame):

    image = frame.to_ndarray(format="bgr24")

    processed_frame, result = detector.process_frame(image)

    with lock:
        state.update(result)

    if result["status"] == "DROWSY":
        start_alarm()
    else:
        stop_alarm()

    return av.VideoFrame.from_ndarray(
        processed_frame,
        format="bgr24"
    )

# ---------------------------------------------------------
# METERED TURN CONFIGURATION
# ---------------------------------------------------------

metered = st.secrets["metered"]

ice_servers = [
    {
        "urls": "turns:global.relay.metered.ca:443?transport=tcp",
        "username": metered["username"],
        "credential": metered["credential"]
    },
    {
        "urls": "turn:global.relay.metered.ca:443?transport=tcp",
        "username": metered["username"],
        "credential": metered["credential"]
    },
    {
        "urls": "turn:global.relay.metered.ca:80?transport=tcp",
        "username": metered["username"],
        "credential": metered["credential"]
    }
]

rtc_configuration = {
    "iceServers": ice_servers
}
# ---------------------------------------------------------
# CAMERA
# ---------------------------------------------------------

ctx = webrtc_streamer(
    key="drowsiness-camera",
    video_frame_callback=video_frame_callback,
    media_stream_constraints={
        "video": True,
        "audio": False
    },
    frontend_rtc_configuration=rtc_configuration,
    server_rtc_configuration=rtc_configuration
)
# ---------------------------------------------------------
# LIVE STATUS
# ---------------------------------------------------------

status_placeholder = st.empty()
metrics_placeholder = st.empty()


if ctx.state.playing:

    while ctx.state.playing:

        with lock:
            current = state.copy()

        status = current["status"]

        if status == "AWAKE":
            icon = "🟢"
            message = "Driver is alert."

        elif status == "EYES CLOSED":
            icon = "🟡"
            message = "Eyes are closed."

        elif status == "DROWSY":
            icon = "🔴"
            message = "Drowsiness detected! Please take a break."

        else:
            icon = "⚪"
            message = "Waiting for detection..."

        status_placeholder.markdown(
            f"""
            <div class="status">

                <div class="status-title">
                    DRIVER STATUS
                </div>

                <div class="status-value">
                    {icon} {status}
                </div>

                <div>
                    {message}
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )

        cnn_confidence = (
            current["left_prediction"]
            + current["right_prediction"]
        ) / 2

        with metrics_placeholder.container():

            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown(
                    f"""
                    <div class="metric">
                        <div class="metric-title">👁 EAR</div>
                        <div class="metric-value">
                            {current["ear"]:.2f}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            with col2:
                st.markdown(
                    f"""
                    <div class="metric">
                        <div class="metric-title">🧠 CNN Confidence</div>
                        <div class="metric-value">
                            {cnn_confidence:.2f}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            with col3:
                st.markdown(
                    f"""
                    <div class="metric">
                        <div class="metric-title">⏱ Closed Frames</div>
                        <div class="metric-value">
                            {current["closed_frames"]}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

        time.sleep(0.1)


# ---------------------------------------------------------
# FOOTER
# ---------------------------------------------------------

st.markdown(
    """
    <div class="footer">
        DrowsyGuard AI • Drive Safe, Stay Alert
    </div>
    """,
    unsafe_allow_html=True
)