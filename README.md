# 🚗 Driver Risk Monitoring System

An AI-powered computer vision and telemetry prototype designed to monitor driver fatigue, distraction, and unsafe driving conditions in real-time. By processing video feeds using MediaPipe Face Landmarker and combining facial indicators with simulated vehicle telemetry, the system calculates a unified **Driver Risk Score**.

## 🚀 Live Demo

👉 [Try the app here](https://surucu-risk-izleme.streamlit.app)

## 🛠️ Tech Stack

- **Python** – core programming language
- **MediaPipe (FaceLandmarker)** – 3D facial landmark detection (478 landmark points)
- **OpenCV** – video processing and frame extraction
- **Streamlit** – web interface and deployment
- **Matplotlib & NumPy** – dynamic data processing and real-time risk timeline plotting

## 📊 About the Project

The application evaluates driver safety in real-time by combining face mesh analytics with vehicle speed and maneuver telemetry.

Fatigue is measured using the **Eye Aspect Ratio (EAR)** for eye closure (computing the PERCLOS metric) and the **Mouth Aspect Ratio (MAR)** for yawning frequency. Distraction is tracked by monitoring nose-to-eye alignment to detect head pose and gaze deviation.

These visual risk factors are fused with simulated vehicle telemetries (such as high speeds or sudden maneuvers normally collected via OBD-II) to calculate a unified risk score on a **0–100 scale**:

- 🟢 **0 – 20 (Low Risk):** Driver is alert and attentive.
- 🟡 **21 – 40 (Moderate Risk):** Warning state; driver should stay focused.
- 🔴 **40+ (High Risk):** Critical alert; driver must pull over and rest.

## 💻 Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## 📁 Repository Structure

```text
surucu-risk-izleme/
├── app.py                  # Main Streamlit application & risk engine
├── face_landmarker.task    # MediaPipe Face Landmarker model asset
├── requirements.txt        # Python package dependencies
├── packages.txt            # Linux system dependencies for headless deployment
└── config.toml             # Custom Streamlit UI theme configuration
```

## ⚠️ Note

This tool provides an AI-based visual pre-assessment and does not replace an official automotive safety or driver assistance system (ADAS).

## 👤 Author

Yiğit Efe USTA – Computer Engineering Student
