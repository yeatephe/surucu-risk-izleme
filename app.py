import streamlit as st
import cv2
import numpy as np
import tempfile
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

st.set_page_config(page_title="Sürücü Risk İzleme", page_icon="🚗", layout="centered")

# ---- Görünüm ----
st.markdown("""
<style>
h1 {
    background: linear-gradient(90deg, #EF4444, #F87171);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    font-weight: 800 !important;
}
h2 { border-left: 4px solid #EF4444; padding-left: 12px; }
.stButton > button {
    background: linear-gradient(90deg, #DC2626, #EF4444);
    color: white; border: none; border-radius: 10px;
    padding: 0.55rem 1.4rem; font-weight: 600;
}
div[data-testid="stMetric"] {
    background: #1A2332; border: 1px solid #2A3B52;
    border-radius: 12px; padding: 12px 16px;
}
</style>
""", unsafe_allow_html=True)

# Nokta indeksleri
SOL_GOZ = [362, 385, 387, 263, 373, 380]
SAG_GOZ = [33, 160, 158, 133, 153, 144]
AGIZ_DIKEY = [13, 14]
AGIZ_YATAY = [61, 291]
BURUN_UCU = 1
EAR_ESIK = 0.20
MAR_ESIK = 0.6
BAKIS_ESIK = 0.15


@st.cache_resource
def yukle_detector():
    base = python.BaseOptions(model_asset_path="face_landmarker.task")
    opts = vision.FaceLandmarkerOptions(base_options=base, num_faces=1)
    return vision.FaceLandmarker.create_from_options(opts)


detector = yukle_detector()


def nokta_al(n, i, w, h):
    return np.array([n[i].x * w, n[i].y * h])

def ear_hesapla(n, idx, w, h):
    p = [nokta_al(n, i, w, h) for i in idx]
    dikey = np.linalg.norm(p[1]-p[5]) + np.linalg.norm(p[2]-p[4])
    yatay = np.linalg.norm(p[0]-p[3])
    return dikey / (2.0 * yatay)

def mar_hesapla(n, w, h):
    ust = nokta_al(n, AGIZ_DIKEY[0], w, h); alt = nokta_al(n, AGIZ_DIKEY[1], w, h)
    sol = nokta_al(n, AGIZ_YATAY[0], w, h); sag = nokta_al(n, AGIZ_YATAY[1], w, h)
    return np.linalg.norm(ust-alt) / np.linalg.norm(sol-sag)

def bakis_sapmasi(n, w, h):
    burun = nokta_al(n, BURUN_UCU, w, h)
    sol = nokta_al(n, 33, w, h); sag = nokta_al(n, 263, w, h)
    orta = (sol + sag) / 2
    return abs((burun[0] - orta[0]) / (np.linalg.norm(sag - sol) + 1e-6))

def hiz_risk(h, m):
    k = 0
    if h > 130: k += 25
    elif h > 100: k += 12
    if m: k += 20
    return k


st.title("Sürücü Risk İzleme Sistemi")
st.markdown(
    "<p style='color:#94A3B8;'>Sürücü videosunu yükle; yapay zeka yorgunluk, dikkat dağınıklığı "
    "ve (simüle) araç verisini birleştirip risk skoru hesaplasın.</p>", unsafe_allow_html=True)

st.info("ℹ️ Bu bir prototiptir; gerçek bir araç güvenlik sistemi yerine geçmez. "
        "Araç hızı verisi burada simüle edilmiştir (gerçekte OBD portundan gelir).")

yuklenen = st.file_uploader("Sürücü videosu yükle", type=["mp4", "mov", "avi"])

if yuklenen is not None and st.button("Analiz Et", type="primary"):
    with st.spinner("Video işleniyor, lütfen bekleyin..."):
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(yuklenen.read())

        vt = cv2.VideoCapture(tfile.name)
        fps = vt.get(cv2.CAP_PROP_FPS) or 25
        kare_say = vt.get(cv2.CAP_PROP_FRAME_COUNT)
        vt.release()
        sure = max(1, int(kare_say / fps) + 1)

        np.random.seed(0)
        hizlar, manevralar = [], []
        hiz = 100
        for s in range(sure):
            hiz = max(0, min(180, hiz + np.random.randint(-10, 15)))
            hizlar.append(hiz)
            manevralar.append(1 if abs(np.random.randn()) > 1.8 else 0)

        video = cv2.VideoCapture(tfile.name)
        kapali = esneme = dikkat = yuz = 0
        hiz_katkilari, risk_gecmis = [], []
        kare_no = 0

        while True:
            ok, kare = video.read()
            if not ok:
                break
            kare_no += 1
            if kare_no % 3 != 0:  # her 3 karede 1 (hiz icin)
                continue
            rgb = cv2.cvtColor(kare, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            sonuc = detector.detect(mp_image)
            saniye = min(int(kare_no / fps), len(hizlar) - 1)
            anlik = 0
            if sonuc.face_landmarks:
                yuz += 1
                n = sonuc.face_landmarks[0]
                h, w = kare.shape[:2]
                ear = (ear_hesapla(n, SOL_GOZ, w, h) + ear_hesapla(n, SAG_GOZ, w, h)) / 2
                mar = mar_hesapla(n, w, h)
                sapma = bakis_sapmasi(n, w, h)
                if ear < EAR_ESIK: kapali += 1; anlik += 50
                if mar > MAR_ESIK: esneme += 1; anlik += 20
                if sapma > BAKIS_ESIK: dikkat += 1; anlik += 20
                anlik += hiz_risk(hizlar[saniye], manevralar[saniye])
                hiz_katkilari.append(hiz_risk(hizlar[saniye], manevralar[saniye]))
            risk_gecmis.append(min(100, anlik))
        video.release()

        perclos = kapali / yuz * 100 if yuz else 0
        esneme_o = esneme / yuz * 100 if yuz else 0
        dikkat_o = dikkat / yuz * 100 if yuz else 0
        ort_hiz = np.mean(hiz_katkilari) if hiz_katkilari else 0
        risk = min(100, perclos * 1.0 + esneme_o * 0.5 + dikkat_o * 0.6 + ort_hiz)

    st.header("Sonuç")
    c1, c2, c3 = st.columns(3)
    c1.metric("👁️ Göz kapalı (PERCLOS)", f"%{perclos:.0f}")
    c2.metric("🥱 Esneme", f"%{esneme_o:.0f}")
    c3.metric("🧭 Dikkat dağınık", f"%{dikkat_o:.0f}")

    st.progress(int(risk) / 100)
    st.subheader(f"🎯 Birleşik Risk Skoru: {risk:.0f}/100")
    if risk > 40:
        st.error("🔴 YÜKSEK RİSK — Sürücü dinlenmeli, hız düşürülmeli!")
    elif risk > 20:
        st.warning("🟡 ORTA RİSK — Dikkatli olun.")
    else:
        st.success("🟢 DÜŞÜK RİSK — Sürücü uyanık ve dikkatli.")

    # Risk grafigi
    fig, ax = plt.subplots(figsize=(9, 3))
    ax.plot(risk_gecmis, color="#EF4444", linewidth=1.5)
    ax.axhline(40, color="orange", linestyle="--", label="Yüksek risk eşiği")
    ax.set_xlabel("Kare"); ax.set_ylabel("Risk (0-100)")
    ax.set_title("Zaman içinde risk skoru"); ax.legend()
    fig.tight_layout()
    st.pyplot(fig)

    st.caption(f"İşlenen yüz kareleri: {yuz} • Araç hız katkısı: +{ort_hiz:.0f}")
