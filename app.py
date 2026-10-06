import os

# Must be set BEFORE importing TensorFlow/Keras-based libraries
os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tempfile

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from keras_facenet import FaceNet
from deepface import DeepFace

st.set_page_config(
    page_title="IVA Vision Lab",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.stApp {
    background:
        radial-gradient(circle at 92% 3%, rgba(255,105,180,.16), transparent 25%),
        radial-gradient(circle at 78% 92%, rgba(147,197,253,.18), transparent 28%),
        linear-gradient(135deg, #fffafd 0%, #fff7fb 48%, #f8f7ff 100%);
    color: #30243b;
}
.main .block-container {
    max-width: 1280px;
    padding-top: 2.5rem;
    padding-bottom: 3rem;
}
[data-testid="stHeader"] { background: rgba(255,250,253,.86); }

[data-testid="stSidebar"] {
    background:
        radial-gradient(circle at 10% 5%, rgba(255,255,255,.78), transparent 25%),
        linear-gradient(180deg, #fff0f8 0%, #f3edff 54%, #eaf7ff 100%);
    border-right: 1px solid #f0d9ea;
}
[data-testid="stSidebar"] > div:first-child { padding-top: 2rem; }
[data-testid="stSidebar"] * { color: #40344d; }
[data-testid="stSidebar"] hr { border-color: #e5d8e9 !important; }

.sidebar-logo { display:flex; align-items:center; gap:10px; margin-bottom:4px; }
.sidebar-logo-icon {
    width:40px; height:40px; border-radius:14px;
    display:flex; align-items:center; justify-content:center;
    background:linear-gradient(135deg,#ff4fa3,#ff6b81);
    box-shadow:0 8px 18px rgba(255,79,163,.24);
    font-size:21px;
}
.sidebar-title { font-size:22px; font-weight:900; color:#33263e; }
.sidebar-subtitle { color:#8a7b96 !important; font-size:14px; margin-top:5px; margin-bottom:24px; }
.sidebar-section { font-size:16px; font-weight:850; color:#493551; margin:4px 0 15px 0; }

[data-testid="stSidebar"] [data-baseweb="slider"] [role="slider"] {
    background:#ff4d88 !important; border-color:#ff4d88 !important;
}
[data-testid="stSidebar"] [data-baseweb="slider"] > div > div {
    background:#eadfe9 !important;
}

.hero-card {
    position:relative; overflow:hidden;
    border:1px solid #f2c5df; border-radius:28px;
    padding:28px 32px; margin-bottom:24px;
    background:
        radial-gradient(circle at 94% 22%, rgba(255,105,180,.18), transparent 22%),
        radial-gradient(circle at 80% 92%, rgba(96,165,250,.16), transparent 25%),
        linear-gradient(135deg,#fff0f8 0%,#f7f0ff 55%,#eef8ff 100%);
    box-shadow:0 16px 42px rgba(174,95,153,.12);
}
.hero-card:after {
    content:""; position:absolute; width:110px; height:110px;
    right:32px; top:24px; border-radius:34px;
    background:linear-gradient(135deg,rgba(255,78,164,.18),rgba(96,165,250,.18));
    transform:rotate(15deg);
}
.hero-top { display:flex; align-items:center; gap:9px; position:relative; z-index:2; }
.hero-logo {
    width:34px; height:34px; display:flex; align-items:center; justify-content:center;
    border-radius:12px; background:linear-gradient(135deg,#ff4fa3,#ff758c);
    box-shadow:0 7px 18px rgba(255,79,163,.22); font-size:18px;
}
.hero-title { font-size:30px; line-height:1.1; font-weight:900; color:#30243d; }
.hero-text {
    max-width:720px; margin-top:10px; font-size:16px; line-height:1.65;
    color:#78687f; position:relative; z-index:2;
}
.hero-badge {
    display:inline-block; margin-top:15px; padding:7px 12px; border-radius:999px;
    background:rgba(255,255,255,.78); border:1px solid #f0c8df;
    color:#c13d78; font-size:12px; font-weight:800; position:relative; z-index:2;
}

.upload-card {
    border:1px solid #f3c6df; border-radius:24px; padding:24px;
    background:linear-gradient(135deg,rgba(255,238,248,.94),rgba(247,239,255,.90));
    box-shadow:0 12px 32px rgba(217,100,162,.10);
}
.upload-title { font-size:22px; font-weight:900; color:#3c2d48; margin-bottom:4px; }
.upload-description { color:#89798f; font-size:14px; line-height:1.6; margin-bottom:15px; }

.tip-card {
    border:1px solid #cfe1fa; border-radius:22px; padding:22px;
    background:linear-gradient(135deg,#eef7ff,#f5f0ff);
    box-shadow:0 10px 28px rgba(92,145,205,.09); min-height:190px;
}
.tip-icon { font-size:24px; margin-bottom:8px; }
.tip-title { font-size:16px; font-weight:900; color:#38506e; margin-bottom:8px; }
.tip-text { color:#6f7f93; font-size:14px; line-height:1.65; }

.empty-card {
    border:1px solid #ead9ef; border-radius:24px; padding:26px; margin-top:20px;
    background:
        radial-gradient(circle at 92% 20%, rgba(255,79,163,.12), transparent 23%),
        linear-gradient(135deg,#fff5fb,#f4f0ff 65%,#eef8ff);
    box-shadow:0 12px 32px rgba(135,101,161,.08);
}
.empty-icon { font-size:30px; margin-bottom:5px; }
.empty-title { font-size:26px; font-weight:900; color:#3b2c47; margin-bottom:5px; }
.empty-text { color:#84748e; font-size:15px; }

section[data-testid="stFileUploader"] {
    border:2px dashed #f28abb !important; border-radius:18px !important;
    padding:8px !important; background:rgba(255,255,255,.74) !important;
    box-shadow:0 7px 22px rgba(226,101,163,.08);
}
section[data-testid="stFileUploader"] [data-testid="stFileUploaderDropzone"] {
    background:#fffafe !important; border-radius:13px !important;
}
section[data-testid="stFileUploader"] button,
.stButton > button {
    background:linear-gradient(135deg,#ff4fa3,#ff5e78) !important;
    color:white !important; border:0 !important; border-radius:12px !important;
    font-weight:800 !important; box-shadow:0 7px 18px rgba(255,79,163,.22) !important;
}
section[data-testid="stFileUploader"] button:hover,
.stButton > button:hover {
    background:linear-gradient(135deg,#ed3d93,#f04e68) !important;
    transform:translateY(-1px);
}

.stTabs [data-baseweb="tab-list"] {
    gap:7px; background:#f7edf8; padding:7px; border-radius:15px;
}
.stTabs [data-baseweb="tab"] { color:#765a79; font-weight:800; border-radius:10px; }
.stTabs [aria-selected="true"] {
    background:linear-gradient(135deg,#ff4fa3,#ff5e78) !important; color:white !important;
}

div[data-testid="stMetric"] {
    background:rgba(255,255,255,.88); border:1px solid #f0c9df; border-radius:17px;
    padding:15px; box-shadow:0 8px 22px rgba(188,99,153,.08);
}
div[data-testid="stMetricLabel"] { color:#87677d !important; }
div[data-testid="stMetricValue"] { color:#d53b78 !important; }

.result-card {
    border:1px solid #f0c6dc; border-radius:18px; padding:20px;
    background:rgba(255,255,255,.90); box-shadow:0 9px 25px rgba(190,92,147,.08);
}
.small-label {
    color:#c13d78; font-size:12px; font-weight:900;
    text-transform:uppercase; letter-spacing:.08em; margin-bottom:10px;
}
[data-testid="stDataFrame"] { border:1px solid #edc5dc; border-radius:14px; overflow:hidden; }
h1,h2,h3 { color:#44324f; }
[data-testid="stAlert"] { border-radius:14px; }
.footer { text-align:center; color:#9b899f; font-size:12px; padding:14px 0 5px; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_facenet():
    return FaceNet()


@st.cache_resource
def load_face_cascade():
    return cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )


def detect_faces(image_bgr, scale_factor, min_neighbors):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    return load_face_cascade().detectMultiScale(
        gray, scaleFactor=scale_factor, minNeighbors=min_neighbors, minSize=(30, 30)
    )


def draw_face_boxes(image_bgr, faces):
    output = image_bgr.copy()
    for i, (x, y, w, h) in enumerate(faces, 1):
        box_color = (90, 80, 255)
        cv2.rectangle(output, (x, y), (x + w, y + h), box_color, 3)
        cv2.putText(
            output, f"Face {i}", (x, max(y - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, box_color, 2
        )
    return cv2.cvtColor(output, cv2.COLOR_BGR2RGB)


def get_embeddings(image_bgr, faces):
    model = load_facenet()
    embeddings = []
    for x, y, w, h in faces:
        face = image_bgr[y:y + h, x:x + w]
        if face.size == 0:
            continue
        face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        embeddings.append(model.embeddings([face_rgb])[0])
    return embeddings


def template_match(main_bgr, template_bgr):
    main_gray = cv2.cvtColor(main_bgr, cv2.COLOR_BGR2GRAY)
    template_gray = cv2.cvtColor(template_bgr, cv2.COLOR_BGR2GRAY)
    th, tw = template_gray.shape[:2]
    mh, mw = main_gray.shape[:2]
    if th > mh or tw > mw:
        return None
    result = cv2.matchTemplate(main_gray, template_gray, cv2.TM_CCOEFF_NORMED)
    _, score, _, location = cv2.minMaxLoc(result)
    x, y = location
    marked = main_bgr.copy()
    cv2.rectangle(marked, (x, y), (x + tw, y + th), (60, 120, 255), 4)
    return {
        "score": float(score),
        "location": (int(x), int(y)),
        "size": (int(tw), int(th)),
        "image": cv2.cvtColor(marked, cv2.COLOR_BGR2RGB),
    }


def run_deepface(image_bgr):
    temp = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
    path = temp.name
    temp.close()
    try:
        cv2.imwrite(path, image_bgr)
        result = DeepFace.analyze(
            img_path=path,
            actions=["age", "gender", "emotion"],
            enforce_detection=False,
        )
        return result[0] if isinstance(result, list) else result
    finally:
        if os.path.exists(path):
            os.remove(path)


# ---------------- Sidebar ----------------
with st.sidebar:
    st.markdown("""
    <div class="sidebar-logo">
        <div class="sidebar-logo-icon">🧠</div>
        <div class="sidebar-title">IVA Vision Lab</div>
    </div>
    <div class="sidebar-subtitle">Computer Vision Dashboard</div>
    """, unsafe_allow_html=True)

    st.divider()

    st.markdown(
        '<div class="sidebar-section">⚙️ Analysis Controls</div>',
        unsafe_allow_html=True
    )

    scale_factor = st.slider("Face detection scale", 1.05, 1.30, 1.10, 0.05)
    min_neighbors = st.slider("Detection sensitivity", 3, 10, 5)
    show_embedding = st.checkbox("Show FaceNet vector", True)

    st.divider()

    st.markdown(
        '<div class="sidebar-section">🧩 Pipeline</div>',
        unsafe_allow_html=True
    )

    st.markdown("""
    <div style="line-height:2.0;font-size:14px;">
        <div>🔵 <b>01</b> Face Detection</div>
        <div>🟣 <b>02</b> Face Embedding</div>
        <div>🟢 <b>03</b> Template Search</div>
        <div>🟠 <b>04</b> Facial Attributes</div>
    </div>
    """, unsafe_allow_html=True)


# ---------------- Hero ----------------
st.markdown("""
<div class="hero-card">
    <div class="hero-top">
        <div class="hero-logo">🧠</div>
        <div class="hero-title">IVA Vision Lab</div>
    </div>
    <div class="hero-text">
        A redesigned workspace for face detection, embeddings,
        template matching and facial attribute analysis.
    </div>
    <div class="hero-badge">✦ AI-Powered Computer Vision Workspace</div>
</div>
""", unsafe_allow_html=True)


# ---------------- Upload area ----------------
left, right = st.columns([2.15, 0.85], gap="large")

with left:
    st.markdown("""
    <div class="upload-card">
        <div class="upload-title">📤 Upload Main Image</div>
        <div class="upload-description">
            Choose a clear image to begin your AI computer vision analysis.
            <br>Supported formats: JPG, JPEG, PNG.
        </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Choose an image",
        type=["jpg", "jpeg", "png"],
        help="Upload the image you want IVA Vision Lab to analyze.",
    )

    st.markdown("</div>", unsafe_allow_html=True)

with right:
    st.markdown("""
    <div class="tip-card">
        <div class="tip-icon">💡</div>
        <div class="tip-title">Quick Tip</div>
        <div class="tip-text">
            For best results, use a clear image with visible faces
            and good lighting.
        </div>
    </div>
    """, unsafe_allow_html=True)


# ---------------- Empty state ----------------
if uploaded_file is None:
    st.markdown("""
    <div class="empty-card">
        <div class="empty-icon">✨</div>
        <div class="empty-title">Start a new analysis</div>
        <div class="empty-text">
            Upload an image above to generate all result views.
            Your face detection, FaceNet, template matching and
            DeepFace tools will appear here.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(
        '<div class="footer">IVA Vision Lab • Viola-Jones • FaceNet • Template Matching • DeepFace</div>',
        unsafe_allow_html=True
    )
    st.stop()


# ---------------- Image preparation ----------------
pil_image = Image.open(uploaded_file).convert("RGB")
image_bgr = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
faces = detect_faces(image_bgr, scale_factor, min_neighbors)

st.success(f"Loaded: {uploaded_file.name}")


# ---------------- Summary ----------------
c1, c2, c3 = st.columns(3)
c1.metric("Image Width", f"{pil_image.width}px")
c2.metric("Image Height", f"{pil_image.height}px")
c3.metric("Faces Found", len(faces))


tabs = st.tabs([
    "📊 Overview",
    "🎯 Face Detection",
    "🔢 FaceNet",
    "🧩 Template Match",
    "🙂 DeepFace",
])


# ---------------- Overview ----------------
with tabs[0]:
    st.subheader("Original Image")
    a, b = st.columns([2, 1])

    with a:
        st.image(pil_image, caption="Uploaded source image", use_container_width=True)

    with b:
        st.markdown(
            '<div class="result-card"><div class="small-label">Pipeline Status</div>',
            unsafe_allow_html=True
        )
        st.write("✅ Image loaded")
        st.write(f"🎯 {len(faces)} face(s) detected")
        st.write("🔢 FaceNet ready")
        st.write("🧩 Template matching ready")
        st.write("🙂 DeepFace ready")
        st.markdown("</div>", unsafe_allow_html=True)

    st.subheader("Detected Face Map")

    if len(faces):
        st.image(
            draw_face_boxes(image_bgr, faces),
            caption="All detected faces",
            use_container_width=True
        )
    else:
        st.warning("No face was detected in the uploaded image.")


# ---------------- Face Detection ----------------
with tabs[1]:
    st.subheader("🎯 Viola-Jones Face Detection")

    if not len(faces):
        st.warning("No faces detected. Adjust sensitivity in the sidebar.")
    else:
        a, b = st.columns([2, 1])

        with a:
            st.image(
                draw_face_boxes(image_bgr, faces),
                caption="Viola-Jones bounding boxes",
                use_container_width=True
            )

        with b:
            st.metric("Detected faces", len(faces))
            rows = [
                {"Face": i, "X": int(x), "Y": int(y), "Width": int(w), "Height": int(h)}
                for i, (x, y, w, h) in enumerate(faces, 1)
            ]
            st.dataframe(
                pd.DataFrame(rows),
                hide_index=True,
                use_container_width=True
            )


# ---------------- FaceNet ----------------
with tabs[2]:
    st.subheader("🔢 FaceNet Embedding Output")

    if not len(faces):
        st.warning("FaceNet needs at least one detected face.")
    else:
        with st.spinner("Generating FaceNet embeddings..."):
            embeddings = get_embeddings(image_bgr, faces)

        st.success(f"Generated {len(embeddings)} embedding(s).")

        for i, embedding in enumerate(embeddings, 1):
            with st.expander(f"Face {i} — {len(embedding)} dimensions", expanded=True):
                x, y = st.columns(2)
                x.metric("Embedding dimension", len(embedding))
                y.metric("Vector norm", f"{float(np.linalg.norm(embedding)):.4f}")

                if show_embedding:
                    df = pd.DataFrame({
                        "Index": np.arange(1, min(11, len(embedding) + 1)),
                        "Value": embedding[:10],
                    })
                    st.dataframe(
                        df,
                        hide_index=True,
                        use_container_width=True
                    )
                else:
                    st.info("Enable 'Show FaceNet vector' in the sidebar.")


# ---------------- Template Matching ----------------
with tabs[3]:
    st.subheader("🧩 Template Matching")
    st.write("Upload a smaller image and search for it inside the main image.")

    template_file = st.file_uploader(
        "Choose template image",
        type=["jpg", "jpeg", "png"],
        key="new_template",
    )

    if template_file is None:
        st.info("No template selected yet.")
    else:
        template_pil = Image.open(template_file).convert("RGB")
        template_bgr = cv2.cvtColor(np.array(template_pil), cv2.COLOR_RGB2BGR)
        result = template_match(image_bgr, template_bgr)

        if result is None:
            st.error("The template must be smaller than the main image.")
        else:
            a, b = st.columns([2, 1])

            with a:
                st.image(
                    result["image"],
                    caption="Best template location",
                    use_container_width=True
                )

            with b:
                score = result["score"]
                st.metric("Matching score", f"{score:.3f}")
                st.write(f"Location: {result['location']}")
                st.write(
                    f"Template size: {result['size'][0]} × {result['size'][1]}"
                )

                if score >= 0.70:
                    st.success("Strong match")
                elif score >= 0.50:
                    st.warning("Possible match")
                else:
                    st.error("Weak match")


# ---------------- DeepFace ----------------
with tabs[4]:
    st.subheader("🙂 DeepFace Facial Analysis")

    if st.button("Run DeepFace Analysis", type="primary", use_container_width=True):
        try:
            with st.spinner("DeepFace is analyzing the image..."):
                analysis = run_deepface(image_bgr)

            a, b, c = st.columns(3)
            a.metric("Age", analysis.get("age", "N/A"))
            b.metric("Gender", analysis.get("dominant_gender", "N/A"))
            c.metric("Dominant emotion", analysis.get("dominant_emotion", "N/A"))

            scores = analysis.get("emotion", {})

            if scores:
                st.markdown("### Emotion Score Distribution")

                emotion_df = pd.DataFrame({
                    "Emotion": list(scores.keys()),
                    "Score": list(scores.values()),
                }).set_index("Emotion")

                st.bar_chart(emotion_df)

                st.markdown("### Raw Analysis")
                st.json({
                    "age": analysis.get("age"),
                    "gender": analysis.get("dominant_gender"),
                    "dominant_emotion": analysis.get("dominant_emotion"),
                    "emotion_scores": scores,
                })

        except Exception as exc:
            st.error("DeepFace could not complete the analysis.")
            st.exception(exc)
    else:
        st.info("Click the button above to run DeepFace.")


st.divider()
st.markdown(
    '<div class="footer">IVA Vision Lab • Viola-Jones • FaceNet • Template Matching • DeepFace</div>',
    unsafe_allow_html=True
)
