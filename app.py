import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2


# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(
    page_title="Fake Image Detection System",
    page_icon="🧠",
    layout="wide"
)

# -------------------------------------------------
# CUSTOM UI STYLE
# -------------------------------------------------
st.markdown("""
<style>

.block-container{
    padding-top:2rem;
    max-width:1200px;
    margin:auto;
}

.title{
    text-align:center;
    font-size:40px;
    font-weight:700;
    background: linear-gradient(90deg,#00c6ff,#0072ff);
    -webkit-background-clip:text;
    -webkit-text-fill-color:transparent;
}

.subtitle{
    text-align:center;
    color:#94a3b8;
    margin-bottom:30px;
}

.card{
    background:#0f2740;
    padding:18px;
    border-radius:12px;
    margin-bottom:15px;
}

.result-real{
    background:#1cc88a;
    padding:14px;
    border-radius:10px;
    text-align:center;
    font-weight:bold;
    color:white;
}

.result-fake{
    background:#e74a3b;
    padding:14px;
    border-radius:10px;
    text-align:center;
    font-weight:bold;
    color:white;
}

.sidebar-card{
    background:#1f2a38;
    padding:20px;
    border-radius:10px;
    color:white;
}

            .analysis-card{
background:#12263f;
padding:14px;
border-radius:10px;
margin-bottom:12px;
font-size:13px;
}
         
</style>
""", unsafe_allow_html=True)


# -------------------------------------------------
# HEADER
# -------------------------------------------------
st.markdown('<div class="title">Fake Image Detection System</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Explainable AI Image Forensic Dashboard</div>', unsafe_allow_html=True)


# -------------------------------------------------
# LOAD MODEL
# -------------------------------------------------
model = tf.keras.models.load_model("fake_image_model.h5")
IMG_SIZE = 128


# -------------------------------------------------
# DEFAULT VARIABLES
# -------------------------------------------------
label = "No image analyzed yet"
confidence = 0
texture_text = ""
edge_text = ""
light_text = ""
sym_text = ""

texture_flag = "neutral"
edge_flag = "neutral"
light_flag = "neutral"


# -------------------------------------------------
# IMAGE ANALYSIS FUNCTIONS
# -------------------------------------------------
def texture_analysis(image):

    global texture_flag

    img = np.array(image)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    score = cv2.Laplacian(gray, cv2.CV_64F).var()

    if score > 400:
        texture_flag = "real"
        return f"""
Strong natural texture detected (score: {score:.1f})

This image contains rich micro-details such as skin pores, fabric
patterns and natural surface variations typical in real photographs.
"""
    elif score < 50:
        texture_flag = "ai"
        return f"""
Very low texture variation detected (score: {score:.1f})

Surfaces appear overly smooth which is common in AI-generated images.
"""
    else:
        texture_flag = "neutral"
        return f"""
Moderate texture variation detected (score: {score:.1f})

Both real photographs and AI images may produce this pattern.
"""
    return texture_text, score

def edge_analysis(image):

    global edge_flag

    img = np.array(image)

    edges = cv2.Canny(img,100,200)
    edge_pixels = np.sum(edges > 0)

    if edge_pixels > 20000:
        edge_flag = "real"
        return f"""
High edge density detected ({edge_pixels} pixels)

Natural images contain irregular edges caused by hair strands,
fabric folds and camera noise.
"""
    elif edge_pixels < 6000:
        edge_flag = "ai"
        return f"""
Very low edge density detected ({edge_pixels} pixels)

Edges appear overly smooth which may indicate AI-generated content.
"""
    else:
        edge_flag = "neutral"
        return f"""
Moderate edge density detected ({edge_pixels} pixels)

The image structure appears reasonably natural.
"""
    return text, edge_pixels


def lighting_analysis(image):

    global light_flag

    img = np.array(image)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    contrast = np.std(gray)

    if contrast > 45:
        light_flag = "real"
        return f"""
Natural lighting variation detected (contrast: {contrast:.1f})

Real photos usually contain uneven lighting caused by shadows
and reflections.
"""
    else:
        light_flag = "ai"
        return f"""
Lighting appears unusually uniform (contrast: {contrast:.1f})

AI generated images sometimes produce flat lighting patterns.
"""
    return text, contrast

def symmetry_analysis(image):

    img = np.array(image)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    h,w = gray.shape

    mid = w//2

    left = gray[:,:mid]
    right = np.fliplr(gray[:,mid:])

    min_w = min(left.shape[1],right.shape[1])

    left = left[:,:min_w]
    right = right[:,:min_w]

    diff = np.mean(np.abs(left-right))

    return f"""
Facial symmetry difference value: {diff:.1f}

Human faces normally contain small asymmetries. Extremely perfect
symmetry may sometimes indicate synthetic generation.
"""  
    return text, diff

# -------------------------------------------------
# IMAGE SOURCE
# -------------------------------------------------
image_source = st.radio(
    "Select Image Source",
    ["Upload Image","Capture From Camera"]
)

image = None

if image_source == "Upload Image":

    uploaded_file = st.file_uploader(
        "Upload Image",
        type=["jpg","jpeg","png"]
    )

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")


if image_source == "Capture From Camera":

    cam = st.camera_input("Capture Image")

    if cam:
        image = Image.open(cam).convert("RGB")


# -------------------------------------------------
# MAIN DASHBOARD
# -------------------------------------------------
if image is not None:

    # -------- TOP ROW --------
    col1,col2 = st.columns([1.2,1])

    # IMAGE
    with col1:
        st.markdown("### Uploaded Image")
        st.image(image, width=200)

    # PREDICTION
    with col2:

        st.markdown("### Prediction")

        img = image.resize((IMG_SIZE,IMG_SIZE))
        img = np.array(img)/255.0
        img = np.expand_dims(img,axis=0)

        prediction = model.predict(img)

        raw_score = float(prediction[0][0])

        ai_prob = raw_score
        real_prob = 1 - raw_score

        st.write("Raw Model Score:",round(raw_score,4))

        c1,c2 = st.columns(2)

        c1.metric("AI Generated",f"{ai_prob*100:.2f}%")
        c2.metric("Real Image",f"{real_prob*100:.2f}%")


        # FINAL DECISION
        if ai_prob > 0.60:
            label = "AI GENERATED IMAGE"
            confidence = ai_prob

        elif 0.55 <= ai_prob <= 0.60:
            label = "LIKELY AI GENERATED"
            confidence = ai_prob

        elif 0.50 <= ai_prob < 0.55:
            label = "LIKELY REAL IMAGE"
            confidence = real_prob

        else:
            label = "REAL IMAGE"
            confidence = real_prob


        if "AI" in label:
            st.markdown(
                '<div class="result-fake">⚠ '+label+'</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                '<div class="result-real">✅ '+label+'</div>',
                unsafe_allow_html=True
            )

        st.progress(confidence)
        st.metric("Confidence Score",f"{confidence*100:.2f}%")


# -------------------------------------------------
# SIDEBAR IMAGE ANALYSIS
# -------------------------------------------------

st.sidebar.markdown("## 🔎 Image Analysis")

if image is not None:

    st.sidebar.markdown(
        f"""
<div class="analysis-card">
<b>Texture Analysis</b><br><br>
{texture_analysis(image)}
</div>
""",
        unsafe_allow_html=True
    )

    st.sidebar.markdown(
        f"""
<div class="analysis-card">
<b>Edge Density</b><br><br>
{edge_analysis(image)}
</div>
""",
        unsafe_allow_html=True
    )

    st.sidebar.markdown(
        f"""
<div class="analysis-card">
<b>Lighting Analysis</b><br><br>
{lighting_analysis(image)}
</div>
""",
        unsafe_allow_html=True
    )

    st.sidebar.markdown(
        f"""
<div class="analysis-card">
<b>Facial Symmetry</b><br><br>
{symmetry_analysis(image)}
</div>
""",
        unsafe_allow_html=True
    )

# -------------------------------------------------
# FORENSIC REPORT (BOTTOM)
# -------------------------------------------------

if image is not None:
    st.markdown("## 📄 Image Forensic Report")

    report = f"""
<div class="sidebar-report">
<b>Prediction Result:</b>  <br>
 <b>                                        {label} </b>
<br>
<br>
<b>Confidence Score:</b> <br>
                                {confidence*100:.2f} %
<br><br>
<b>Conclusion:</b><br>
The deep learning model analyzed texture patterns,
edge density, lighting distribution and facial symmetry
to determine whether the image is AI-generated or real.
<br><br>
<b>Disclaimer:</b><br>
AI-assisted predictions may not always be 100% accurate.
Always verify results using additional forensic tools.
</div>
"""

    st.markdown(report, unsafe_allow_html=True)