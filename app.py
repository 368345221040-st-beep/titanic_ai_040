"""
ระบบทำนายการรอดชีวิตผู้โดยสารเรือไททานิก (Titanic Survival Prediction)
พัฒนาโดย: นาย ศักรินทร์ เกิดศิริ 040

วิธีรัน:
    streamlit run app.py

หมายเหตุสำคัญเรื่อง Feature:
    โมเดล titanic_mlp.keras รับ input 5 ค่า ในไฟล์นี้ตั้งสมมติฐานว่าลำดับฟีเจอร์คือ
        [Pclass, Sex(0=หญิง,1=ชาย), Age, FamilySize(SibSp+Parch), Fare]
    และไม่มีการ scale ค่าก่อนป้อนเข้าโมเดล
    ถ้าตอนเทรนโมเดลมีการ StandardScaler/MinMaxScaler หรือฟีเจอร์คนละชุด
    ให้แก้ไขได้ที่ฟังก์ชัน build_feature_vector() ด้านล่างจุดเดียว
"""

import numpy as np
import streamlit as st
import tensorflow as tf

# ============================================================
# ค่าคงที่ที่ควรแก้ไขให้ตรงกับผลการเทรนจริง
# ============================================================
MODEL_PATH = "titanic_mlp.keras"
MODEL_ACCURACY = 0.82  # TODO: แก้ไขค่านี้ให้ตรงกับ accuracy จริงจากการเทรน/ทดสอบโมเดล
APP_TITLE = "ระบบทำนายการรอดชีวิตผู้โดยสารเรือไททานิก"
DEVELOPER_NAME = "นาย ศักรินทร์ เกิดศิริ 040"


# ============================================================
# การตั้งค่าหน้าเว็บ + สไตล์มินิมอล
# ============================================================
st.set_page_config(
    page_title=APP_TITLE,
    page_icon="🚢",
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
        :root {
            --bg: #fafaf9;
            --card: #ffffff;
            --primary: #2f6690;
            --primary-dark: #16425b;
            --accent: #d9dcd6;
            --text: #2b2b2b;
            --muted: #6b7280;
        }
        html, body, [class*="css"] {
            font-family: "Segoe UI", "Sarabun", sans-serif;
        }
        .stApp {
            background-color: var(--bg);
        }
        .app-header {
            text-align: center;
            padding: 1.4rem 1rem 1rem 1rem;
            margin-bottom: 1.2rem;
            background: linear-gradient(135deg, var(--primary-dark), var(--primary));
            border-radius: 16px;
            color: white;
        }
        .app-header h1 {
            font-size: 1.6rem;
            margin: 0;
            font-weight: 700;
        }
        .app-header p {
            margin: 0.3rem 0 0 0;
            font-size: 0.9rem;
            opacity: 0.9;
        }
        .card {
            background: var(--card);
            border: 1px solid var(--accent);
            border-radius: 14px;
            padding: 1.4rem;
            margin-bottom: 1rem;
        }
        .result-box {
            border-radius: 14px;
            padding: 1.2rem;
            text-align: center;
            margin-top: 0.6rem;
        }
        .result-survive {
            background-color: #e6f4ea;
            border: 1px solid #34a853;
            color: #1e7e34;
        }
        .result-not-survive {
            background-color: #fdecea;
            border: 1px solid #d93025;
            color: #a1291d;
        }
        .footer-box {
            text-align: center;
            margin-top: 2rem;
            padding-top: 1rem;
            border-top: 1px solid var(--accent);
            color: var(--muted);
            font-size: 0.85rem;
        }
        .accuracy-badge {
            display: inline-block;
            background-color: #eef3f8;
            color: var(--primary-dark);
            border: 1px solid var(--primary);
            border-radius: 999px;
            padding: 0.3rem 0.9rem;
            font-size: 0.85rem;
            font-weight: 600;
        }
        div.stButton > button {
            background-color: var(--primary);
            color: white;
            border-radius: 10px;
            border: none;
            padding: 0.55rem 1.2rem;
            font-weight: 600;
            width: 100%;
        }
        div.stButton > button:hover {
            background-color: var(--primary-dark);
            color: white;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# โหลดโมเดล (cache ไว้ไม่ให้โหลดซ้ำทุกครั้งที่กดปุ่ม)
# ============================================================
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


def build_feature_vector(pclass: int, sex: int, age: float, family_size: int, fare: float) -> np.ndarray:
    """
    รวมค่าที่รับจากผู้ใช้ให้เป็น input vector ขนาด (1, 5)
    เรียงลำดับ: [Pclass, Sex, Age, FamilySize, Fare]
    แก้ไขที่นี่จุดเดียวหากลำดับ/การ scale ฟีเจอร์จริงไม่ตรงกับสมมติฐาน
    """
    features = np.array([[pclass, sex, age, family_size, fare]], dtype=np.float32)
    return features


# ============================================================
# ส่วนหัวของหน้าเว็บ
# ============================================================
st.markdown(
    f"""
    <div class="app-header">
        <h1>🚢 {APP_TITLE}</h1>
        <p>กรอกข้อมูลผู้โดยสารเพื่อทำนายโอกาสในการรอดชีวิตด้วย Neural Network</p>
    </div>
    """,
    unsafe_allow_html=True,
)

try:
    model = load_model()
    model_ready = True
except Exception as e:
    model_ready = False
    st.error(f"ไม่สามารถโหลดโมเดลได้: {e}")


# ============================================================
# ฟอร์มรับข้อมูล
# ============================================================
with st.container():
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📋 ข้อมูลผู้โดยสาร")

    col1, col2 = st.columns(2)

    with col1:
        pclass_label = st.selectbox(
            "ชั้นโดยสาร (Pclass)",
            options=["ชั้น 1 (Upper)", "ชั้น 2 (Middle)", "ชั้น 3 (Lower)"],
        )
        pclass = int(pclass_label.split()[1]) if False else {"ชั้น 1 (Upper)": 1, "ชั้น 2 (Middle)": 2, "ชั้น 3 (Lower)": 3}[pclass_label]

        sex_label = st.radio("เพศ (Sex)", options=["ชาย", "หญิง"], horizontal=True)
        sex = 1 if sex_label == "ชาย" else 0

    with col2:
        age = st.slider("อายุ (Age)", min_value=0, max_value=90, value=30)
        fare = st.number_input("ค่าโดยสาร (Fare)", min_value=0.0, max_value=600.0, value=32.0, step=1.0)

    st.markdown("**สมาชิกครอบครัวที่เดินทางด้วย**")
    col3, col4 = st.columns(2)
    with col3:
        sibsp = st.number_input("พี่น้อง/คู่สมรส (SibSp)", min_value=0, max_value=10, value=0, step=1)
    with col4:
        parch = st.number_input("พ่อแม่/ลูก (Parch)", min_value=0, max_value=10, value=0, step=1)

    family_size = sibsp + parch

    st.markdown("</div>", unsafe_allow_html=True)

    predict_clicked = st.button("🔮 ทำนายผล", use_container_width=True, disabled=not model_ready)


# ============================================================
# แสดงผลการทำนาย
# ============================================================
if predict_clicked and model_ready:
    features = build_feature_vector(pclass, sex, age, family_size, fare)
    prediction = model.predict(features, verbose=0)
    survival_prob = float(prediction[0][0])

    if survival_prob >= 0.5:
        st.markdown(
            f"""
            <div class="result-box result-survive">
                <h2>✅ มีแนวโน้ม "รอดชีวิต"</h2>
                <p>ความน่าจะเป็นในการรอดชีวิต: <b>{survival_prob*100:.1f}%</b></p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
            <div class="result-box result-not-survive">
                <h2>⚠️ มีแนวโน้ม "ไม่รอดชีวิต"</h2>
                <p>ความน่าจะเป็นในการรอดชีวิต: <b>{survival_prob*100:.1f}%</b></p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.progress(min(max(survival_prob, 0.0), 1.0))


# ============================================================
# ส่วนท้าย: ความแม่นยำของระบบ + ชื่อผู้พัฒนา
# ============================================================
st.markdown(
    f"""
    <div class="footer-box">
        <span class="accuracy-badge">🎯 ความแม่นยำของโมเดล (Accuracy): {MODEL_ACCURACY*100:.1f}%</span>
        <p style="margin-top:0.8rem;">พัฒนาโดย: {DEVELOPER_NAME}</p>
    </div>
    """,
    unsafe_allow_html=True,
)
