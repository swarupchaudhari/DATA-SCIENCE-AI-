# streamlit run app.py

import streamlit as st
import pandas as pd
import pickle

# =========================
# PAGE CONFIG
# =========================

st.set_page_config(
    page_title="AI Placement Predictor",
    page_icon="🎓",
    layout="wide"
)

# =========================
# CUSTOM CSS
# =========================

st.markdown("""
<style>

/* Main Background */

.stApp{
    background: linear-gradient(to bottom right, #020617, #071226, #0f172a);
    color: white;
}

/* Main Title */

.main-title{
    text-align:center;
    font-size:58px;
    font-weight:800;
    color:white;
    margin-bottom:5px;
    text-shadow:0px 0px 20px #00e5ff;
}

/* Subtitle */

.sub-title{
    text-align:center;
    color:#d1d5db;
    font-size:24px;
    margin-bottom:30px;
}

/* Glass Card */

.glass-card{
    background: rgba(255,255,255,0.03);
    border:1px solid rgba(255,255,255,0.08);
    border-radius:22px;
    padding:25px;
    backdrop-filter: blur(10px);
    box-shadow:0px 0px 25px rgba(0,0,0,0.5);
}

/* Section Heading */

.section-title{
    font-size:28px;
    font-weight:bold;
    color:#00e5ff;
    margin-bottom:20px;
}

/* Labels */

label{
    color:white !important;
    font-weight:600 !important;
}

/* Inputs */

[data-testid="stNumberInput"] input{
    background-color:#020617 !important;
    color:white !important;
    border-radius:10px !important;
    border:1px solid #334155 !important;
}

[data-testid="stSelectbox"] div{
    background-color:#020617 !important;
    color:white !important;
    border-radius:10px !important;
}

/* Slider */

.stSlider{
    padding-top:10px;
    padding-bottom:10px;
}

/* Button */

.stButton>button{
    width:100%;
    height:70px;
    border:none;
    border-radius:18px;
    background: linear-gradient(to right, #00d2ff, #8b5cf6);
    color:white;
    font-size:28px;
    font-weight:bold;
    transition:0.4s;
    box-shadow:0px 0px 20px rgba(0,229,255,0.5);
}

.stButton>button:hover{
    transform:scale(1.03);
    background: linear-gradient(to right, #8b5cf6, #00d2ff);
}

/* Result Success */

.success-card{
    background: linear-gradient(to right, #064e3b, #16a34a);
    padding:35px;
    border-radius:25px;
    text-align:center;
    font-size:35px;
    font-weight:bold;
    color:white;
    margin-top:20px;
    box-shadow:0px 0px 25px rgba(0,255,100,0.4);
}

/* Result Failure */

.fail-card{
    background: linear-gradient(to right, #7f1d1d, #dc2626);
    padding:35px;
    border-radius:25px;
    text-align:center;
    font-size:35px;
    font-weight:bold;
    color:white;
    margin-top:20px;
    box-shadow:0px 0px 25px rgba(255,0,0,0.4);
}

/* Small Info Box */

.info-box{
    background: rgba(255,255,255,0.03);
    border:1px solid rgba(255,255,255,0.08);
    padding:20px;
    border-radius:20px;
    margin-top:20px;
}

/* Footer */

.footer{
    text-align:center;
    margin-top:40px;
    color:#94a3b8;
    font-size:18px;
}

</style>
""", unsafe_allow_html=True)

# =========================
# LOAD MODEL
# =========================

model = pickle.load(open("knn.pkl", "rb"))

# =========================
# TITLE
# =========================

st.markdown("""
<h1 class='main-title'>
🎓 AI STUDENT PLACEMENT PREDICTION SYSTEM
</h1>

<p class='sub-title'>
Predict your placement chances with AI ✨
</p>
""", unsafe_allow_html=True)

# =========================
# MAIN LAYOUT
# =========================

left, right = st.columns([1.3, 1])

# =========================
# LEFT SIDE
# =========================

with left:

    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)

    st.markdown("""
    <div class='section-title'>
    📋 STUDENT DETAILS
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:

        age = st.number_input("🎂 Age", 18, 30)

        gender = st.selectbox(
            "👤 Gender",
            ["Male", "Female"]
        )

        degree = st.selectbox(
            "🎓 Degree",
            ["B.Tech", "B.Sc", "BCA", "M.Tech", "MBA"]
        )

        branch = st.selectbox(
            "🏫 Branch",
            ["CSE", "IT", "ECE", "Mechanical", "Civil"]
        )

        cgpa = st.slider(
            "📊 CGPA",
            0.0,
            10.0,
            7.5
        )

        internships = st.number_input(
            "💼 Internships",
            0,
            10
        )

        projects = st.number_input(
            "🛠 Projects",
            0,
            20
        )

    with col2:

        coding_skills = st.slider(
            "💻 Coding Skills",
            1,
            10,
            7
        )

        communication_skills = st.slider(
            "🗣 Communication Skills",
            1,
            10,
            7
        )

        aptitude_score = st.slider(
            "🧠 Aptitude Score",
            0,
            100,
            75
        )

        soft_skills = st.slider(
            "🤝 Soft Skills",
            1,
            10,
            7
        )

        certifications = st.number_input(
            "📜 Certifications",
            0,
            20
        )

        backlogs = st.number_input(
            "❌ Backlogs",
            0,
            10
        )

    st.write("")

    predict = st.button("🚀 PREDICT PLACEMENT")

    st.markdown("""
    <div class='info-box'>
    💡 Tip: Higher CGPA, Skills and Aptitude Score improve placement chances!
    </div>
    """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

# =========================
# ENCODING
# =========================

gender = 1 if gender == "Male" else 0

degree_mapping = {
    "B.Tech": 0,
    "B.Sc": 1,
    "BCA": 2,
    "M.Tech": 3,
    "MBA": 4
}

branch_mapping = {
    "CSE": 0,
    "IT": 1,
    "ECE": 2,
    "Mechanical": 3,
    "Civil": 4
}

degree = degree_mapping[degree]
branch = branch_mapping[branch]

# =========================
# RIGHT SIDE RESULT
# =========================

with right:

    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)

    st.markdown("""
    <div class='section-title'>
    📊 PREDICTION RESULT
    </div>
    """, unsafe_allow_html=True)

    if predict:

        input_data = pd.DataFrame([[
            age,
            gender,
            degree,
            branch,
            cgpa,
            internships,
            projects,
            coding_skills,
            communication_skills,
            aptitude_score,
            soft_skills,
            certifications,
            backlogs
        ]], columns=[
            "Age",
            "Gender",
            "Degree",
            "Branch",
            "CGPA",
            "Internships",
            "Projects",
            "Coding_Skills",
            "Communication_Skills",
            "Aptitude_Test_Score",
            "Soft_Skills_Rating",
            "Certifications",
            "Backlogs"
        ])

        prediction = model.predict(input_data)

        if prediction[0] == 1:

            st.balloons()

            st.markdown("""
            <div class='success-card'>

            🏆 STUDENT IS LIKELY TO BE PLACED 🏆

            <br><br>

            ⭐⭐⭐ CONGRATULATIONS ⭐⭐⭐

            <br><br>

            🚀 Bright Career Ahead 🚀

            </div>
            """, unsafe_allow_html=True)

            st.write("")

            st.subheader("📈 Placement Probability")

            st.progress(100)

            st.markdown("""
            <div class='info-box'>

            ✅ Excellent Academic and Technical Performance 💯

            </div>
            """, unsafe_allow_html=True)

        else:

            st.markdown("""
            <div class='fail-card'>

            ❌ PLACEMENT CHANCES ARE LOW ❌

            <br><br>

            📚 NEEDS IMPROVEMENT 📚

            <br><br>

            💪 KEEP PRACTICING 💪

            </div>
            """, unsafe_allow_html=True)

            st.write("")

            st.subheader("📈 Placement Probability")

            st.progress(40)

            st.markdown("""
            <div class='info-box'>

            ⚡ Improve Coding Skills, Communication and Aptitude

            </div>
            """, unsafe_allow_html=True)

    else:

        st.image(
            "https://cdn-icons-png.flaticon.com/512/3135/3135715.png",
            width=220
        )

        st.markdown("""
        <div class='info-box'>

        🎯 Fill student details and click on Predict Placement

        </div>
        """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

# =========================
# FOOTER
# =========================

st.markdown("""
<div class='footer'>

✨ Developed with Machine Learning & Streamlit ✨

</div>
""", unsafe_allow_html=True)