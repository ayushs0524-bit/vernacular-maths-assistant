import re
import pandas as pd
import streamlit as st
from sarvamai import SarvamAI


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Vernacular Maths Assistant",
    page_icon="📚",
    layout="centered"
)


# ============================================================
# CONFIGURATION
# ============================================================

DATA_FILE = "Hindi_Santali_Maths_Dataset_Starter.xlsx"


# ============================================================
# SARVAM AI CLIENT
# ============================================================

try:
    sarvam_key = st.secrets["sk_p87syka3_uMW7J8EipungKX3djRZb4a8x"]

    sarvam_client = SarvamAI(
        api_subscription_key=sarvam_key
    )

    sarvam_available = True
    sarvam_error = None

except Exception as e:

    sarvam_client = None
    sarvam_available = False
    sarvam_error = str(e)


# ============================================================
# LOAD DATASET
# ============================================================

@st.cache_data
def load_data():

    df = pd.read_excel(
        DATA_FILE,
        sheet_name="Core_Seed_300"
    )

    # Prototype scope: Class 3 Mathematics
    df = df[df["Class"] == 3].copy()

    return df


df = load_data()


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def clean(value):

    if pd.isna(value):
        return ""

    return str(value).strip()


def find_match(text):

    text_norm = re.sub(
        r"\s+",
        " ",
        text.strip()
    )

    exact = df[
        df["Hindi_Text"]
        .astype(str)
        .str.strip()
        == text_norm
    ]

    if not exact.empty:
        return exact.iloc[0]

    return None


# ============================================================
# AI PEDAGOGY ENGINE
# ============================================================

def generate_ai_explanation(question):

    if not sarvam_available:

        return None, f"Sarvam API is not configured: {sarvam_error}"


    system_prompt = """
You are an expert primary-school mathematics teacher.

Your job is to explain Class 3 Mathematics questions to
children in simple Hindi.

Follow these rules:

1. Use very simple, child-friendly Hindi.
2. Explain the concept step by step.
3. Do not use advanced mathematical terminology.
4. Clearly show the calculation.
5. Give the final answer clearly.
6. Keep the explanation concise.
7. Use familiar examples when useful.
8. Never invent information that is not required.
9. Make sure the mathematical answer is correct.

Format your response like this:

समझते हैं:
<simple explanation>

हल:
<step-by-step calculation>

उत्तर:
<final answer>
"""


    user_prompt = f"""
This is a Class 3 Mathematics question:

{question}

Explain and solve it for a Class 3 student.
"""


    try:

        response = sarvam_client.chat.completions(

            model="sarvam-105b",

            messages=[
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": user_prompt
                }
            ],

            temperature=0.2,

            max_tokens=500,

            reasoning_effort=None
        )


        explanation = (
            response
            .choices[0]
            .message
            .content
        )

        return explanation, None


    except Exception as e:

        return None, str(e)


# ============================================================
# SARVAM HINDI → SANTALI TRANSLATION
# ============================================================

def translate_to_santali(hindi_text):

    if not sarvam_available:

        return None, "Sarvam API is not configured correctly."


    try:

        response = sarvam_client.text.translate(

            input=hindi_text,

            source_language_code="hi-IN",

            target_language_code="sat-IN",

            model="sarvam-translate:v1"
        )


        return response.translated_text, None


    except Exception as e:

        return None, str(e)


# ============================================================
# PRACTICE QUESTION GENERATOR
# ============================================================

def generate_practice_question(question):

    if not sarvam_available:

        return None


    prompt = f"""
Create ONE similar Class 3 Mathematics practice question
based on this original question:

{question}

Rules:
- Keep it suitable for Class 3.
- Use simple Hindi.
- Change the numbers.
- Test the same mathematical concept.
- Do NOT provide the answer.
- Return only the question.
"""


    try:

        response = sarvam_client.chat.completions(

            model="sarvam-105b",

            messages=[
                {
                    "role": "system",
                    "content":
                    "You create simple Class 3 Mathematics practice questions."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],

            temperature=0.4,

            max_tokens=150,

            reasoning_effort=None
        )


        return (
            response
            .choices[0]
            .message
            .content
            .strip()
        )


    except Exception:

        return None


# ============================================================
# MAIN UI
# ============================================================

st.title(
    "📚 AI Vernacular Maths Assistant"
)

st.caption(
    "AI-powered pedagogy • Class 3 Mathematics • Hindi → Santali"
)


st.info(
    "The system generates a child-friendly Hindi explanation "
    "using Sarvam AI and then translates it into Santali."
)


# ============================================================
# USER INPUT
# ============================================================

question = st.text_area(

    "Enter a Class 3 Mathematics question in Hindi",

    placeholder=(
        "उदाहरण: एक टोकरी में 8 आम हैं और "
        "दूसरी टोकरी में 5 आम हैं। कुल कितने आम हैं?"
    ),

    height=120
)


# ============================================================
# GENERATE BUTTON
# ============================================================

if st.button(
    "Generate Explanation",
    type="primary"
):

    if not question.strip():

        st.warning(
            "Please enter a question."
        )

        st.stop()


    # ========================================================
    # AI PEDAGOGY
    # ========================================================

    st.subheader(
        "🧠 AI Hindi Explanation"
    )


    with st.spinner(
        "AI is preparing a child-friendly explanation..."
    ):

        hindi_explanation, pedagogy_error = (
            generate_ai_explanation(question)
        )


    if pedagogy_error:

        st.error(
            "AI explanation failed."
        )

        st.caption(
            f"Error: {pedagogy_error}"
        )

        st.stop()


    st.write(
        hindi_explanation
    )


    # ========================================================
    # SANTALI TRANSLATION
    # ========================================================

    st.subheader(
        "🌐 Santali Translation"
    )


    with st.spinner(
        "Translating Hindi → Santali..."
    ):

        santali_output, translation_error = (
            translate_to_santali(
                hindi_explanation
            )
        )


    if translation_error:

        st.error(
            "Translation failed."
        )

        st.caption(
            f"Error: {translation_error}"
        )

    else:

        st.success(
            "Translation generated using Sarvam AI"
        )

        st.write(
            santali_output
        )


    # ========================================================
    # PRACTICE QUESTION
    # ========================================================

    st.subheader(
        "📝 Practice Question"
    )


    with st.spinner(
        "Generating a similar practice question..."
    ):

        practice_question = (
            generate_practice_question(question)
        )


    if practice_question:

        st.write(
            practice_question
        )

    else:

        st.info(
            "Practice question could not be generated."
        )


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header(
    "Project Status"
)


st.sidebar.success(
    "Dataset loaded"
)


if sarvam_available:

    st.sidebar.success(
        "Sarvam AI connected"
    )

else:

    st.sidebar.error(
        "Sarvam API not configured"
    )

    if sarvam_error:

        st.sidebar.caption(
            f"Debug: {sarvam_error}"
        )


st.sidebar.info(
    "AI Pedagogy: Sarvam 105B"
)

st.sidebar.info(
    "Translation: Hindi → Santali"
)

st.sidebar.info(
    "Scope: Class 3 Mathematics"
)


# ============================================================
# PROTOTYPE INFORMATION
# ============================================================

st.divider()


with st.expander(
    "Prototype information"
):

    st.write(
        f"Class 3 records loaded: **{len(df)}**"
    )

    st.write(
        "AI pedagogy engine: **Sarvam 105B**"
    )

    st.write(
        "Translation engine: **Sarvam Translate**"
    )

    st.write(
        "Target language: **Santali (sat-IN)**"
    )

    st.write(
        "The starter dataset is used as supporting "
        "educational data. Language-specific translations "
        "still require native-speaker validation."
    )
