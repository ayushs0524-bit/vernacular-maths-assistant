import streamlit as st
import pandas as pd
import re
from sarvamai import SarvamAI


# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="AI Vernacular Maths Assistant",
    page_icon="📚",
    layout="wide"
)


# --------------------------------------------------
# TITLE
# --------------------------------------------------

st.title("📚 AI Vernacular Maths Assistant")
st.caption(
    "AI-powered pedagogy • Class 3 Mathematics • Hindi → Santali"
)


# --------------------------------------------------
# SARVAM API CONFIGURATION
# --------------------------------------------------

sarvam_key = st.secrets.get("SARVAM_API_KEY")

sarvam_available = False
sarvam_client = None
sarvam_error = None

if not sarvam_key:
    sarvam_error = (
        "SARVAM_API_KEY is missing from Streamlit Secrets. "
        "Add a secret named exactly SARVAM_API_KEY."
    )
else:
    try:
        sarvam_client = SarvamAI(
            api_subscription_key=sarvam_key
        )
        sarvam_available = True
    except Exception as e:
        sarvam_error = str(e)


# --------------------------------------------------
# LOAD DATASET
# --------------------------------------------------

DATA_FILE = "Hindi_Santali_Maths_Dataset_Starter.xlsx"

try:
    df = pd.read_excel(
        DATA_FILE,
        sheet_name="Core_Seed_300"
    )

    # Keep only Class 3 records
    class3_df = df[
        df["Class"].astype(str).str.contains("3", na=False)
    ].copy()

    dataset_loaded = True

except Exception as e:
    dataset_loaded = False
    class3_df = pd.DataFrame()
    dataset_error = str(e)


# --------------------------------------------------
# FIND DATASET MATCH
# --------------------------------------------------

def find_match(question):
    if not dataset_loaded or class3_df.empty:
        return None

    question_clean = question.strip().lower()

    for _, row in class3_df.iterrows():
        hindi_text = str(row.get("Hindi_Text", "")).strip().lower()

        if hindi_text == question_clean:
            return row

    return None


# --------------------------------------------------
# AI PEDAGOGY ENGINE
# --------------------------------------------------

def generate_ai_explanation(question):

    if not sarvam_available:
        return None, "Sarvam AI is not connected."

    system_prompt = """
You are an expert Class 3 primary-school mathematics teacher.

Explain mathematics to a Class 3 student in very simple Hindi.

Rules:
- First explain the idea in simple words.
- Then show the calculation step by step.
- Use simple language.
- Avoid advanced mathematical terminology.
- Use familiar examples when useful.
- Make sure the final answer is correct.
- Keep the explanation concise.

Use exactly this structure:

समझते हैं:
<simple explanation>

हल:
<step-by-step calculation>

उत्तर:
<final answer>
"""

    user_prompt = f"""
Class 3 Mathematics question:

{question}

Explain this question for a Class 3 student.
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

        answer = response.choices[0].message.content

        return answer, None

    except Exception as e:
        return None, str(e)


# --------------------------------------------------
# HINDI → SANTALI TRANSLATION
# --------------------------------------------------

def translate_to_santali(hindi_text):

    if not sarvam_available:
        return None, "Sarvam AI is not connected."

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


# --------------------------------------------------
# PRACTICE QUESTION GENERATOR
# --------------------------------------------------

def generate_practice_question(question):

    if not sarvam_available:
        return None, "Sarvam AI is not connected."

    system_prompt = """
You are a Class 3 primary-school mathematics teacher.

Create ONE new practice question based on the student's question.

Rules:
- Keep the same mathematical concept.
- Change the numbers.
- Keep the difficulty suitable for Class 3.
- Use simple Hindi.
- Do not provide the answer.
- Return only the practice question.
"""

    user_prompt = f"""
Original question:

{question}

Create one similar practice question.
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
            temperature=0.4,
            max_tokens=200,
            reasoning_effort=None
        )

        practice = response.choices[0].message.content

        return practice, None

    except Exception as e:
        return None, str(e)


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

with st.sidebar:

    st.header("⚙️ System Status")

    if dataset_loaded:
        st.success("Dataset loaded")
    else:
        st.error("Dataset not loaded")
        st.caption(dataset_error)

    if sarvam_available:
        st.success("Sarvam AI connected")
    else:
        st.error("Sarvam API not configured")

        if sarvam_error:
            st.caption(
                f"Debug: {sarvam_error}"
            )

    st.divider()

    st.info("AI Pedagogy: Sarvam 105B")
    st.info("Translation: Hindi → Santali")
    st.info("Scope: Class 3 Mathematics")


# --------------------------------------------------
# USER INPUT
# --------------------------------------------------

st.subheader("🧮 Ask a Class 3 Maths Question")

question = st.text_area(
    "Enter the question in Hindi",
    placeholder="उदाहरण: 25 और 17 को जोड़ने पर कितना होगा?",
    height=120
)


# --------------------------------------------------
# GENERATE BUTTON
# --------------------------------------------------

generate_button = st.button(
    "🚀 Generate Explanation",
    type="primary"
)


# --------------------------------------------------
# MAIN PROCESSING
# --------------------------------------------------

if generate_button:

    if not question.strip():

        st.warning(
            "कृपया पहले एक गणित का प्रश्न लिखें।"
        )

    else:

        # ------------------------------------------
        # DATASET MATCH
        # ------------------------------------------

        matched_row = find_match(question)

        if matched_row is not None:

            st.info(
                "📚 This question was found in the local Class 3 dataset."
            )

        # ------------------------------------------
        # AI EXPLANATION
        # ------------------------------------------

        with st.spinner(
            "🧠 AI is creating the explanation..."
        ):

            explanation, explanation_error = (
                generate_ai_explanation(question)
            )

        if explanation:

            st.subheader("🧠 AI Hindi Explanation")

            st.markdown(explanation)

        else:

            st.error(
                f"Could not generate explanation: "
                f"{explanation_error}"
            )

        # ------------------------------------------
        # SANTALI TRANSLATION
        # ------------------------------------------

        if explanation:

            with st.spinner(
                "🌐 Translating explanation into Santali..."
            ):

                santali_text, translation_error = (
                    translate_to_santali(explanation)
                )

            if santali_text:

                st.subheader("🌐 Santali Translation")

                st.write(santali_text)

            else:

                st.error(
                    f"Translation failed: "
                    f"{translation_error}"
                )

        # ------------------------------------------
        # PRACTICE QUESTION
        # ------------------------------------------

        with st.spinner(
            "📝 Creating a practice question..."
        ):

            practice_question, practice_error = (
                generate_practice_question(question)
            )

        if practice_question:

            st.subheader("📝 Practice Question")

            st.write(practice_question)

        else:

            st.error(
                f"Practice question generation failed: "
                f"{practice_error}"
            )


# --------------------------------------------------
# PROTOTYPE INFORMATION
# --------------------------------------------------

with st.expander("ℹ️ About this prototype"):

    st.write(
        """
        This prototype demonstrates an AI-powered vernacular
        mathematics learning assistant for Class 3 students.

        Current capabilities:

        • Class 3 Mathematics
        • Hindi input
        • AI-generated simple Hindi explanation
        • Hindi → Santali translation
        • Practice question generation
        • Local starter dataset integration

        AI services are powered by Sarvam AI.
        """
    )
