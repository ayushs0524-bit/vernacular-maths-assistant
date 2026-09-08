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
    sarvam_client = SarvamAI(
        api_subscription_key=st.secrets["sk_p87syka3_uMW7J8EipungKX3djRZb4a8x"]
    )
    sarvam_available = True

except Exception:
    sarvam_client = None
    sarvam_available = False


# ============================================================
# LOAD DATASET
# ============================================================

@st.cache_data
def load_data():

    df = pd.read_excel(
        DATA_FILE,
        sheet_name="Core_Seed_300"
    )

    # Prototype scope:
    # Class 3 Mathematics only
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
# SIMPLE MATH CALCULATOR
# ============================================================

def calculate_expression(text):

    """
    Supports simple expressions such as:

    2 + 3
    8 - 5
    3 * 4
    3 x 4
    12 / 3
    """

    match = re.search(
        r"(\d+)\s*([+\-*x×÷/])\s*(\d+)",
        text
    )

    if not match:
        return None

    a = int(match.group(1))
    operator = match.group(2)
    b = int(match.group(3))

    if operator == "+":

        result = a + b

    elif operator == "-":

        result = a - b

    elif operator in ("*", "x", "×"):

        result = a * b

    elif operator in ("/", "÷"):

        if b == 0:
            return None

        result = a / b

        if result.is_integer():
            result = int(result)

    else:

        return None

    return a, operator, b, result


# ============================================================
# HINDI PEDAGOGY / EXPLANATION
# ============================================================

def generate_hindi_explanation(question):

    calculation = calculate_expression(question)

    if calculation:

        a, operator, b, result = calculation

        operator_words = {

            "+": "जोड़ने",
            "-": "घटाने",
            "*": "गुणा करने",
            "x": "गुणा करने",
            "×": "गुणा करने",
            "/": "भाग देने",
            "÷": "भाग देने"

        }

        operation = operator_words[operator]

        explanation = (
            f"{a} और {b} को {operation} पर "
            f"{result} प्राप्त होता है।"
        )

        return explanation

    match = find_match(question)

    if match is not None:

        topic = clean(match["Topic"])

        return (
            f"यह प्रश्न Class 3 Mathematics के "
            f"{topic} topic से संबंधित है।"
        )

    return (
        "यह प्रश्न अभी हमारे छोटे prototype knowledge base "
        "में उपलब्ध नहीं है। आगे AI pedagogy engine के माध्यम से "
        "इसे सरल और उम्र-उपयुक्त तरीके से समझाया जाएगा।"
    )


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
# MAIN UI
# ============================================================

st.title("📚 AI Vernacular Maths Assistant")

st.caption(
    "Prototype • Class 3 Mathematics • Hindi → Santali"
)


st.info(
    "This prototype uses Sarvam AI for Hindi → Santali "
    "translation. The system is designed for mother "
    "tongue-based primary mathematics education."
)


# ============================================================
# USER INPUT
# ============================================================

question = st.text_area(

    "Enter a Class 3 Mathematics question in Hindi",

    placeholder="उदाहरण: 2 + 3 = ?",

    height=100

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


    # --------------------------------------------------------
    # FIND DATASET MATCH
    # --------------------------------------------------------

    match = find_match(question)


    # --------------------------------------------------------
    # HINDI EXPLANATION
    # --------------------------------------------------------

    st.subheader(
        "🇮🇳 Hindi Explanation"
    )

    hindi_explanation = generate_hindi_explanation(
        question
    )

    st.write(
        hindi_explanation
    )


    # --------------------------------------------------------
    # SANTALI TRANSLATION
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # DATASET INFORMATION
    # --------------------------------------------------------

    if match is not None:

        st.caption(
            f"Topic: {clean(match['Topic'])} "
            f"| Dataset ID: {clean(match['ID'])}"
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
        "Source sheet: **Core_Seed_300**"
    )

    st.write(
        "The starter dataset contains Class 1–3 "
        "primary mathematics material. Language-specific "
        "translations require native-speaker validation."
    )

    st.write(
        "Current translation layer: **Sarvam AI**"
    )

    st.write(
        "Target language: **Santali (sat-IN)**"
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
        "Sarvam API connected"
    )

else:

    st.sidebar.error(
        "Sarvam API not configured"
    )


st.sidebar.info(
    "Translation: Hindi → Santali"
)

st.sidebar.info(
    "Scope: Class 3 Mathematics"
)

st.sidebar.markdown(
    "**Next:** Add voice interaction"
)
