import re
import pandas as pd
import streamlit as st

DATA_FILE = "Hindi_Santali_Maths_Dataset_Starter.xlsx"

st.set_page_config(
    page_title="Vernacular Maths Assistant",
    page_icon="📚",
    layout="centered"
)

@st.cache_data
def load_data():
    df = pd.read_excel(DATA_FILE, sheet_name="Core_Seed_300")
    # Prototype scope: Class 3 Mathematics only
    df = df[df["Class"] == 3].copy()
    return df

df = load_data()

def clean(value):
    if pd.isna(value):
        return ""
    return str(value).strip()

def find_match(text):
    text_norm = re.sub(r"\s+", " ", text.strip())
    exact = df[df["Hindi_Text"].astype(str).str.strip() == text_norm]
    if not exact.empty:
        return exact.iloc[0]

    # Try a normalized expression match, e.g. "2 + 3 = ?" with Hindi numerals converted.
    return None

def calculate_expression(text):
    # Supports simple arithmetic expressions such as 2 + 3, 8 - 5, 3 * 4, 12 / 3
    m = re.search(r"(\d+)\s*([+\-*x×÷/])\s*(\d+)", text)
    if not m:
        return None

    a = int(m.group(1))
    op = m.group(2)
    b = int(m.group(3))

    if op == "+":
        result = a + b
    elif op == "-":
        result = a - b
    elif op in ("*", "x", "×"):
        result = a * b
    elif op in ("/", "÷"):
        if b == 0:
            return None
        result = a / b
        if result.is_integer():
            result = int(result)
    else:
        return None

    return a, op, b, result

st.title("📚 AI Vernacular Maths Assistant")
st.caption("Prototype • Class 3 Mathematics • Hindi → Santali")

st.info(
    "This prototype currently uses the local starter dataset. "
    "BHASHINI NMT will replace the local translation layer after API approval."
)

question = st.text_area(
    "Enter a Class 3 Mathematics question in Hindi",
    placeholder="उदाहरण: 2 + 3 = ?",
    height=100
)

if st.button("Generate Explanation", type="primary"):
    if not question.strip():
        st.warning("Please enter a question.")
        st.stop()

    match = find_match(question)

    st.subheader("Hindi Explanation")

    calc = calculate_expression(question)

    if calc:
        a, op, b, result = calc
        op_word = {
            "+": "जोड़ने",
            "-": "घटाने",
            "*": "गुणा करने",
            "x": "गुणा करने",
            "×": "गुणा करने",
            "/": "भाग देने",
            "÷": "भाग देने",
        }[op]
        hindi_explanation = f"{a} और {b} को {op_word} पर {result} प्राप्त होता है।"
        st.write(hindi_explanation)
    elif match is not None:
        st.write(f"यह प्रश्न Class 3 Mathematics के {clean(match['Topic'])} topic से संबंधित है।")
    else:
        st.write(
            "यह प्रश्न अभी हमारे छोटे prototype knowledge base में नहीं है। "
            "अगले चरण में AI pedagogy engine इसे सरल, उम्र-उपयुक्त तरीके से समझाएगा।"
        )

    st.subheader("Santali Output")

    if match is not None:
        # Prefer the Ol Chiki field when available.
        santali = clean(match.get("Santali_OlChiki", ""))
        if not santali:
            santali = clean(match.get("Santali_Text", ""))

        if santali:
            st.success(santali)
        else:
            st.warning("A Santali translation is not available for this row yet.")
    elif calc:
        # The dataset contains translated number terms; for now show a transparent
        # prototype fallback rather than pretending an unverified sentence translation.
        st.warning(
            "Local translation for this complete sentence is not available yet. "
            "BHASHINI NMT will provide the sentence-level Hindi → Santali translation."
        )
    else:
        st.warning(
            "No local Santali translation found. BHASHINI NMT will be connected here."
        )

    if match is not None:
        st.caption(
            f"Topic: {clean(match['Topic'])}  |  Dataset ID: {clean(match['ID'])}"
        )

st.divider()

with st.expander("Prototype dataset status"):
    st.write(f"Class 3 records loaded: **{len(df)}**")
    st.write("Source sheet: **Core_Seed_300**")
    st.write(
        "The dataset itself states that language-specific translations require "
        "native-speaker validation. Do not present all rows as professionally validated."
    )

st.sidebar.header("Project Status")
st.sidebar.success("Dataset loaded")
st.sidebar.info("BHASHINI API: Pending approval")
st.sidebar.info("Hugging Face access: Pending")
st.sidebar.markdown("**Next:** connect BHASHINI NMT")
