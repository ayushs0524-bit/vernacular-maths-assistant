import streamlit as st
import pandas as pd
import json
import re
import hashlib
import os
from io import BytesIO
from pathlib import Path
import tempfile
import wave

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    PageBreak,
)
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

from sarvamai import SarvamAI


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Vernacular Maths Assistant",
    page_icon="📚",
    layout="wide",
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 42px;
        font-weight: 800;
        text-align: center;
        margin-bottom: 5px;
    }

    .subtitle {
        text-align: center;
        font-size: 18px;
        color: #666;
        margin-bottom: 25px;
    }

    .feature-card {
        padding: 18px;
        border-radius: 15px;
        border: 1px solid rgba(128,128,128,0.25);
        margin-bottom: 15px;
    }

    .success-box {
        padding: 15px;
        border-radius: 12px;
        background: rgba(0, 180, 100, 0.08);
        border: 1px solid rgba(0, 180, 100, 0.3);
    }

    .small-text {
        font-size: 13px;
        color: #777;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">📚 Vernacular Maths Assistant</div>',
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="subtitle">
    AI-Powered Hindi → Santali Mathematics Learning Assistant
    <br>
    Class 3 • Mathematics • Addition & Subtraction + Multi-topic Practice
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SARVAM API
# ============================================================

# IMPORTANT:
# Keep your API key in Streamlit Secrets.
#
# Example:
#
# SARVAM_API_KEY = "your-key"
#
# OR if you are currently using:
#
# vernacular_maths = "your-key"
#
# change the line below accordingly.

sarvam_key = st.secrets.get("SARVAM_API_KEY")

sarvam_available = False
sarvam_client = None
sarvam_error = None

if sarvam_key:

    try:
        sarvam_client = SarvamAI(
            api_subscription_key=sarvam_key
        )

        sarvam_available = True

    except Exception as e:
        sarvam_error = str(e)

else:

    sarvam_error = "SARVAM_API_KEY not found in Streamlit Secrets."


# ============================================================
# DATASET
# ============================================================

DATASET_FILE = "Hindi_Santali_Maths_Dataset_Starter.xlsx"


@st.cache_data
def load_dataset():

    if not os.path.exists(DATASET_FILE):
        return None, None

    try:

        core_df = pd.read_excel(
            DATASET_FILE,
            sheet_name="Core_Seed_300"
        )

        validation_df = pd.read_excel(
            DATASET_FILE,
            sheet_name="Native_Validation_200"
        )

        return core_df, validation_df

    except Exception:
        return None, None


core_df, validation_df = load_dataset()


if core_df is not None:

    class3_df = core_df[
        core_df["Class"].astype(str).str.contains(
            "3",
            na=False
        )
    ].copy()

else:

    class3_df = pd.DataFrame()


# ============================================================
# CURRICULUM / LEARNING OUTCOME MAP
# ============================================================

CURRICULUM = {

    "Addition": {
        "learning_outcome":
            "Student can add two or more numbers and solve simple addition word problems.",
        "skills":
            "Number addition, carrying, mathematical reasoning"
    },

    "Subtraction": {
        "learning_outcome":
            "Student can subtract numbers and understand difference between quantities.",
        "skills":
            "Subtraction, borrowing, comparison"
    },

    "Multiplication": {
        "learning_outcome":
            "Student understands multiplication as repeated addition.",
        "skills":
            "Multiplication tables, repeated addition"
    },

    "Division": {
        "learning_outcome":
            "Student understands simple division as equal sharing.",
        "skills":
            "Equal grouping, division facts"
    },

    "Comparing Numbers": {
        "learning_outcome":
            "Student can compare numbers using greater than, less than and equal to.",
        "skills":
            "Number comparison"
    },

    "Place Value": {
        "learning_outcome":
            "Student understands ones, tens and hundreds place values.",
        "skills":
            "Place value and number decomposition"
    },

    "Fractions": {
        "learning_outcome":
            "Student understands simple fractions as equal parts of a whole.",
        "skills":
            "Half, quarter and simple fractions"
    },

    "Money": {
        "learning_outcome":
            "Student can solve simple money-related addition and subtraction problems.",
        "skills":
            "Rupees, paise and transactions"
    },

    "Time": {
        "learning_outcome":
            "Student can understand basic time concepts and read a clock.",
        "skills":
            "Hours, minutes and daily activities"
    },

    "Measurement": {
        "learning_outcome":
            "Student can use basic units of length, weight and capacity.",
        "skills":
            "Measurement and comparison"
    },

    "Geometry": {
        "learning_outcome":
            "Student can identify and describe basic geometric shapes.",
        "skills":
            "Shapes, sides and simple geometry"
    },
}


# ============================================================
# LOCAL CACHE
# ============================================================

if "answer_cache" not in st.session_state:
    st.session_state.answer_cache = {}

if "translation_cache" not in st.session_state:
    st.session_state.translation_cache = {}

if "worksheet_cache" not in st.session_state:
    st.session_state.worksheet_cache = {}


def cache_key(text):

    return hashlib.md5(
        text.strip().lower().encode("utf-8")
    ).hexdigest()


# ============================================================
# RAG RETRIEVAL
# ============================================================

def retrieve_context(question, top_k=5):

    """
    Lightweight local RAG.

    Searches the user's Excel knowledge base using:
    - Hindi text
    - topic
    - sentence type
    - English gloss

    This keeps the prototype simple and does not require
    a vector database.
    """

    if class3_df.empty:
        return []

    question_lower = question.lower()

    scored_rows = []

    for _, row in class3_df.iterrows():

        score = 0

        fields = [
            str(row.get("Hindi_Text", "")),
            str(row.get("Topic", "")),
            str(row.get("Sentence_Type", "")),
            str(row.get("English_Gloss", "")),
        ]

        combined = " ".join(fields).lower()

        # Exact Hindi sentence
        if question_lower.strip() == str(
            row.get("Hindi_Text", "")
        ).lower().strip():

            score += 100

        # Word overlap
        question_words = set(
            re.findall(
                r"[\u0900-\u097F]+",
                question_lower
            )
        )

        row_words = set(
            re.findall(
                r"[\u0900-\u097F]+",
                combined
            )
        )

        overlap = question_words.intersection(row_words)

        score += len(overlap) * 5

        if score > 0:

            scored_rows.append(
                (score, row)
            )

    scored_rows.sort(
        key=lambda x: x[0],
        reverse=True
    )

    results = []

    for score, row in scored_rows[:top_k]:

        results.append(
            {
                "score": score,
                "hindi": str(
                    row.get("Hindi_Text", "")
                ),
                "santali": str(
                    row.get("Santali_Text", "")
                ),
                "topic": str(
                    row.get("Topic", "")
                ),
                "sentence_type": str(
                    row.get("Sentence_Type", "")
                ),
                "english": str(
                    row.get("English_Gloss", "")
                ),
            }
        )

    return results


def format_rag_context(results):

    if not results:
        return "No matching local curriculum example was found."

    context = []

    for item in results:

        context.append(
            f"""
Hindi example:
{item['hindi']}

Santali:
{item['santali']}

Topic:
{item['topic']}

Sentence type:
{item['sentence_type']}
"""
        )

    return "\n---\n".join(context)


# ============================================================
# EXACT DATASET MATCH
# ============================================================

def find_match(question):

    if class3_df.empty:
        return None

    q = question.strip().lower()

    for _, row in class3_df.iterrows():

        hindi = str(
            row.get("Hindi_Text", "")
        ).strip().lower()

        if hindi == q:
            return row

    return None


# ============================================================
# AI CHAT HELPER
# ============================================================

def call_sarvam(
    system_prompt,
    user_prompt,
    max_tokens=600,
    temperature=0.2
):

    if not sarvam_available:
        raise RuntimeError(
            "Sarvam AI is not connected."
        )

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

        temperature=temperature,

        max_tokens=max_tokens,

        reasoning_effort=None
    )

    return response.choices[0].message.content.strip()


# ============================================================
# AI STEP-BY-STEP PEDAGOGY
# ============================================================

def generate_ai_explanation(question):

    key = cache_key(question)

    if key in st.session_state.answer_cache:
        return st.session_state.answer_cache[key]

    rag_results = retrieve_context(question)

    rag_context = format_rag_context(
        rag_results
    )

    system_prompt = """
You are an expert Class 3 primary-school mathematics teacher.

Your job is to explain mathematics to children in very simple Hindi.

Rules:

1. Use age-appropriate Hindi.
2. Explain the idea before the final answer.
3. Show calculation step by step.
4. Do not use advanced mathematical terminology.
5. Use familiar examples where useful.
6. Always verify the arithmetic.
7. Keep the explanation concise.
8. Follow the retrieved curriculum context when relevant.

Return exactly this structure:

समझते हैं:
<simple explanation>

हल:
<step-by-step calculation>

उत्तर:
<final answer>

सीखने का उद्देश्य:
<one short learning outcome>
"""

    user_prompt = f"""
Student question:

{question}

Retrieved curriculum examples:

{rag_context}

Explain this question for a Class 3 student.
"""

    result = call_sarvam(
        system_prompt,
        user_prompt,
        max_tokens=600,
        temperature=0.2
    )

    st.session_state.answer_cache[key] = result

    return result


# ============================================================
# HINDI → SANTALI TRANSLATION
# ============================================================

def translate_to_santali(hindi_text):

    key = cache_key(hindi_text)

    if key in st.session_state.translation_cache:
        return st.session_state.translation_cache[key]

    # First check local dataset
    match = find_match(hindi_text)

    if match is not None:

        santali = str(
            match.get("Santali_Text", "")
        ).strip()

        if santali and santali.lower() != "nan":

            st.session_state.translation_cache[key] = santali

            return santali

    if not sarvam_available:
        return "Santali translation unavailable because Sarvam AI is not connected."

    system_prompt = """
You are a careful Hindi-to-Santali educational translator.

Translate educational mathematics content from Hindi into Santali.

Rules:

1. Preserve mathematical meaning exactly.
2. Do not change numbers.
3. Do not invent information.
4. Use child-friendly Santali.
5. Prefer the Santali terminology from the provided reference examples.
6. Return only the Santali translation.
"""

    rag_results = retrieve_context(
        hindi_text,
        top_k=5
    )

    reference = format_rag_context(
        rag_results
    )

    user_prompt = f"""
Translate this Hindi educational text into Santali:

{hindi_text}

Reference examples from our local curriculum dataset:

{reference}
"""

    try:

        response = sarvam_client.text.translate(

            input=hindi_text,

            source_language_code="hi-IN",

            target_language_code="sat-IN",

            model="sarvam-translate:v1"
        )

        translated = response.translated_text.strip()

        st.session_state.translation_cache[key] = translated

        return translated

    except Exception as e:

        return f"Santali translation failed: {e}"


# ============================================================
# PRACTICE QUESTION
# ============================================================

def generate_practice_question(question):

    system_prompt = """
You are a Class 3 mathematics teacher.

Create ONE new practice question based on the same mathematical concept.

Rules:

- Same concept as the original question.
- Change the numbers.
- Keep difficulty appropriate for Class 3.
- Use simple Hindi.
- Do not provide the answer.
- Return only the question.
"""

    user_prompt = f"""
Original question:

{question}

Create one similar practice question.
"""

    return call_sarvam(
        system_prompt,
        user_prompt,
        max_tokens=200,
        temperature=0.4
    )


# ============================================================
# ANSWER GENERATOR
# ============================================================

def generate_answer(question):

    system_prompt = """
You are a Class 3 mathematics teacher.

Solve the following question.

Return ONLY the final numerical or short answer.
Do not explain.
"""

    return call_sarvam(
        system_prompt,
        question,
        max_tokens=100,
        temperature=0.1
    )


# ============================================================
# MULTI-TOPIC WORKSHEET
# ============================================================

def generate_worksheet():

    if "worksheet" in st.session_state.worksheet_cache:

        return st.session_state.worksheet_cache[
            "worksheet"
        ]

    topics = list(CURRICULUM.keys())

    topic_text = ", ".join(topics)

    system_prompt = """
You are an expert Class 3 mathematics worksheet designer.

Create exactly 5 questions.

Use DIFFERENT mathematical topics.

Possible topics include:
addition,
subtraction,
multiplication,
division,
comparing numbers,
place value,
fractions,
money,
time,
measurement,
geometry.

Difficulty should vary between easy, medium and challenging.

Return ONLY valid JSON.

Format:

{
  "questions": [
    {
      "topic": "...",
      "difficulty": "...",
      "question": "..."
    }
  ]
}
"""

    user_prompt = f"""
Generate a Class 3 mathematics worksheet.

Use different topics from:

{topic_text}

Generate exactly 5 questions.
"""

    raw = call_sarvam(
        system_prompt,
        user_prompt,
        max_tokens=900,
        temperature=0.4
    )

    # Remove accidental markdown fences
    raw = raw.replace(
        "```json",
        ""
    ).replace(
        "```",
        ""
    ).strip()

    data = json.loads(raw)

    st.session_state.worksheet_cache[
        "worksheet"
    ] = data

    return data


# ============================================================
# LEARNING OUTCOME
# ============================================================

def get_learning_outcome(topic):

    topic = str(topic).strip()

    if topic in CURRICULUM:

        return CURRICULUM[
            topic
        ]["learning_outcome"]

    return (
        "Student develops age-appropriate "
        "mathematical reasoning skills."
    )


# ============================================================
# PDF FONT
# ============================================================

def setup_pdf_font():

    possible_fonts = [

        "/usr/share/fonts/truetype/noto/"
        "NotoSansDevanagari-Regular.ttf",

        "/usr/share/fonts/opentype/noto/"
        "NotoSansDevanagari-Regular.ttf",

        "/usr/share/fonts/truetype/dejavu/"
        "DejaVuSans.ttf",
    ]

    for font_path in possible_fonts:

        if os.path.exists(font_path):

            try:

                pdfmetrics.registerFont(
                    TTFont(
                        "AppFont",
                        font_path
                    )
                )

                return "AppFont"

            except Exception:
                pass

    return "Helvetica"


# ============================================================
# CREATE BILINGUAL PDF
# ============================================================

def create_pdf(
    worksheet_data,
    translated_questions,
    answers
):

    font_name = setup_pdf_font()

    buffer = BytesIO()

    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=1.5 * cm,
        leftMargin=1.5 * cm,
        topMargin=1.5 * cm,
        bottomMargin=1.5 * cm,
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "Title",
        parent=styles["Title"],
        fontName=font_name,
        fontSize=20,
        alignment=TA_CENTER,
        spaceAfter=10,
    )

    subtitle_style = ParagraphStyle(
        "Subtitle",
        parent=styles["Normal"],
        fontName=font_name,
        fontSize=11,
        alignment=TA_CENTER,
        spaceAfter=15,
    )

    question_style = ParagraphStyle(
        "Question",
        parent=styles["Normal"],
        fontName=font_name,
        fontSize=11,
        leading=16,
        spaceAfter=5,
    )

    small_style = ParagraphStyle(
        "Small",
        parent=styles["Normal"],
        fontName=font_name,
        fontSize=9,
        leading=13,
    )

    story = []

    # --------------------------------------------------------
    # TITLE
    # --------------------------------------------------------

    story.append(
        Paragraph(
            "CLASS 3 — MATHEMATICS WORKSHEET",
            title_style
        )
    )

    story.append(
        Paragraph(
            "Hindi ↔ Santali Bilingual Learning Worksheet",
            subtitle_style
        )
    )

    story.append(
        Paragraph(
            "Name: ________________________________",
            question_style
        )
    )

    story.append(
        Paragraph(
            "Date: _________________________________",
            question_style
        )
    )

    story.append(Spacer(1, 15))

    # --------------------------------------------------------
    # QUESTIONS
    # --------------------------------------------------------

    for i, item in enumerate(
        worksheet_data["questions"]
    ):

        topic = item.get(
            "topic",
            "Mathematics"
        )

        difficulty = item.get(
            "difficulty",
            "Medium"
        )

        hindi_question = item.get(
            "question",
            ""
        )

        santali_question = translated_questions[
            i
        ]

        outcome = get_learning_outcome(
            topic
        )

        story.append(
            Paragraph(
                f"<b>Q{i + 1}. {topic}</b> "
                f"({difficulty})",
                question_style
            )
        )

        story.append(
            Paragraph(
                f"<b>Hindi:</b> {hindi_question}",
                question_style
            )
        )

        story.append(
            Paragraph(
                f"<b>Santali:</b> {santali_question}",
                question_style
            )
        )

        story.append(
            Paragraph(
                f"<b>Learning Outcome:</b> {outcome}",
                small_style
            )
        )

        story.append(
            Spacer(1, 8)
        )

        story.append(
            Paragraph(
                "Answer: __________________________________________",
                question_style
            )
        )

        story.append(
            Spacer(1, 15)
        )

    # --------------------------------------------------------
    # ANSWER KEY
    # --------------------------------------------------------

    story.append(PageBreak())

    story.append(
        Paragraph(
            "ANSWER KEY",
            title_style
        )
    )

    story.append(
        Paragraph(
            "Teacher / Parent Reference",
            subtitle_style
        )
    )

    for i, answer in enumerate(answers):

        story.append(
            Paragraph(
                f"<b>{i + 1}. Answer:</b> {answer}",
                question_style
            )
        )

        story.append(
            Spacer(1, 5)
        )

    # --------------------------------------------------------
    # BUILD
    # --------------------------------------------------------

    doc.build(story)

    buffer.seek(0)

    return buffer


# ============================================================
# VOICE → HINDI TEXT
# ============================================================

def transcribe_hindi_voice(audio_file):
    """
    Convert Streamlit's recorded audio into a clean WAV file and
    send the file path to Sarvam Saaras STT.

    Streamlit/browser recordings can sometimes arrive with a MIME type
    such as audio/vnd.wave. Writing a real .wav file to disk makes the
    upload metadata unambiguous for the Sarvam SDK.
    """

    if not sarvam_available:
        return (
            None,
            "Sarvam AI is not connected."
        )

    temp_path = None

    try:
        # --------------------------------------------------------
        # 1. Read the recording from Streamlit
        # --------------------------------------------------------

        audio_bytes = audio_file.getvalue()

        if not audio_bytes:
            return (
                None,
                "No audio was recorded. Please record your question again."
            )

        # --------------------------------------------------------
        # 2. Validate and rebuild the WAV container
        # --------------------------------------------------------
        # The screenshot error showed:
        #     Invalid file type: audio/vnd.wave
        #
        # The audio itself is WAV, but its MIME metadata is not one
        # Sarvam accepts. Rebuilding the WAV gives us a clean file
        # with a .wav extension and standard RIFF/WAVE headers.

        input_buffer = BytesIO(audio_bytes)

        try:
            with wave.open(input_buffer, "rb") as wav_in:
                channels = wav_in.getnchannels()
                sample_width = wav_in.getsampwidth()
                sample_rate = wav_in.getframerate()
                frames = wav_in.readframes(wav_in.getnframes())

        except wave.Error as e:
            return (
                None,
                f"Recorded audio is not a valid WAV file: {e}"
            )

        if not frames:
            return (
                None,
                "The recording contains no audio. Please record again."
            )

        # --------------------------------------------------------
        # 3. Write a real temporary .wav file
        # --------------------------------------------------------

        with tempfile.NamedTemporaryFile(
            suffix=".wav",
            delete=False
        ) as temp_file:

            temp_path = temp_file.name

        with wave.open(temp_path, "wb") as wav_out:
            wav_out.setnchannels(channels)
            wav_out.setsampwidth(sample_width)
            wav_out.setframerate(sample_rate)
            wav_out.writeframes(frames)

        # --------------------------------------------------------
        # 4. Send the actual WAV file to Sarvam
        # --------------------------------------------------------
        # Passing an opened .wav file follows Sarvam's documented
        # Python SDK usage and lets the SDK/server infer the correct
        # audio type from the real file.

        with open(temp_path, "rb") as wav_file:
            response = (
                sarvam_client
                .speech_to_text
                .transcribe(
                    file=wav_file,
                    model="saaras:v3",
                    language_code="hi-IN",
                    mode="transcribe"
                )
            )

        transcript = getattr(
            response,
            "transcript",
            ""
        )

        if not transcript:
            return (
                None,
                "Sarvam received the audio but returned an empty transcript."
            )

        return (
            transcript.strip(),
            None
        )

    except Exception as e:
        # Keep the UI error short and useful instead of dumping the
        # complete HTTP headers returned by the API.
        error_text = str(e)

        if "Invalid file type" in error_text:
            error_message = (
                "Sarvam rejected the audio format. "
                "The recording was converted to standard WAV, "
                "so please try recording once more."
            )
        elif "400" in error_text:
            error_message = (
                "Sarvam rejected the voice request (HTTP 400). "
                "Please record a short Hindi question and try again."
            )
        elif "401" in error_text or "403" in error_text:
            error_message = (
                "Sarvam authentication failed. "
                "Please check SARVAM_API_KEY in Streamlit Secrets."
            )
        elif "429" in error_text:
            error_message = (
                "Sarvam rate limit reached. "
                "Please wait a moment and try again."
            )
        else:
            error_message = (
                f"Voice transcription failed: {error_text}"
            )

        return None, error_message

    finally:
        # Remove the temporary WAV file after the API call.
        if temp_path:
            try:
                os.remove(temp_path)
            except OSError:
                pass



# ============================================================
# VOICE OUTPUT — HINDI
# ============================================================

def generate_hindi_audio(text):

    if not sarvam_available:

        return None, "Sarvam AI is not connected."

    try:

        audio = (
            sarvam_client
            .text_to_speech
            .convert(
                text=text,
                model="bulbul:v3",
                language_code="hi-IN",
                speaker="priya"
            )
        )

        return audio, None

    except Exception as e:

        return None, str(e)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("⚙️ System Status")

    if sarvam_available:

        st.success(
            "🟢 Sarvam AI Connected"
        )

    else:

        st.error(
            "🔴 Sarvam AI Not Connected"
        )

    if core_df is not None:

        st.success(
            f"📚 Dataset Loaded\n\n"
            f"{len(core_df)} core records"
        )

    else:

        st.warning(
            "Dataset not found."
        )

    st.markdown("---")

    st.markdown(
        """
        ### 🧠 AI Modules

        ✅ Sarvam-105B  
        ✅ Hindi → Santali  
        ✅ Saaras Speech-to-Text  
        ✅ Bulbul Hindi Voice Output  
        ✅ Local RAG  
        ✅ Curriculum Alignment  
        ✅ Worksheet Generation  
        """
    )

    st.markdown("---")

    st.caption(
        "SIH 2026 Prototype • Class 3 Mathematics"
    )


# ============================================================
# MAIN INPUT SECTION
# ============================================================

st.header("1️⃣ Ask a Mathematics Question")

input_method = st.radio(
    "Choose input method",
    [
        "⌨️ Type",
        "🎤 Voice"
    ],
    horizontal=True
)


question = ""


# ============================================================
# TEXT INPUT
# ============================================================

if input_method == "⌨️ Type":

    question = st.text_area(
        "Enter your Class 3 maths question in Hindi",
        placeholder=(
            "उदाहरण: 25 और 17 को जोड़ने पर कितना होगा?"
        ),
        height=120
    )


# ============================================================
# VOICE INPUT
# ============================================================

else:

    audio = st.audio_input(
        "🎤 Record your Hindi maths question"
    )

    if audio is not None:

        with st.spinner(
            "🎧 Converting your voice to Hindi text..."
        ):

            voice_question, voice_error = (
                transcribe_hindi_voice(audio)
            )

        if voice_question:

            question = voice_question

            st.success(
                "✅ Voice converted successfully!"
            )

            st.markdown(
                "### 🗣️ You said:"
            )

            st.info(
                voice_question
            )

        else:

            st.error(
                f"Voice transcription failed: "
                f"{voice_error}"
            )


# ============================================================
# GENERATE BUTTON
# ============================================================

generate_button = st.button(
    "🚀 Generate AI Learning Response",
    type="primary",
    use_container_width=True
)


# ============================================================
# MAIN AI RESPONSE
# ============================================================

if generate_button:

    if not question.strip():

        st.warning(
            "Please enter or speak a question first."
        )

    elif not sarvam_available:

        st.error(
            "Sarvam AI is not connected. "
            "Please check Streamlit Secrets."
        )

    else:

        # ----------------------------------------------------
        # RAG
        # ----------------------------------------------------

        rag_results = retrieve_context(
            question,
            top_k=5
        )

        # ----------------------------------------------------
        # STEP 1: AI PEDAGOGY
        # ----------------------------------------------------

        with st.spinner(
            "🧠 Generating child-friendly explanation..."
        ):

            explanation = (
                generate_ai_explanation(
                    question
                )
            )

        st.subheader(
            "🧠 Step-by-Step Explanation"
        )

        st.markdown(
            explanation
        )

        # ----------------------------------------------------
        # STEP 2: SANTALI TRANSLATION
        # ----------------------------------------------------

        with st.spinner(
            "🌐 Translating into Santali..."
        ):

            santali_translation = (
                translate_to_santali(
                    explanation
                )
            )

        st.subheader(
            "🌿 Santali Translation"
        )

        st.info(
            santali_translation
        )

        # ----------------------------------------------------
        # STEP 3: LEARNING OUTCOME
        # ----------------------------------------------------

        st.subheader(
            "🎯 Curriculum Alignment"
        )

        detected_topic = "Addition"

        if rag_results:

            detected_topic = (
                rag_results[0]["topic"]
            )

        st.success(
            get_learning_outcome(
                detected_topic
            )
        )

        # ----------------------------------------------------
        # STEP 4: RAG CONTEXT
        # ----------------------------------------------------

        with st.expander(
            "🔎 View Retrieved Curriculum Context"
        ):

            if rag_results:

                for result in rag_results:

                    st.markdown(
                        f"""
                        **Topic:** {result['topic']}

                        **Hindi:** {result['hindi']}

                        **Santali:** {result['santali']}
                        """
                    )

                    st.markdown("---")

            else:

                st.info(
                    "No local dataset match found."
                )

        # ----------------------------------------------------
        # STEP 5: VOICE OUTPUT
        # ----------------------------------------------------

        st.subheader(
            "🔊 Listen to Explanation"
        )

        # We speak the Hindi explanation because
        # Bulbul v3 currently supports Hindi but not Santali.

        with st.spinner(
            "🔊 Generating Hindi voice..."
        ):

            audio_output, audio_error = (
                generate_hindi_audio(
                    explanation
                )
            )

        if audio_output:

            # Depending on SDK response shape,
            # audio_output may expose .audios.

            try:

                if hasattr(
                    audio_output,
                    "audios"
                ):

                    audio_bytes = (
                        audio_output.audios[0]
                    )

                    st.audio(
                        audio_bytes,
                        format="audio/wav"
                    )

                elif isinstance(
                    audio_output,
                    bytes
                ):

                    st.audio(
                        audio_output,
                        format="audio/wav"
                    )

                else:

                    st.info(
                        "Audio generated successfully."
                    )

            except Exception:

                st.warning(
                    "Audio generated, "
                    "but could not be displayed."
                )

        else:

            st.warning(
                f"Voice output unavailable: "
                f"{audio_error}"
            )

        # ----------------------------------------------------
        # STEP 6: PRACTICE QUESTION
        # ----------------------------------------------------

        st.subheader(
            "✏️ Practice Question"
        )

        with st.spinner(
            "Generating similar practice question..."
        ):

            practice_question = (
                generate_practice_question(
                    question
                )
            )

        st.write(
            practice_question
        )

        # ----------------------------------------------------
        # STEP 7: PRACTICE QUESTION TRANSLATION
        # ----------------------------------------------------

        with st.spinner(
            "Translating practice question..."
        ):

            practice_santali = (
                translate_to_santali(
                    practice_question
                )
            )

        st.markdown(
            "**🌿 Santali:**"
        )

        st.info(
            practice_santali
        )


# ============================================================
# WORKSHEET SECTION
# ============================================================

st.markdown("---")

st.header(
    "2️⃣ AI Bilingual Worksheet Generator"
)

st.write(
    """
    Generate a Class 3 worksheet containing
    multiple mathematics topics, Santali translations,
    learning outcomes and an answer key.
    """
)

if st.button(
    "📄 Generate 5-Question Worksheet",
    use_container_width=True
):

    if not sarvam_available:

        st.error(
            "Sarvam AI is not connected."
        )

    else:

        try:

            with st.spinner(
                "🧠 Creating multi-topic worksheet..."
            ):

                worksheet = generate_worksheet()

            translated_questions = []
            answers = []

            progress = st.progress(0)

            total = len(
                worksheet["questions"]
            )

            for i, item in enumerate(
                worksheet["questions"]
            ):

                hindi_q = item[
                    "question"
                ]

                # Santali
                santali_q = (
                    translate_to_santali(
                        hindi_q
                    )
                )

                translated_questions.append(
                    santali_q
                )

                # Answer
                answer = generate_answer(
                    hindi_q
                )

                answers.append(
                    answer
                )

                progress.progress(
                    (i + 1) / total
                )

            progress.empty()

            # ------------------------------------------------
            # DISPLAY WORKSHEET
            # ------------------------------------------------

            st.subheader(
                "📚 Worksheet Preview"
            )

            for i, item in enumerate(
                worksheet["questions"]
            ):

                topic = item.get(
                    "topic",
                    "Mathematics"
                )

                difficulty = item.get(
                    "difficulty",
                    "Medium"
                )

                hindi_q = item.get(
                    "question",
                    ""
                )

                santali_q = (
                    translated_questions[i]
                )

                st.markdown(
                    f"""
                    ### Q{i + 1}. {topic}

                    **Difficulty:** {difficulty}

                    **Hindi:**  
                    {hindi_q}

                    **Santali:**  
                    {santali_q}

                    **🎯 Learning Outcome:**  
                    {get_learning_outcome(topic)}
                    """
                )

                st.markdown("---")

            # ------------------------------------------------
            # PDF
            # ------------------------------------------------

            with st.spinner(
                "📄 Creating printable bilingual PDF..."
            ):

                pdf_file = create_pdf(
                    worksheet,
                    translated_questions,
                    answers
                )

            st.success(
                "✅ Bilingual worksheet PDF created!"
            )

            st.download_button(
                label="⬇️ Download Bilingual PDF",
                data=pdf_file,
                file_name=(
                    "Class_3_Hindi_Santali_"
                    "Bilingual_Worksheet.pdf"
                ),
                mime="application/pdf",
                use_container_width=True
            )

            # ------------------------------------------------
            # ANSWER KEY
            # ------------------------------------------------

            with st.expander(
                "🔑 View Answer Key"
            ):

                for i, answer in enumerate(
                    answers
                ):

                    st.write(
                        f"**Q{i + 1}:** {answer}"
                    )

        except Exception as e:

            st.error(
                f"Worksheet generation failed: {e}"
            )


# ============================================================
# OFFLINE / CACHE SECTION
# ============================================================

st.markdown("---")

st.header(
    "3️⃣ Offline & Caching Support"
)

st.info(
    """
    The prototype caches successful AI results during the
    current Streamlit session and uses the local Excel
    curriculum dataset for retrieval.

    Internet is still required for new Sarvam AI requests.
    A completely offline AI system would require deploying
    local ASR/translation/LLM models on the device.
    """
)

col1, col2, col3 = st.columns(3)

with col1:

    st.metric(
        "Cached Explanations",
        len(
            st.session_state.answer_cache
        )
    )

with col2:

    st.metric(
        "Cached Translations",
        len(
            st.session_state.translation_cache
        )
    )

with col3:

    st.metric(
        "Dataset Records",
        len(class3_df)
        if not class3_df.empty
        else 0
    )


# ============================================================
# RAG / SYSTEM ARCHITECTURE
# ============================================================

st.markdown("---")

st.header(
    "4️⃣ Prototype Architecture"
)

st.markdown(
    """
    **User Input**
    
    ⬇️
    
    🎤 Hindi Voice / ⌨️ Hindi Text
    
    ⬇️
    
    **Sarvam Saaras — Speech-to-Text**
    
    ⬇️
    
    **Local Curriculum RAG**
    
    ⬇️
    
    **Sarvam-105B**
    
    ⬇️
    
    🧠 Step-by-Step Pedagogy
    
    ⬇️
    
    🌿 Hindi → Santali Translation
    
    ⬇️
    
    🔊 Hindi Voice Output
    
    ⬇️
    
    ✏️ Practice Question
    
    ⬇️
    
    📄 Bilingual Worksheet + PDF
    """
)


# ============================================================
# SIH DEMO FLOW
# ============================================================

st.markdown("---")

st.header(
    "5️⃣ SIH Demo Flow"
)

st.markdown(
    """
    ### 🎬 Recommended 2–3 Minute Demo

    **Step 1 — Teacher asks a question**

    🎤 Speak:

    > 25 और 17 को जोड़ने पर कितना होगा?

    **Step 2 — Speech recognition**

    System converts Hindi speech → Hindi text.

    **Step 3 — Curriculum retrieval**

    System retrieves relevant Class 3 mathematics
    examples from the local curriculum dataset.

    **Step 4 — AI pedagogy**

    Sarvam-105B generates a child-friendly,
    step-by-step explanation.

    **Step 5 — Vernacular bridge**

    Hindi explanation → Santali.

    **Step 6 — Voice**

    Hindi explanation is converted into speech.

    **Step 7 — Personalised practice**

    AI generates another question using the same concept.

    **Step 8 — Worksheet**

    Generate five questions across different
    Class 3 mathematics topics.

    **Step 9 — PDF**

    Download a printable Hindi–Santali worksheet
    with learning outcomes and answer key.
    """
)


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.markdown(
    """
    <div style="text-align:center;color:#777">

    **SIH 2026 Prototype**

    AI-Powered Vernacular Pedagogy & Real-Time Translation

    Hindi → Santali • Class 3 Mathematics

    </div>
    """,
    unsafe_allow_html=True
)
