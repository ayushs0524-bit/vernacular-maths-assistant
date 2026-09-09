import streamlit as st
import pandas as pd
import json
import re
import os
import base64
import difflib
from io import BytesIO
from datetime import datetime

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
    PageBreak
)
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from sarvamai import SarvamAI


# ==================================================
# PAGE CONFIG
# ==================================================

st.set_page_config(
    page_title="AI Vernacular Maths Assistant",
    page_icon="📚",
    layout="wide"
)

# Light UI polish (item 10) — kept minimal so it can't break on any platform.
st.markdown(
    """
    <style>
    .block-container { padding-top: 2rem; }
    .stTabs [data-baseweb="tab-list"] { gap: 6px; }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("📚 AI Vernacular Maths Assistant")
st.caption(
    "AI-powered pedagogy • Class 3 Mathematics • Hindi → Santali • "
    "SIH 26042 Prototype"
)


# ==================================================
# SARVAM API CONFIGURATION
# ==================================================

sarvam_key = None
try:
    sarvam_key = st.secrets.get("SARVAM_API_KEY")
except Exception:
    # st.secrets raises if no secrets.toml exists at all on some platforms —
    # treat that the same as "no key configured" instead of crashing the app.
    sarvam_key = None

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
        sarvam_client = SarvamAI(api_subscription_key=sarvam_key)
        sarvam_available = True
    except Exception as e:
        sarvam_error = str(e)


# ==================================================
# LOAD DATASET (also doubles as the offline fallback — item 8)
# ==================================================

DATA_FILE = "Hindi_Santali_Maths_Dataset_Starter.xlsx"


@st.cache_data(show_spinner=False)
def load_dataset(path):
    df = pd.read_excel(path, sheet_name="Core_Seed_300")
    class3 = df[df["Class"].astype(str).str.contains("3", na=False)].copy()
    return class3


try:
    if os.path.exists(DATA_FILE):
        class3_df = load_dataset(DATA_FILE)
        dataset_loaded = True
        dataset_error = None
    else:
        class3_df = pd.DataFrame()
        dataset_loaded = False
        dataset_error = (
            f"'{DATA_FILE}' not found next to app.py. "
            "Place the Excel dataset in the same folder before running."
        )
except Exception as e:
    dataset_loaded = False
    class3_df = pd.DataFrame()
    dataset_error = str(e)


def _col(row, *names, default=""):
    """Safely read the first matching column name that exists on a row."""
    for name in names:
        if name in row and pd.notna(row[name]):
            return str(row[name])
    return default


# --------------------------------------------------
# EXACT DATASET MATCH
# --------------------------------------------------

def find_match(question):
    if not dataset_loaded or class3_df.empty:
        return None

    question_clean = question.strip().lower()

    for _, row in class3_df.iterrows():
        hindi_text = _col(row, "Hindi_Text").strip().lower()
        if hindi_text == question_clean:
            return row

    return None


# --------------------------------------------------
# RAG-LITE: retrieve similar dataset rows as grounding context (item 9)
# --------------------------------------------------

def retrieve_similar_context(question, top_k=3):
    """
    Lightweight retrieval-augmented-generation step: instead of a vector DB,
    we score every dataset question against the input using string similarity
    and feed the closest matches to the AI as grounding examples. This keeps
    generated explanations aligned with the seed curriculum's style, topics,
    and (where present) learning-outcome tags.
    """
    if not dataset_loaded or class3_df.empty:
        return []

    question_clean = question.strip().lower()
    scored = []

    for _, row in class3_df.iterrows():
        hindi_text = _col(row, "Hindi_Text")
        if not hindi_text:
            continue
        score = difflib.SequenceMatcher(
            None, question_clean, hindi_text.lower()
        ).ratio()
        scored.append((score, row))

    scored.sort(key=lambda x: x[0], reverse=True)

    results = []
    for score, row in scored[:top_k]:
        if score < 0.2:
            continue
        results.append({
            "hindi": _col(row, "Hindi_Text"),
            "santali": _col(row, "Santali_Text", "Santali_Ol_Chiki"),
            "topic": _col(row, "Topic", "Subtopic", default="General"),
            "learning_outcome": _col(
                row, "Learning_Outcome", "LO", "Outcome", default=""
            ),
        })

    return results


def build_context_block(context_rows):
    if not context_rows:
        return ""

    lines = ["Reference examples from the validated Class 3 curriculum seed set:"]
    for i, c in enumerate(context_rows, start=1):
        line = f"{i}. Hindi: {c['hindi']}"
        if c["topic"]:
            line += f" | Topic: {c['topic']}"
        if c["learning_outcome"]:
            line += f" | Learning outcome: {c['learning_outcome']}"
        lines.append(line)

    return "\n".join(lines)


# ==================================================
# AI PEDAGOGY ENGINE (cached — item 8 offline/cost efficiency)
# ==================================================

@st.cache_data(show_spinner=False, ttl=3600)
def generate_ai_explanation(question, context_block):
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
- If reference examples are provided, stay consistent with their topic,
  difficulty level, and terminology, but do not copy them verbatim.

Use exactly this structure:

समझते हैं:
<simple explanation>

हल:
<step-by-step calculation>

उत्तर:
<final answer>
"""

    user_prompt = f"""
{context_block}

Class 3 Mathematics question:

{question}

Explain this question for a Class 3 student.
"""

    try:
        response = sarvam_client.chat.completions(
            model="sarvam-105b",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.2,
            max_tokens=500,
            reasoning_effort=None
        )
        return response.choices[0].message.content, None
    except Exception as e:
        return None, str(e)


# ==================================================
# HINDI → SANTALI TRANSLATION (cached)
# ==================================================

@st.cache_data(show_spinner=False, ttl=3600)
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


# ==================================================
# VOICE: HINDI SPEECH → TEXT
# ==================================================

def transcribe_hindi_voice(audio_file):
    if not sarvam_available:
        return None, "Sarvam AI is not connected."

    try:
        audio_bytes = audio_file.getvalue()
        response = sarvam_client.speech_to_text.transcribe(
            file=audio_bytes,
            model="saaras:v3",
            language_code="hi-IN"
        )
        return response.transcript, None
    except Exception as e:
        return None, str(e)


# ==================================================
# VOICE OUTPUT: SANTALI TEXT → SPEECH (item 7)
# ==================================================

def synthesize_santali_speech(text):
    """
    Best-effort TTS wrapper. Sarvam's TTS endpoint/SDK surface can differ by
    SDK version, so every attempt is guarded — if TTS isn't available in your
    installed sarvamai version, this returns a clear message instead of
    crashing the app on deploy.
    """
    if not sarvam_available:
        return None, "Sarvam AI is not connected."

    if not hasattr(sarvam_client, "text_to_speech"):
        return None, (
            "This version of the sarvamai SDK does not expose a "
            "text_to_speech endpoint. Update the sarvamai package or check "
            "Sarvam's docs for the current TTS method name."
        )

    try:
        response = sarvam_client.text_to_speech.convert(
            text=text,
            target_language_code="sat-IN",
            model="bulbul:v2"
        )

        audio_field = getattr(response, "audios", None)
        if not audio_field:
            return None, "TTS call succeeded but returned no audio data."

        audio_b64 = audio_field[0]
        audio_bytes = base64.b64decode(audio_b64)
        return audio_bytes, None

    except Exception as e:
        return None, str(e)


# ==================================================
# PRACTICE QUESTION GENERATOR (cached)
# ==================================================

@st.cache_data(show_spinner=False, ttl=3600)
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
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.4,
            max_tokens=200,
            reasoning_effort=None
        )
        return response.choices[0].message.content, None
    except Exception as e:
        return None, str(e)


# ==================================================
# AI WORKSHEET GENERATOR
# ==================================================

def generate_worksheet():
    if not sarvam_available:
        return None, "Sarvam AI is not connected."

    system_prompt = """
You are an expert Class 3 primary-school mathematics teacher.

Create exactly 5 mathematics questions.

The questions must cover DIFFERENT Class 3 mathematics concepts.

Possible concepts:
- Addition
- Subtraction
- Multiplication
- Division
- Comparing numbers
- Place value
- Basic fractions
- Money
- Time
- Measurement
- Simple word problems
- Basic geometry

Rules:
- Suitable only for Class 3.
- Do not use Class 4 or higher mathematics.
- Mix calculation questions and word problems.
- Use simple Hindi.
- Use different numbers.
- Include easy, medium and challenging questions.
- Every question must have one clear numerical answer.
- Do NOT provide answers inside the questions.

Return ONLY valid JSON.

Use exactly this structure:

{
  "worksheet_title": "कक्षा 3 गणित अभ्यास पत्र",
  "questions": [
    {"number": 1, "topic": "Addition", "difficulty": "Easy", "question": "..."},
    {"number": 2, "topic": "Subtraction", "difficulty": "Easy", "question": "..."},
    {"number": 3, "topic": "Multiplication", "difficulty": "Medium", "question": "..."},
    {"number": 4, "topic": "Division", "difficulty": "Medium", "question": "..."},
    {"number": 5, "topic": "Word Problem", "difficulty": "Challenging", "question": "..."}
  ]
}
"""

    try:
        response = sarvam_client.chat.completions(
            model="sarvam-105b",
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": (
                        "Generate a Class 3 Mathematics worksheet "
                        "covering different topics."
                    )
                }
            ],
            temperature=0.5,
            max_tokens=1200,
            reasoning_effort=None
        )

        worksheet_text = response.choices[0].message.content
        worksheet_text = re.sub(r"```json|```", "", worksheet_text).strip()
        worksheet_data = json.loads(worksheet_text)
        return worksheet_data, None

    except Exception as e:
        return None, str(e)


# ==================================================
# GENERATE ANSWER FOR WORKSHEET
# ==================================================

def generate_answer(question):
    if not sarvam_available:
        return "Unavailable"

    prompt = f"""
Solve this Class 3 mathematics question.

Return ONLY the final numerical answer.
Do not provide explanation.

Question:
{question}
"""

    try:
        response = sarvam_client.chat.completions(
            model="sarvam-105b",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=50,
            reasoning_effort=None
        )
        return response.choices[0].message.content.strip()
    except Exception:
        return "Unavailable"


# ==================================================
# FONTS FOR PDF (Hindi + Santali/Ol Chiki rendering)
# ==================================================
#
# IMPORTANT DEPLOYMENT NOTE:
# Streamlit Community Cloud (and most fresh Linux containers) do NOT ship
# Noto Devanagari/Ol Chiki fonts by default. If no matching font is found,
# ReportLab falls back to Helvetica, which cannot render Hindi or Ol Chiki
# glyphs — the PDF will generate without errors but the text will look blank
# or show boxes. To fix this for your real deployment:
#   1. Download NotoSansDevanagari-Regular.ttf and NotoSansOlChiki-Regular.ttf
#   2. Put them in a "fonts/" folder next to app.py
#   3. Redeploy — the code below checks that folder first.

FONT_CANDIDATES_DEVANAGARI = [
    "fonts/NotoSansDevanagari-Regular.ttf",
    "/usr/share/fonts/truetype/noto/NotoSansDevanagari-Regular.ttf",
    "/usr/share/fonts/opentype/noto/NotoSansDevanagari-Regular.ttf",
]

FONT_CANDIDATES_OLCHIKI = [
    "fonts/NotoSansOlChiki-Regular.ttf",
    "/usr/share/fonts/truetype/noto/NotoSansOlChiki-Regular.ttf",
    "/usr/share/fonts/opentype/noto/NotoSansOlChiki-Regular.ttf",
]


@st.cache_resource(show_spinner=False)
def setup_pdf_fonts():
    """
    Registers whatever fonts are available and returns a dict describing
    what worked, so the UI can warn the user instead of silently producing
    a blank-looking PDF.
    """
    result = {"devanagari": None, "olchiki": None, "warning": None}

    for path in FONT_CANDIDATES_DEVANAGARI:
        if os.path.exists(path):
            try:
                pdfmetrics.registerFont(TTFont("NotoDevanagari", path))
                result["devanagari"] = "NotoDevanagari"
                break
            except Exception:
                continue

    for path in FONT_CANDIDATES_OLCHIKI:
        if os.path.exists(path):
            try:
                pdfmetrics.registerFont(TTFont("NotoOlChiki", path))
                result["olchiki"] = "NotoOlChiki"
                break
            except Exception:
                continue

    if not result["devanagari"] or not result["olchiki"]:
        result["warning"] = (
            "Devanagari/Ol Chiki font files were not found next to app.py "
            "(expected in a 'fonts/' folder). PDF text may render blank for "
            "Hindi and/or Santali. See the sidebar for details."
        )

    return result


def _font_or_fallback(name):
    return name if name else "Helvetica"


# ==================================================
# PDF BUILDER — shared style setup
# ==================================================

def _get_pdf_styles(font_name):
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "WorksheetTitle", parent=styles["Title"], fontName=font_name,
        fontSize=20, leading=24, alignment=TA_CENTER, spaceAfter=10
    )
    subtitle_style = ParagraphStyle(
        "Subtitle", parent=styles["Normal"], fontName=font_name,
        fontSize=11, leading=15, alignment=TA_CENTER, spaceAfter=15
    )
    question_style = ParagraphStyle(
        "Question", parent=styles["Normal"], fontName=font_name,
        fontSize=11, leading=17, spaceAfter=8
    )
    small_style = ParagraphStyle(
        "Small", parent=styles["Normal"], fontName=font_name,
        fontSize=9, leading=13
    )
    return title_style, subtitle_style, question_style, small_style


# ==================================================
# PDF #1 — SINGLE QUESTION (bilingual explanation card)
# ==================================================

def create_single_question_pdf(question, explanation, santali_explanation,
                                practice_question, fonts):
    buffer = BytesIO()
    font_name = _font_or_fallback(fonts["devanagari"])

    document = SimpleDocTemplate(
        buffer, pagesize=A4,
        rightMargin=1.5 * cm, leftMargin=1.5 * cm,
        topMargin=1.5 * cm, bottomMargin=1.5 * cm
    )

    title_style, subtitle_style, question_style, small_style = (
        _get_pdf_styles(font_name)
    )

    story = [
        Paragraph("CLASS 3 — MATHEMATICS", title_style),
        Paragraph("हिंदी ↔ संताली | Hindi ↔ Santali", subtitle_style),
        Spacer(1, 8),
        Paragraph("<b>प्रश्न / Question:</b>", question_style),
        Paragraph(question, question_style),
        Spacer(1, 6),
        Paragraph("<b>समझाइए / AI Explanation (Hindi):</b>", question_style),
        Paragraph(explanation.replace("\n", "<br/>"), question_style),
        Spacer(1, 6),
    ]

    if santali_explanation:
        story.append(
            Paragraph("<b>संताली अनुवाद / Santali Translation:</b>", question_style)
        )
        story.append(
            Paragraph(santali_explanation.replace("\n", "<br/>"), question_style)
        )
        story.append(Spacer(1, 6))

    if practice_question:
        story.append(
            Paragraph("<b>अभ्यास प्रश्न / Practice Question:</b>", question_style)
        )
        story.append(Paragraph(practice_question, question_style))
        story.append(Paragraph(
            "उत्तर: __________________________________________", question_style
        ))

    story.append(Spacer(1, 20))
    story.append(Paragraph(
        f"Generated by AI Vernacular Maths Assistant · "
        f"{datetime.now().strftime('%d %b %Y, %H:%M')}",
        small_style
    ))

    document.build(story)
    buffer.seek(0)
    return buffer.getvalue()


# ==================================================
# PDF #2 — WORKSHEET (multi-question + answer key)
# ==================================================

def create_worksheet_pdf(worksheet_data, santali_questions, answers, fonts):
    buffer = BytesIO()
    font_name = _font_or_fallback(fonts["devanagari"])

    document = SimpleDocTemplate(
        buffer, pagesize=A4,
        rightMargin=1.5 * cm, leftMargin=1.5 * cm,
        topMargin=1.5 * cm, bottomMargin=1.5 * cm
    )

    title_style, subtitle_style, question_style, small_style = (
        _get_pdf_styles(font_name)
    )

    story = [
        Paragraph("CLASS 3 — MATHEMATICS WORKSHEET", title_style),
        Paragraph("हिंदी ↔ संताली | Hindi ↔ Santali", subtitle_style),
    ]

    student_table = Table(
        [[
            Paragraph("<b>नाम:</b> __________________________", question_style),
            Paragraph("<b>तारीख:</b> __________________", question_style)
        ]],
        colWidths=[10 * cm, 7 * cm]
    )
    student_table.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8)
    ]))
    story.append(student_table)

    story.append(Paragraph("प्रश्नों को ध्यान से पढ़ें और हल करें।", question_style))
    story.append(Spacer(1, 8))

    for i, item in enumerate(worksheet_data["questions"]):
        question_number = item.get("number", i + 1)
        topic = item.get("topic", "Mathematics")
        difficulty = item.get("difficulty", "Medium")
        hindi_question = item.get("question", "")
        santali_question = santali_questions[i] if i < len(santali_questions) else ""

        story.append(Paragraph(
            f"<b>प्रश्न {question_number}</b> ({topic} • {difficulty})",
            question_style
        ))
        story.append(Paragraph(f"<b>हिंदी:</b> {hindi_question}", question_style))
        story.append(Paragraph(f"<b>संताली:</b> {santali_question}", question_style))
        story.append(Paragraph(
            "उत्तर: __________________________________________", question_style
        ))
        story.append(Spacer(1, 10))

    story.append(PageBreak())
    story.append(Paragraph("ANSWER KEY", title_style))
    story.append(Paragraph("शिक्षक के लिए उत्तर सूची", subtitle_style))

    answer_data = [[
        Paragraph("<b>Question</b>", small_style),
        Paragraph("<b>Answer</b>", small_style)
    ]]
    for i, answer in enumerate(answers):
        answer_data.append([
            Paragraph(str(i + 1), small_style),
            Paragraph(str(answer), small_style)
        ])

    answer_table = Table(answer_data, colWidths=[4 * cm, 10 * cm])
    answer_table.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.5, "black"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8)
    ]))
    story.append(answer_table)
    story.append(Spacer(1, 20))
    story.append(Paragraph("Generated by AI Vernacular Maths Assistant", small_style))

    document.build(story)
    buffer.seek(0)
    return buffer.getvalue()


# ==================================================
# SIDEBAR — SYSTEM STATUS
# ==================================================

pdf_fonts = setup_pdf_fonts()

with st.sidebar:
    st.header("⚙️ System Status")

    if dataset_loaded:
        st.success("Dataset loaded (offline fallback ready)")
    else:
        st.error("Dataset not loaded")
        if dataset_error:
            st.caption(dataset_error)

    if sarvam_available:
        st.success("Sarvam AI connected")
    else:
        st.error("Sarvam API not configured")
        if sarvam_error:
            st.caption(f"Debug: {sarvam_error}")

    if pdf_fonts["warning"]:
        st.warning(pdf_fonts["warning"])
    else:
        st.success("PDF fonts loaded")

    st.divider()
    st.info("AI Pedagogy: Sarvam 105B")
    st.info("Translation: Hindi → Santali")
    st.info("Retrieval: dataset-grounded (RAG-lite)")
    st.info("Scope: Class 3 Mathematics")

    if not sarvam_available and dataset_loaded:
        st.caption(
            "Offline mode: AI generation is unavailable, but exact dataset "
            "matches will still show pre-existing Hindi/Santali pairs."
        )


# ==================================================
# TABS — item 10: clearer demo flow
# ==================================================

tab_ask, tab_worksheet, tab_about = st.tabs(
    ["🧮 Ask a Question", "📝 Worksheet Generator", "ℹ️ About"]
)


# ==================================================
# TAB 1 — ASK A QUESTION
# ==================================================

with tab_ask:
    st.subheader("⌨️ Type Your Question")

    typed_question = st.text_area(
        "Enter a Class 3 maths question in Hindi",
        placeholder="उदाहरण: 25 और 17 को जोड़ने पर कितना होगा?",
        height=120
    )

    st.subheader("🎤 Or Ask by Voice")
    audio = st.audio_input("Record your Hindi maths question")

    voice_question = None
    voice_santali_quick = None

    if audio is not None:
        with st.spinner("🎧 Converting your voice to Hindi text..."):
            voice_question, voice_error = transcribe_hindi_voice(audio)

        if voice_question:
            st.success("✅ Voice converted successfully!")
            st.markdown("### 🗣️ You said (Hindi):")
            st.info(voice_question)

            # Item 6: voice input goes straight through to Santali too.
            with st.spinner("🌐 Translating your spoken question to Santali..."):
                voice_santali_quick, voice_translate_error = (
                    translate_to_santali(voice_question)
                )

            if voice_santali_quick:
                st.markdown("### 🌐 In Santali:")
                st.info(voice_santali_quick)
            else:
                st.error(f"Could not translate voice input: {voice_translate_error}")
        else:
            st.error(f"Voice transcription failed: {voice_error}")

    question = voice_question if voice_question else typed_question

    generate_button = st.button("🚀 Generate Explanation", type="primary")

    if generate_button:
        if not question or not question.strip():
            st.warning("कृपया पहले एक गणित का प्रश्न लिखें।")
        else:
            matched_row = find_match(question)

            if matched_row is not None:
                st.info("📚 This question was found in the local Class 3 dataset.")

            if not sarvam_available and matched_row is not None:
                # Offline fallback: show whatever the dataset already has.
                st.subheader("📚 Offline Dataset Answer")
                dataset_santali = _col(
                    matched_row, "Santali_Text", "Santali_Ol_Chiki"
                )
                if dataset_santali:
                    st.write(dataset_santali)
                else:
                    st.warning(
                        "Sarvam AI is unavailable and no pre-stored Santali "
                        "text exists for this question."
                    )
            elif not sarvam_available:
                st.error(
                    "Sarvam AI is not connected, and this question isn't in "
                    "the local dataset, so no offline answer is available."
                )
            else:
                context_rows = retrieve_similar_context(question)
                context_block = build_context_block(context_rows)

                with st.spinner("🧠 AI is creating the explanation..."):
                    explanation, explanation_error = generate_ai_explanation(
                        question, context_block
                    )

                santali_text = None
                practice_question = None

                if explanation:
                    st.subheader("🧠 AI Hindi Explanation")
                    st.markdown(explanation)

                    if context_rows:
                        with st.expander("📖 Curriculum examples used for grounding"):
                            st.text(context_block)

                    with st.spinner("🌐 Translating explanation into Santali..."):
                        santali_text, translation_error = translate_to_santali(
                            explanation
                        )

                    if santali_text:
                        st.subheader("🌐 Santali Translation")
                        st.write(santali_text)

                        audio_bytes, tts_error = synthesize_santali_speech(
                            santali_text
                        )
                        if audio_bytes:
                            st.audio(audio_bytes, format="audio/wav")
                        else:
                            st.caption(f"🔊 Voice output unavailable: {tts_error}")
                    else:
                        st.error(f"Translation failed: {translation_error}")

                    with st.spinner("📝 Creating a practice question..."):
                        practice_question, practice_error = (
                            generate_practice_question(question)
                        )

                    if practice_question:
                        st.subheader("📝 Practice Question")
                        st.write(practice_question)
                    else:
                        st.error(
                            f"Practice question generation failed: {practice_error}"
                        )

                    st.divider()
                    single_pdf = create_single_question_pdf(
                        question, explanation, santali_text,
                        practice_question, pdf_fonts
                    )
                    st.download_button(
                        label="📥 Download This Question as PDF",
                        data=single_pdf,
                        file_name="Class_3_Question_Bilingual.pdf",
                        mime="application/pdf"
                    )
                else:
                    st.error(f"Could not generate explanation: {explanation_error}")


# ==================================================
# TAB 2 — WORKSHEET GENERATOR
# ==================================================

with tab_worksheet:
    st.header("📝 AI Bilingual Worksheet Generator")
    st.write(
        "Generate a Class 3 Mathematics worksheet covering different "
        "topics automatically."
    )
    st.caption(
        "The AI selects questions from multiple Class 3 mathematics concepts."
    )

    generate_worksheet_button = st.button("📄 Generate Bilingual Worksheet")

    if generate_worksheet_button:
        if not sarvam_available:
            st.error(
                "Sarvam AI is not connected — worksheet generation needs "
                "the live API and can't use the offline dataset fallback."
            )
        else:
            with st.spinner("🧠 Creating Class 3 Maths worksheet..."):
                worksheet_data, worksheet_error = generate_worksheet()

            if worksheet_data:
                st.subheader("🇮🇳 Hindi Worksheet")
                for item in worksheet_data["questions"]:
                    st.markdown(
                        f"**Q{item['number']} — {item['topic']} "
                        f"({item['difficulty']})**"
                    )
                    st.write(item["question"])

                santali_questions = []
                st.subheader("🌐 Santali Worksheet")

                for item in worksheet_data["questions"]:
                    with st.spinner(f"Translating question {item['number']}..."):
                        translated, translation_error = translate_to_santali(
                            item["question"]
                        )

                    if translated:
                        santali_questions.append(translated)
                        st.markdown(f"**Q{item['number']} — संताली**")
                        st.write(translated)
                    else:
                        santali_questions.append("Translation unavailable.")
                        st.error(
                            f"Translation failed for question "
                            f"{item['number']}: {translation_error}"
                        )

                with st.spinner("✅ Creating answer key..."):
                    answers = [
                        generate_answer(item["question"])
                        for item in worksheet_data["questions"]
                    ]

                with st.spinner("📄 Creating printable PDF..."):
                    try:
                        pdf_data = create_worksheet_pdf(
                            worksheet_data, santali_questions, answers, pdf_fonts
                        )
                        st.success("🎉 Bilingual worksheet PDF is ready!")
                        st.download_button(
                            label="📥 Download Bilingual Worksheet PDF",
                            data=pdf_data,
                            file_name="Class_3_Bilingual_Maths_Worksheet.pdf",
                            mime="application/pdf"
                        )
                    except Exception as e:
                        st.error(f"PDF generation failed: {e}")
            else:
                st.error(f"Worksheet generation failed: {worksheet_error}")


# ==================================================
# TAB 3 — ABOUT / ROADMAP
# ==================================================

with tab_about:
    st.write(
        """
        This prototype demonstrates an AI-powered vernacular mathematics
        learning assistant for Class 3 students.

        **Current capabilities**
        - Class 3 Mathematics, Hindi input, voice or text
        - AI-generated simple Hindi explanation, grounded in curriculum
          examples retrieved from the seed dataset (RAG-lite)
        - Hindi → Santali translation (text and, where available, voice)
        - Practice question generation
        - AI-generated multi-topic worksheet with answer key
        - Printable bilingual PDFs — single question and full worksheet
        - Offline fallback to the local dataset when the API is unreachable

        AI services are powered by Sarvam AI. Santali translations are
        machine-generated and have not yet been validated by native
        speakers — this is a prototype, not a certified translation tool.
        """
    )

    st.subheader("Roadmap")
    st.markdown(
        """
        1. ✅ Hindi → Santali translation
        2. ✅ AI step-by-step pedagogy
        3. ✅ AI practice question
        4. ✅ AI multi-topic worksheet
        5. ✅ Printable bilingual PDF worksheet (+ single-question PDF)
        6. ✅ Voice input — Hindi speech → Santali
        7. ✅ Voice output (best-effort; depends on Sarvam TTS availability)
        8. ✅ Offline/caching support (dataset fallback + response caching)
        9. ✅ RAG-lite + curriculum grounding (upgrade path to full
           learning-outcome-aligned RAG once tagged data is available)
        10. ✅ Tabbed SIH-ready demo flow
        """
    )