import streamlit as st
import pandas as pd
import json
import re
from io import BytesIO
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

    class3_df = df[
        df["Class"].astype(str).str.contains("3", na=False)
    ].copy()

    dataset_loaded = True
    dataset_error = None

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

        hindi_text = str(
            row.get("Hindi_Text", "")
        ).strip().lower()

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
# AI WORKSHEET GENERATOR
# --------------------------------------------------

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
    {
      "number": 1,
      "topic": "Addition",
      "difficulty": "Easy",
      "question": "..."
    },
    {
      "number": 2,
      "topic": "Subtraction",
      "difficulty": "Easy",
      "question": "..."
    },
    {
      "number": 3,
      "topic": "Multiplication",
      "difficulty": "Medium",
      "question": "..."
    },
    {
      "number": 4,
      "topic": "Division",
      "difficulty": "Medium",
      "question": "..."
    },
    {
      "number": 5,
      "topic": "Word Problem",
      "difficulty": "Challenging",
      "question": "..."
    }
  ]
}
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
                    "content": (
                        "Generate a Class 3 Mathematics "
                        "worksheet covering different topics."
                    )
                }
            ],
            temperature=0.5,
            max_tokens=1200,
            reasoning_effort=None
        )

        worksheet_text = response.choices[0].message.content

        # Remove markdown code fences if AI adds them
        worksheet_text = re.sub(
            r"```json|```",
            "",
            worksheet_text
        ).strip()

        worksheet_data = json.loads(worksheet_text)

        return worksheet_data, None

    except Exception as e:
        return None, str(e)


# --------------------------------------------------
# GENERATE ANSWER FOR WORKSHEET
# --------------------------------------------------

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
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.1,
            max_tokens=50,
            reasoning_effort=None
        )

        return response.choices[0].message.content.strip()

    except Exception:
        return "Unavailable"


# --------------------------------------------------
# REGISTER FONT FOR HINDI
# --------------------------------------------------

def setup_pdf_font():

    font_paths = [
        "/usr/share/fonts/truetype/noto/NotoSansDevanagari-Regular.ttf",
        "/usr/share/fonts/opentype/noto/NotoSansDevanagari-Regular.ttf",
        "/usr/share/fonts/truetype/noto/NotoSansDevanagari-Regular.ttf"
    ]

    for path in font_paths:

        try:

            pdfmetrics.registerFont(
                TTFont("NotoDevanagari", path)
            )

            return "NotoDevanagari"

        except Exception:
            continue

    return "Helvetica"


# --------------------------------------------------
# CREATE PDF WORKSHEET
# --------------------------------------------------

def create_pdf(worksheet_data, santali_questions, answers):

    buffer = BytesIO()

    font_name = setup_pdf_font()

    document = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=1.5 * cm,
        leftMargin=1.5 * cm,
        topMargin=1.5 * cm,
        bottomMargin=1.5 * cm
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "WorksheetTitle",
        parent=styles["Title"],
        fontName=font_name,
        fontSize=20,
        leading=24,
        alignment=TA_CENTER,
        spaceAfter=10
    )

    subtitle_style = ParagraphStyle(
        "Subtitle",
        parent=styles["Normal"],
        fontName=font_name,
        fontSize=11,
        leading=15,
        alignment=TA_CENTER,
        spaceAfter=15
    )

    question_style = ParagraphStyle(
        "Question",
        parent=styles["Normal"],
        fontName=font_name,
        fontSize=11,
        leading=17,
        spaceAfter=8
    )

    small_style = ParagraphStyle(
        "Small",
        parent=styles["Normal"],
        fontName=font_name,
        fontSize=9,
        leading=13
    )

    story = []

    # ----------------------------------------------
    # HEADER
    # ----------------------------------------------

    story.append(
        Paragraph(
            "CLASS 3 — MATHEMATICS WORKSHEET",
            title_style
        )
    )

    story.append(
        Paragraph(
            "हिंदी ↔ संताली | Hindi ↔ Santali",
            subtitle_style
        )
    )

    student_table = Table(
        [
            [
                Paragraph(
                    "<b>नाम:</b> __________________________",
                    question_style
                ),
                Paragraph(
                    "<b>तारीख:</b> __________________",
                    question_style
                )
            ]
        ],
        colWidths=[10 * cm, 7 * cm]
    )

    student_table.setStyle(
        TableStyle(
            [
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 8)
            ]
        )
    )

    story.append(student_table)

    story.append(
        Paragraph(
            "प्रश्नों को ध्यान से पढ़ें और हल करें।",
            question_style
        )
    )

    story.append(Spacer(1, 8))

    # ----------------------------------------------
    # QUESTIONS
    # ----------------------------------------------

    for i, item in enumerate(
        worksheet_data["questions"]
    ):

        question_number = item.get(
            "number",
            i + 1
        )

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

        santali_question = ""

        if i < len(santali_questions):
            santali_question = santali_questions[i]

        story.append(
            Paragraph(
                f"<b>प्रश्न {question_number}</b> "
                f"({topic} • {difficulty})",
                question_style
            )
        )

        story.append(
            Paragraph(
                f"<b>हिंदी:</b> {hindi_question}",
                question_style
            )
        )

        story.append(
            Paragraph(
                f"<b>संताली:</b> {santali_question}",
                question_style
            )
        )

        story.append(
            Paragraph(
                "उत्तर: __________________________________________",
                question_style
            )
        )

        story.append(Spacer(1, 10))

    # ----------------------------------------------
    # ANSWER KEY
    # ----------------------------------------------

    story.append(PageBreak())

    story.append(
        Paragraph(
            "ANSWER KEY",
            title_style
        )
    )

    story.append(
        Paragraph(
            "शिक्षक के लिए उत्तर सूची",
            subtitle_style
        )
    )

    answer_data = [
        [
            Paragraph("<b>Question</b>", small_style),
            Paragraph("<b>Answer</b>", small_style)
        ]
    ]

    for i, answer in enumerate(answers):

        answer_data.append(
            [
                Paragraph(
                    str(i + 1),
                    small_style
                ),
                Paragraph(
                    str(answer),
                    small_style
                )
            ]
        )

    answer_table = Table(
        answer_data,
        colWidths=[4 * cm, 10 * cm]
    )

    answer_table.setStyle(
        TableStyle(
            [
                ("GRID", (0, 0), (-1, -1), 0.5, "black"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                ("TOPPADDING", (0, 0), (-1, -1), 8),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 8)
            ]
        )
    )

    story.append(answer_table)

    story.append(Spacer(1, 20))

    story.append(
        Paragraph(
            "Generated by AI Vernacular Maths Assistant",
            small_style
        )
    )

    document.build(story)

    buffer.seek(0)

    return buffer.getvalue()


# ==================================================
# SIDEBAR
# ==================================================

with st.sidebar:

    st.header("⚙️ System Status")

    if dataset_loaded:
        st.success("Dataset loaded")
    else:
        st.error("Dataset not loaded")

        if dataset_error:
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


# ==================================================
# SECTION 1 — AI MATHS ASSISTANT
# ==================================================

st.header("🧮 AI Maths Assistant")

question = st.text_area(
    "Enter a Class 3 maths question in Hindi",
    placeholder="उदाहरण: 25 और 17 को जोड़ने पर कितना होगा?",
    height=120
)

generate_button = st.button(
    "🚀 Generate Explanation",
    type="primary"
)


if generate_button:

    if not question.strip():

        st.warning(
            "कृपया पहले एक गणित का प्रश्न लिखें।"
        )

    else:

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


# ==================================================
# SECTION 2 — BILINGUAL WORKSHEET
# ==================================================

st.divider()

st.header("📝 AI Bilingual Worksheet Generator")

st.write(
    "Generate a Class 3 Mathematics worksheet "
    "covering different topics automatically."
)

st.caption(
    "The AI selects questions from multiple Class 3 "
    "mathematics concepts."
)


generate_worksheet_button = st.button(
    "📄 Generate Bilingual Worksheet"
)


if generate_worksheet_button:

    if not sarvam_available:

        st.error(
            "Sarvam AI is not connected."
        )

    else:

        # ------------------------------------------
        # GENERATE QUESTIONS
        # ------------------------------------------

        with st.spinner(
            "🧠 Creating Class 3 Maths worksheet..."
        ):

            worksheet_data, worksheet_error = (
                generate_worksheet()
            )

        if worksheet_data:

            st.subheader(
                "🇮🇳 Hindi Worksheet"
            )

            # Display Hindi questions
            for item in worksheet_data["questions"]:

                st.markdown(
                    f"**Q{item['number']} — "
                    f"{item['topic']} "
                    f"({item['difficulty']})**"
                )

                st.write(
                    item["question"]
                )

            # --------------------------------------
            # SANTALI TRANSLATIONS
            # --------------------------------------

            santali_questions = []

            st.subheader(
                "🌐 Santali Worksheet"
            )

            for item in worksheet_data["questions"]:

                with st.spinner(
                    f"Translating question {item['number']}..."
                ):

                    translated, translation_error = (
                        translate_to_santali(
                            item["question"]
                        )
                    )

                if translated:

                    santali_questions.append(
                        translated
                    )

                    st.markdown(
                        f"**Q{item['number']} — "
                        f"संताली**"
                    )

                    st.write(translated)

                else:

                    santali_questions.append(
                        "Translation unavailable."
                    )

                    st.error(
                        f"Translation failed for "
                        f"question {item['number']}: "
                        f"{translation_error}"
                    )

            # --------------------------------------
            # ANSWERS
            # --------------------------------------

            with st.spinner(
                "✅ Creating answer key..."
            ):

                answers = []

                for item in worksheet_data["questions"]:

                    answer = generate_answer(
                        item["question"]
                    )

                    answers.append(answer)

            # --------------------------------------
            # CREATE PDF
            # --------------------------------------

            with st.spinner(
                "📄 Creating printable PDF..."
            ):

                try:

                    pdf_data = create_pdf(
                        worksheet_data,
                        santali_questions,
                        answers
                    )

                    st.success(
                        "🎉 Bilingual worksheet PDF is ready!"
                    )

                    st.download_button(
                        label="📥 Download Bilingual PDF",
                        data=pdf_data,
                        file_name=(
                            "Class_3_Bilingual_Maths_Worksheet.pdf"
                        ),
                        mime="application/pdf"
                    )

                except Exception as e:

                    st.error(
                        f"PDF generation failed: {e}"
                    )

        else:

            st.error(
                f"Worksheet generation failed: "
                f"{worksheet_error}"
            )


# ==================================================
# PROTOTYPE INFORMATION
# ==================================================

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
        • AI-generated multi-topic worksheet
        • Bilingual worksheet output
        • Printable PDF worksheet
        • Local starter dataset integration

        AI services are powered by Sarvam AI.
        """
    )
