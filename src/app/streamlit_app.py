
import sys
from pathlib import Path
import streamlit as st
import duckdb
from openai import OpenAI

import json

# -------------------------------------------------------------------
# Ensure src/ is importable
# -------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

# -------------------------------------------------------------------
# Local imports
# -------------------------------------------------------------------
# from agent.sql_agent import ask_question, classify_intent
# from agent.rag_agent import answer_with_rag
# from agent.brain_agent import answer_with_brain
from agent.unified_agent import * 
# -------------------------------------------------------------------
# OpenAI client (conversation routing)
# -------------------------------------------------------------------
llm_client = OpenAI()

def llm_is_followup(user_message: str, last_assistant_message: str | None):
    """
    Ask the LLM if the user message is a contextual follow-up.

    Returns:
    {
      "is_followup": true | false,
      "action": "table" | "chart" | "text" | null
    }
    """
    system = """
You are a conversational router.
Determine if the user message refers to the previous assistant response.

If YES:
- is_followup = true
- infer the expected action

Possible actions:
- table  : show a table
- chart  : show a chart
- text   : explain / reformulate
- null   : nothing specific

Respond STRICTLY in valid JSON.
"""

    messages = [
        {"role": "system", "content": system},
        {
            "role": "user",
            "content": f"""
User message:
"{user_message}"

Previous assistant message:
"{last_assistant_message or ''}"
"""
        }
    ]

    resp = llm_client.chat.completions.create(
        model="gpt-4.1-mini",
        temperature=0,
        messages=messages,
    )

    return json.loads(resp.choices[0].message.content)

# -------------------------------------------------------------------
# Streamlit UI CONFIG (DARK STYLE)
# -------------------------------------------------------------------
st.set_page_config(
    page_title="CEI 2025 Chat",
    page_icon="🗳️",
    layout="wide"
)

# -------------------------------------------------------------------
# Global dark theme overrides
# -------------------------------------------------------------------
st.markdown(
    """
    <style>
        :root { color-scheme: dark; }

        [data-testid="stAppViewContainer"] {
            background: #030712;
            color: #ffffff !important;
        }

        [data-testid="stSidebar"] {
            background: #020617;
            color: #F1F5F9 !important;
        }

        .title, .subtitle, .stMarkdown, p, span, div {
            color: #ffffff !important;
        }

        .title {
            font-size: 3rem !important;
            font-weight: 900 !important;
            margin-bottom: 0.6rem;
        }

        .subtitle {
            font-size: 1.35rem !important;
            color: #e2e8f0 !important;
            margin-bottom: 1.8rem;
        }

        .stChatMessage[data-testid="stChatMessage-assistant"] {
            background-color: #1e3a8a !important;
            border-radius: 14px;
            padding: 12px;
            margin-bottom: 10px;
        }

        .stChatMessage[data-testid="stChatMessage-user"] {
            background-color: #065f46 !important;
            border-radius: 14px;
            padding: 12px;
            margin-bottom: 10px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------------------------------------------------------
# TABLE DISPLAY LOGIC (EXISTING – KEPT)
# -------------------------------------------------------------------
def wants_table(question: str) -> bool:
    q = question.lower()
    keywords = [
        "tableau", "detail", "détail", "chiffre", "stat",
        "participation", "inscrits", "votants", "taux",
        "liste", "montre", "affiche",
    ]
    return any(k in q for k in keywords)

DISPLAY_COLS = [
    "region",
    "code_circonscription",
    "circonscription",
    "nb_bv",
    "inscrits",
    "votants",
    "taux_participation",
]

# -------------------------------------------------------------------
# HEADER
# -------------------------------------------------------------------
st.markdown(
    "<div class='title'>🇨🇮 🗳️ CEI 2025 — Chat d’analyse des résultats</div>",
    unsafe_allow_html=True
)
st.markdown(
    "<div class='subtitle'>Pose tes questions sur les résultats électoraux.</div>",
    unsafe_allow_html=True
)

# -------------------------------------------------------------------
# SIDEBAR HISTORIQUE
# -------------------------------------------------------------------
st.sidebar.markdown("### 🔁 Historique récent")

try:
    con = duckdb.connect("db/results.duckdb")
    hist = con.execute(
        "SELECT ts, question FROM history ORDER BY ts DESC LIMIT 20"
    ).fetch_df()
    con.close()

    if not hist.empty:
        for _, row in hist.iterrows():
            st.sidebar.markdown(
                f"• **{row['ts']}**<br>"
                f"<span style='font-size:0.85rem;color:#94a3b8;'>{row['question']}</span>",
                unsafe_allow_html=True,
            )
    else:
        st.sidebar.caption("Aucune question enregistrée.")
except Exception:
    st.sidebar.caption("Historique indisponible.")

# -------------------------------------------------------------------
# CHAT SESSION MEMORY
# -------------------------------------------------------------------
# -------------------------------------------------------------------
# CHAT SESSION MEMORY
# -------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Bonjour 👋 Je suis ton assistant CEI 2025. Pose-moi une question.",
            "display": "Résumé texte"
        }
    ]

# 🧠 Mémoire conversationnelle (OBLIGATOIRE)
if "last_answer" not in st.session_state:
    st.session_state.last_answer = None

if "last_facts" not in st.session_state:
    st.session_state.last_facts = None


for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("source_rows"):
            st.dataframe(msg["source_rows"], use_container_width=True)

# -------------------------------------------------------------------
# USER INPUT + MODE
# -------------------------------------------------------------------
col1, col2 = st.columns([4, 2])

with col1:
    user_question = st.chat_input("Écris ta question ici…")

with col2:
    display_for_this_input = st.radio(
        "Mode",
        ("Résumé texte", "Graphique"),
        index=0,
        horizontal=True
    )

if not user_question:
    st.stop()

st.session_state.messages.append(
    {"role": "user", "content": user_question, "display": display_for_this_input}
)

with st.chat_message("user"):
    st.markdown(user_question)

# -------------------------------------------------------------------
# CONTEXTUAL FOLLOW-UP (LLM-DRIVEN, ADDED)
# -------------------------------------------------------------------
last_assistant_msg = None
for m in reversed(st.session_state.messages):
    if m["role"] == "assistant":
        last_assistant_msg = m["content"]
        break

decision = llm_is_followup(user_question, last_assistant_msg)

if (
    decision.get("is_followup") is True
    and "last_rows" in st.session_state
    and st.session_state.last_rows is not None
):
    rows = st.session_state.last_rows

    if decision.get("action") == "table":
        rows_display = (
            rows[DISPLAY_COLS]
            .drop_duplicates()
            .reset_index(drop=True)
        )

        with st.chat_message("assistant"):
            st.markdown("Voici le tableau correspondant 👇")
            st.dataframe(rows_display, use_container_width=True)

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": "Voici le tableau correspondant 👇",
                "source_rows": rows_display,
                "display": "Résumé texte",
            }
        )

        st.stop()

# -------------------------------------------------------------------
# PROCESS QUESTION (EXISTING FLOW – UNCHANGED)
# -------------------------------------------------------------------
# intent = classify_intent(user_question)

with st.chat_message("assistant"):
    with st.spinner("Je réfléchis…"):

        assistant_content = ""
        rows_display = None
        assistant_chart = None
        
        
        result = ask(user_question)

        assistant_content = result.get("text", "")
        # 🧠 Mémoire conversationnelle
        st.session_state.last_answer = assistant_content

        if result.get("engine") in {"sql", "rag"}:
            st.session_state.last_facts = {
                "engine": result.get("engine"),
                "question": user_question,
                "raw": result.get("sql") or result.get("sources"),
                "text": assistant_content
            }

        rows = result.get("data")
        assistant_chart = result.get("chart")

        if rows is not None and not rows.empty:
            st.session_state.last_rows = rows


        # if intent in {"seats", "ranking", "turnout", "chart"}:
        #     result = ask_question(user_question)
        #     assistant_content = result.get("text", "")
        #     assistant_chart = result.get("chart")

        # else:
        #     answer, rows = answer_with_brain(user_question)
        #     # answer, rows = answer_with_rag(user_question)
        #     assistant_content = answer

            # 🔑 Save context for conversation
            if rows is not None and not rows.empty:
                st.session_state.last_rows = rows

            # Existing logic kept
            if (
                rows is not None
                and not rows.empty
                and wants_table(user_question)
                and display_for_this_input == "Résumé texte"
            ):
                rows_display = (
                    rows[DISPLAY_COLS]
                    .drop_duplicates()
                    .reset_index(drop=True)
                )

        # --- DISPLAY ---
        st.markdown(assistant_content)

        
        if assistant_chart and display_for_this_input == "Graphique":
            st.altair_chart(assistant_chart, use_container_width=True)

        # if assistant_chart and display_for_this_input == "Graphique":
        #     st.image(assistant_chart, caption="Graphique basé sur les résultats")

        if rows_display is not None:
            st.dataframe(rows_display, use_container_width=True)

# -------------------------------------------------------------------
# SAVE MESSAGE
# -------------------------------------------------------------------
st.session_state.messages.append(
    {
        "role": "assistant",
        "content": assistant_content,
        "source_rows": rows_display,
        "display": display_for_this_input
    }
)


