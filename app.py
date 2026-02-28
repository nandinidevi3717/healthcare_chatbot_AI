import streamlit as st
import pickle
import google.generativeai as genai
import requests
import numpy as np
import time

#  API CONFIGURATION
GEMINI_API_KEY = ""
genai.configure(api_key=GEMINI_API_KEY)
model_ai = genai.GenerativeModel("models/gemini-2.5-flash")

#  LOAD ML MODELS

# RAW GitHub file URLs
MODEL_URL = "https://raw.githubusercontent.com/nandinidevi3717/healthcare_chatbot_AI/main/disease_model.pkl"
SYMPTOM_URL = "https://raw.githubusercontent.com/nandinidevi3717/healthcare_chatbot_AI/main/symptom_list.pkl"

try:
    # Download model file
    model_response = requests.get(MODEL_URL)
    model = pickle.loads(model_response.content)

    # Download symptom list file
    symptom_response = requests.get(SYMPTOM_URL)
    symptom_list = pickle.loads(symptom_response.content)

except Exception as e:
    st.error("Error loading model files from GitHub.")
    st.error(str(e))


# ADVANCED UI / CUSTOM CSS
st.set_page_config(
    page_title="Next-Gen|Medical Assistant", layout="wide", page_icon="❄️"
)

st.markdown(
    """
<style>
    /* Main Background */
    .stApp {
        background: radial-gradient(circle at top right, #0a192f, #020617);
        color: #e2e8f0;
    }
    
    /* Glowing Sidebar */
    section[data-testid="stSidebar"] {
        background-color: rgba(15, 23, 42, 0.8) !important;
        border-right: 1px solid rgba(0, 255, 225, 0.2);
    }

    /* Chat Bubbles */
    .stChatMessage {
        background-color: rgba(30, 41, 59, 0.5) !important;
        border-radius: 15px !important;
        border: 1px solid rgba(0, 255, 225, 0.1) !important;
        margin-bottom: 10px !important;
        transition: all 0.3s ease;
    }
    
    .stChatMessage:hover {
        border: 1px solid rgba(0, 255, 225, 0.4) !important;
        box-shadow: 0 0 15px rgba(0, 255, 225, 0.1);
    }

    /* Input Box Glowing */
    .stChatInputContainer {
        border-top: 1px solid rgba(0, 255, 225, 0.3) !important;
    }

    /* Custom Header */
    .med-header {
        font-family: 'Inter', sans-serif;
        background: linear-gradient(90deg, #00ffe1, #7000ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem;
        font-weight: 800;
        text-align: center;
        margin-bottom: 0px;
        filter: drop-shadow(0 0 10px rgba(0, 255, 225, 0.3));
    }

    /* Status Cards */
    .med-card {
        background: rgba(255, 255, 255, 0.05);
        padding: 20px;
        border-radius: 15px;
        border-left: 5px solid #00ffe1;
        margin: 10px 0;
    }
</style>
""",
    unsafe_allow_html=True,
)

#  STATE MANAGEMENT
if "step" not in st.session_state:
    st.session_state.step = "welcome"
if "messages" not in st.session_state:
    st.session_state.messages = []
if "user_data" not in st.session_state:
    st.session_state.user_data = {
        "symptoms": "",
        "age": "",
        "gender": "",
        "duration": "",
        "history": "",
        "meds": "",
        "allergies": "",
    }

# SIDEBAR UI
with st.sidebar:
    st.markdown("## ⚚ MediScan Dashboard")
    st.info(
        " 🖥️ System: Online\n\n✦ Model: Gemini-2.5-flash\n\n👩🏻‍🏫 Diagnostic: Active"
    )
    st.divider()
    st.write("📋 **Current Session Profile**")
    st.write(f"👤 Age: {st.session_state.user_data['age']}")
    st.write(f"⚧ Gender: {st.session_state.user_data['gender']}")
    if st.button("🔄 Reset Consultation"):
        st.session_state.step = "welcome"
        st.session_state.messages = []
        st.session_state.user_data = {k: "" for k in st.session_state.user_data}
        st.rerun()

#  CHAT LOGIC
st.markdown('<h class="med-header">NovaCure-AI 𔒝</h>', unsafe_allow_html=True)
st.caption("Advanced Medical Diagnostic Intelligence • Professional Grade ⚛")

# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- Step-by-Step Questions ---
questions = {
    "welcome": "Hello! I am your AI Medical Assistant 🤖ིྀ.  To help you better, please describe your **symptoms** in detail (e.g., 'I have a sharp headache and fever').",
    "age": "Got it. How old are you? 🎂",
    "gender": "What is your gender? (Male/Female/Other) ⚧",
    "duration": "How long have you been experiencing these symptoms? (e.g., 3 days, 1 week) ⏳",
    "history": "Do you have any known medical history? (Diabetes, BP, Asthma, etc. Type 'None' if applicable) 📜",
    "meds": "Are you currently taking any medications? 💊",
    "allergies": "Finally, do you have any allergies? 🚫",
}

# Auto-post first greeting
if not st.session_state.messages:
    st.session_state.messages.append(
        {"role": "assistant", "content": questions["welcome"]}
    )
    st.rerun()

# Handle User Input
if prompt := st.chat_input("Type your response here..."):
    # Display user message
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Store data based on current step
    current_step = st.session_state.step
    if current_step == "welcome":
        st.session_state.user_data["symptoms"] = prompt
        st.session_state.step = "age"
    elif current_step == "age":
        st.session_state.user_data["age"] = prompt
        st.session_state.step = "gender"
    elif current_step == "gender":
        st.session_state.user_data["gender"] = prompt
        st.session_state.step = "duration"
    elif current_step == "duration":
        st.session_state.user_data["duration"] = prompt
        st.session_state.step = "history"
    elif current_step == "history":
        st.session_state.user_data["history"] = prompt
        st.session_state.step = "meds"
    elif current_step == "meds":
        st.session_state.user_data["meds"] = prompt
        st.session_state.step = "allergies"
    elif current_step == "allergies":
        st.session_state.user_data["allergies"] = prompt
        st.session_state.step = "analyze"

    # Post next question if not done
    if st.session_state.step != "analyze":
        next_q = questions[st.session_state.step]
        st.session_state.messages.append({"role": "assistant", "content": next_q})

    st.rerun()

#  FINAL ANALYSIS
if st.session_state.step == "analyze":
    with st.status(
        "Initializing Medical Neural Engine ــــــــــﮩ٨ـ", expanded=True
    ) as status:

        # 1. Extract Symptoms via NLP
        st.write("🔍 Extracting clinical markers...")
        nlp_prompt = f"You are a clinical medical NLP assistant trained to identify medically relevant symptoms only.Task:Extract ONLY medically recognized symptoms from the sentence below.Rules:- Ignore non-medical words, emotions, casual expressions, and irrelevant information.- Convert symptoms into standard medical terminology when possible.Example: high temperature → fever- Do not include diseases, diagnoses, treatments, or medications.- Do not include body parts unless they indicate a symptom (e.g., chest pain is valid).- Remove duplicates.- Return output in lowercase.- Output format: comma-separated list only.- If no symptoms are found, return: none(say them to give relavant symptoms)- If duration is mentioned, do NOT include duration in output.- If negation is present (e.g., no fever), exclude that symptom.Sentence: {st.session_state.user_data['symptoms']}"
        # (Simplified prompt logic for speed, keeping your exact prompt rules internally)
        try:
            nlp_res = model_ai.generate_content(nlp_prompt).text.lower()

            # 2. ML Prediction
            st.write("🧬 Cross-referencing symptom database...")
            input_vector = [0] * len(symptom_list)
            for s in nlp_res.split(","):
                s = s.strip()
                if s in symptom_list:
                    input_vector[symptom_list.index(s)] = 1

            prediction = model.predict([input_vector])[0]

            # 3. Gemini Advice
            st.write("📜 Generating professional advisory...")
            u = st.session_state.user_data
            med_prompt = f"""
Act as a physician and provide a SHORT medical summary.

Patient:
Age: {u['age']}
Gender: {u['gender']}
Symptoms: {nlp_res}
Duration: {u['duration']}
History: {u['history']}
Medications: {u['meds']}
Allergies: {u['allergies']}
Predicted Condition: {nlp_res}

STRICT RULES:

- Output MUST be a Markdown table with exactly 2 columns.
- Headers: Section | Advice
- Maximum 20 words per row.
- No paragraphs.
- No HTML tags.
- No bullet symbols.
- Keep total response under 150 words.
- Give general adult guidance only (no exact prescription).
- Avoid controlled drugs.

Sections (rows):

1. Condition
2. Medicines (name + how often + after/before food)
3. Home Care
4. Diet
5. Signs to immidiate doctor concern
6. Prevention

For Medicines row:

- Format strictly as:
  Medicines name – Frequency – Food instruction – Duration note
- Use short phrases only.
- No phrases like “as directed” or “if indicated”.
- No vague wording.
- Maximum 18 words total.
- Example format:
  Paracetamol(medicines name) – every 6–8 hours (if still not feel good and also based on maximum usage of that medicine per day) – after food (or before food)– max 3 days(indicate maximum days can tolerate with this medicines)

After table, add this exact disclaimer:

"This information is generated by an AI system for educational purposes only and is not a substitute for professional medical diagnosis or treatment. Please consult a licensed healthcare provider for personalized medical advice."

Return only the table and disclaimer.
"""
            advice = model_ai.generate_content(med_prompt).text

            status.update(
                label="✅ Analysis Complete!", state="complete", expanded=False
            )

            # Display Result in Chat
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": f"### 😷 Diagnostic Prediction: **{nlp_res}**\n\n{advice}",
                }
            )
            st.session_state.step = "complete"
            st.rerun()

        except Exception as e:
            st.error(f"Engine Error: {str(e)}")

# Disclaimer Footer
st.markdown("---")
st.caption(" ‼️ Note | For Professional Reference Only")