# 🏥 Swasthya Mitra — UPKSK Citizen Health Assistant

> A citizen-centric RAG chatbot that helps people in Uttar Pradesh find
> the right hospital and specialist doctor based on their symptoms,
> using **UPKSK (Uttar Pradesh ke Swasthya Kendra)** data.

## How It Works

1. **Citizen describes symptoms** — e.g., "I fell down and my leg is hurting badly"
2. **AI asks follow-up questions** — e.g., "Can you move your leg? Is there swelling?"
3. **Identifies the specialist** — Maps symptoms → Orthopedist (fracture suspected)
4. **Recommends hospitals** — Shows nearby government hospitals where an orthopedist is available, with OPD timings and doctor names

## Tech Stack

| Layer            | Technology                         |
| ---------------- | ---------------------------------- |
| **Orchestration** | LangChain                         |
| **Vector DB**     | ChromaDB (local, persistent)      |
| **Embeddings**    | HuggingFace `all-MiniLM-L6-v2`   |
| **LLM**           | Google Gemini (free tier)         |
| **UI**            | Streamlit                         |

---

## 🚀 Setup & Run

### Step 1 — Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 2 — Set Your API Key

**Option A** — Enter it in the app sidebar.

**Option B** — Create a `.env` file:

```
GOOGLE_API_KEY=your_key_here
```

Get a free key at https://aistudio.google.com/apikey

### Step 3 — Run the App

```bash
streamlit run app.py
```

Opens at `http://localhost:8501`

---

## 📂 Project Structure

```
rag/
├── app.py                      # Main application (RAG + UI)
├── health_insurance_data.txt   # UPKSK hospital & doctor knowledge base
├── requirements.txt            # Python dependencies
├── .env.example                # API key template
├── chroma_db/                  # Auto-created vector store
└── README.md                   # This file
```

---

## 🔑 Features

- **Symptom Triage** — AI asks follow-up questions to identify the right specialist
- **Hospital Finder** — Recommends government hospitals with the needed specialist
- **13 Specialties** — Orthopedics, Cardiology, Neurology, Pediatrics, and more
- **5 Districts** — Lucknow, Kanpur, Varanasi, Agra, Prayagraj
- **Emergency Guidance** — Immediate 108 ambulance advice for critical symptoms
- **Government Schemes** — Info on Ayushman Bharat, Janani Suraksha, etc.
- **Chat Memory** — Supports follow-up questions
- **Model Selector** — Switch between Gemini models if one is rate-limited

---

## 🚨 Emergency Numbers

| Service | Number |
|---------|--------|
| Ambulance | **108** |
| UPKSK Helpline | **1800-180-5145** |
| National Health | **104** |
| Emergency | **112** |
