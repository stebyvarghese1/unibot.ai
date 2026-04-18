<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=24,30,20&height=240&section=header&text=%F0%9F%8E%93%20Unibot.AI&fontSize=80&fontColor=ffffff&fontAlignY=42&desc=The%20Intelligent%20Academic%20Layer%20%E2%80%94%20RAG-powered%2C%20hallucination-free.&descAlignY=63&descSize=16&descColor=94a3b8&animation=fadeIn" width="100%"/>

</div>

<br>

<div align="center">

   RAG  ·  Zero Hallucinations  ·  FAISS  ·  VLM  ·  Privacy-First

</div>

<br>
<div align="center">


[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-2.0+-000000?style=for-the-badge&logo=flask&logoColor=white)](https://flask.palletsprojects.com)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97-Hugging%20Face-FFD21E?style=for-the-badge)](https://huggingface.co)
[![Supabase](https://img.shields.io/badge/Supabase-3ECF8E?style=for-the-badge&logo=supabase&logoColor=white)](https://supabase.com)
[![pgvector](https://img.shields.io/badge/PGVector-Persistent-0064ff?style=for-the-badge&logo=postgresql&logoColor=white)](https://github.com/pgvector/pgvector)
[![License](https://img.shields.io/badge/License-MIT-fbbf24?style=for-the-badge)](LICENSE)

<br>

[![Stars](https://img.shields.io/github/stars/stebyvarghese1/unibot.ai?style=flat-square&color=fbbf24&label=⭐%20Stars)](https://github.com/stebyvarghese1/unibot.ai/stargazers)
&nbsp;
[![Forks](https://img.shields.io/github/forks/stebyvarghese1/unibot.ai?style=flat-square&color=38bdf8&label=Forks)](https://github.com/stebyvarghese1/unibot.ai/forks)
&nbsp;
[![Built by](https://img.shields.io/badge/by-Steby%20Varghese-a78bfa?style=flat-square)](https://github.com/stebyvarghese1)

</div>

<br>
<br>

---

<div align="center">

## &nbsp;&nbsp;&nbsp;` 01 `&nbsp;&nbsp; THE PRODUCT

</div>

<br>

<div align="center">

<table>
<tr>
<td align="center" width="50%">
<img src="https://github.com/user-attachments/assets/0e5e7c8e-5e8c-4174-afce-5df3b5223793" width="100%"/>
<br><br>
<b>💬 STUDENT INTERFACE</b>
<br>
<sub>Sleek glassmorphic dark-mode chat UI for instant academic answers.</sub>
</td>
<td align="center" width="50%">
<img src="https://github.com/user-attachments/assets/19cd26a4-9c0d-491e-b577-8eed0cc819b8" width="100%"/>
<br><br>
<b>👤 USER PROFILE</b>
<br>
<sub>Persistent session history — pick up right where you left off.</sub>
</td>
</tr>
</table>

<br>

<img src="https://github.com/user-attachments/assets/672b98f1-bb55-4406-9191-166141092e80" width="96%"/>
<br><br>
<b>⚙️ ADMIN COMMAND CENTER</b>
<br>
<sub>Full document management, real-time re-indexing, and system health monitoring.</sub>

</div>

<br>
<br>

---

<div align="center">

## &nbsp;&nbsp;&nbsp;` 02 `&nbsp;&nbsp; THE IDEA

</div>

<br>

> ### *"Not a chatbot. An Academic Intelligence Layer — where every answer is grounded in your institution's own documents."*

<br>

**Unibot.AI** uses **Retrieval-Augmented Generation (RAG)** to serve answers that are 100% bound to your university's uploaded content. No internet guessing. No hallucinations. Just precise, document-grounded intelligence.

<br>

<div align="center">

```
  ┌─────────────────────────────────────────────────────────────────┐
  │                                                                 │
  │   🚫  ZERO HALLUCINATIONS   Responses bound to your documents   │
  │   🔒  PERSISTENT BRAIN      Supabase pgvector Knowledge Base    │
  │   ⚡  PERFORMANCE TUNED      Gzip + Singleton + Eager Loading    │
  │                                                                 │
  └─────────────────────────────────────────────────────────────────┘
```

</div>

<br>
<br>

---

<div align="center">

## &nbsp;&nbsp;&nbsp;` 03 `&nbsp;&nbsp; FEATURES

</div>

<br>

**👨‍🎓 For Students**

<div align="center">

| &nbsp; | Feature | Description |
|--------|---------|-------------|
| 🎯 | **Adaptive Querying** | Instant answers on courses, exams, grades, and schedules |
| 🖼️ | **Multimodal Vision** | Upload a photo of notes or a timetable — BLIP VLM reads it |
| 🎨 | **Premium Dark UI** | Glassmorphic interface built for the modern student |
| 💾 | **Session History** | Persistent conversations — your context is never lost |

</div>

<br>

**🛡️ For Administrators**

<div align="center">

| &nbsp; | Feature | Description |
|--------|---------|-------------|
| 📄 | **Async Ingestion** | Background PDF/Docx processing — no UI blocking |
| 🌐 | **Smart Scraping** | Non-blocking web crawling with persistent auto-sync |
| 🔄 | **Live Re-indexing** | Rebuild your persistent vector library with one click |
| 📊 | **Gzip Compression** | Optimized transfers for large knowledge bases |

</div>

<br>
<br>

---

<div align="center">

## &nbsp;&nbsp;&nbsp;` 04 `&nbsp;&nbsp; HOW IT WORKS

</div>

<br>

<div align="center">

```
  ┌─────────────────────────────────────────────────────────────────┐
  │                        DATA FLOW                                │
  ├─────────────────────────────────────────────────────────────────┤
  │                                                                 │
  │  STEP 1  INGESTION                                              │
  │          Async threading → sentence-transformers → vectors      │
  │                                                                 │
  │  STEP 2  INDEXING                                               │
  │          Persistent storage in Supabase pgvector (PostgreSQL)   │
  │                                                                 │
  │  STEP 3  RETRIEVAL                                              │
  │          Eager relationship loading → sub-second fetching       │
  │                                                                 │
  │  STEP 4  SYNTHESIS                                              │
  │          Flask + Gzip Compression → Mistral LLM → UI            │
  │                                                                 │
  │  STEP 5  VISUALS                                                │
  │          Images → BLIP VLM → description → chat logic           │
  │                                                                 │
  └─────────────────────────────────────────────────────────────────┘
```

</div>

<br>

<div align="center">

```
  ┌──────────────────┐        ┌───────────────────┐       ┌───────────────────┐
  │  User Interface  │◀──────▶│   Flask Backend    │──────▶│  Hugging Face API │
  │  Desktop/Mobile  │        │   SQLAlchemy ORM   │       │  LLM Inference    │
  └──────────────────┘        └────────┬──────────┘       └───────────────────┘
                                       │
                         ┌─────────────┴──────────┐
                         │                        │
                         ▼                        ▼
               ┌─────────────────┐     ┌──────────────────┐
               │  Supabase DB    │◀───▶│  Supabase Vector   │
               │  (PostgreSQL)   │     │  (PGVector)        │
               └─────────────────┘     └──────────────────┘
```

</div>

<br>
<br>

---

<div align="center">

## &nbsp;&nbsp;&nbsp;` 05 `&nbsp;&nbsp; TECH STACK

</div>

<br>

<div align="center">

| Layer | Technology |
|-------|-----------|
| 🎨 **Frontend** | HTML5 · CSS3 · Glassmorphic UI · Dark Mode |
| ⚙️ **Backend** | Python 3.10+ · Flask · SQLAlchemy |
| 🧠 **AI / NLP** | all-MiniLM-L6-v2 · Google Gemma-2b-it LLM |
| 🗄️ **Database** | Supabase (PostgreSQL) · SQLAlchemy (Eager Loading) |
| 🔍 **Vector Search** | Supabase pgvector (Persistent) |
| ⚡ **Performance** | Flask-Compress (Gzip) · Singleton Service Pattern |
| 📄 **Doc Parsing** | PDF · DOCX · PPTX — semantic chunking |

</div>

<br>
<br>

---

<div align="center">

## &nbsp;&nbsp;&nbsp;` 06 `&nbsp;&nbsp; GETTING STARTED

</div>

<br>

**Prerequisites** — Python `3.10+` · Supabase account · Hugging Face API token

<br>

**Step 1 — Clone**
```bash
git clone https://github.com/stebyvarghese1/unibot.ai.git
cd unibot.ai
```

**Step 2 — Environment**
```bash
python -m venv venv
source venv/bin/activate      # Linux / macOS
venv\Scripts\activate         # Windows

pip install -r requirements.txt
```

**Step 3 — Configure** → create `.env` in project root
```env
SUPABASE_URL=your_supabase_url
SUPABASE_KEY=your_supabase_anon_key
DATABASE_URL=your_postgresql_connection_string
HUGGINGFACE_API_TOKEN=your_token_here
HF_LLM_MODEL=google/gemma-2b-it
HF_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
```

**Step 4 — Launch**
```bash
python run.py
```

<div align="center">

> 🟢 &nbsp;**Unibot is live — your academic intelligence layer is ready.**

</div>

<br>
<br>

---

<div align="center">

## &nbsp;&nbsp;&nbsp;` 07 `&nbsp;&nbsp; ROADMAP

</div>

<br>

<div align="center">

```
  🔲  Multilingual support — regional university languages
  🔲  Voice interaction — ask Unibot by speaking naturally
  🔲  Department isolation — sub-bots per faculty or department
  🔲  Analytics dashboard — deep insights on student query patterns
```

</div>

<br>
<br>

---

<br>

<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=24,30,20&height=130&section=footer&animation=fadeIn" width="100%"/>

### Built for the future of education by [Steby Varghese](https://github.com/stebyvarghese1)

[![GitHub](https://img.shields.io/badge/GitHub-stebyvarghese1-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/stebyvarghese1)
&nbsp;
[![Portfolio](https://img.shields.io/badge/Portfolio-Visit-a78bfa?style=flat-square&logo=firefox&logoColor=white)](https://portfolio-v3ia.onrender.com/)
&nbsp;
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0A66C2?style=flat-square&logo=linkedin&logoColor=white)](https://linkedin.com/in/steby-varghese)

<br>

**⭐ Star this repo if Unibot impressed you — it keeps the mission going!**

Licensed under [MIT](LICENSE)

</div>
