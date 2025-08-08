# 🧠 Candidate Recommendation Engine


---

- **Deployment**: Hugging Face spaces - https://huggingface.co/spaces/VaibhaviSavani1910/resume-matching

---

## 📌 Overview
This project is a **Candidate Recommendation Engine** that ranks candidate resumes based on their relevance to a given job description.  
It uses **vector embeddings** and **semantic search** to find the most relevant resumes, and then applies an **LLM evaluation** step to generate structured feedback and summaries.

The app is built with **Gradio** for an interactive UI and **MongoDB Atlas Vector Search** for scalable similarity queries.

---

## 🚀 Features
✅ Upload multiple candidate resumes in **PDF** format  
✅ Paste a job description into a text box  
✅ **Chunking for better embeddings** – resumes are split into 1000-character overlapping segments before embedding  
✅ **Generate embeddings** using OpenAI’s `text-embedding-ada-002` model  
✅ Store and search embeddings in **MongoDB Atlas Vector Search**  
✅ Retrieve top matches based on **cosine similarity**  
✅ **BONUS**: AI-generated JSON evaluation of each candidate, including:
- Top matching skills/criteria
- Score for each criterion
- Justification for each score
- Overall summary of fit

✅ Clean, responsive UI with a loader animation  
✅ Option to **delete old resumes** from the database  

---

## 🛠️ Tech Stack
- **Frontend/UI**: [Gradio](https://www.gradio.app/)  
- **LLM & Embeddings**:  
  - OpenAI `text-embedding-ada-002` for vector generation  
  - gemini-2.0-flash for candidate evaluation  
- **Vector Database**: [MongoDB Atlas Vector Search](https://www.mongodb.com/atlas/vector-search)  
- **PDF Parsing & Chunking**: `PyPDFLoader` + `RecursiveCharacterTextSplitter` (LangChain)  
- **Backend Language**: Python 3.13  
- **Deployment**: Hugging Face spaces - https://huggingface.co/spaces/VaibhaviSavani1910/resume-matching

---

## ⚙️ How It Works

### 1. **Resume Upload & Processing**
- User uploads PDF resumes.
- Each resume is parsed into text using **PyPDFLoader**.
- Resumes are **split into 1000-character chunks** for better embedding quality.

### 2. **Embedding & Indexing**
- Each chunk is converted into a vector using **OpenAI Embeddings**.
- Vectors are stored in **MongoDB Atlas Vector Search**.
- An **index marker document** is added and polled until MongoDB indexing completes.

### 3. **Matching Pipeline**
- Job description is embedded and compared with all stored chunks via **cosine similarity**.
- Top relevant chunks are grouped by **Resume ID**.
- Each resume’s **best similarity score** is used to rank candidates.

### 4. **AI-Powered Evaluation (Bonus Feature)**
- For each top resume, full text is retrieved.
- **gemini-2.0-flash** is prompted to:
  - Identify top 2–3 relevant skills
  - Assign scores out of 10
  - Give justifications
  - Summarize overall fit
- Output is strictly structured in **valid JSON**.

### 5. **Display Results**
- Shows:
  - Resume ID
  - Candidate name (placeholder in this version)
  - Similarity score
  - Summary of fit
  - Detailed criteria table (skill, score, justification)


## Watch demo video

(demo.mov)

