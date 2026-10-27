# DSU Student Assistant

A chatbot that answers questions about Dayananda Sagar University (admissions, courses,
rules, hostel, academic calendar ...) using the official website (dsu.edu.in) and DSU PDF
documents. It runs fully on your own computer: Streamlit + FAISS + a local Ollama model.

## Features
- Answers only from the DSU website and documents, and shows the sources it used
- ChatGPT-style chat: one prompt at a time, Stop button, Edit a prompt, loading indicator
- Refresh button to start a new chat; "Download Chat" (in the menu) saves the chat as a PDF

## Setup
1. Install Python 3.10+ and [Ollama](https://ollama.com), then pull the model:
   `ollama pull qwen2.5:7b`
2. Create a virtual environment and install the packages:
   ```
   python -m venv .venv
   .venv\Scripts\activate
   pip install -r requirements.txt
   ```
3. Put your PDFs in `data/pdfs/` (the academic calendar file must start with `AC_`).
4. Crawl the website (optional but recommended):
   `python crawl_website.py`
5. Build the search index:
   `python ingest.py`
6. Start the app:
   `streamlit run app.py`

Re-run steps 4 and 5 whenever the website or the PDFs change.

## Project files
| File | Purpose |
|------|---------|
| `app.py` | Streamlit chat app |
| `ingest.py` | Builds the FAISS index from PDFs and website text |
| `crawl_website.py` | Downloads text from dsu.edu.in |
| `requirements.txt` | Python packages |
