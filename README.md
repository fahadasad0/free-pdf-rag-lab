# Free PDF RAG Lab (Streamlit)

A beginner-friendly Retrieval-Augmented Generation (RAG) demo. Users upload a PDF, the app extracts text, chunks it, creates embeddings locally, retrieves relevant passages, and asks a language model to answer with page citations.

## Important note about "free and unlimited"

There is no dependable hosted GPT/LLM API that is both free and unlimited. This project uses:
- **Embeddings:** `sentence-transformers/all-MiniLM-L6-v2` (open-source; downloaded and run by the app).
- **Text generation:** Groq API free tier, subject to model availability, rate limits, and usage limits. You can use another compatible provider by changing the generation section in `app.py`.

The app does not require a paid service, but free-tier quotas may change. The first load downloads the embedding model and can take a few minutes. Streamlit Community Cloud has resource limits, so very large PDFs may not fit.

## Files

- `app.py` — Streamlit frontend and RAG pipeline.
- `requirements.txt` — Python dependencies.

## Run locally

1. Install Python 3.10 or 3.11.
2. Open a terminal in this folder.
3. Create and activate a virtual environment (optional but recommended):
   - Windows: `py -m venv .venv` then `.venv\Scripts\activate`
   - macOS/Linux: `python3 -m venv .venv` then `source .venv/bin/activate`
4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
5. Get a Groq API key from the Groq Console. Use its free tier and check the current quotas/model availability. Never commit the key to GitHub.
6. Run:
   ```bash
   streamlit run app.py
   ```
7. In the browser, upload a text-based PDF, click **Build / rebuild PDF index**, enter your Groq key in the sidebar, then ask a question.

## Publish to GitHub

1. Create a new repository on GitHub (for example, `free-pdf-rag-lab`).
2. Upload `app.py`, `requirements.txt`, and this `README.md` to the repository root.
3. Make sure no API keys or `.env` files are committed. This app asks for the key in the sidebar instead.

## Deploy on Streamlit Community Cloud (free tier)

1. Push the repository to GitHub.
2. Sign in at https://share.streamlit.io/ using GitHub.
3. Select **Create app** (or **Deploy an app**).
4. Choose your repository, branch (usually `main`), and main file path `app.py`.
5. Open **Advanced settings** only if you need to select a Python version supported by Streamlit Cloud; choose Python 3.11 if available.
6. Click **Deploy** and wait for dependency installation. The first start may be slow because the embedding model must download.
7. Open the deployed app URL. Upload a PDF, click **Build / rebuild PDF index**, and enter your Groq API key in the sidebar.
8. Keep the repository public if you want a straightforward free deployment. Do not put the API key in source code, README, GitHub commits, or screenshots.

Official guides:
- Streamlit Community Cloud: https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app
- Groq API docs: https://console.groq.com/docs/quickstart
- Sentence Transformers: https://www.sbert.net/

## What "train the RAG" means here

This lab does **not** train or fine-tune an LLM. It builds a document index (embeddings) from your PDF and retrieves relevant passages at question time. This is the standard beginner RAG workflow.

## Limitations

- Text-based PDFs only; scanned image PDFs need OCR, which is not included.
- The index exists in app memory and is lost when the app restarts or the session ends.
- The free generation API has quotas and rate limits; it is not unlimited.
- Avoid uploading confidential documents to a hosted app unless you understand the hosting and model provider's data policies.
