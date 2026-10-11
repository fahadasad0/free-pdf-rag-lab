import os
import tempfile
from pathlib import Path

import streamlit as st
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import numpy as np
from groq import Groq

st.set_page_config(page_title="Free PDF RAG Lab", page_icon="📚", layout="wide")
st.title("📚 PDF Question Answering — Simple RAG Lab")
st.caption("Upload a PDF, build a searchable index, then ask questions grounded in its content.")

@st.cache_resource
def load_embedder():
    # Free, open-source embedding model. First run downloads model weights.
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def extract_pdf(uploaded_file):
    reader = PdfReader(uploaded_file)
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            pages.append((i, text))
    return pages

def chunk_pages(pages, chunk_size=900, overlap=150):
    chunks = []
    for page_num, text in pages:
        text = " ".join(text.split())
        start = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            piece = text[start:end].strip()
            if piece:
                chunks.append({"text": piece, "page": page_num})
            if end == len(text):
                break
            start = end - overlap
    return chunks

def make_index(chunks, embedder):
    texts = [c["text"] for c in chunks]
    vectors = embedder.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return np.asarray(vectors, dtype=np.float32)

def retrieve(question, chunks, vectors, embedder, top_k=4):
    qvec = embedder.encode([question], normalize_embeddings=True)[0]
    scores = vectors @ qvec
    indices = np.argsort(scores)[::-1][:min(top_k, len(chunks))]
    return [(chunks[int(i)], float(scores[int(i)])) for i in indices]

with st.sidebar:
    st.header("1. Upload & index")
    uploaded = st.file_uploader("Choose a PDF", type=["pdf"])
    chunk_size = st.slider("Chunk size (characters)", 400, 1600, 900, 100)
    overlap = st.slider("Chunk overlap", 0, 300, 150, 25)
    top_k = st.slider("Retrieved passages", 1, 8, 4)
    st.markdown("---")
    st.header("2. Free answer model")
    st.write("This deployable version uses Groq's API free tier for text generation. It is free within provider limits, not unlimited.")
    api_key = st.text_input("Groq API key", type="password", help="Create a key in the Groq console. Do not commit it to GitHub.")
    model_name = st.selectbox(
        "Model",
        ["llama-3.1-8b-instant", "openai/gpt-oss-20b"],
        index=0,
        help="Availability can change; choose a model currently enabled for your account."
    )
    build = st.button("Build / rebuild PDF index", type="primary", use_container_width=True)

if "chunks" not in st.session_state:
    st.session_state.chunks = None
    st.session_state.vectors = None
    st.session_state.pdf_name = None

if build:
    if uploaded is None:
        st.error("Upload a PDF first.")
    else:
        try:
            with st.spinner("Reading PDF and creating embeddings..."):
                pages = extract_pdf(uploaded)
                chunks = chunk_pages(pages, chunk_size, overlap)
                if not chunks:
                    st.error("No selectable text was found. This PDF may be scanned; OCR is not included in this simple lab.")
                else:
                    embedder = load_embedder()
                    vectors = make_index(chunks, embedder)
                    st.session_state.chunks = chunks
                    st.session_state.vectors = vectors
                    st.session_state.pdf_name = uploaded.name
                    st.success(f"Indexed {len(chunks)} text chunks from {len(pages)} pages.")
        except Exception as e:
            st.error(f"Could not process PDF: {e}")

if st.session_state.chunks:
    st.success(f"Current document: **{st.session_state.pdf_name}** · {len(st.session_state.chunks)} chunks indexed")
    question = st.text_input("Ask a question about your PDF", placeholder="What are the main conclusions?")
    ask = st.button("Ask question")
    if ask:
        if not question.strip():
            st.warning("Type a question first.")
        elif not api_key.strip():
            st.warning("Enter your Groq API key in the sidebar. The key is only used for this session; don't hard-code it.")
        else:
            with st.spinner("Finding relevant passages and generating an answer..."):
                embedder = load_embedder()
                matches = retrieve(question, st.session_state.chunks, st.session_state.vectors, embedder, top_k)
                context = "\n\n".join(
                    f"[Page {item['page']}]\n{item['text']}" for item, score in matches
                )
                prompt = f"""You are a careful PDF assistant. Answer using ONLY the context below.
If the answer is not supported by the context, say: "I couldn't find that in the uploaded PDF."
Cite supporting page numbers like [Page 3]. Do not invent quotations or facts.

CONTEXT:
{context}

QUESTION: {question}
ANSWER:"""
                try:
                    client = Groq(api_key=api_key.strip())
                    completion = client.chat.completions.create(
                        model=model_name,
                        messages=[
                            {"role": "system", "content": "Answer faithfully from supplied context and cite page numbers."},
                            {"role": "user", "content": prompt},
                        ],
                        temperature=0.1,
                        max_tokens=900,
                    )
                    st.subheader("Answer")
                    st.write(completion.choices[0].message.content)
                    with st.expander("Show retrieved PDF passages"):
                        for n, (item, score) in enumerate(matches, start=1):
                            st.markdown(f"**Passage {n} · Page {item['page']} · similarity {score:.3f}**")
                            st.write(item["text"])
                except Exception as e:
                    st.error(f"Generation failed: {e}. Check your API key, model availability, and free-tier quota.")
else:
    st.info("Upload a PDF in the sidebar and click **Build / rebuild PDF index** to begin.")

with st.expander("How this RAG demo works"):
    st.markdown("""
    1. **Extract** selectable text from the uploaded PDF.
    2. **Chunk** text into overlapping passages.
    3. **Embed** passages with `all-MiniLM-L6-v2` and keep vectors in memory.
    4. **Retrieve** the most similar passages for a question using cosine similarity.
    5. **Generate** a grounded answer using a free-tier hosted language model.

    This is retrieval-augmented generation (RAG), not model fine-tuning. The PDF is indexed for this app session.
    """)
