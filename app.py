# app.py
import streamlit as st
import requests, json, os, io
from bs4 import BeautifulSoup

# PDF support
try:
    import pdfplumber
    PDF_LIB = "pdfplumber"
except:
    from PyPDF2 import PdfReader
    PDF_LIB = "pypdf2"

st.set_page_config(page_title="Text Summarizer", layout="wide")
st.title("📝 Text / Link / PDF Summarizer")

# ---------------- Sidebar ----------------
with st.sidebar:
    st.header("Settings")
    min_length = st.slider("Minimum summary length (tokens)", 10, 200, 30)
    max_length = st.slider("Maximum summary length (tokens)", 30, 500, 120)
    model_choice = st.selectbox(
        "Model",
        ["facebook/bart-large-cnn"]
    )

# ---------------- Inputs ----------------
col1, col2 = st.columns([2, 1])
with col1:
    input_mode = st.radio("Input type", ["URL / Link", "Paste text", "Upload PDF"])
    if input_mode == "URL / Link":
        url = st.text_input("Enter URL of article / post")
    elif input_mode == "Paste text":
        pasted_text = st.text_area("Paste your text or post here", height=200)
    else:
        uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])

with col2:
    truncate = st.number_input("Truncate input to first N chars (0 = none)", min_value=0, value=0, step=500)

# ---------------- Helper Functions ----------------
def extract_text_from_url(u: str) -> str:
    try:
        r = requests.get(u, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
        soup = BeautifulSoup(r.text, "html.parser")
        article = soup.find("article")
        if article:
            return " ".join(p.get_text(strip=True) for p in article.find_all("p"))
        else:
            ps = soup.find_all("p")
            return " ".join(p.get_text(strip=True) for p in ps)
    except Exception as e:
        return f"ERROR fetching URL: {e}"

def extract_text_from_pdf(file_bytes) -> str:
    if PDF_LIB == "pdfplumber":
        text = []
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            for page in pdf.pages:
                text.append(page.extract_text() or "")
        return "\n".join(text)
    else:
        reader = PdfReader(io.BytesIO(file_bytes))
        text = []
        for page in reader.pages:
            text.append(page.extract_text() or "")
        return "\n".join(text)

def analyze_with_hf(text, model_id="facebook/bart-large-cnn"):
    HF_TOKEN = os.getenv("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")
    if not HF_TOKEN:
        return "Error: Hugging Face token not found in environment variable HF_TOKEN"
    headers = {
        "Authorization": f"Bearer {HF_TOKEN}",
        "Content-Type": "application/json"
    }
    payload = {"inputs": text}
    try:
        response = requests.post(
            f"https://api-inference.huggingface.co/models/{model_id}",
            headers=headers,
            json=payload
        )
        if response.status_code == 200:
            output = response.json()
            if isinstance(output, list) and "generated_text" in output[0]:
                return output[0]["generated_text"]
            elif isinstance(output, list) and "summary_text" in output[0]:
                return output[0]["summary_text"]
            return str(output)
        else:
            return f"Error {response.status_code}: {response.text}"
    except Exception as e:
        return f"Error connecting to Hugging Face: {e}"

# ---------------- Compose input text ----------------
input_text = ""
if input_mode == "URL / Link" and url:
    st.info("Fetching and extracting text from URL...")
    input_text = extract_text_from_url(url)
elif input_mode == "Paste text" and pasted_text:
    input_text = pasted_text
elif input_mode == "Upload PDF" and uploaded_file is not None:
    bytes_data = uploaded_file.read()
    st.info("Extracting text from PDF...")
    input_text = extract_text_from_pdf(bytes_data)

if truncate > 0:
    input_text = input_text[:truncate]

if input_text:
    st.write("### Input Preview")
    st.write(input_text[:5000] + ("..." if len(input_text) > 5000 else ""))

# ---------------- Summarize ----------------
if st.button("Summarize"):
    if not input_text.strip():
        st.error("No input text found. Please provide a URL, text, or PDF.")
    else:
        st.info("Summarizing via Hugging Face Inference API...")
        max_chunk_chars = 4000
        chunks = [input_text[i:i+max_chunk_chars] for i in range(0, len(input_text), max_chunk_chars)]
        summaries = []
        for ch in chunks:
            summary = analyze_with_hf(ch, model_id=model_choice)
            summaries.append(summary)
        final_summary = "\n\n".join(summaries)
        st.subheader("Summary")
        st.write(final_summary)
        st.download_button("Download Summary (TXT)", final_summary, file_name="summary.txt")
