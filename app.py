import os
import zipfile
import tempfile

# Disable tokenizers parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from dotenv import load_dotenv
from PyPDF2 import PdfReader
import cassio
import gradio as gr

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.cassandra import Cassandra
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate


# ---------------- ENV / CONFIG ---------------- #
load_dotenv()

ASTRA_DB_APPLICATION_TOKEN = os.getenv("ASTRA_DB_APPLICATION_TOKEN")
ASTRA_DB_ID = os.getenv("ASTRA_DB_ID")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not ASTRA_DB_APPLICATION_TOKEN or not ASTRA_DB_ID:
    raise ValueError("Astra DB credentials are missing in environment variables.")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY is missing in environment variables.")

cassio.init(
    token=ASTRA_DB_APPLICATION_TOKEN,
    database_id=ASTRA_DB_ID,
)

# ---------------- MODELS ---------------- #
llm = ChatGroq(api_key=GROQ_API_KEY, model="llama-3.1-8b-instant")
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Astra vector store
astra_vector_store = Cassandra(
    embedding=embedding,
    table_name="folder_pdf_qna"
)

# ---------------- PROMPT ---------------- #
prompt = ChatPromptTemplate.from_template("""
You are an assistant that answers questions based on the retrieved document context.

Rules:
- Use the provided context as the primary source of information.
- You may summarize or rephrase, but do not invent new facts.
- If the answer is not present in the context, respond with:
  "The answer is not available in the provided documents."

Context:
{context}

Question:
{input}

Answer (clear and concise):
""")


# ---------------- ZIP PROCESSING / INDEX BUILD ---------------- #
def process_zip(zip_file):
    """Process an uploaded ZIP file containing PDFs and build a RAG chain."""
    if zip_file is None:
        return "Please upload a ZIP file containing PDFs.", None

    # Extract ZIP to temporary directory
    temp_dir = tempfile.mkdtemp()
    try:
        with zipfile.ZipFile(zip_file.name, "r") as z:
            z.extractall(temp_dir)
    except Exception as e:
        return f"Failed to extract ZIP file: {e}", None

    # Collect all PDFs (recursive walk)
    pdf_files = []
    for root, _, files in os.walk(temp_dir):
        for f in files:
            if f.lower().endswith(".pdf"):
                pdf_files.append(os.path.join(root, f))

    if not pdf_files:
        return "The uploaded ZIP does not contain any PDF files.", None

    all_text = ""

    # Read each PDF
    for pdf_path in pdf_files:
        try:
            reader = PdfReader(pdf_path)
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    all_text += text + "\n"
        except Exception as e:
            # Skip problematic files but continue processing others
            print(f"Error reading {pdf_path}: {e}")

    if not all_text.strip():
        return "Unable to extract text from the PDFs.", None

    # Chunk the text using a recursive splitter for better boundaries
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " "],
    )
    chunks = splitter.split_text(all_text)

    if not chunks:
        return "No text chunks could be generated from the PDFs.", None

    # Insert chunks into Astra vector store
    astra_vector_store.add_texts(chunks)

    # Build RAG chain
    retriever = astra_vector_store.as_retriever(
        search_kwargs={"k": 6}  # retrieve top-6 relevant chunks
    )
    combine = create_stuff_documents_chain(llm=llm, prompt=prompt)
    qa_chain = create_retrieval_chain(retriever, combine)

    status_msg = (
        f"Successfully processed {len(pdf_files)} PDF file(s). "
        f"Indexed {len(chunks)} text chunks."
    )
    return status_msg, qa_chain


def answer(query, chain):
    """Answer a question using the previously built RAG chain."""
    if chain is None:
        return "Please upload and process a ZIP file first."

    query = (query or "").strip()
    if not query:
        return "Please enter a question."

    try:
        result = chain.invoke({"input": query})
        return result.get("answer", "No answer was returned by the model.")
    except Exception as e:
        return f"Error while generating answer: {e}"


# ---------------- GRADIO UI ---------------- #
with gr.Blocks() as app:
    gr.Markdown("## Upload a ZIP of PDFs and Ask Questions")
    gr.Markdown(
        "Upload a ZIP file containing one or more PDF documents. "
        "The system will index all PDFs and then answer questions based on their content."
    )

    zip_file = gr.File(label="Upload ZIP file", file_types=[".zip"])
    load_btn = gr.Button("Process ZIP")

    status = gr.Markdown()
    chain_state = gr.State(None)

    load_btn.click(
        fn=process_zip,
        inputs=zip_file,
        outputs=[status, chain_state],
    )

    q = gr.Textbox(label="Question", placeholder="Enter your question here")
    ask = gr.Button("Get Answer")
    ans = gr.Markdown()

    ask.click(
        fn=answer,
        inputs=[q, chain_state],
        outputs=ans,
    )

app.launch()
