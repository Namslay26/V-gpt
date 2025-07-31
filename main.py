import streamlit as st
from google import genai
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader, PyMuPDFLoader, UnstructuredPowerPointLoader, UnstructuredWordDocumentLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
import os
import random
from datetime import datetime, time
from langchain.document_loaders import PyMuPDFLoader, UnstructuredPowerPointLoader, UnstructuredWordDocumentLoader

GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
# --- Setup ---
client = genai.Client(api_key=GEMINI_API_KEY)
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# --- File paths ---
DATA_FILE = os.path.join(os.path.dirname(__file__), "data", "basic_networking.md")
if not os.path.exists(DATA_FILE):
    with open(DATA_FILE, "w") as f:
        f.write("")

# --- Load documents ---
loader = TextLoader(DATA_FILE)
docs = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
documents = text_splitter.split_documents(docs)

# --- Create vector DB ---
vectordb = Chroma.from_documents(documents, embedding=embedding_model, persist_directory="db")
vectordb.persist()
retriever = vectordb.as_retriever()

# --- Ask Gemini ---
def ask_gemini(question, retries=3):
    formatted_q = f"You are V-GPT, a wise and experienced computer networking expert. Answer this:\n{question}"
    for attempt in range(retries):
        try:
            formatted_q = f"You are V-GPT, a wise and experienced computer networking expert. Answer this:\n{question}"
            response = client.models.generate_content(model="gemini-2.5-pro", contents = formatted_q)
            return response.text
        except (
            Exception
        ) as e:  # instead of raising the exception, you can let the model handle it
            function_response = {'error': str(e)}
    st.error("Gemini API is unavailable. Please try again later.")
    return None

# --- Extract Q&A Pairs (for quiz + Daily Drop) ---
def parse_qa_pairs(text):
    qas = []
    chunks = text.split("### Q:")
    for chunk in chunks[1:]:
        parts = chunk.strip().split("\n", 1)
        if len(parts) == 2:
            question = parts[0].strip()
            answer = parts[1].strip().split("---")[0].strip()
            qas.append((question, answer))
    return qas

qa_pairs = parse_qa_pairs(docs[0].page_content if docs else "")

# --- Streamlit UI ---
st.set_page_config(page_title="V-GPT", page_icon="🧠")
st.title("🧠 V-GPT: Your Personal Networking Guru")

tab1, tab2, tab3,tab4 = st.tabs(["💬 Ask V-GPT", "🧠 Teach V-GPT", "📝 Quiz Mode","📂 Upload Documents"])

# --- Daily Dad Drop ---
with st.container():
    if qa_pairs:
        seed = int(datetime.now().strftime("%Y%m%d"))
        random.seed(seed)
        q, a = random.choice(qa_pairs)
        st.info(f"📅 **Daily Dad Drop**\n\n💡 {a}")

# --- Tab 1: Ask V-GPT ---
with tab1:
    with st.form("ask_form"):
            user_question = st.text_input("Ask NetDad anything about computer networks...")
            submitted = st.form_submit_button("Ask")

            if submitted and user_question:
                relevant_docs = retriever.get_relevant_documents(user_question)
                context = "\n".join([doc.page_content for doc in relevant_docs])
                combined_input = f"Context:\n{context}\n\nQuestion: {user_question}"
                answer = ask_gemini(combined_input)
                st.markdown("**NetDad says:**")
                st.write(answer)

# --- Tab 2: Add New Q&A ---
with tab2:
    st.subheader("📥 Add a new question-answer")
    with st.form("add_qa_form"):
        new_q = st.text_area("New Question")
        new_a = st.text_area("Answer")
        new_tags = st.text_input("Tags (comma separated)", placeholder="e.g., wifi, troubleshooting")
        submitted = st.form_submit_button("Add")

        if submitted and new_q and new_a:
            tag_text = f"[Tags: {new_tags.strip()}]" if new_tags.strip() else ""
            with open(DATA_FILE, "a") as f:
                f.write(f"\n\n### Q: {new_q}\n{new_a}\n{tag_text}\n---\n")
            st.success("New Q&A added! Please restart the app to refresh the knowledge base.")

# --- Tab 3: Quiz Mode ---
with tab3:
    st.subheader("📝 Quiz Time!")
    if st.button("Start Quiz") and qa_pairs:
        quiz_qas = random.sample(qa_pairs, min(3, len(qa_pairs)))
        score = 0
        for idx, (q, a) in enumerate(quiz_qas, 1):
            st.markdown(f"**Q{idx}: {q}**")
            user_ans = st.text_input(f"Your answer for Q{idx}", key=f"quiz_{idx}")
            if user_ans:
                if user_ans.lower().strip() in a.lower():
                    st.success("✅ Correct (roughly matched)")
                    score += 1
                else:
                    st.error(f"❌ Not quite. Answer: {a}")
        st.markdown(f"### 🎯 Your score: {score}/{len(quiz_qas)}")
    elif not qa_pairs:
        st.warning("No Q&A data available yet. Add some in 'Teach V-GPT' tab!")



# --- Tab 4: Upload Documents ---
with tab4:
    st.subheader("📂 Upload Documents to Teach V-GPT")

    uploaded_files = st.file_uploader(
        "Upload PDF, PPTX, DOCX, TXT, or Markdown files",
        type=["pdf", "pptx", "docx", "txt", "md"],
        accept_multiple_files=True
    )

    if uploaded_files:
        for uploaded_file in uploaded_files:
            ext = uploaded_file.name.split(".")[-1].lower()
            temp_path = os.path.join("temp", uploaded_file.name)
            os.makedirs("temp", exist_ok=True)

            # Save to temp file
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            # Pick loader based on file type
            try:
                if ext == "pdf":
                    loader = PyMuPDFLoader(temp_path)
                elif ext == "pptx":
                    loader = UnstructuredPowerPointLoader(temp_path)
                elif ext == "docx":
                    loader = UnstructuredWordDocumentLoader(temp_path)
                elif ext in ["txt", "md"]:
                    loader = TextLoader(temp_path)
                else:
                    st.error(f"Unsupported file: {uploaded_file.name}")
                    continue

                docs = loader.load()
                splits = text_splitter.split_documents(docs)
                vectordb.add_documents(splits)
                vectordb.persist()

                st.success(f"✅ Added {uploaded_file.name} to V-GPT's memory.")

            except Exception as e:
                st.error(f"❌ Error with {uploaded_file.name}: {str(e)}")
