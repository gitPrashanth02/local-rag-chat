"""Chat page for the RAG API.

This is only the frontend: it sends questions and uploaded PDFs to the FastAPI
backend (api.py) and shows the answers. Start the backend first, then this page:

    python src/api.py                       # terminal 1: backend
    streamlit run src/streamlit_app.py      # terminal 2: chat page
"""
import os
import time

import requests
import streamlit as st

API_URL = os.environ.get("RAG_API_URL", "http://127.0.0.1:8000")
REQUEST_TIMEOUT = 300   # seconds; answers take ~40 s on CPU
UPLOAD_TIMEOUT = 1800   # seconds; parsing and embedding a PDF can take many minutes

SUGGESTIONS = {
    ":material/summarize: Summarize the document": "Give a short summary of this document.",
    ":material/list: What are the main points?": "What are the main points of this document?",
    ":material/flag: What does it conclude?": "What conclusions does this document reach?",
}

st.set_page_config(page_title="Document chat", page_icon=":material/forum:")


# ---------------------------------------------------------------------------
# Talking to the backend
# ---------------------------------------------------------------------------
@st.cache_data(ttl=10, show_spinner=False)
def get_health(api_url: str) -> dict | None:
    """Ask the API whether it is running. Cached for 10 s so every rerun doesn't call it."""
    try:
        response = requests.get(f"{api_url}/health", timeout=3)
        response.raise_for_status()
        return response.json()
    except requests.RequestException:
        return None


def call_api(path: str, timeout: int, **kwargs) -> dict:
    """POST to the API. Returns the JSON, or raises RuntimeError with a readable message."""
    try:
        response = requests.post(f"{API_URL}{path}", timeout=timeout, **kwargs)
    except requests.ConnectionError:
        raise RuntimeError(f"Can't reach the API at {API_URL}. Start it with `python src/api.py`.")
    except requests.Timeout:
        raise RuntimeError(f"The API took longer than {timeout} seconds to respond.")

    if not response.ok:
        try:
            detail = response.json().get("detail", response.text)
        except ValueError:
            detail = response.text
        raise RuntimeError(f"The API returned an error ({response.status_code}): {detail}")
    return response.json()


def ask_api(question: str, top_k: int) -> dict:
    """Send a question to POST /ask."""
    return call_api("/ask", REQUEST_TIMEOUT, json={"question": question, "top_k": top_k})


def upload_api(name: str, data: bytes) -> dict:
    """Send a PDF to POST /documents as a file upload (multipart form)."""
    return call_api("/documents", UPLOAD_TIMEOUT, files={"file": (name, data, "application/pdf")})


def cancel_upload():
    """Forget the attached PDF (Cancel button, or the popup closed with X / Esc)."""
    st.session_state.pop("pending_upload", None)
    st.session_state.pop("question_after_ingest", None)


@st.dialog("Ingest document?", icon=":material/upload_file:", on_dismiss=cancel_upload)
def ingest_dialog():
    """Popup shown after a PDF is attached in the prompt box. Buttons in here rerun only the popup."""
    upload = st.session_state.pending_upload
    st.markdown(f"**{upload['name']}** ({len(upload['data']) / 1024 / 1024:.1f} MB)")
    st.caption("This replaces the current document and clears the chat. Indexing can take several minutes.")
    if question := st.session_state.get("question_after_ingest"):
        st.caption(f"Your question will be asked afterwards: *{question}*")

    with st.container(horizontal=True):
        ingest = st.button("Ingest", icon=":material/check:", type="primary")
        cancel = st.button("Cancel")

    if cancel:
        cancel_upload()
        st.rerun()
    if ingest:
        with st.spinner(f"Indexing {upload['name']}…"):
            try:
                info = upload_api(upload["name"], upload["data"])
            except RuntimeError as error:
                st.error(str(error), icon=":material/error:")
                return  # keep the popup open so the user sees why
        st.session_state.messages = []   # old answers were about the old document
        st.session_state.upload_notice = f"Indexed **{info['document']}**: {info['chunks']} chunks in {info['seconds']:.0f} s"
        if question := st.session_state.get("question_after_ingest"):
            st.session_state.pending_question = question   # asked automatically on the next run
        cancel_upload()                  # done with the file
        get_health.clear()               # sidebar shows the new document name
        st.rerun()                       # full rerun closes the popup


def use_suggestion():
    """Runs when a suggestion is clicked: remember its question, then unselect the pill
    so it doesn't fire again later (for example after "Clear chat")."""
    choice = st.session_state.suggestion
    if choice:
        st.session_state.pending_question = SUGGESTIONS[choice]
    st.session_state.suggestion = None


def show_sources(sources: list[dict]):
    """List the chunks the answer was based on, inside a collapsed expander."""
    if not sources:
        return
    pages = sorted({s["page"] for s in sources})
    label = f"Sources: page{'s' if len(pages) > 1 else ''} {', '.join(map(str, pages))}"
    with st.expander(label, icon=":material/description:"):
        for s in sources:
            st.caption(f"Page {s['page']} · chunk {s['chunk_id']} · relevance {s['score']:.2f}")
            st.text(s["text"][:600] + ("…" if len(s["text"]) > 600 else ""))


# ---------------------------------------------------------------------------
# Session state: the chat history lives here between reruns
# ---------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []


# ---------------------------------------------------------------------------
# Sidebar: backend status and settings
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Settings")
    health = get_health(API_URL)
    uploads_allowed = bool(health and health.get("uploads", True))   # the API can turn uploads off
    if health:
        st.success(f"Document: **{health.get('document', '?')}**\n\n{health['chunks']} chunks · `{health['llm']}`",
                   title="API connected", icon=":material/check_circle:")
    else:
        st.error("Start the backend with `python src/api.py` and wait for *API is ready*.",
                 title="API not reachable", icon=":material/error:")
        if st.button("Check again", icon=":material/refresh:", width="stretch"):
            get_health.clear()  # forget the cached "not running" result
            st.rerun()

    top_k = st.slider("Chunks to use", min_value=1, max_value=10, value=5,
                      help="How many document chunks the model reads before answering.")
    if st.button("Clear chat", icon=":material/delete:", width="stretch"):
        st.session_state.messages = []
        st.rerun()

    if notice := st.session_state.pop("upload_notice", None):
        st.success(notice, icon=":material/task_alt:")


# ---------------------------------------------------------------------------
# Main chat area
# ---------------------------------------------------------------------------
st.title("Chat with your document")
st.caption("Answers come only from the PDF, generated by a local model."
           + (" Attach a PDF in the message box to switch documents." if uploads_allowed else ""))

# 1. Show the conversation so far
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            show_sources(msg.get("sources", []))

# 2. Get a new question: typed, a suggestion clicked, or one sent together with a PDF
submission =st.chat_input("Ask a question, or attach a PDF" if uploads_allowed else "Ask a question",
                           submit_mode="disable", disabled=health is None,
                           accept_file=uploads_allowed, file_type=["pdf"], max_upload_size=50)
# With accept_file, chat_input returns an object with .text and .files; without it, a plain string
prompt = submission.text if uploads_allowed and submission else submission

if uploads_allowed and submission and submission.files:
    # Save the file now: chat_input returns it only in this run, but the popup's buttons cause new runs
    file = submission.files[0]
    st.session_state.pending_upload = {"name": file.name, "data": file.getvalue()}
    st.session_state.question_after_ingest = prompt or None
    prompt = None   # the question waits until the new document is ingested

prompt = prompt or st.session_state.pop("pending_question", None)

if "pending_upload" in st.session_state:
    ingest_dialog()

if not st.session_state.messages and not prompt and "pending_upload" not in st.session_state:
    st.pills("Try asking:", list(SUGGESTIONS), key="suggestion", on_change=use_suggestion,
             label_visibility="collapsed", disabled=health is None)

# 3. Answer it
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        start = time.perf_counter()
        with st.status(":shimmer[Searching the document and writing an answer]", type="compact") as status:
            result, error_message = None, None
            try:
                result = ask_api(prompt, top_k)
                status.update(label=f"Answered in {time.perf_counter() - start:.0f} seconds", state="complete")
            except RuntimeError as error:
                error_message = str(error)
                status.update(label="Something went wrong", state="error")

        if result:
            st.markdown(result["answer"])
            show_sources(result["sources"])
            st.session_state.messages.append(
                {"role": "assistant", "content": result["answer"], "sources": result["sources"]}
            )
        else:
            st.error(error_message, icon=":material/error:")
            st.session_state.messages.pop()  # drop the unanswered question so it can be asked again
