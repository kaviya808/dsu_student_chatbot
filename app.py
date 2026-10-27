import datetime
import html
import io
import os
import queue
import re
import threading
import time

import streamlit as st
import streamlit.components.v1 as components
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate

try:  # PDF export of the chat (pip install reportlab)
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer
    HAVE_PDF = True
except ImportError:
    HAVE_PDF = False

# Top-right "Refresh" button, placed next to Deploy. The PDF download itself is wired into
# Streamlit's three-dot menu (it replaces "Record screen" with "Download Chat", see below).
SHOW_DOWNLOAD_BUTTON = False  # True = also show a visible "Download PDF" button next to Refresh
TOPBAR_RIGHT = "9rem"         # raise this if the Refresh button overlaps Deploy

st.set_page_config(page_title="DSU Student Assistant", page_icon="🎓", layout="centered")

st.title("🎓 DSU Student Assistant")
st.caption("Ask questions about Admissions, Courses, Rules, Hostel, Academic Calendar etc.")


@st.cache_resource
def load_resources():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    llm = OllamaLLM(model="qwen2.5:7b", temperature=0)
    calendar_docs = [
        d for d in vectorstore.docstore._dict.values()
        if d.page_content.startswith("[Document: AC_")
    ]
    return vectorstore, llm, calendar_docs


vectorstore, llm, calendar_docs = load_resources()

prompt_template = """
You are a helpful and polite assistant for Dayananda Sagar University (DSU).
Answer the question using only the information provided in the context below.
The context comes from the official DSU website (dsu.edu.in) and DSU documents.
If the answer is not present in the context, say: "I couldn't find this on the DSU website or in the university documents. Please check www.dsu.edu.in or contact the university."
Copy dates, numbers, fees, phone numbers, e-mail addresses and names exactly as written in the context.
Do not invent links. Only mention a link if it appears in the context.
Academic calendar lines look like: Term | Event | Year: dates. "Odd Semester" and "Even Semester" are different terms.
If the question does not say odd or even semester, give the answer for both.

Context:
{context}

Question: {question}

Answer:
"""

PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
retriever = vectorstore.as_retriever(search_kwargs={"k": 6})
chain = PROMPT | llm


# ---------- Helpers (retrieval) ----------
def norm(text):
    return re.sub(r"[^a-z0-9]", "", text.lower())


def program_of(doc):
    m = re.search(r"PROGRAM\s*:\s*([^<\n|]+)", doc.page_content)
    return m.group(1).strip() if m else ""


def calendar_pages_for(query):
    q = norm(query)
    best_len, hits = 0, []
    for d in calendar_docs:
        for part in re.split(r"&|,|/|\band\b", program_of(d), flags=re.I):
            p = norm(part)
            if len(p) >= 3 and p in q:
                if len(p) > best_len:
                    best_len, hits = len(p), [d]
                elif len(p) == best_len and d not in hits:
                    hits.append(d)
    return hits


def is_calendar(doc):
    return doc.page_content.startswith("[Document: AC_")


ORDINALS = {
    "1st": "I Year", "first": "I Year", "2nd": "II Year", "second": "II Year",
    "3rd": "III Year", "third": "III Year", "4th": "IV Year", "fourth": "IV Year",
    "5th": "V Year", "fifth": "V Year",
}
NUMBERS = {"1": "I Year", "2": "II Year", "3": "III Year", "4": "IV Year", "5": "V Year"}
ROMANS = {"i": "I Year", "ii": "II Year", "iii": "III Year", "iv": "IV Year", "v": "V Year"}


def year_in_query(query):
    q = query.lower()
    if "year" not in q:
        return None
    for word, label in ORDINALS.items():
        if re.search(rf"\b{word}\b", q):
            return label
    m = re.search(r"\byear\s*([1-5])\b|\b([1-5])\s*year\b", q)
    if m:
        return NUMBERS[m.group(1) or m.group(2)]
    m = re.search(r"\b(iv|iii|ii|i|v)\s*year\b", q)
    if m:
        return ROMANS[m.group(1)]
    return None


def semester_in_query(query):
    q = query.lower()
    odd, even = bool(re.search(r"\bodd\b", q)), bool(re.search(r"\beven\b", q))
    if odd and not even:
        return "Odd Semester"
    if even and not odd:
        return "Even Semester"
    return None


EVENT_LINE = re.compile(r" \| (?:I|II|III|IV|V) Year: ")


def narrow_calendar(doc, year, sem):
    keep = []
    for line in doc.page_content.split("\n"):
        if EVENT_LINE.search(line):
            if year and f" | {year}: " not in line:
                continue
            if sem and not line.startswith(sem):
                continue
        keep.append(line)
    return Document(page_content="\n".join(keep), metadata=doc.metadata)


def keyword_search_docs(query, k=3):
    q_lower = query.lower()
    matches = []
    terms = [t for t in q_lower.split() if len(t) > 2]
    if not terms:
        return []

    for doc in vectorstore.docstore._dict.values():
        content_lower = doc.page_content.lower()
        score = sum(1 for term in terms if term in content_lower)
        if q_lower in content_lower:
            score += 15
        if score > 0:
            matches.append((score, doc))

    matches.sort(key=lambda x: x[0], reverse=True)
    return [doc for score, doc in matches[:k]]


def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)


def retrieve_docs(query):
    """Vector search + keyword search + program-specific calendar page."""
    docs = retriever.invoke(query)
    keyword_hits = keyword_search_docs(query, k=3)
    seen_contents = set()
    merged_docs = []
    for d in keyword_hits + docs:
        if d.page_content not in seen_contents:
            seen_contents.add(d.page_content)
            merged_docs.append(d)
    docs = merged_docs[:6]

    cal_hits = calendar_pages_for(query)
    if cal_hits:
        year, sem = year_in_query(query), semester_in_query(query)
        cal_hits = [narrow_calendar(d, year, sem) for d in cal_hits]
        others = [d for d in docs if not is_calendar(d)][:3]
        docs = cal_hits + others
    return docs


def source_urls(docs):
    urls = []
    for d in docs:
        u = d.metadata.get("source", "")
        if u.startswith("http") and u not in urls:
            urls.append(u)
    return urls[:5]


# ---------- Helpers (streaming in a background thread) ----------
def start_stream(inputs):
    """Run the LLM in a background thread and hand the text back through a queue.
    The main script keeps polling the queue, so the Stop / Edit buttons stay responsive
    even while the model is still 'thinking' and has not produced any text yet."""
    q = queue.Queue()
    stop_event = threading.Event()

    def worker():
        try:
            for chunk in chain.stream(inputs):
                if stop_event.is_set():
                    break
                q.put(("chunk", str(chunk)))
            q.put(("done", ""))
        except Exception as e:  # e.g. Ollama is not running
            q.put(("error", f"{e.__class__.__name__}: {e}"))

    threading.Thread(target=worker, daemon=True).start()
    return q, stop_event


def render_sources(urls, source_docs):
    if urls:
        st.markdown("**Web sources:** " + " , ".join(f"[{u}]({u})" for u in urls))
    if source_docs:
        with st.expander("Show Source Documents"):
            for i, text in enumerate(source_docs, 1):
                st.markdown(f"**Source {i}:**")
                st.text(text)
                st.divider()


# ---------- Helpers (PDF export) ----------
EMOJI_RE = re.compile("[\U00010000-\U0010FFFF\u2600-\u27BF\u2B00-\u2BFF\u23CF-\u23FF\uFE0F\u200D]")


def register_pdf_fonts():
    """Use Arial (Windows) or DejaVu if available so rupee signs / accents print correctly."""
    win = os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts")
    candidates = [
        (os.path.join(win, "arial.ttf"), os.path.join(win, "arialbd.ttf"),
         os.path.join(win, "ariali.ttf"), os.path.join(win, "arialbi.ttf")),
        ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
         "/usr/share/fonts/truetype/dejavu/DejaVuSans-Oblique.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-BoldOblique.ttf"),
        ("/Library/Fonts/Arial.ttf", "/Library/Fonts/Arial Bold.ttf",
         "/Library/Fonts/Arial Italic.ttf", "/Library/Fonts/Arial Bold Italic.ttf"),
    ]
    for regular, bold, italic, bold_italic in candidates:
        if os.path.exists(regular):
            bold = bold if os.path.exists(bold) else regular
            italic = italic if os.path.exists(italic) else regular
            bold_italic = bold_italic if os.path.exists(bold_italic) else bold
            if "ChatFont" not in pdfmetrics.getRegisteredFontNames():
                pdfmetrics.registerFont(TTFont("ChatFont", regular))
                pdfmetrics.registerFont(TTFont("ChatFont-Bold", bold))
                pdfmetrics.registerFont(TTFont("ChatFont-Italic", italic))
                pdfmetrics.registerFont(TTFont("ChatFont-BoldItalic", bold_italic))
                pdfmetrics.registerFontFamily(
                    "ChatFont", normal="ChatFont", bold="ChatFont-Bold",
                    italic="ChatFont-Italic", boldItalic="ChatFont-BoldItalic",
                )
            return "ChatFont", "ChatFont-Bold", True
    return "Helvetica", "Helvetica-Bold", False


def pdf_text(raw, unicode_ok):
    """Plain chat text -> safe reportlab paragraph markup."""
    t = EMOJI_RE.sub("", raw or "")
    if not unicode_ok:  # built-in Helvetica only knows Windows-1252 characters
        t = t.encode("cp1252", "replace").decode("cp1252")
    t = html.escape(t, quote=False)
    t = re.sub(r"\[([^\]]+)\]\((https?://[^)\s]+)\)", r"\1 (\2)", t)  # [text](url) -> text (url)
    out = []
    for line in t.split("\n"):
        line = re.sub(r"^#{1,6}\s*", "", line)
        line = re.sub(r"^\s*[-*]\s+", "\u2022 " if unicode_ok else "- ", line)
        line = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", line)
        line = re.sub(r"`([^`]+)`", r"\1", line)
        out.append(line)
    return "<br/>".join(out)


def build_chat_pdf(messages):
    """Whole conversation -> PDF bytes."""
    font, bold, unicode_ok = register_pdf_fonts()
    title = ParagraphStyle("t", fontName=bold, fontSize=16, leading=20, spaceAfter=2)
    meta = ParagraphStyle("m", fontName=font, fontSize=9, leading=12, textColor=colors.grey, spaceAfter=10)
    you = ParagraphStyle("you", fontName=bold, fontSize=10, leading=13, textColor=colors.HexColor("#1f4e9c"), spaceBefore=6)
    bot = ParagraphStyle("bot", fontName=bold, fontSize=10, leading=13, textColor=colors.HexColor("#b35c00"), spaceBefore=6)
    body = ParagraphStyle("b", fontName=font, fontSize=10.5, leading=14.5, spaceAfter=2)
    small = ParagraphStyle("s", fontName=font, fontSize=8.5, leading=11, textColor=colors.grey)

    def footer(canvas, doc):
        canvas.saveState()
        canvas.setFont(font, 8)
        canvas.setFillColor(colors.grey)
        canvas.drawCentredString(A4[0] / 2, 1.0 * cm, f"DSU Student Assistant  -  page {doc.page}")
        canvas.restoreState()

    story = [
        Paragraph("DSU Student Assistant - Chat transcript", title),
        Paragraph("Saved on " + datetime.datetime.now().strftime("%d %b %Y, %H:%M"), meta),
    ]
    for m in messages:
        if m["role"] == "user":
            story.append(Paragraph("You", you))
            story.append(Paragraph(pdf_text(m["content"], unicode_ok) or "(empty)", body))
        else:
            story.append(Paragraph("Assistant", bot))
            story.append(Paragraph(pdf_text(m["content"], unicode_ok) or "(no text)", body))
            if m.get("stopped"):
                story.append(Paragraph("(response stopped)", small))
            for u in m.get("urls", []):
                safe = html.escape(u, quote=True)
                story.append(Paragraph(f'Source: <link href="{safe}" color="blue">{safe}</link>', small))
        story.append(Spacer(1, 8))

    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm, topMargin=2 * cm, bottomMargin=2 * cm,
        title="DSU Student Assistant - chat", author="DSU Student Assistant",
    )
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    return buf.getvalue()


# ---------- Session state ----------
ss = st.session_state
ss.setdefault("messages", [])        # finished chat messages
ss.setdefault("generating", False)   # True from "send" until the answer is saved
ss.setdefault("gen_started", False)  # True once the generation code has started running
ss.setdefault("pending_query", None)
ss.setdefault("partial", "")         # text streamed so far (survives an interrupted run)
ss.setdefault("cur_urls", [])
ss.setdefault("cur_docs", [])
ss.setdefault("editing_idx", None)   # index of the user message being edited
ss.setdefault("input_nonce", 0)    # changing it resets (empties) the typing box


def start_generation(text, cut_at=None):
    """Queue a new prompt. If cut_at is given (edit), drop that message and everything after it."""
    if cut_at is not None:
        ss.messages = ss.messages[:cut_at]
    ss.messages.append({"role": "user", "content": text})
    ss.pending_query = text
    ss.generating = True
    ss.gen_started = False
    ss.partial = ""
    ss.cur_urls, ss.cur_docs = [], []
    ss.editing_idx = None
    st.rerun()


# ---------- 1) A previous run was interrupted (Stop / Edit / Refresh clicked) ----------
# Streamlit aborts a running script when a button is clicked. Whatever text had been
# streamed so far is kept as a "stopped" answer.
if ss.generating and ss.gen_started:
    ss.messages.append({
        "role": "assistant",
        "content": ss.partial,
        "stopped": True,
        "urls": ss.cur_urls,
        "source_docs": ss.cur_docs,
    })
    ss.generating = False
    ss.gen_started = False
    ss.pending_query = None
    ss.partial = ""

# ---------- Top-right: Refresh button + "Download Chat" in Streamlit's menu ----------
hide_pdf_css = "" if SHOW_DOWNLOAD_BUTTON else """
    .st-key-pdf_btn { position: absolute; width: 1px; height: 1px; overflow: hidden; opacity: 0; pointer-events: none; }
"""
st.markdown(
    f"""<style>
    .st-key-topbar {{
        position: fixed; top: 0.55rem; right: {TOPBAR_RIGHT}; z-index: 1000000;
        width: auto !important; flex-direction: row !important; align-items: center; gap: 0.5rem !important;
    }}
    .st-key-topbar [data-testid="stElementContainer"] {{ width: auto !important; }}
    .st-key-topbar button {{ min-height: 2rem; padding: 0.15rem 0.8rem; }}
    .st-key-menupatch {{ position: absolute; width: 0; height: 0; overflow: hidden; }}
    [data-testid="stChatMessage"] {{ position: relative; }}
    [data-testid="stChatMessage"]:has([class*="st-key-editbtn_"]) {{ padding-right: 3rem; }}
    [class*="st-key-editbtn_"] {{
        position: absolute; top: 0.45rem; right: 0.5rem; z-index: 2;
        width: auto !important; opacity: 0.5; transition: opacity 0.15s;
    }}
    [class*="st-key-editbtn_"] button {{
        min-height: 1.7rem; height: 1.7rem; width: 1.7rem; padding: 0;
        font-size: 0.8rem; border: none; background: transparent;
    }}
    [class*="st-key-editbtn_"] button:hover {{ background: rgba(128, 128, 128, 0.2); }}
    [data-testid="stChatMessage"]:hover [class*="st-key-editbtn_"] {{ opacity: 1; }}
    {hide_pdf_css}
    </style>""",
    unsafe_allow_html=True,
)

# Streamlit's three-dot menu cannot be edited from Python, so a tiny script renames the
# "Record screen" entry to "Download Chat" each time the menu is opened, and makes it
# click the (hidden) download button below.
MENU_PATCH_HTML = """
<script>
const main = function () {
  if (window.__dsuMenuPatch) {
    document.removeEventListener('click', window.__dsuMenuPatch.onClick, true);
  }
  const LABEL = 'Download Chat';

  function patch() {
    const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT);
    let node;
    while ((node = walker.nextNode())) {
      const t = node.nodeValue.trim().toLowerCase();
      if (t === 'record screen' || t === 'record a screencast') {
        node.nodeValue = LABEL;
        const el = node.parentElement;
        const item = el.closest('[role="menuitem"], [role="option"], li, button') || el;
        item.setAttribute('data-dsu-download', '1');
      }
    }
  }

  function onClick(e) {
    const target = e.target;
    if (!target || !target.closest) return;
    const item = target.closest('[data-dsu-download="1"]');
    if (item) {
      e.preventDefault();
      e.stopPropagation();
      e.stopImmediatePropagation();
      const btn = document.querySelector('[class*="st-key-pdf_btn"] button, [class*="st-key-pdf_btn"] a');
      if (btn && !btn.disabled) {
        btn.click();
      } else {
        window.alert('There is no chat to download yet. Ask a question first.');
      }
      document.dispatchEvent(new KeyboardEvent('keydown', {key: 'Escape', code: 'Escape', keyCode: 27, bubbles: true}));
      return;
    }
    if (target.closest('[data-testid="stToolbar"], [data-testid="stMainMenu"], [data-testid="stHeader"]')) {
      [60, 200, 500].forEach(function (ms) { setTimeout(patch, ms); });
    }
  }

  document.addEventListener('click', onClick, true);
  window.__dsuMenuPatch = { onClick: onClick };
};

const old = window.parent.document.getElementById('dsu-menu-patch');
if (old) old.remove();
const s = window.parent.document.createElement('script');
s.id = 'dsu-menu-patch';
s.textContent = '(' + main.toString() + ')();';
window.parent.document.head.appendChild(s);
</script>
"""
with st.container(key="menupatch"):
    try:
        components.html(MENU_PATCH_HTML, height=0)
    except Exception:
        pass  # the menu just keeps its normal "Record screen" entry

with st.container(key="topbar"):
    if st.button("🔄 Refresh", key="refresh_btn", help="Clear this chat and start a new one"):
        ss.messages = []
        ss.editing_idx = None
        ss.input_nonce += 1  # a new key gives a brand-new, empty typing box
        for k in [k for k in ss.keys() if str(k).startswith("edit_text_")]:
            del ss[k]
        st.rerun()

    pdf_bytes, pdf_help = b"", "Download this chat as a PDF"
    if not HAVE_PDF:
        pdf_help = "Install the PDF library first:  pip install reportlab"
    elif ss.messages:
        try:
            pdf_bytes = build_chat_pdf(ss.messages)
        except Exception as e:
            pdf_help = f"Could not build the PDF ({e.__class__.__name__})"
    else:
        pdf_help = "Ask a question first - then you can download the chat"
    pdf_args = dict(
        label="⬇ Download PDF", data=pdf_bytes, key="pdf_btn", help=pdf_help,
        file_name=datetime.datetime.now().strftime("DSU_chat_%Y%m%d_%H%M.pdf"),
        mime="application/pdf", disabled=not pdf_bytes,
    )
    try:
        st.download_button(on_click="ignore", **pdf_args)  # "ignore": downloading never interrupts an answer
    except TypeError:  # older Streamlit without on_click="ignore"
        st.download_button(**pdf_args)

# ---------- 2) Chat history ----------
for idx, msg in enumerate(ss.messages):
    with st.chat_message(msg["role"]):
        if msg["role"] == "user":
            if ss.editing_idx == idx:
                new_text = st.text_area(
                    "Edit your prompt", value=msg["content"],
                    key=f"edit_text_{idx}", label_visibility="collapsed",
                )
                c1, c2, _ = st.columns([1, 1, 3])
                if c1.button("Send", key=f"send_{idx}", type="primary") and new_text.strip():
                    start_generation(new_text.strip(), cut_at=idx)
                if c2.button("Cancel", key=f"cancel_{idx}"):
                    ss.editing_idx = None
                    st.rerun()
            else:
                st.markdown(msg["content"])
                if st.button("✏️", key=f"editbtn_{idx}", help="Edit prompt"):
                    ss.editing_idx = idx
                    st.rerun()
        else:
            if msg["content"].strip():
                st.markdown(msg["content"])
            if msg.get("stopped"):
                st.caption("⏹ Response stopped" if msg["content"].strip()
                           else "⏹ Response stopped before any text was generated")
            render_sources(msg.get("urls", []), msg.get("source_docs", []))

# ---------- 3) Chat input: only ONE prompt at a time ----------
# While an answer is being generated the input box is disabled.
user_input = st.chat_input(
    "Type your question here...", disabled=ss.generating, key=f"chat_input_{ss.input_nonce}"
)
if user_input and not ss.generating:
    start_generation(user_input.strip())

# ---------- 4) Generate the answer ----------
if ss.generating:
    ss.gen_started = True
    query = ss.pending_query

    with st.chat_message("assistant"):
        loading = st.empty()      # animated "loading" indicator
        answer_slot = st.empty()  # the streaming answer
        stop_slot = st.empty()    # Stop button
        stop_slot.button("⏹ Stop generating", key="stop_btn")

        loading.markdown("🔎 **Searching the DSU website and documents…**")
        try:
            docs = retrieve_docs(query)
        except Exception as e:
            loading.empty()
            stop_slot.empty()
            ss.messages.append({
                "role": "assistant",
                "content": f"⚠️ Something went wrong while searching: `{e.__class__.__name__}: {e}`",
            })
            ss.generating = False
            ss.gen_started = False
            ss.pending_query = None
            st.rerun()

        ss.cur_urls = source_urls(docs)
        ss.cur_docs = [
            d.page_content[:1500] + ("..." if len(d.page_content) > 1500 else "") for d in docs
        ]

        q, stop_event = start_stream({"context": format_docs(docs), "question": query})
        full, error, started = "", None, time.time()
        try:
            while True:
                try:
                    kind, value = q.get(timeout=0.25)
                except queue.Empty:
                    # Nothing yet: update the indicator (this also lets Streamlit notice a click on Stop)
                    if not full:
                        secs = int(time.time() - started)
                        dots = "." * (int(time.time() * 2) % 4)
                        loading.markdown(f"⏳ **Thinking{dots}** ({secs}s)")
                    continue
                if kind == "chunk":
                    if not full:
                        loading.empty()  # first words arrived -> hide the loading indicator
                    full += value
                    ss.partial = full
                    answer_slot.markdown(full + "▌")  # blinking-cursor effect
                elif kind == "error":
                    error = value
                    break
                else:  # done
                    break
        finally:
            stop_event.set()  # also runs when Streamlit aborts this run (Stop / Edit clicked)

        # ---- reached only when the answer finished normally ----
        loading.empty()
        stop_slot.empty()
        if error:
            full = (full + "\n\n" if full else "") + (
                f"⚠️ The language model failed: `{error}`. "
                "Make sure Ollama is running and the model is pulled (`ollama pull qwen2.5:7b`)."
            )
        answer_slot.markdown(full)

    ss.messages.append({
        "role": "assistant",
        "content": full,
        "stopped": False,
        "urls": ss.cur_urls,
        "source_docs": ss.cur_docs,
    })
    ss.generating = False
    ss.gen_started = False
    ss.pending_query = None
    ss.partial = ""
    st.rerun()  # re-draw from history so the input box and Edit buttons are enabled again