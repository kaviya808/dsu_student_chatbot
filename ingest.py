import glob
import json
import os
import re

YEAR_NAMES = ["I Year", "II Year", "III Year", "IV Year", "V Year"]
STARS = {"", "*", "**", "***", "****", "*****", "******", "-", "NA", "N/A"}


def is_stars(cell):
    c = cell.strip()
    return c in STARS or set(c) <= {"*"}


def calendar_page_to_text(raw, name):
    """Turn one academic-calendar page (a markdown table) into plain lines like:
    'Odd Semester 2025-26 | Semester End Examinations | I Year: from 29.12.2025 to 19.01.2026'
    so the LLM does not have to read a wide table. Returns None if the page can't be parsed."""
    raw = raw.replace("<br>", " ").replace("<br/>", " ")
    lines = raw.split("\n")

    # 1) join table rows that were broken over several lines, collect non-table text
    rows, other, buf = [], [], ""
    for ln in lines:
        s = ln.strip()
        if buf:
            buf += " " + s
            if s.endswith("|"):
                rows.append(buf)
                buf = ""
            continue
        if s.startswith("|"):
            if s.endswith("|") and len(s) > 1:
                rows.append(s)
            else:
                buf = s
        elif s:
            other.append(re.sub(r"<[^>]+>|^#+\s*", "", s).strip())
    if buf:
        rows.append(buf)

    text_all = " ".join(other)
    m = re.search(r"ACADEMIC CALENDAR\s*(\d{4}-\d{2})", text_all, re.I)
    ay = m.group(1) if m else ""
    m = re.search(r"PROGRAM\s*:\s*(.+)", text_all, re.I)
    program = re.sub(r"\s+", " ", m.group(1)).strip() if m else ""

    # 2) parse rows
    parsed = []
    for r in rows:
        cells = [c.strip() for c in r.strip().strip("|").split("|")]
        if not cells or all(set(c) <= {"-", ":", " "} for c in cells):
            continue  # separator row
        first = cells[0].upper()
        if first == "EVENTS" or any(c.upper() in ("FROM", "TO") for c in cells):
            continue  # header rows
        label = re.sub(r"\s+", " ", cells[0]).strip()
        data = cells[1:]
        if all(c == "" for c in data) and parsed:
            parsed[-1][0] += " " + label  # wrapped label, e.g. '... Semesters' + '2026-27'
            continue
        parsed.append([label, data])

    # 3) build sentences
    term = f"Academic year {ay}".strip()
    out = []
    for label, data in parsed:
        low = label.lower()
        row_term = term
        if "commencement" in low and "odd" in low:
            y = re.search(r"\d{4}-\d{2}", label)
            term = row_term = f"Odd Semester {y.group(0) if y else ay}".strip()
        elif "commencement" in low and "even" in low:
            term = row_term = f"Even Semester {ay}".strip()
        elif "summer" in low:
            row_term = f"Summer Term {ay}".strip()
        n_years = (len(data) + 1) // 2
        for yi in range(min(n_years, len(YEAR_NAMES))):
            frm = data[2 * yi] if 2 * yi < len(data) else ""
            to = data[2 * yi + 1] if 2 * yi + 1 < len(data) else ""
            if is_stars(frm) and is_stars(to):
                dates = "no date given"
            elif is_stars(to):
                dates = frm
            elif is_stars(frm):
                dates = to
            else:
                dates = f"from {frm} to {to}"
            out.append(f"{row_term} | {label} | {YEAR_NAMES[yi]}: {dates}")

    if len(out) < 3:
        return None

    header = [f"[Document: {name}]", "DAYANANDA SAGAR UNIVERSITY", f"ACADEMIC CALENDAR {ay}"]
    header.append(f"PROGRAM : {program}")
    extra = [o for o in other if o and not re.search(r"DAYANANDA SAGAR UNIVERSITY|ACADEMIC CALENDAR|PROGRAM\s*:", o, re.I)]
    return "\n".join(header + extra + [""] + out)


def main():
    import pymupdf4llm
    from langchain_core.documents import Document
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_community.vectorstores import FAISS

    documents = []        # PDF pages that will be split into chunks
    calendar_pages = []   # academic calendar pages, kept whole

    # ---- 1) your own PDFs in data/pdfs ---------------------------------
    print("Loading PDFs from data/pdfs/ ...")
    for path in glob.glob("data/pdfs/*.pdf"):
        name = os.path.splitext(os.path.basename(path))[0]
        pages = pymupdf4llm.to_markdown(path, page_chunks=True)
        for i, p in enumerate(pages):
            meta = {"source": path, "page": i}
            if name.startswith("AC_"):
                # One page = one program. Convert the table to plain lines and
                # keep the page WHOLE (never split it).
                text = calendar_page_to_text(p["text"], name)
                if text is None:
                    text = f"[Document: {name}]\n{p['text']}"
                calendar_pages.append(Document(page_content=text, metadata=meta))
            else:
                documents.append(Document(
                    page_content=f"[Document: {name}]\n{p['text']}",
                    metadata=meta,
                ))
    print(f"  local PDF pages: {len(documents) + len(calendar_pages)}")

    # ---- 2) PDFs downloaded from the website (crawl_website.py --pdfs) --
    manifest = {}
    manifest_path = "data/website_pdfs/_manifest.json"
    if os.path.exists(manifest_path):
        with open(manifest_path, encoding="utf-8") as f:
            manifest = json.load(f)
    web_pdf_count = 0
    for path in glob.glob("data/website_pdfs/*.pdf"):
        fname = os.path.basename(path)
        url = manifest.get(fname, path)
        try:
            pages = pymupdf4llm.to_markdown(path, page_chunks=True)
        except Exception as e:  # damaged or scanned PDFs
            print(f"  ! could not read {fname}: {e.__class__.__name__}")
            continue
        for i, p in enumerate(pages):
            documents.append(Document(
                page_content=f"[Document: {os.path.splitext(fname)[0]}] ({url})\n{p['text']}",
                metadata={"source": url, "page": i},
            ))
        web_pdf_count += 1
    print(f"  website PDFs: {web_pdf_count}")

    # ---- 3) website pages (crawl_website.py) ----------------------------
    web_chunks = []
    web_path = "data/website/pages.jsonl"
    if os.path.exists(web_path):
        web_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        with open(web_path, encoding="utf-8") as f:
            for line in f:
                page = json.loads(line)
                for chunk in web_splitter.split_text(page["text"]):
                    # every chunk carries its page title + URL so it can be cited
                    web_chunks.append(Document(
                        page_content=f"[Web page: {page['title']}] ({page['url']})\n{chunk}",
                        metadata={"source": page["url"], "title": page["title"], "type": "web"},
                    ))
        print(f"  website text chunks: {len(web_chunks)}")
    else:
        print("  (no website data found - run 'python crawl_website.py' first)")

    # ---- 4) split PDFs, combine, embed ----------------------------------
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents) + calendar_pages + web_chunks
    print(f"Created {len(chunks)} chunks in total")

    print("Creating embeddings (this can take several minutes)...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    print("Building FAISS index...")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local("faiss_index")

    print("\nIngestion completed successfully!")
    print("FAISS index saved in 'faiss_index' folder.")


if __name__ == "__main__":
    main()
