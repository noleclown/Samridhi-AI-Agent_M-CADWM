import os
import time
import requests
import tempfile
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from langchain.schema import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# ==============================
# CONFIG
# ==============================

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DOCS_PATH   = os.path.join(BASE_DIR, "documents")
FAISS_PATH  = os.path.join(BASE_DIR, "faiss_index")

TARGET_SITE = "http://cadwm.gov.in/"
MAX_PAGES   = 200       # max HTML pages to crawl
MAX_PDFS    = 100       # max linked PDFs to download
DELAY       = 0.5       # seconds between requests (be polite)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}

STRIP_TAGS = ["script", "style", "nav", "footer", "header",
              "aside", "noscript", "iframe", "form", "button"]

SKIP_EXTS = {".jpg", ".jpeg", ".png", ".gif", ".svg", ".webp",
             ".zip", ".rar", ".tar", ".gz", ".exe", ".mp4",
             ".mp3", ".wav", ".css", ".js"}


# ==============================
# HELPERS
# ==============================

def clean_url(url):
    p = urlparse(url)
    return p._replace(fragment="").geturl().rstrip("/")

def is_internal(url, domain):
    return urlparse(url).netloc == domain

def get_ext(url):
    _, ext = os.path.splitext(urlparse(url).path)
    return ext.lower()


# ==============================
# DEEP TEXT EXTRACTION
# ==============================

def extract_text(soup, url):
    parts = []

    # Page title
    if soup.title:
        parts.append(f"PAGE TITLE: {soup.title.get_text(strip=True)}")

    # Meta description
    meta = soup.find("meta", attrs={"name": "description"})
    if meta and meta.get("content"):
        parts.append(f"DESCRIPTION: {meta['content']}")

    # Strip noise
    for tag in soup(STRIP_TAGS):
        tag.decompose()

    # Tables — preserve as structured text
    for table in soup.find_all("table"):
        rows = []
        headers = [th.get_text(strip=True) for th in table.find_all("th")]
        for tr in table.find_all("tr"):
            cells = [td.get_text(strip=True) for td in tr.find_all(["td","th"])]
            if not any(cells):
                continue
            if headers and len(cells) == len(headers):
                rows.append(" | ".join(f"{h}: {v}" for h, v in zip(headers, cells)))
            else:
                rows.append(" | ".join(c for c in cells if c))
        if rows:
            parts.append("TABLE DATA:\n" + "\n".join(rows))
        table.decompose()

    # Definition lists
    for dl in soup.find_all("dl"):
        pairs = []
        for dt, dd in zip(dl.find_all("dt"), dl.find_all("dd")):
            pairs.append(f"{dt.get_text(strip=True)}: {dd.get_text(strip=True)}")
        if pairs:
            parts.append("DEFINITIONS:\n" + "\n".join(pairs))
        dl.decompose()

    # Body text with deduplication
    body = soup.get_text(separator="\n", strip=True)
    seen, lines = set(), []
    for line in body.splitlines():
        line = line.strip()
        if line and len(line) > 3 and line not in seen:
            seen.add(line)
            lines.append(line)
    parts.append("\n".join(lines))

    return "\n\n".join(filter(None, parts))


# ==============================
# SCRAPE ONE HTML PAGE
# ==============================

def scrape_html(url, session, domain):
    try:
        resp = session.get(url, headers=HEADERS, timeout=12)
        resp.raise_for_status()
        if "text/html" not in resp.headers.get("Content-Type", ""):
            return None, [], []

        soup  = BeautifulSoup(resp.text, "html.parser")
        text  = extract_text(soup, url)

        # Collect child links
        html_links, pdf_links = [], []
        # Re-parse for links (soup was mutated during extract_text)
        soup2 = BeautifulSoup(resp.text, "html.parser")
        for a in soup2.find_all("a", href=True):
            href = clean_url(urljoin(url, a["href"]))
            ext  = get_ext(href)
            if not is_internal(href, domain):
                continue
            if ext == ".pdf":
                pdf_links.append(href)
            elif ext not in SKIP_EXTS and "#" not in href:
                html_links.append(href)

        if len(text) < 80:
            return None, html_links, pdf_links

        doc = Document(
            page_content=text,
            metadata={"source": url, "type": "website_html"}
        )
        return doc, html_links, pdf_links

    except Exception as e:
        print(f"    ⚠ HTML [{url}]: {e}")
        return None, [], []


# ==============================
# DOWNLOAD + PARSE SITE PDF
# ==============================

def scrape_pdf(url, session):
    try:
        resp = session.get(url, headers=HEADERS, timeout=25, stream=True)
        resp.raise_for_status()

        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            for chunk in resp.iter_content(8192):
                tmp.write(chunk)
            tmp_path = tmp.name

        pages = PyPDFLoader(tmp_path).load()
        os.unlink(tmp_path)

        for p in pages:
            p.metadata["source"] = url
            p.metadata["type"]   = "website_pdf"

        print(f"    📥 {url} ({len(pages)} pages)")
        return pages

    except Exception as e:
        print(f"    ⚠ PDF [{url}]: {e}")
        return []


# ==============================
# FULL CRAWLER
# ==============================

def crawl():
    domain       = urlparse(TARGET_SITE).netloc
    visited_html = set()
    visited_pdf  = set()
    html_queue   = [clean_url(TARGET_SITE)]
    pdf_queue    = []
    html_docs    = []
    pdf_docs     = []
    session      = requests.Session()

    print(f"\n{'='*55}")
    print(f"  🌐 Deep crawl of {TARGET_SITE}")
    print(f"  Max HTML pages : {MAX_PAGES} | Max PDFs : {MAX_PDFS}")
    print(f"{'='*55}\n")

    # ── Phase 1: HTML pages ────────────────────────────────────────
    while html_queue and len(visited_html) < MAX_PAGES:
        url = html_queue.pop(0)
        if url in visited_html:
            continue
        visited_html.add(url)

        print(f"  HTML [{len(visited_html):03d}/{MAX_PAGES}] {url}")
        doc, h_links, p_links = scrape_html(url, session, domain)

        if doc:
            html_docs.append(doc)

        for lnk in h_links:
            lnk = clean_url(lnk)
            if lnk not in visited_html and lnk not in html_queue:
                html_queue.append(lnk)
        for lnk in p_links:
            lnk = clean_url(lnk)
            if lnk not in visited_pdf and lnk not in pdf_queue:
                pdf_queue.append(lnk)

        time.sleep(DELAY)

    print(f"\n  ✅ HTML phase done  — {len(html_docs)} pages with content\n")

    # ── Phase 2: PDFs found on site ────────────────────────────────
    print(f"  Found {len(pdf_queue)} PDF links on site.")
    to_dl = pdf_queue[:MAX_PDFS]
    print(f"  Downloading {len(to_dl)} PDFs...\n")

    for url in to_dl:
        if url in visited_pdf:
            continue
        visited_pdf.add(url)
        pages = scrape_pdf(url, session)
        pdf_docs.extend(pages)
        time.sleep(DELAY)

    print(f"\n  ✅ PDF phase done   — {len(pdf_docs)} pages from {len(visited_pdf)} PDFs")
    return html_docs, pdf_docs


# ==============================
# LOCAL PDFs
# ==============================

def load_local_pdfs():
    docs = []
    if not os.path.exists(DOCS_PATH):
        print("  No documents/ folder — skipping local PDFs")
        return docs
    files = [f for f in os.listdir(DOCS_PATH) if f.endswith(".pdf")]
    print(f"\n📂 Loading {len(files)} local PDFs...")
    for f in files:
        print(f"  {f}")
        try:
            pages = PyPDFLoader(os.path.join(DOCS_PATH, f)).load()
            for p in pages:
                p.metadata["type"] = "local_pdf"
            docs.extend(pages)
        except Exception as e:
            print(f"  ⚠ {f}: {e}")
    print(f"  ✅ {len(docs)} pages loaded")
    return docs


# ==============================
# MAIN
# ==============================

if __name__ == "__main__":

    all_docs = []

    # 1. Local PDFs
    all_docs.extend(load_local_pdfs())

    # 2. Website crawl (HTML + linked PDFs)
    html_docs, site_pdf_docs = crawl()
    all_docs.extend(html_docs)
    all_docs.extend(site_pdf_docs)

    if not all_docs:
        print("\n❌ No documents found.")
        exit(1)

    # Stats
    local  = sum(1 for d in all_docs if d.metadata.get("type") == "local_pdf")
    web_h  = sum(1 for d in all_docs if d.metadata.get("type") == "website_html")
    web_p  = sum(1 for d in all_docs if d.metadata.get("type") == "website_pdf")
    print(f"\n{'='*55}")
    print(f"  📊 Documents collected")
    print(f"     Local PDFs   : {local} pages")
    print(f"     Website HTML : {web_h} pages")
    print(f"     Website PDFs : {web_p} pages")
    print(f"     TOTAL        : {len(all_docs)}")
    print(f"{'='*55}\n")

    # 3. Split
    print("✂️  Splitting into chunks...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        separators=["\n\n", "\n", "।", ".", " ", ""]
    )
    chunks = splitter.split_documents(all_docs)
    print(f"   → {len(chunks)} chunks\n")

    # 4. Embed
    print("🔢 Creating embeddings (may take a few minutes)...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        cache_folder=os.path.join(BASE_DIR, "models")
    )

    # 5. Build + save FAISS
    print("\n💾 Building FAISS index...")
    vector_db = FAISS.from_documents(chunks, embeddings)
    vector_db.save_local(FAISS_PATH)

    print(f"\n{'='*55}")
    print(f"  ✅ FAISS index built successfully!")
    print(f"     Chunks indexed : {len(chunks)}")
    print(f"     Saved to       : {FAISS_PATH}")
    print(f"{'='*55}")
