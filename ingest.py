import os
import requests
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

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DOCS_PATH  = os.path.join(BASE_DIR, "documents")
FAISS_PATH = os.path.join(BASE_DIR, "faiss_index")

HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; SamridhiBot/1.0)"}

# ==============================
# WEBSITE SCRAPER
# ==============================

def scrape_page(url):
    try:
        resp = requests.get(url, headers=HEADERS, timeout=10)
        if resp.status_code != 200:
            print(f"  Skipped {url} (status {resp.status_code})")
            return None
        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header",
                          "aside", "noscript", "iframe"]):
            tag.decompose()
        text = soup.get_text(separator="\n", strip=True)
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        text = "\n".join(lines)
        return text if len(text) > 100 else None
    except Exception as e:
        print(f"  Error scraping {url}: {e}")
        return None


def crawl_website(base_url, max_pages=60):
    visited, to_visit, docs = set(), [base_url], []
    domain = urlparse(base_url).netloc

    while to_visit and len(visited) < max_pages:
        url = to_visit.pop(0)
        if url in visited:
            continue
        visited.add(url)
        print(f"  Scraping: {url}")
        text = scrape_page(url)
        if text:
            docs.append(Document(
                page_content=text,
                metadata={"source": url, "type": "website"}
            ))
        try:
            resp = requests.get(url, headers=HEADERS, timeout=10)
            soup = BeautifulSoup(resp.text, "html.parser")
            for a in soup.find_all("a", href=True):
                href = urljoin(url, a["href"])
                parsed = urlparse(href)
                if (parsed.netloc == domain
                        and href not in visited
                        and href not in to_visit
                        and "#" not in href
                        and not any(href.endswith(x) for x in
                                    [".pdf",".jpg",".png",".zip",
                                     ".xlsx",".doc",".docx"])):
                    to_visit.append(href)
        except Exception:
            pass

    print(f"  Crawled {len(docs)} pages from {base_url}")
    return docs


# ==============================
# PDF LOADER
# ==============================

def load_pdfs():
    docs = []
    if not os.path.exists(DOCS_PATH):
        print("  No documents/ folder — skipping PDFs")
        return docs
    for file in os.listdir(DOCS_PATH):
        if file.endswith(".pdf"):
            print(f"  Loading: {file}")
            try:
                loader = PyPDFLoader(os.path.join(DOCS_PATH, file))
                docs.extend(loader.load())
            except Exception as e:
                print(f"  Error loading {file}: {e}")
    print(f"  Loaded {len(docs)} PDF pages")
    return docs


# ==============================
# MAIN
# ==============================

if __name__ == "__main__":
    all_docs = []

    print("\n📄 Loading PDFs...")
    all_docs.extend(load_pdfs())

    print("\n🌐 Crawling cadwm.gov.in...")
    all_docs.extend(crawl_website("http://cadwm.gov.in/", max_pages=60))

    if not all_docs:
        print("❌ No documents found.")
        exit(1)

    print(f"\n✂️  Splitting {len(all_docs)} documents...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = splitter.split_documents(all_docs)
    print(f"   → {len(chunks)} chunks")

    print("\n🔢 Creating embeddings...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        cache_folder=os.path.join(BASE_DIR, "models")
    )

    print("\n💾 Saving FAISS index...")
    vector_db = FAISS.from_documents(chunks, embeddings)
    vector_db.save_local(FAISS_PATH)

    print(f"\n✅ Done! {len(chunks)} chunks indexed (PDFs + website).")
