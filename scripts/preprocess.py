# scripts/preprocess.py
import json, re, hashlib
from pathlib import Path
from pypdf import PdfReader
from pypdf.errors import PdfReadError, DependencyError

RAW_DIR = Path("reference")
OUT_PATH = Path("data_processed/corpus.jsonl")
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

CHUNK_SIZE = 2000
OVERLAP = 300

def stable_id(*parts: str) -> str:
    return hashlib.md5("::".join(parts).encode("utf-8")).hexdigest()[:10]

def clean_text(s: str) -> str:
    if not s:
        return ""
    s = s.replace("\r", "\n")
    # compress blank 
    s = re.sub(r"[ \t]{2,}", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    # merge non-paragraph newline
    s = re.sub(r"(?<![.!?。！？:])\n(?!\n)", " ", s)
    return s.strip()

def pdf_to_text(path: Path) -> str:
    try:
        reader = PdfReader(str(path))

        # process encrypted pdf
        if getattr(reader, "is_encrypted", False):
            try:
                reader.decrypt("")
            except Exception:
                print(f"Skip encrypted PDF (cannot decrypt): {path}")
                return ""

        pages_text = []
        bad_pages = 0

        for i, p in enumerate(reader.pages):
            try:
                pages_text.append(p.extract_text() or "")
            except Exception as e:
                bad_pages += 1
                print(f"Page extract failed: {path} page={i} err={type(e).__name__}")
                continue

        text = "\n".join(pages_text).strip()
        if bad_pages > 0:
            print(f"{path}: skipped {bad_pages} bad pages")

        if len(reader.pages) > 0 and bad_pages / len(reader.pages) > 0.5:
            print(f"Skip PDF (too many bad pages): {path}")
            return ""
        
        return text

    except (PdfReadError, DependencyError) as e:
        print(f"Skip PDF (read/decrypt error): {path} | {e}")
        return ""
    except Exception as e:
        print(f"Skip PDF (unexpected error): {path} | {type(e).__name__}: {e}")
        return ""

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = OVERLAP):
    chunks = []
    i = 0
    n = len(text)
    while i < n:
        j = min(n, i + chunk_size)
        chunk = text[i:j].strip()
        if chunk:
            chunks.append(chunk)
        if j == n:
            break
        i = max(0, j - overlap)
    return chunks

def subject_from_path(path: Path) -> str:
    # reference/<subject>/... 
    # e.g. reference/math/xxx.pdf -> math
    parts = path.parts
    if "reference" in parts:
        idx = parts.index("reference")
        if idx + 1 < len(parts):
            return parts[idx + 1].strip()
    return "unknown"

def infer_grade(path: Path) -> int | None:
    """
    Infer grade (1-6) from your file naming patterns.
    Handles:
      - Grade3, grade-5, grade_2, grade 4
      - 3rdGrade, 5thgradefull (no space ok)
      - Fourth-Grade (word-based)
      - ManualGrade2, Spectrum_LA_Grade4...
      - 1stmathcomplete... / 2ndscience...
    Avoids false matches like math320_*.pdf.
    """
    s = path.name.lower()

    # grade3 / grade-3 / grade_3 / grade 3 / ...grade3...
    m = re.search(r"\bgrade[\s\-_]*([1-6])\b", s)
    if m:
        return int(m.group(1))

    # 2) ordinal + grade：3rdgrade / 5thgradefull / 4th grade 
    m = re.search(r"\b([1-6])(st|nd|rd|th)\s*grade\b", s)
    if m:
        return int(m.group(1))
    m = re.search(r"\b([1-6])(st|nd|rd|th)\s*grade", s)  # 去掉末尾边界，支持 5thgradefull
    if m:
        return int(m.group(1))

    # 3) ordinal + subject/word：1stmathcompleteblm.pdf 
    m = re.search(r"\b([1-6])(st|nd|rd|th)(?=[a-z])", s)
    if m:
        return int(m.group(1))

    # 4) first/second/third/fourth/fifth + -?grade
    word_to_num = {
        "first": 1,
        "second": 2,
        "third": 3,
        "fourth": 4,
        "fifth": 5,
        "sixth": 6,
    }
    m = re.search(r"\b(first|second|third|fourth|fifth|sixth)[\s\-_]*grade\b", s)
    if m:
        return word_to_num[m.group(1)]

    # "...FourthGrade"（no line）
    m = re.search(r"\b(first|second|third|fourth|fifth|sixth)grade\b", s)
    if m:
        return word_to_num[m.group(1)]

    # 6) g3
    m = re.search(r"\bg([1-6])\b", s)
    if m:
        return int(m.group(1))

    return None


def main():
    count_files = 0
    count_chunks = 0
    skipped_scan_like = 0

    grade_count = {}

    with OUT_PATH.open("w", encoding="utf-8") as out:
        for path in RAW_DIR.rglob("*"):
            if path.is_dir():
                continue
            if path.suffix.lower() not in [".pdf", ".txt"]:
                continue

            subject = subject_from_path(path)
            source = str(path.relative_to(RAW_DIR))

            if path.suffix.lower() == ".pdf":
                raw = pdf_to_text(path)
            else:
                raw = path.read_text(encoding="utf-8", errors="ignore")

            text = clean_text(raw)

            if len(text) < 200:
                skipped_scan_like += 1
                continue

            grade = infer_grade(path)
            chunks = chunk_text(text)
            for k, c in enumerate(chunks):
                doc_id = f"{path.stem}_{k:04d}_{stable_id(source, str(k))}"
                rec = {
                    "doc_id": doc_id,
                    "subject": subject,   
                    "grade": grade,        
                    "source": source,
                    "text": c,
                }
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                g = grade if grade is not None else "None"
                grade_count[g] = grade_count.get(g, 0) + 1
                count_chunks += 1

            count_files += 1

    print(f"Processed files: {count_files}")
    print(f"Wrote chunks: {count_chunks} -> {OUT_PATH}")
    print("Grade distribution (chunks):", grade_count)
    if skipped_scan_like:
        print(f"Skipped {skipped_scan_like} files (too little extracted text; Might be a scanned pdf)")

if __name__ == "__main__":
    main()