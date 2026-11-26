"""Load the knowledge base (the official Christite Handbook PDF + any extra .md notes)
and split it into retrievable, citable chunks."""
import re
from dataclasses import dataclass
from pathlib import Path

from . import config

PAGE_OFFSET = 6   # PDF page 7 is printed page 1 in the handbook


@dataclass
class Chunk:
    id: str
    source: str      # file name
    title: str       # document title
    section: str     # handbook chapter / markdown H2
    text: str
    page: int | None = None   # printed handbook page

    @property
    def citation(self) -> str:
        page = f", p.{self.page}" if self.page else ""
        return f"{self.section}{page}"


def _windows(text: str, size: int = config.CHUNK_WORDS, overlap: int = config.CHUNK_OVERLAP):
    words = text.split()
    if len(words) <= size:
        return [" ".join(words)]
    out, step = [], size - overlap
    for start in range(0, len(words), step):
        out.append(" ".join(words[start:start + size]))
        if start + size >= len(words):
            break
    return out


# ------------------------------------------------------------------ PDF
_FOOTER = re.compile(r"^\s*\d*\s*(Handbook 2026-27|CHRIST \(Deemed to be University\))\s*\d*\s*$")
def parse_toc(pages_text):
    """Read the 'Table of Contents' page into [(start_printed_page, chapter_title)]."""
    toc_text = next((t for t in pages_text if "Table of Contents" in t), "")
    entries, pending = [], None
    for ln in toc_text.splitlines():
        ln = ln.strip()
        m = re.match(r"^(\d{1,2})\.\s+(.*?)\s+(\d{1,3})$", ln)
        if m:
            title = re.sub(r"\(.*?\)", "", m.group(2)).strip()
            entries.append((int(m.group(3)), title))
            pending = None
        elif re.match(r"^(\d{1,2})\.\s+\S", ln):          # title whose page number wraps to next line
            pending = re.sub(r"^\d{1,2}\.\s+", "", ln).strip()
        elif pending and re.search(r"(\d{1,3})$", ln):
            entries.append((int(re.search(r"(\d{1,3})$", ln).group(1)), pending))
            pending = None
    return sorted(entries)


def load_pdf(path: Path):
    from pypdf import PdfReader
    reader = PdfReader(str(path))
    title = "Christite Handbook 2026-27"
    pages_text = [p.extract_text() or "" for p in reader.pages]
    toc = parse_toc(pages_text)
    chunks = []
    for idx, raw in enumerate(pages_text):
        if "Table of Contents" in raw:
            continue
        printed = idx + 1 - PAGE_OFFSET
        section = next((t for start, t in reversed(toc) if printed >= start), "Front matter")
        lines = [ln for ln in raw.splitlines() if not _FOOTER.match(ln)]
        body = " ".join(lines)
        if len(body.split()) < 25:          # cover / blank / signature pages
            continue
        for i, piece in enumerate(_windows(body)):
            chunks.append(Chunk(
                id=f"handbook-p{printed}-{i}", source=path.name, title=title,
                section=section, page=printed if printed > 0 else None,
                text=f"{section}: {piece}"))
    return chunks


# ------------------------------------------------------------------ Markdown (optional extra notes)
def load_markdown(path: Path):
    raw = path.read_text(encoding="utf-8")
    title_m = re.search(r"^# (.+)$", raw, re.M)
    title = title_m.group(1).strip() if title_m else path.stem
    chunks = []
    for part in re.split(r"^## ", raw, flags=re.M)[1:]:
        heading, _, body = part.partition("\n")
        for i, piece in enumerate(_windows(body.strip())):
            chunks.append(Chunk(id=f"{path.stem}#{heading.strip()}-{i}", source=path.name,
                                title=title, section=f"{title} - {heading.strip()}",
                                text=f"{title} - {heading.strip()}: {piece}"))
    return chunks


def load_chunks(kb_dir: Path = config.KB_DIR):
    chunks = []
    # If precomputed json chunks exist, prefer loading them directly (fast & cloud-compatible)
    chunks_json = kb_dir / "handbook_chunks.json"
    has_pdf = any(p.suffix.lower() == ".pdf" for p in kb_dir.iterdir()) if kb_dir.exists() else False

    if chunks_json.exists() and not has_pdf:
        import json
        items = json.loads(chunks_json.read_text(encoding="utf-8"))
        for it in items:
            chunks.append(Chunk(
                id=it["id"], source=it["source"], title=it["title"],
                section=it["section"], text=it["text"], page=it.get("page")
            ))
        return chunks

    if kb_dir.exists():
        for path in sorted(kb_dir.iterdir()):
            if path.suffix.lower() == ".pdf":
                chunks += load_pdf(path)
            elif path.suffix.lower() == ".md":
                chunks += load_markdown(path)

    # Fallback to json if PDF wasn't present or yielded nothing
    if not chunks and chunks_json.exists():
        import json
        items = json.loads(chunks_json.read_text(encoding="utf-8"))
        for it in items:
            chunks.append(Chunk(
                id=it["id"], source=it["source"], title=it["title"],
                section=it["section"], text=it["text"], page=it.get("page")
            ))

    return chunks
