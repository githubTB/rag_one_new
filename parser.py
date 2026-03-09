"""
parser.py — 文档解析模块

功能：
- 每个 chunk 携带完整溯源元数据（文件名、页码、标题路径、block 类型）
- 按语义边界切块（标题归属、整表为一块）
- PDF 用 PyMuPDF + pdfplumber 提取文字和表格
- 图片 OCR：优先 PaddleOCR（中文），降级到 Tesseract
- Word 内嵌图片、PDF 内嵌图片全部提取并 OCR
- Excel 每个 Sheet 作为一个 table chunk
"""

import os
import re
import io
import hashlib
import tempfile
from dataclasses import dataclass, field
from langchain_core.documents import Document

# ── 可选依赖检测 ──────────────────────────────────────────────

try:
    import fitz  # PyMuPDF
    HAS_PYMUPDF = True
except ImportError:
    HAS_PYMUPDF = False
    print("⚠️  PyMuPDF 未安装: pip install pymupdf")

try:
    import pdfplumber
    HAS_PDFPLUMBER = True
except ImportError:
    HAS_PDFPLUMBER = False

try:
    from docx import Document as DocxDocument
    HAS_DOCX = True
except ImportError:
    HAS_DOCX = False
    print("⚠️  python-docx 未安装: pip install python-docx")

try:
    import openpyxl
    HAS_OPENPYXL = True
except ImportError:
    HAS_OPENPYXL = False
    print("⚠️  openpyxl 未安装: pip install openpyxl")

# ── PaddleOCR ────────────────────────────────────────────────
try:
    from paddleocr import PaddleOCR
    _paddle_instance = None
    HAS_PADDLE = True
except Exception:
    HAS_PADDLE = False

# ── Tesseract ────────────────────────────────────────────────
HAS_TESSERACT = False
_TESS_LANG = "eng"

try:
    import pytesseract
    from PIL import Image, ImageFilter, ImageEnhance
    _available_langs = pytesseract.get_languages(config="")
    HAS_TESSERACT = True
    _found = [l for l in ["chi_sim", "chi_tra", "eng"] if l in _available_langs]
    _TESS_LANG = "+".join(_found) if _found else "eng"
    print(f"✅ Tesseract 可用，语言: {_TESS_LANG}")
except Exception as _e:
    pass

try:
    from PIL import Image, ImageFilter, ImageEnhance
    HAS_PIL = True
except ImportError:
    HAS_PIL = False


# ── 自定义异常 ───────────────────────────────────────────────

class OCRUnavailableError(RuntimeError):
    """OCR 引擎未安装时抛出，携带安装指引"""
    INSTALL_GUIDE = (
        "未检测到可用的 OCR 引擎，无法识别图片文字。\n"
        "请安装以下任意一个 OCR 引擎后重启服务：\n\n"
        "  方案 A — PaddleOCR（推荐，中文识别率高）：\n"
        "    pip install paddlepaddle paddleocr\n\n"
        "  方案 B — Tesseract（轻量）：\n"
        "    # Ubuntu/Debian\n"
        "    sudo apt-get install tesseract-ocr tesseract-ocr-chi-sim\n"
        "    pip install pytesseract\n\n"
        "    # macOS\n"
        "    brew install tesseract tesseract-lang\n"
        "    pip install pytesseract"
    )

    def __init__(self, fname: str = ""):
        self.fname = fname
        super().__init__(self.INSTALL_GUIDE)


# ── Chunk 数据结构 ────────────────────────────────────────────

@dataclass
class Chunk:
    """带完整溯源信息的文本块"""
    text: str
    file_name: str
    file_path: str
    file_hash: str
    page: int = 1
    block_type: str = "text"          # text | heading | table | image
    heading_path: str = ""
    heading_level: int = 0
    table_headers: list = field(default_factory=list)
    chunk_index: int = 0

    def to_langchain_doc(self) -> Document:
        return Document(
            page_content=self.text,
            metadata={
                "file_name":     self.file_name,
                "file_path":     self.file_path,
                "file_hash":     self.file_hash,
                "page":          self.page,
                "block_type":    self.block_type,
                "heading_path":  self.heading_path,
                "heading_level": self.heading_level,
                "table_headers": "|".join(self.table_headers),
                "chunk_index":   self.chunk_index,
            }
        )


# ── 工具函数 ──────────────────────────────────────────────────

def file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def clean_text(text: str) -> str:
    """去除中文字符间多余空格，规范化换行"""
    text = re.sub(r'([\u4e00-\u9fff])\s+([\u4e00-\u9fff])', r'\1\2', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def detect_heading_level(text: str, style_name: str = "") -> int:
    """返回标题级别（1-6），0 表示非标题"""
    if style_name:
        s = style_name.lower().replace(" ", "")
        m = re.match(r'heading(\d)', s) or re.match(r'标题(\d)', s)
        if m:
            return int(m.group(1))
        if "title" in s:    return 1
        if "subtitle" in s: return 2

    if not text or len(text) > 120:
        return 0

    if re.match(r'^第[一二三四五六七八九十百]+[章节部分]', text): return 1
    if re.match(r'^[一二三四五六七八九十]+[、．.]', text):        return 2
    if re.match(r'^\d+\.\d+\.\d+\s', text):                       return 3
    if re.match(r'^\d+\.\d+\s', text):                             return 2
    if re.match(r'^\d+[\.、]\s', text):                            return 2
    return 0


def smart_chunk_text(text: str, max_size: int = 800, overlap: int = 100) -> list[str]:
    """按段落语义边界切块，避免字符数硬切"""
    if len(text) <= max_size:
        return [text]

    paragraphs = [p.strip() for p in re.split(r'\n\n+', text) if p.strip()]
    chunks, current, current_len = [], [], 0

    for para in paragraphs:
        if current_len + len(para) > max_size and current:
            chunks.append("\n\n".join(current))
            current = current[-1:]
            current_len = len(current[0]) if current else 0
        current.append(para)
        current_len += len(para)

    if current:
        chunks.append("\n\n".join(current))

    return chunks


# ── 标题栈管理 ────────────────────────────────────────────────

def _update_heading_stack(stack: list, level: int, text: str):
    while stack and stack[-1][0] >= level:
        stack.pop()
    stack.append((level, text))


def _heading_path(stack: list) -> str:
    return " > ".join(t for _, t in stack)


# ── OCR ──────────────────────────────────────────────────────

def _preprocess_image(img):
    """图像预处理：灰度→放大→锐化→增强对比度→二值化"""
    if not HAS_PIL:
        return img
    img = img.convert("L")
    w, h = img.size
    if w < 1200:
        scale = 1200 / w
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    img = img.filter(ImageFilter.SHARPEN)
    img = ImageEnhance.Contrast(img).enhance(2.0)
    img = img.point(lambda p: 255 if p > 140 else 0, "1")
    return img


def _ocr_with_paddle(img_path: str) -> str:
    global _paddle_instance
    if _paddle_instance is None:
        _paddle_instance = PaddleOCR(use_angle_cls=True, lang="ch")
    try:
        result = _paddle_instance.ocr(img_path, angle_classification=True)
    except TypeError:
        result = _paddle_instance.ocr(img_path)
    lines = []
    if result:
        for line in result:
            if line:
                for item in line:
                    if item and len(item) >= 2:
                        lines.append(item[1][0])
    return "\n".join(lines)


def _ocr_image_obj(img) -> str:
    img = _preprocess_image(img)
    if HAS_PADDLE:
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            img.convert("RGB").save(f.name)
            tmp = f.name
        try:
            return _ocr_with_paddle(tmp)
        finally:
            os.unlink(tmp)
    elif HAS_TESSERACT:
        return pytesseract.image_to_string(img, lang=_TESS_LANG, config="--psm 6")
    return ""


def _ocr_bytes(img_bytes: bytes, ext: str = ".png") -> str:
    if not HAS_PIL:
        return ""
    try:
        img = Image.open(io.BytesIO(img_bytes))
        return _ocr_image_obj(img)
    except Exception as e:
        print(f"    ⚠️  图片 OCR 失败: {e}")
        return ""


def _ocr_page(page) -> str:
    """对 PyMuPDF 页面做 OCR（扫描页专用）"""
    try:
        mat = fitz.Matrix(2, 2)
        pix = page.get_pixmap(matrix=mat)
        return _ocr_bytes(pix.tobytes("png"), ".png")
    except Exception as e:
        print(f"  OCR 失败: {e}")
    return ""


# ── PDF 解析 ──────────────────────────────────────────────────

def _extract_pdf_page_images(page, fname, file_path, fhash,
                              page_num, heading_path, start_idx) -> list[Chunk]:
    """提取 PDF 页面内嵌图片并 OCR"""
    chunks = []
    if not HAS_PYMUPDF or (not HAS_PADDLE and not HAS_TESSERACT):
        return chunks
    try:
        seen_xref = set()
        for img_info in page.get_images(full=True):
            xref = img_info[0]
            if xref in seen_xref:
                continue
            seen_xref.add(xref)
            try:
                base_img = page.parent.extract_image(xref)
                if base_img.get("width", 0) < 100 or base_img.get("height", 0) < 100:
                    continue
                img_text = _ocr_bytes(base_img["image"], "." + base_img.get("ext", "png"))
                if img_text.strip():
                    for t in smart_chunk_text(clean_text(img_text)):
                        if t.strip():
                            chunks.append(Chunk(
                                text=t, file_name=fname, file_path=file_path,
                                file_hash=fhash, page=page_num, block_type="image",
                                heading_path=heading_path,
                                chunk_index=start_idx + len(chunks)
                            ))
            except Exception as e:
                print(f"    ⚠️  PDF 图片提取失败: {e}")
    except Exception as e:
        print(f"  ⚠️  页面图片列表获取失败: {e}")
    return chunks


def parse_pdf(file_path: str) -> list[Chunk]:
    """PyMuPDF + pdfplumber 解析 PDF，扫描页自动 OCR"""
    chunks = []
    fhash = file_sha256(file_path)
    fname = os.path.basename(file_path)
    idx = 0

    if not HAS_PYMUPDF:
        print(f"⚠️  PyMuPDF 不可用，跳过 {fname}")
        return chunks

    doc = fitz.open(file_path)
    heading_stack = []
    plumber_doc = pdfplumber.open(file_path) if HAS_PDFPLUMBER else None

    for page_num in range(len(doc)):
        page = doc[page_num]
        page_display = page_num + 1
        text_raw = page.get_text("text")

        # 扫描页 → OCR
        if len(text_raw.strip()) < 50:
            ocr_text = _ocr_page(page)
            if ocr_text:
                for t in smart_chunk_text(clean_text(ocr_text)):
                    chunks.append(Chunk(
                        text=t, file_name=fname, file_path=file_path,
                        file_hash=fhash, page=page_display, block_type="image",
                        chunk_index=idx, heading_path=_heading_path(heading_stack)
                    ))
                    idx += 1
            continue

        # 表格（pdfplumber）
        table_texts = set()
        if plumber_doc:
            try:
                for table in plumber_doc.pages[page_num].extract_tables():
                    if not table:
                        continue
                    headers = [str(c or "").strip() for c in table[0]]
                    rows = [" | ".join(str(c or "").strip() for c in row)
                            for row in table[1:] if any(row)]
                    table_text = " | ".join(headers) + "\n" + "\n".join(rows)
                    table_texts.add(table_text[:200])
                    chunks.append(Chunk(
                        text=table_text, file_name=fname, file_path=file_path,
                        file_hash=fhash, page=page_display, block_type="table",
                        table_headers=headers, heading_path=_heading_path(heading_stack),
                        chunk_index=idx
                    ))
                    idx += 1
            except Exception as e:
                print(f"  ⚠️  表格提取失败 page {page_display}: {e}")

        # 页内嵌入图片 OCR
        img_chunks = _extract_pdf_page_images(
            page, fname, file_path, fhash, page_display,
            _heading_path(heading_stack), idx
        )
        chunks.extend(img_chunks)
        idx += len(img_chunks)

        # 文字块（按字体大小识别标题）
        blocks = page.get_text("dict")["blocks"]
        for block in blocks:
            if block.get("type") != 0:
                continue
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    t = span["text"].strip()
                    if not t or t in table_texts:
                        continue
                    size = span.get("size", 12)
                    flags = span.get("flags", 0)
                    is_bold = bool(flags & 2 ** 4)
                    if size >= 16 or (size >= 13 and is_bold and len(t) < 80):
                        level = 1 if size >= 18 else (2 if size >= 15 else 3)
                        level = detect_heading_level(t) or level
                        _update_heading_stack(heading_stack, level, t)
                        chunks.append(Chunk(
                            text=t, file_name=fname, file_path=file_path,
                            file_hash=fhash, page=page_display, block_type="heading",
                            heading_level=level, heading_path=_heading_path(heading_stack),
                            chunk_index=idx
                        ))
                        idx += 1

        # 普通文字段落
        heading_texts = {c.text for c in chunks
                         if c.page == page_display and c.block_type == "heading"}
        lines = [l for l in clean_text(text_raw).split("\n")
                 if l.strip() and l.strip() not in heading_texts]
        for t in smart_chunk_text("\n".join(lines)):
            if t.strip():
                chunks.append(Chunk(
                    text=t, file_name=fname, file_path=file_path,
                    file_hash=fhash, page=page_display, block_type="text",
                    heading_path=_heading_path(heading_stack), chunk_index=idx
                ))
                idx += 1

    doc.close()
    if plumber_doc:
        plumber_doc.close()
    return chunks


# ── Word 解析 ─────────────────────────────────────────────────

def _extract_docx_paragraph_images(element, word_doc) -> list[tuple[bytes, str]]:
    """提取 Word 段落内嵌图片"""
    results = []
    try:
        ns = {"a": "http://schemas.openxmlformats.org/drawingml/2006/main"}
        for blip in element.findall(".//a:blip", ns):
            embed = blip.get(
                "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}embed"
            )
            if not embed:
                continue
            rels = word_doc.part.rels
            if embed not in rels:
                continue
            rel = rels[embed]
            if "image" not in rel.reltype.lower():
                continue
            try:
                img_part = rel.target_part
                ext = os.path.splitext(img_part.partname)[-1].lower() or ".png"
                results.append((img_part.blob, ext))
            except Exception as e:
                print(f"    ⚠️  Word 图片读取失败: {e}")
    except Exception as e:
        print(f"  ⚠️  Word 图片解析失败: {e}")
    return results


def parse_docx(file_path: str) -> list[Chunk]:
    """python-docx 按 body 顺序解析，保留标题层级和表格"""
    chunks = []
    fhash = file_sha256(file_path)
    fname = os.path.basename(file_path)
    idx = 0

    if not HAS_DOCX:
        return chunks

    word = DocxDocument(file_path)
    heading_stack = []

    for element in word.element.body:
        tag = element.tag.split("}")[-1]

        if tag == "p":
            from docx.text.paragraph import Paragraph
            para = Paragraph(element, word)

            # 嵌入图片 OCR
            for img_bytes, img_ext in _extract_docx_paragraph_images(element, word):
                img_text = _ocr_bytes(img_bytes, img_ext)
                if img_text.strip():
                    for t in smart_chunk_text(clean_text(img_text)):
                        if t.strip():
                            chunks.append(Chunk(
                                text=t, file_name=fname, file_path=file_path,
                                file_hash=fhash, page=1, block_type="image",
                                heading_path=_heading_path(heading_stack), chunk_index=idx
                            ))
                            idx += 1

            text = para.text.strip()
            if not text:
                continue

            style = para.style.name if para.style else ""
            level = detect_heading_level(text, style)

            if level > 0:
                _update_heading_stack(heading_stack, level, text)
                chunks.append(Chunk(
                    text=text, file_name=fname, file_path=file_path,
                    file_hash=fhash, page=1, block_type="heading",
                    heading_level=level, heading_path=_heading_path(heading_stack),
                    chunk_index=idx
                ))
                idx += 1
            else:
                for t in smart_chunk_text(text):
                    if t.strip():
                        chunks.append(Chunk(
                            text=t, file_name=fname, file_path=file_path,
                            file_hash=fhash, page=1, block_type="text",
                            heading_path=_heading_path(heading_stack), chunk_index=idx
                        ))
                        idx += 1

        elif tag == "tbl":
            from docx.table import Table
            table = Table(element, word)
            if not table.rows:
                continue
            headers = [c.text.strip() for c in table.rows[0].cells]
            data_rows = [
                " | ".join(c.text.strip() for c in row.cells)
                for row in table.rows[1:] if any(c.text.strip() for c in row.cells)
            ]
            table_text = " | ".join(headers) + "\n" + "\n".join(data_rows)
            chunks.append(Chunk(
                text=table_text, file_name=fname, file_path=file_path,
                file_hash=fhash, page=1, block_type="table",
                table_headers=headers, heading_path=_heading_path(heading_stack),
                chunk_index=idx
            ))
            idx += 1

    return chunks


# ── Excel 解析 ────────────────────────────────────────────────

def _fmt_cell(val) -> str:
    if val is None:
        return ""
    if isinstance(val, float):
        if val == int(val):
            return str(int(val))
        import decimal
        return str(decimal.Decimal(str(val)).normalize())
    return str(val).strip()


def parse_excel(file_path: str) -> list[Chunk]:
    """每个 Sheet 作为一个 table chunk"""
    chunks = []
    fhash = file_sha256(file_path)
    fname = os.path.basename(file_path)

    if not HAS_OPENPYXL:
        return chunks

    wb = openpyxl.load_workbook(file_path, data_only=True)
    for idx, sheet_name in enumerate(wb.sheetnames):
        ws = wb[sheet_name]
        rows = list(ws.iter_rows(values_only=True))
        if not rows:
            continue
        headers = [str(c or "").strip() for c in rows[0]]
        data_rows = [
            " | ".join(_fmt_cell(c) for c in row)
            for row in rows[1:] if any(c is not None for c in row)
        ]
        table_text = f"[Sheet: {sheet_name}]\n" + " | ".join(headers) + "\n" + "\n".join(data_rows)
        chunks.append(Chunk(
            text=table_text, file_name=fname, file_path=file_path,
            file_hash=fhash, page=idx + 1, block_type="table",
            table_headers=headers, heading_path=f"Sheet: {sheet_name}",
            chunk_index=idx
        ))
    return chunks


# ── 独立图片 OCR ──────────────────────────────────────────────

def parse_image(file_path: str) -> list[Chunk]:
    """独立图片文件 OCR。无 OCR 引擎时抛出 OCRUnavailableError。"""
    fhash = file_sha256(file_path)
    fname = os.path.basename(file_path)

    if not HAS_PIL:
        raise OCRUnavailableError(fname)  # Pillow 也算无引擎

    if not HAS_PADDLE and not HAS_TESSERACT:
        raise OCRUnavailableError(fname)

    try:
        img = Image.open(file_path)
        text = _ocr_image_obj(img)
    except OCRUnavailableError:
        raise
    except Exception as e:
        print(f"    ❌ 图片处理失败 {fname}: {e}")
        return []

    if not text.strip():
        print(f"    ⚠️  OCR 未识别到文字: {fname}")
        return []

    return [
        Chunk(
            text=t, file_name=fname, file_path=file_path,
            file_hash=fhash, page=1, block_type="image", chunk_index=i
        )
        for i, t in enumerate(smart_chunk_text(clean_text(text))) if t.strip()
    ]


# ── 统一入口 ──────────────────────────────────────────────────

SUPPORTED_EXTS = {
    ".pdf":  parse_pdf,
    ".docx": parse_docx,
    ".xlsx": parse_excel,
    ".xls":  parse_excel,
    ".png":  parse_image,
    ".jpg":  parse_image,
    ".jpeg": parse_image,
    ".tiff": parse_image,
    ".bmp":  parse_image,
}


def parse_file(file_path: str) -> list[Chunk]:
    ext = os.path.splitext(file_path)[1].lower()
    parser = SUPPORTED_EXTS.get(ext)
    if not parser:
        print(f"⚠️  不支持的格式: {file_path}")
        return []
    print(f"  📄 解析 {os.path.basename(file_path)} ({ext})")
    return parser(file_path)


def parse_directory(dir_path: str) -> list[Chunk]:
    all_chunks = []
    for root, _, files in os.walk(dir_path):
        for fname in sorted(files):
            fpath = os.path.join(root, fname)
            try:
                all_chunks.extend(parse_file(fpath))
            except Exception as e:
                print(f"  ❌ 解析失败 {fname}: {e}")
    print(f"✅ 共解析 {len(all_chunks)} 个 chunk")
    return all_chunks