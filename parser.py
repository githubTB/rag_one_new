"""
parser.py — 替换原来的 loader.py + pipeline.py

核心改进：
1. 每个 chunk 携带完整溯源元数据（文件名、页码、标题路径、block类型）
2. 按语义边界切块（标题归属、整表为一块），不按字符数硬切
3. PDF 用 PyMuPDF，效果远好于 PyPDFLoader
4. 图片 OCR 用 PaddleOCR（中文效果好），降级到 pytesseract
5. 表格整体作为一个 chunk，保留结构
6. 嵌入图片处理：Word 内嵌图片、PDF 内嵌图片 全部提取并 OCR
"""

import os
import re
import io
import hashlib
import tempfile
from dataclasses import dataclass, field
from typing import Optional
from langchain_core.documents import Document

# ── 可选依赖 ──────────────────────────────────────────────────
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
    from docx.oxml.ns import qn
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

# 先设置Paddle相关的缓存目录为当前目录
import os

# 设置多个相关环境变量
os.environ['PADDLE_HOME'] = os.path.join(os.getcwd(), '.paddle')
os.environ['PADDLEX_HOME'] = os.path.join(os.getcwd(), '.paddlex')
os.environ['PADDLEOCR_HOME'] = os.path.join(os.getcwd(), '.paddleocr')
os.environ['HOME'] = os.getcwd()  # 临时覆盖HOME目录

# 确保目录存在
for dir_path in [os.environ['PADDLE_HOME'], os.environ['PADDLEX_HOME'], os.environ['PADDLEOCR_HOME']]:
    os.makedirs(dir_path, exist_ok=True)

# 先测试langchain.docstore模块是否存在
try:
    print("🔍 尝试导入langchain.docstore...")
    from langchain.docstore import InMemoryDocstore
    print("✅ langchain.docstore导入成功")
except ImportError as e:
    print(f"❌ langchain.docstore导入失败: {e}")
    # 尝试从langchain_community导入
    try:
        print("🔍 尝试从langchain_community导入...")
        from langchain_community.docstore import InMemoryDocstore
        print("✅ langchain_community.docstore导入成功")
        # 添加到sys.modules，让paddleocr能够找到它
        import sys
        import langchain_community.docstore
        sys.modules['langchain.docstore'] = langchain_community.docstore
    except ImportError as e2:
        print(f"❌ langchain_community.docstore导入失败: {e2}")

# 尝试导入PaddleOCR，如果失败则禁用
try:
    print("🔍 尝试导入PaddleOCR...")
    from paddleocr import PaddleOCR
    print("✅ PaddleOCR导入成功")
    _paddle_instance = None
    HAS_PADDLE = True
    print(f"✅ HAS_PADDLE设置为: {HAS_PADDLE}")
except ImportError as e:
    print(f"❌ PaddleOCR导入失败 (ImportError): {e}")
    HAS_PADDLE = False
except Exception as e:
    print(f"❌ PaddleOCR初始化失败 (Exception): {e}")
    import traceback
    traceback.print_exc()
    HAS_PADDLE = False
print(f"📊 最终HAS_PADDLE值: {HAS_PADDLE}")

try:
    from PIL import Image, ImageFilter, ImageEnhance
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

# Tesseract：import 成功不代表能用，需要实际验证并检测可用语言包
HAS_TESSERACT = False
_TESS_LANG = "eng"  # 默认只用英文，成功检测到中文包才加

try:
    import pytesseract
    if HAS_PIL:
        # 实际调用一次，确认 tesseract binary 可用
        _available_langs = pytesseract.get_languages(config="")
        HAS_TESSERACT = True
        # 动态选最佳语言组合
        _lang_candidates = ["chi_sim", "chi_tra", "eng"]
        _found = [l for l in _lang_candidates if l in _available_langs]
        _TESS_LANG = "+".join(_found) if _found else "eng"
        print(f"✅ Tesseract 可用，语言包: {_available_langs}，将使用: {_TESS_LANG}")
except Exception as _tess_err:
    HAS_TESSERACT = False
    print(f"⚠️  Tesseract 不可用: {_tess_err}")


# ── Chunk 数据结构 ────────────────────────────────────────────

@dataclass
class Chunk:
    """带完整溯源信息的文本块"""
    text: str                          # 文本内容
    file_name: str                     # 原始文件名
    file_path: str                     # 完整路径
    file_hash: str                     # 文件 SHA256（用于去重）
    page: int = 1                      # 页码（从1开始）
    block_type: str = "text"           # text | heading | table | image
    heading_path: str = ""             # 标题路径，如 "第一章 > 1.1 背景"
    heading_level: int = 0             # 0=非标题，1-6=标题级别
    table_headers: list = field(default_factory=list)  # 表格表头
    chunk_index: int = 0               # 在文件中的顺序

    def to_langchain_doc(self) -> Document:
        """转为 LangChain Document，元数据完整保留"""
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
                "table_headers": "|".join(self.table_headers) if self.table_headers else "",
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
    """去掉中文字符中间多余空格，保留段落结构"""
    text = re.sub(r'([\u4e00-\u9fff])\s+([\u4e00-\u9fff])', r'\1\2', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def detect_heading_level(text: str, style_name: str = "") -> int:
    """检测标题级别，返回 0（非标题）或 1-6"""
    # Word 样式名优先
    if style_name:
        s = style_name.lower().replace(" ", "")
        for pattern, level in [
            (r'heading(\d)', None), (r'标题(\d)', None)
        ]:
            m = re.match(pattern, s)
            if m:
                return int(m.group(1))
        if "title" in s: return 1
        if "subtitle" in s: return 2

    if not text or len(text) > 120:
        return 0

    # 中文编号规律
    if re.match(r'^第[一二三四五六七八九十百]+[章节部分]', text): return 1
    if re.match(r'^[一二三四五六七八九十]+[、．.]', text):        return 2
    if re.match(r'^\d+\.\d+\.\d+\s', text):                       return 3
    if re.match(r'^\d+\.\d+\s', text):                             return 2
    if re.match(r'^\d+[\.、]\s', text):                            return 2

    return 0


def smart_chunk_text(text: str, max_size: int = 800, overlap: int = 100) -> list[str]:
    """
    对长文本做语义边界切块（段落 → 句子），保留 overlap
    比原来的字符数硬切好很多
    """
    if len(text) <= max_size:
        return [text]

    # 先按段落切
    paragraphs = [p.strip() for p in re.split(r'\n\n+', text) if p.strip()]
    chunks, current, current_len = [], [], 0

    for para in paragraphs:
        if current_len + len(para) > max_size and current:
            chunk_text = "\n\n".join(current)
            chunks.append(chunk_text)
            # overlap：保留最后一段
            current = current[-1:] if current else []
            current_len = len(current[0]) if current else 0
        current.append(para)
        current_len += len(para)

    if current:
        chunks.append("\n\n".join(current))

    return chunks


# ── PDF 解析 ──────────────────────────────────────────────────

def parse_pdf(file_path: str) -> list[Chunk]:
    """
    用 PyMuPDF 解析 PDF，自动检测扫描页走 OCR
    比原来的 PyPDFLoader 保留更多结构信息
    """
    chunks = []
    fhash = file_sha256(file_path)
    fname = os.path.basename(file_path)
    idx = 0

    if not HAS_PYMUPDF:
        print(f"⚠️  PyMuPDF 不可用，跳过 {fname}")
        return chunks

    doc = fitz.open(file_path)
    heading_stack = []  # 维护当前标题路径栈

    # 同时用 pdfplumber 提取表格
    plumber_doc = pdfplumber.open(file_path) if HAS_PDFPLUMBER else None

    for page_num in range(len(doc)):
        page = doc[page_num]
        page_display = page_num + 1

        # ── 检测是否为扫描页 ──
        text_raw = page.get_text("text")
        if len(text_raw.strip()) < 50:
            # 扫描页 → OCR
            ocr_text = _ocr_page(page)
            if ocr_text:
                for t in smart_chunk_text(clean_text(ocr_text)):
                    chunks.append(Chunk(
                        text=t, file_name=fname, file_path=file_path,
                        file_hash=fhash, page=page_display,
                        block_type="image", chunk_index=idx,
                        heading_path=_heading_path(heading_stack)
                    ))
                    idx += 1
            continue

        # ── 提取表格（pdfplumber）──
        table_texts = set()
        if plumber_doc:
            try:
                plumber_page = plumber_doc.pages[page_num]
                for table in plumber_page.extract_tables():
                    if not table: continue
                    headers = [str(c or "").strip() for c in table[0]]
                    rows = []
                    for row in table[1:]:
                        r = [str(c or "").strip() for c in row]
                        if any(r):
                            rows.append(" | ".join(r))
                    table_text = " | ".join(headers) + "\n" + "\n".join(rows)
                    table_texts.add(table_text[:200])  # 用于去重
                    chunks.append(Chunk(
                        text=table_text, file_name=fname, file_path=file_path,
                        file_hash=fhash, page=page_display,
                        block_type="table", table_headers=headers,
                        heading_path=_heading_path(heading_stack), chunk_index=idx
                    ))
                    idx += 1
            except Exception as e:
                print(f"  ⚠️  表格提取失败 page {page_display}: {e}")

        # ── 提取页内嵌入的图片并 OCR ──────────────────────────
        img_chunks = _extract_pdf_page_images(page, fname, file_path, fhash,
                                               page_display, _heading_path(heading_stack), idx)
        if img_chunks:
            chunks.extend(img_chunks)
            idx += len(img_chunks)

        # ── 提取文字块，按字体大小识别标题 ──
        blocks = page.get_text("dict")["blocks"]
        for block in blocks:
            if block.get("type") != 0:  # 只处理文字块
                continue
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    t = span["text"].strip()
                    if not t or t in table_texts:
                        continue
                    size = span.get("size", 12)
                    flags = span.get("flags", 0)
                    is_bold = bool(flags & 2**4)

                    # 通过字体大小 + 加粗推断标题
                    if size >= 16 or (size >= 13 and is_bold and len(t) < 80):
                        level = 1 if size >= 18 else (2 if size >= 15 else 3)
                        level = detect_heading_level(t) or level
                        _update_heading_stack(heading_stack, level, t)
                        chunks.append(Chunk(
                            text=t, file_name=fname, file_path=file_path,
                            file_hash=fhash, page=page_display,
                            block_type="heading", heading_level=level,
                            heading_path=_heading_path(heading_stack), chunk_index=idx
                        ))
                        idx += 1
                    # else: 普通文字，下面按段落合并

        # ── 普通文字段落 ──
        full_text = clean_text(page.get_text("text"))
        # 去掉已经作为标题提取的行
        heading_texts = {c.text for c in chunks if c.page == page_display and c.block_type == "heading"}
        lines = [l for l in full_text.split("\n") if l.strip() and l.strip() not in heading_texts]
        para_text = "\n".join(lines)

        for t in smart_chunk_text(para_text):
            if t.strip():
                chunks.append(Chunk(
                    text=t, file_name=fname, file_path=file_path,
                    file_hash=fhash, page=page_display,
                    block_type="text", heading_path=_heading_path(heading_stack),
                    chunk_index=idx
                ))
                idx += 1

    doc.close()
    if plumber_doc:
        plumber_doc.close()

    return chunks


# ── Word 解析 ─────────────────────────────────────────────────

def parse_docx(file_path: str) -> list[Chunk]:
    """
    用 python-docx 按 body 顺序遍历，完整保留标题层级和表格
    比原来的 Docx2txtLoader 保留更多结构
    """
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
            # 段落
            from docx.text.paragraph import Paragraph
            para = Paragraph(element, word)

            # ── 先检查段落内是否有嵌入图片 ──────────────────
            for img_bytes, img_ext in _extract_docx_paragraph_images(element, word):
                img_text = _ocr_bytes(img_bytes, img_ext)
                if img_text.strip():
                    for t in smart_chunk_text(clean_text(img_text)):
                        if t.strip():
                            chunks.append(Chunk(
                                text=t, file_name=fname, file_path=file_path,
                                file_hash=fhash, page=1, block_type="image",
                                heading_path=_heading_path(heading_stack),
                                chunk_index=idx
                            ))
                            idx += 1

            # ── 段落文字 ──────────────────────────────────────
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
            # 表格
            from docx.table import Table
            table = Table(element, word)
            rows = table.rows
            if not rows:
                continue

            headers = [c.text.strip() for c in rows[0].cells]
            data_rows = []
            for row in rows[1:]:
                r = [c.text.strip() for c in row.cells]
                if any(r):
                    data_rows.append(" | ".join(r))

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

def parse_excel(file_path: str) -> list[Chunk]:
    """每个 Sheet 作为一个 table chunk，保留表头和数字精度"""
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
        data_rows = []
        for row in rows[1:]:
            r = [_fmt_cell(c) for c in row]
            if any(r):
                data_rows.append(" | ".join(r))

        table_text = f"[Sheet: {sheet_name}]\n"
        table_text += " | ".join(headers) + "\n"
        table_text += "\n".join(data_rows)

        chunks.append(Chunk(
            text=table_text, file_name=fname, file_path=file_path,
            file_hash=fhash, page=idx + 1, block_type="table",
            table_headers=headers, heading_path=f"Sheet: {sheet_name}",
            chunk_index=idx
        ))

    return chunks


def _fmt_cell(val) -> str:
    if val is None: return ""
    if isinstance(val, float):
        if val == int(val): return str(int(val))
        import decimal
        return str(decimal.Decimal(str(val)).normalize())
    return str(val).strip()


# ── 图像预处理 ────────────────────────────────────────────────

def _preprocess_image(img: "Image.Image") -> "Image.Image":
    """
    图像预处理，提升 OCR 识别率
    灰度 → 放大到最小宽度 → 锐化 → 二值化
    """
    if not HAS_PIL:
        return img

    # 1. 转灰度
    img = img.convert("L")

    # 2. 放大（低分辨率图识别率差）
    w, h = img.size
    if w < 1200:
        scale = 1200 / w
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

    # 3. 锐化
    img = img.filter(ImageFilter.SHARPEN)

    # 4. 对比度增强
    img = ImageEnhance.Contrast(img).enhance(2.0)

    # 5. 二值化（简单阈值，适合大多数文档）
    threshold = 140
    img = img.point(lambda p: 255 if p > threshold else 0, "1")

    return img


def _ocr_image_obj(img: "Image.Image") -> str:
    """对 PIL Image 对象做 OCR，内部统一入口"""
    img = _preprocess_image(img)

    if HAS_PADDLE:
        # PaddleOCR 需要文件路径，存临时文件
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            img.convert("RGB").save(f.name)
            tmp = f.name
        try:
            return _ocr_with_paddle(tmp)
        finally:
            os.unlink(tmp)
    elif HAS_TESSERACT:
        if _TESS_LANG == "eng":
            print("    ⚠️  未检测到中文语言包，OCR 中文效果差。")
            print("       安装方法: sudo apt-get install tesseract-ocr-chi-sim")
            print("       或:       pip install paddlepaddle paddleocr  (效果更好)")
        return pytesseract.image_to_string(img, lang=_TESS_LANG,
                                            config="--psm 6")
    return ""


def _ocr_bytes(img_bytes: bytes, ext: str = ".png") -> str:
    """对原始图片字节做 OCR"""
    if not HAS_PIL:
        return ""
    try:
        img = Image.open(io.BytesIO(img_bytes))
        return _ocr_image_obj(img)
    except Exception as e:
        print(f"    ⚠️  图片 OCR 失败: {e}")
        return ""


# ── 独立图片文件 OCR ──────────────────────────────────────────

def parse_image(file_path: str) -> list[Chunk]:
    """
    独立图片文件 OCR（png/jpg/bmp/tiff 等）
    加了图像预处理，识别率明显优于直接 OCR
    """
    fhash = file_sha256(file_path)
    fname = os.path.basename(file_path)

    if not HAS_PIL:
        print(f"⚠️  Pillow 未安装，跳过图片 {fname}: pip install Pillow")
        return []
    if not HAS_PADDLE and not HAS_TESSERACT:
        print(f"⚠️  没有可用的 OCR 引擎，跳过 {fname}")
        print("    安装选项（任选一）:")
        print("    1. Tesseract: sudo apt-get install tesseract-ocr tesseract-ocr-chi-sim")
        print("    2. PaddleOCR: pip install paddlepaddle paddleocr")
        return []

    # 读取图片
    try:
        img = Image.open(file_path)
        print(f"    🖼  OCR 图片 {fname} ({img.size[0]}x{img.size[1]}, {img.mode})")
    except Exception as e:
        print(f"    ❌ 图片文件无法读取 {fname}: {e}")
        return []

    # 执行 OCR
    try:
        text = _ocr_image_obj(img)
    except Exception as e:
        print(f"    ❌ OCR 执行失败 {fname}: {e}")
        return []

    if not text.strip():
        print(f"    ⚠️  OCR 未识别到文字: {fname}")
        return []

    chunks = []
    for idx, t in enumerate(smart_chunk_text(clean_text(text))):
        if t.strip():
            chunks.append(Chunk(
                text=t, file_name=fname, file_path=file_path,
                file_hash=fhash, page=1, block_type="image", chunk_index=idx
            ))
    print(f"    ✅ 提取 {len(chunks)} 个 chunk")
    return chunks


# ── PDF 页内嵌入图片提取 ──────────────────────────────────────

def _extract_pdf_page_images(page, fname: str, file_path: str, fhash: str,
                               page_num: int, heading_path: str,
                               start_idx: int) -> list[Chunk]:
    """
    提取 PDF 页面内嵌的图片并 OCR
    只处理尺寸足够大的图片（过滤装饰性小图标）
    """
    chunks = []
    if not HAS_PYMUPDF or (not HAS_PADDLE and not HAS_TESSERACT):
        return chunks

    MIN_WIDTH, MIN_HEIGHT = 100, 100  # 像素，过滤小图标

    try:
        img_list = page.get_images(full=True)
        seen_xref = set()

        for img_info in img_list:
            xref = img_info[0]
            if xref in seen_xref:
                continue
            seen_xref.add(xref)

            try:
                base_img = page.parent.extract_image(xref)
                img_bytes = base_img["image"]
                img_ext   = "." + base_img.get("ext", "png")
                width     = base_img.get("width", 0)
                height    = base_img.get("height", 0)

                if width < MIN_WIDTH or height < MIN_HEIGHT:
                    continue  # 跳过装饰性小图

                print(f"    🖼  PDF 嵌入图片 xref={xref} ({width}x{height}) page={page_num}")
                img_text = _ocr_bytes(img_bytes, img_ext)

                if img_text.strip():
                    for t in smart_chunk_text(clean_text(img_text)):
                        if t.strip():
                            chunks.append(Chunk(
                                text=t, file_name=fname, file_path=file_path,
                                file_hash=fhash, page=page_num,
                                block_type="image", heading_path=heading_path,
                                chunk_index=start_idx + len(chunks)
                            ))
            except Exception as e:
                print(f"    ⚠️  PDF 图片提取失败 xref={xref}: {e}")
    except Exception as e:
        print(f"  ⚠️  PDF 页面图片列表获取失败 page={page_num}: {e}")

    return chunks


# ── Word 嵌入图片提取 ──────────────────────────────────────────

def _extract_docx_paragraph_images(element, word_doc) -> list[tuple[bytes, str]]:
    """
    从 Word 段落 XML 元素中提取嵌入图片的原始字节
    返回 [(image_bytes, ext), ...]

    Word 图片存储在 /word/media/ 关系里，通过 blip r:embed 引用
    """
    results = []
    try:
        # 查找所有 a:blip 标签（嵌入图片引用）
        ns = {
            "a":   "http://schemas.openxmlformats.org/drawingml/2006/main",
            "r":   "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
            "pic": "http://schemas.openxmlformats.org/drawingml/2006/picture",
        }

        blips = element.findall(".//a:blip", ns)
        if not blips:
            return results

        # 获取文档关系映射 (rId → 图片路径)
        part = word_doc.part
        rels = part.rels

        for blip in blips:
            embed = blip.get("{http://schemas.openxmlformats.org/officeDocument/2006/relationships}embed")
            if not embed or embed not in rels:
                continue

            rel = rels[embed]
            if "image" not in rel.reltype.lower():
                continue

            try:
                img_part = rel.target_part
                img_bytes = img_part.blob
                ext = os.path.splitext(img_part.partname)[-1].lower() or ".png"
                results.append((img_bytes, ext))
                print(f"    🖼  Word 嵌入图片 rId={embed} ({len(img_bytes)//1024}KB)")
            except Exception as e:
                print(f"    ⚠️  Word 图片读取失败 rId={embed}: {e}")

    except Exception as e:
        print(f"  ⚠️  Word 图片解析失败: {e}")

    return results


# ── OCR 底层实现 ──────────────────────────────────────────────

def _ocr_page(page) -> str:
    """对 PyMuPDF 的整页做 OCR（扫描页专用，300 DPI 渲染）"""
    try:
        mat = fitz.Matrix(2, 2)  # 2x 缩放 ≈ 144 DPI，够用且不太慢
        pix = page.get_pixmap(matrix=mat)
        img_bytes = pix.tobytes("png")
        return _ocr_bytes(img_bytes, ".png")
    except Exception as e:
        print(f"  OCR 失败: {e}")
    return ""


def _ocr_with_paddle(img_path: str) -> str:
    global _paddle_instance
    if _paddle_instance is None:
        _paddle_instance = PaddleOCR(use_angle_cls=True, lang="ch")
    # 尝试不同的参数组合
    try:
        # 尝试最新的API
        result = _paddle_instance.ocr(img_path, angle_classification=True)
    except TypeError:
        try:
            # 尝试旧版API
            result = _paddle_instance.ocr(img_path)
        except Exception as e:
            print(f"    ❌ OCR 执行失败: {e}")
            return ""
    lines = []
    if result:
        for line in result:
            if line:
                for item in line:
                    if item and len(item) >= 2:
                        lines.append(item[1][0])
    return "\n".join(lines)


# ── 标题栈管理 ────────────────────────────────────────────────

def _update_heading_stack(stack: list, level: int, text: str):
    """维护标题路径栈，保证层级一致"""
    while stack and stack[-1][0] >= level:
        stack.pop()
    stack.append((level, text))


def _heading_path(stack: list) -> str:
    return " > ".join(t for _, t in stack)


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
    """解析单个文件，返回 Chunk 列表"""
    ext = os.path.splitext(file_path)[1].lower()
    parser = SUPPORTED_EXTS.get(ext)
    if not parser:
        print(f"⚠️  不支持的格式: {file_path}")
        return []
    print(f"  📄 解析 {os.path.basename(file_path)} ({ext})")
    return parser(file_path)


def parse_directory(dir_path: str) -> list[Chunk]:
    """解析目录下所有支持的文件"""
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
