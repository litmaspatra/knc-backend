"""
KnowYourCase Backend — FastAPI
Two endpoints:
  POST /solve-captcha  — receives a CAPTCHA image and returns text using
                         a local CAPTCHA-specific ONNX model
  POST /parse          — receives raw HTML from WebView + CNR, returns structured JSON
                         (parser ported from ecourts/parsers/case_details.py)
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import re
import datetime
import base64
import tempfile
import os
import subprocess
from functools import lru_cache
from bs4 import BeautifulSoup
from typing import Optional

try:
    import cv2
    import numpy as np
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False

try:
    import ddddocr
    DDDDOCR_AVAILABLE = True
except ImportError:
    DDDDOCR_AVAILABLE = False

app = FastAPI(title="KnowYourCase Parser API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["POST","GET"], allow_headers=["*"])

# ── Models ────────────────────────────────────────────────────────────────────

class CaptchaRequest(BaseModel):
    image_b64: str

class CaptchaResponse(BaseModel):
    solved: Optional[str] = None
    confidence: str  # "high" | "low" | "failed"

class ParseRequest(BaseModel):
    cnr: str
    html: str

class HearingItem(BaseModel):
    date: Optional[str] = None
    judge: Optional[str] = None
    purpose: Optional[str] = None
    next_date: Optional[str] = None

class CaseResponse(BaseModel):
    cnr: str
    case_title: Optional[str] = None
    case_number: Optional[str] = None
    case_type: Optional[str] = None
    court_name: Optional[str] = None
    filing_number: Optional[str] = None
    filing_date: Optional[str] = None
    registration_number: Optional[str] = None
    registration_date: Optional[str] = None
    first_hearing_date: Optional[str] = None
    decision_date: Optional[str] = None
    next_hearing_date: Optional[str] = None
    petitioner: Optional[str] = None
    respondent: Optional[str] = None
    petitioner_advocate: Optional[str] = None
    respondent_advocate: Optional[str] = None
    status: Optional[str] = None
    nature_of_disposal: Optional[str] = None
    judges: list = []
    hearings: list = []
    data_completeness: str = "partial"

# ── CAPTCHA solver (ported 1:1 from ecourts/captcha.py by captn3m0) ──────────

CAPTCHA_THRESHOLD = 0.4   # captcha.py: THRESHOLD = 0.4
MAX_PIXEL_VALUE   = 255   # captcha.py: MAX_PIXEL_VALUE = 255


@lru_cache(maxsize=1)
def _captcha_ocr():
    if not DDDDOCR_AVAILABLE:
        return None
    engine = ddddocr.DdddOcr(show_ad=False)
    # eCourts CAPTCHAs use lowercase Latin letters and digits.
    engine.set_ranges(4)
    return engine


def _solve_with_ddddocr(img_bytes: bytes) -> Optional[str]:
    """Run the free CAPTCHA-specific ONNX model in the Render process."""
    engine = _captcha_ocr()
    if engine is None:
        return None
    try:
        text = engine.classification(img_bytes)
        clean = re.sub(r"[^A-Za-z0-9]", "", text or "")
        return clean.lower() if len(clean) == 6 else None
    except Exception:
        return None

def _decaptcha_opencv(img_bytes: bytes) -> Optional[str]:
    """
    Direct port of Captcha.decaptcha() from ecourts/captcha.py.
    Steps mirror the original exactly, including the crop [15:65, 27:190].
    """
    if not OPENCV_AVAILABLE:
        return None
    nparr = np.frombuffer(img_bytes, np.uint8)
    src = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if src is None:
        return None

    threshold_val = int(MAX_PIXEL_VALUE * CAPTCHA_THRESHOLD)

    # 1. Binary threshold
    _, threshold_img = cv2.threshold(src, threshold_val, MAX_PIXEL_VALUE, cv2.THRESH_BINARY)

    # 2. Mask for grey noise lines (colour #707070 from captcha.py)
    lines_color = np.array([0x70, 0x70, 0x70], dtype=np.uint8)
    binary_mask = cv2.inRange(src, lines_color, lines_color)
    masked = cv2.bitwise_and(src, src, mask=binary_mask)

    # 3. Dilate mask lines
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    masked = cv2.dilate(masked, kernel)

    # 4. Inpaint (remove noise lines) — captcha.py: cv2.inpaint(..., 7, cv2.INPAINT_NS)
    masked_gray = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)
    dst = cv2.inpaint(threshold_img, masked_gray, 7, cv2.INPAINT_NS)

    # 5. Dilate small remaining lines
    dst = cv2.dilate(dst, kernel)

    # 6. Gaussian blur + bilateral filter
    dst = cv2.GaussianBlur(dst, (5, 5), 0)
    dst = cv2.bilateralFilter(dst, 5, 75, 75)

    # 7. Grayscale + Otsu threshold
    dst = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
    _, dst = cv2.threshold(dst, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 8. Crop to text region — EXACT from captcha.py line 68
    dst = dst[15:65, 27:190]

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        tmp_path = f.name
    cv2.imwrite(tmp_path, dst)
    try:
        return _run_tesseract(tmp_path)
    finally:
        try: os.unlink(tmp_path)
        except: pass

def _run_tesseract(image_path: str) -> Optional[str]:
    """
    Mirrors captcha.py Popen call:
      tesseract <file> stdout --oem 1 --psm 8 -c tessedit_char_whitelist=abc...0-9
    captcha.py validates len==5; we relax to 4-6 for district courts.
    """
    try:
        proc = subprocess.Popen(
            ["tesseract", image_path, "stdout",
             "--oem", "1", "--psm", "8",
             "-c", "tessedit_char_whitelist=abcdefghijklmnopqrstuvwxyz0123456789"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        proc.wait(timeout=10)
        out, _ = proc.communicate()
        text = out.decode("utf-8").strip()
        clean = re.sub(r"\s+", "", text)
        if 4 <= len(clean) <= 6:
            return clean
        return None
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return None

# ── HTML parser (ported from ecourts/parsers/case_details.py) ─────────────────

def _dt(s: str) -> Optional[str]:
    if not s: return None
    s = s.strip()
    for fmt in ("%d-%m-%Y","%d/%m/%Y","%Y-%m-%d","%dth %B %Y","%dst %B %Y","%dnd %B %Y","%d %B %Y"):
        try: return datetime.datetime.strptime(s, fmt).strftime("%Y-%m-%d")
        except ValueError: pass
    return s

def _t(el) -> str:
    return el.get_text(separator=" ", strip=True).replace("\xa0"," ").strip().lstrip(":").strip() if el else ""

def parse_ecourts_html(cnr: str, html: str) -> CaseResponse:
    soup = BeautifulSoup(html, "html.parser")
    for br in soup.find_all("br"): br.replace_with("\n")
    r = CaseResponse(cnr=cnr)
    filled = 0

    # ── Row-based extraction (works for both district and HC layouts) ─────
    for row in soup.find_all("tr"):
        cells = row.find_all(["td","th"])
        if len(cells) < 2: continue
        key = _t(cells[0]).lower()
        val = _t(cells[-1])
        if not val or len(val) > 300: continue
        if   "case type"            in key: r.case_type = val; filled += 1
        elif "filing number"        in key or "filing no" in key: r.filing_number = val; filled += 1
        elif "filing date"          in key: r.filing_date = _dt(val); filled += 1
        elif "registration number"  in key or "registration no" in key: r.registration_number = val; filled += 1
        elif "registration date"    in key: r.registration_date = _dt(val); filled += 1
        elif "court name"           in key and not r.court_name: r.court_name = val; filled += 1
        elif "case no"              in key and not r.case_number: r.case_number = val; filled += 1
        elif "first hearing"        in key: r.first_hearing_date = _dt(val); filled += 1
        elif "decision date"        in key: r.decision_date = _dt(val); filled += 1
        elif "case status"          in key or "stage of case" in key:
            if not r.status: r.status = val; filled += 1
        elif "nature of disposal"   in key: r.nature_of_disposal = val; filled += 1
        elif "coram"                in key or ("judge" in key and "next" not in key):
            if not r.judges: r.judges = [v.strip() for v in val.split(",") if v.strip()]; filled += 1
        elif "not before me"        in key or "next hearing" in key or "next date" in key:
            if not r.next_hearing_date: r.next_hearing_date = _dt(val); filled += 1

    # ── Span/label pairs (mirrors extract_span_label_dict from case_details.py) ──
    KEYS = ["Number","Station","District","Year","State","Type","Date","Status","Court","Name"]
    for label in soup.select("label"):
        kt = label.get_text(strip=True)
        if not any(k.lower() in kt.lower() for k in KEYS): continue
        value = ""
        for sib in label.next_siblings:
            if getattr(sib,"name",None):
                if sib.name == "label": value = _t(sib)
                break
        if not value and ":" in label.parent.get_text():
            parts = label.parent.get_text().split(":",1)
            if len(parts)>1: value = parts[1].replace("\xa0"," ").strip()
        if not value: continue
        k = kt.lower()
        if "court"     in k and not r.court_name:      r.court_name = value;                       filled += 1
        elif "case type" in k and not r.case_type:     r.case_type = value;                        filled += 1
        elif "filing date" in k and not r.filing_date: r.filing_date = _dt(value);                 filled += 1
        elif "reg" in k and "date" in k and not r.registration_date: r.registration_date = _dt(value); filled += 1
        elif "status"  in k and not r.status:          r.status = value;                           filled += 1

    # ── Parties (mirrors extract_parties from case_details.py) ───────────
    def get_party(cls_name):
        span = soup.find("span", class_=cls_name)
        if not span:
            for s in soup.find_all("span"):
                if cls_name.lower() in " ".join(s.get("class",[])).lower():
                    span = s; break
        if not span: return None, None
        txt = span.get_text(separator="\n",strip=True).replace("\xa0","")
        name = adv = None
        for line in txt.split("\n"):
            line = line.strip()
            m = re.match(r"^\d+\)\s*(.+)", line)
            if m and not name:            name = m.group(1).strip()
            elif re.search(r"advocate\s*[-–]", line, re.I) and not adv:
                adv = re.split(r"advocate\s*[-–]", line, flags=re.I)[-1].strip()
        return name, adv

    pn, pa = get_party("Petitioner_Advocate_table")
    rn, ra = get_party("Respondent_Advocate_table")
    if pn: r.petitioner = pn; filled += 1
    if pa: r.petitioner_advocate = pa; filled += 1
    if rn: r.respondent = rn; filled += 1
    if ra: r.respondent_advocate = ra; filled += 1

    # Case title
    if r.petitioner and r.respondent:
        r.case_title = f"{r.petitioner.split(chr(10))[0].strip()} vs {r.respondent.split(chr(10))[0].strip()}"
    elif r.petitioner:
        r.case_title = r.petitioner.split("\n")[0].strip()

    # ── Hearing history (mirrors extract_hearing from case_details.py) ────
    hh = soup.find("table", id="historyheading")
    if hh:
        ht = hh.find_next_sibling("table") or hh.find_next("table")
        if ht:
            for row in ht.find_all("tr")[1:][:20]:
                cells = row.find_all("td")
                if len(cells) < 4: continue
                r.hearings.append({
                    "judge":     _t(cells[1]) or None,
                    "date":      _dt(_t(cells[2])),
                    "next_date": _dt(_t(cells[3])),
                    "purpose":   _t(cells[4]) if len(cells)>4 else None,
                })
            if r.hearings and not r.next_hearing_date:
                r.next_hearing_date = r.hearings[-1].get("next_date")

    if not r.judges and r.hearings:
        seen: list = []
        for h in r.hearings:
            j = h.get("judge") if isinstance(h,dict) else getattr(h,"judge",None)
            if j and j not in seen: seen.append(j)
        r.judges = seen[:3]

    r.data_completeness = "full" if filled>=10 else "partial" if filled>=4 else "minimal"
    return r

# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "service": "KnowYourCase Parser API",
        "status": "ok",
        "captcha_solver": "ddddocr" if DDDDOCR_AVAILABLE else "tesseract-fallback",
        "opencv": OPENCV_AVAILABLE,
    }

@app.get("/health")
def health():
    return {"status":"ok"}

@app.post("/solve-captcha", response_model=CaptchaResponse)
def solve_captcha(req: CaptchaRequest):
    """
    Accepts a base64 CAPTCHA image. Uses ddddocr first, with the legacy
    OpenCV/Tesseract pipeline as a fallback.
    """
    try:
        img_bytes = base64.b64decode(req.image_b64)
    except Exception:
        raise HTTPException(400, "Invalid base64 image")
    solved = _solve_with_ddddocr(img_bytes)
    if solved:
        return CaptchaResponse(solved=solved, confidence="high")

    solved = _decaptcha_opencv(img_bytes) if OPENCV_AVAILABLE else None
    if solved:
        return CaptchaResponse(solved=solved, confidence="low")
    return CaptchaResponse(solved=None, confidence="failed")

@app.post("/parse", response_model=CaseResponse)
def parse_case(req: ParseRequest):
    """
    Accepts raw eCourts result HTML captured by the Android WebView + CNR.
    Returns structured case data using ecourts/parsers/case_details.py logic.
    """
    cnr = req.cnr.strip().replace("-","").upper()
    if len(cnr) != 16:
        raise HTTPException(400, "Invalid CNR: must be 16 characters")
    if not req.html or len(req.html) < 200:
        raise HTTPException(400, "HTML too short — likely an error page")
    try:
        result = parse_ecourts_html(cnr, req.html)
    except Exception as e:
        raise HTTPException(500, f"Parse error: {e}")
    if not result.case_type and not result.petitioner and not result.status:
        raise HTTPException(422, "Could not extract case data from HTML")
    return result
