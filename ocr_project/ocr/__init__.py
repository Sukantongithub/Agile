"""OCR Image to Text Extraction Package"""

from .ocr import extract_text, extract_text_from_file
from .preprocess import preprocess_image, load_image, preprocess_for_text_type
from .certificate_ocr import extract_certificate_text, preprocess_certificate, enhance_certificate_image_quality
from .mrz_ocr import extract_mrz_text, detect_mrz_region, preprocess_mrz_image, MRZParser, extract_raw_mrz_text
from .document_types import DocumentTypeDetector, DocumentProcessor

__version__ = "1.0.0"
__all__ = [
    "extract_text", "extract_text_from_file", 
    "preprocess_image", "load_image", "preprocess_for_text_type",
    "extract_certificate_text", "preprocess_certificate", "enhance_certificate_image_quality",
    "extract_mrz_text", "detect_mrz_region", "preprocess_mrz_image", "MRZParser", "extract_raw_mrz_text",
    "DocumentTypeDetector", "DocumentProcessor"
]