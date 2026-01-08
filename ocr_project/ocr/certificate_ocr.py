"""Specialized OCR processing for certificates and formal documents"""

import cv2
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Union, Tuple, Optional
import pytesseract

from .preprocess import (
    load_image, pil_to_cv2, cv2_to_pil, to_grayscale, 
    apply_threshold, denoise_image, enhance_contrast, resize_image
)


def preprocess_certificate(image_input: Union[str, Path, Image.Image],
                          enhance_text_areas: bool = False,
                          remove_decorative_elements: bool = False,
                          high_dpi_scaling: bool = True) -> Image.Image:
    """
    Specialized preprocessing for certificates and formal documents.
    
    Args:
        image_input: Image path or PIL Image object
        enhance_text_areas: Apply text area enhancement
        remove_decorative_elements: Try to remove decorative borders/elements
        high_dpi_scaling: Use high DPI scaling for better text recognition
        
    Returns:
        Preprocessed PIL Image optimized for certificate OCR
    """
    # Load image
    if isinstance(image_input, (str, Path)):
        image = load_image(image_input)
    else:
        image = image_input
    
    # Convert to OpenCV format
    cv_image = pil_to_cv2(image)
    original_height, original_width = cv_image.shape[:2]
    
    # High DPI scaling for better text recognition
    if high_dpi_scaling:
        # More conservative scaling for certificates
        scale_factor = 2.0 if min(original_height, original_width) < 800 else 1.5
        cv_image = resize_image(cv_image, scale_factor=scale_factor)
    
    # Convert to grayscale
    if len(cv_image.shape) == 3:
        gray = to_grayscale(cv_image)
    else:
        gray = cv_image.copy()
    
    # Very light enhancement - preserve original quality
    enhanced = enhance_contrast(gray, alpha=1.2, beta=5)
    
    # Light denoising only if needed
    denoised = cv2.bilateralFilter(enhanced, 5, 50, 50)
    
    # Apply morphological operations to enhance text
    if enhance_text_areas:
        # Create kernel for morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
        
        # Apply very light closing to connect broken text
        closed = cv2.morphologyEx(denoised, cv2.MORPH_CLOSE, kernel)
        denoised = closed
    
    # For certificates, often the original image is already well-processed
    # Try light adaptive threshold only if the image appears to need it
    mean_brightness = np.mean(denoised)
    
    if mean_brightness < 200:  # Only threshold if image is not already high contrast
        # Apply very light adaptive threshold
        thresh = cv2.adaptiveThreshold(
            denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 2
        )
        # Blend with original for softer effect
        final_image = cv2.addWeighted(denoised, 0.7, thresh, 0.3, 0)
    else:
        # Keep original processing for high-quality images
        final_image = denoised
    
    # Convert back to PIL
    processed_image = Image.fromarray(final_image, mode='L')
    
    return processed_image


def extract_certificate_text(image_input: Union[str, Path, Image.Image],
                           extract_structured: bool = True,
                           custom_config: Optional[str] = None) -> dict:
    """
    Extract text from certificates with specialized configuration.
    
    Args:
        image_input: Image path or PIL Image object
        extract_structured: Whether to attempt structured extraction
        custom_config: Custom Tesseract configuration
        
    Returns:
        Dictionary containing extracted text and structured data
    """
    # First try raw OCR without preprocessing
    if isinstance(image_input, (str, Path)):
        raw_image = load_image(image_input)
    else:
        raw_image = image_input
    
    # Preprocess the certificate with conservative settings
    processed_image = preprocess_certificate(image_input)
    
    # Certificate-optimized Tesseract configuration
    if custom_config is None:
        configs = [
            '--psm 6',  # Uniform block of text
            '--psm 3',  # Fully automatic page segmentation 
            '--psm 4',  # Single column text
            '--psm 1',  # Automatic page segmentation with OSD
        ]
    else:
        configs = [custom_config]
    
    results = {}
    best_text = ""
    best_confidence = 0
    best_source = "raw"
    
    # Test both raw and processed images with different configurations
    for img_type, image in [("raw", raw_image), ("processed", processed_image)]:
        for i, config in enumerate(configs):
            method_name = f'{img_type}_method_{i+1}'
            try:
                text = pytesseract.image_to_string(image, config=config, lang='eng')
                
                # Get confidence scores
                data = pytesseract.image_to_data(image, config=config, lang='eng', output_type=pytesseract.Output.DICT)
                confidences = [int(conf) for conf in data['conf'] if int(conf) > 0]
                avg_confidence = sum(confidences) / len(confidences) if confidences else 0
                
                results[method_name] = {
                    'text': text.strip(),
                    'confidence': avg_confidence,
                    'config': config,
                    'image_type': img_type
                }
                
                # Prefer longer text with reasonable confidence
                text_length = len(text.strip())
                quality_score = avg_confidence * (1 + text_length / 1000)
                
                if quality_score > (best_confidence * (1 + len(best_text) / 1000)):
                    best_confidence = avg_confidence
                    best_text = text.strip()
                    best_source = img_type
                    
            except Exception as e:
                results[method_name] = {
                    'text': '',
                    'confidence': 0,
                    'error': str(e),
                    'config': config,
                    'image_type': img_type
                }
    
    # Structure extraction for certificates
    structured_data = {}
    if extract_structured and best_text:
        structured_data = extract_certificate_structure(best_text)
    
    return {
        'best_text': best_text,
        'best_confidence': best_confidence,
        'best_source': best_source,
        'all_results': results,
        'structured_data': structured_data
    }


def extract_certificate_structure(text: str) -> dict:
    """
    Extract structured information from certificate text.
    
    Args:
        text: Raw extracted text
        
    Returns:
        Dictionary with structured certificate data
    """
    import re
    
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    
    structured = {
        'certificate_type': '',
        'recipient_name': '',
        'organization': '',
        'date': '',
        'signature_line': '',
        'other_text': []
    }
    
    # Patterns for certificate detection
    cert_patterns = {
        'certificate_type': [
            r'CERTIFICATE\s+OF\s+(\w+)',
            r'(\w+)\s+CERTIFICATE',
            r'CERTIFICATE'
        ],
        'recipient_name': [
            r'PRESENTED\s+TO\s+([A-Za-z\s]+)',
            r'AWARDED\s+TO\s+([A-Za-z\s]+)',
            r'THIS\s+CERTIFIES\s+THAT\s+([A-Za-z\s]+)',
            r'Name\s+([A-Za-z\s]+)',
        ],
        'date': [
            r'\b(\d{1,2}[/-]\d{1,2}[/-]\d{4})\b',
            r'\b(\d{4}[/-]\d{1,2}[/-]\d{1,2})\b',
            r'\b((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{4})\b',
            r'\b(\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{4})\b'
        ]
    }
    
    text_upper = text.upper()
    
    # Extract certificate type
    for pattern in cert_patterns['certificate_type']:
        match = re.search(pattern, text_upper)
        if match:
            if len(match.groups()) > 0:
                structured['certificate_type'] = match.group(1)
            else:
                structured['certificate_type'] = 'CERTIFICATE'
            break
    
    # Extract recipient name
    for pattern in cert_patterns['recipient_name']:
        match = re.search(pattern, text_upper)
        if match:
            name = match.group(1).strip()
            # Clean up the name
            name = re.sub(r'\s+', ' ', name)
            if len(name) > 2 and not re.match(r'^[A-Z\s]+$', name):
                # Convert to title case if not all caps
                name = name.title()
            structured['recipient_name'] = name
            break
    
    # Extract dates
    for pattern in cert_patterns['date']:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            structured['date'] = match.group(1)
            break
    
    # Look for signature line
    signature_keywords = ['signature', 'signed', 'authorized', 'director', 'president', 'manager']
    for line in lines:
        if any(keyword in line.lower() for keyword in signature_keywords):
            structured['signature_line'] = line
            break
    
    # Store remaining text
    structured['other_text'] = [line for line in lines if line.lower() not in [
        structured['certificate_type'].lower(),
        structured['recipient_name'].lower(),
        structured['date'].lower(),
        structured['signature_line'].lower()
    ] and len(line) > 3]
    
    return structured


def enhance_certificate_image_quality(image_input: Union[str, Path, Image.Image]) -> Image.Image:
    """
    Apply advanced image enhancement specifically for certificates.
    
    Args:
        image_input: Image path or PIL Image object
        
    Returns:
        Enhanced PIL Image
    """
    # Load image
    if isinstance(image_input, (str, Path)):
        image = load_image(image_input)
    else:
        image = image_input
    
    cv_image = pil_to_cv2(image)
    
    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    if len(cv_image.shape) == 3:
        # Convert to LAB color space for better contrast enhancement
        lab = cv2.cvtColor(cv_image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        l = clahe.apply(l)
        
        # Merge channels back
        enhanced = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
    else:
        # Grayscale image
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        enhanced = clahe.apply(cv_image)
    
    # Apply unsharp masking for text sharpening
    if len(enhanced.shape) == 3:
        gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
    else:
        gray = enhanced
    
    # Create unsharp mask
    gaussian = cv2.GaussianBlur(gray, (0, 0), 2.0)
    unsharp_mask = cv2.addWeighted(gray, 1.5, gaussian, -0.5, 0)
    
    return Image.fromarray(unsharp_mask, mode='L')


if __name__ == '__main__':
    # Test certificate OCR
    import sys
    
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        print(f"Processing certificate: {image_path}")
        
        result = extract_certificate_text(image_path)
        
        print(f"\nBest Confidence: {result['best_confidence']:.1f}%")
        print(f"\nExtracted Text:")
        print("-" * 50)
        print(result['best_text'])
        
        if result['structured_data']:
            print(f"\nStructured Data:")
            print("-" * 50)
            for key, value in result['structured_data'].items():
                if value:
                    print(f"{key.replace('_', ' ').title()}: {value}")
    else:
        print("Usage: python certificate_ocr.py <image_path>")