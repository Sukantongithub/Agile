"""MRZ (Machine Readable Zone) OCR processing for passports and ID documents"""

import re
import cv2
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Union, Dict, Optional, Tuple
import pytesseract

from .preprocess import load_image, pil_to_cv2, cv2_to_pil, to_grayscale


class MRZParser:
    """Parser for MRZ (Machine Readable Zone) data from passports and ID documents."""
    
    def __init__(self):
        self.mrz_patterns = {
            'passport': {
                'line1': r'^P[A-Z]{3}([A-Z0-9<]{39})$',
                'line2': r'^([A-Z0-9<]{9})([0-9]{1})([A-Z]{3})([0-9]{6})([0-9]{1})([MF<]{1})([0-9]{6})([0-9]{1})([A-Z0-9<]{14})([0-9]{1})$'
            },
            'id_card': {
                'line1': r'^I[A-Z]{3}([A-Z0-9<]{25})([0-9]{1})$',
                'line2': r'^([0-9]{6})([0-9]{1})([MF<]{1})([0-9]{6})([0-9]{1})([A-Z]{3})([A-Z0-9<]{11})([0-9]{1})$',
                'line3': r'^([A-Z0-9<]{30})$'
            },
            'visa': {
                'line1': r'^V[A-Z]{3}([A-Z0-9<]{25})([0-9]{1})$',
                'line2': r'^([A-Z0-9<]{9})([0-9]{1})([A-Z]{3})([0-9]{6})([0-9]{1})([MF<]{1})([0-9]{6})([0-9]{1})([A-Z0-9<]{8})$'
            }
        }
    
    def detect_mrz_type(self, lines: list) -> str:
        """Detect the type of MRZ document based on patterns."""
        if not lines:
            return 'unknown'
        
        first_line = lines[0].upper().replace(' ', '').replace('-', '')
        
        if first_line.startswith('P'):
            return 'passport'
        elif first_line.startswith('I'):
            return 'id_card'
        elif first_line.startswith('V'):
            return 'visa'
        else:
            return 'unknown'
    
    def parse_passport_mrz(self, lines: list) -> dict:
        """Parse passport MRZ data."""
        if len(lines) < 2:
            return {'error': 'Insufficient MRZ lines for passport'}
        
        line1 = lines[0].upper().replace(' ', '').replace('-', '')
        line2 = lines[1].upper().replace(' ', '').replace('-', '')
        
        result = {
            'document_type': 'passport',
            'country_code': '',
            'surname': '',
            'given_names': '',
            'passport_number': '',
            'nationality': '',
            'date_of_birth': '',
            'sex': '',
            'expiry_date': '',
            'personal_number': '',
            'raw_lines': lines
        }
        
        # Parse line 1: P<COUNTRY<SURNAME<<GIVEN_NAMES<<<<<<<<<<<<<<<
        if line1.startswith('P'):
            country_and_names = line1[1:]  # Remove 'P'
            parts = country_and_names.split('<')
            
            if len(parts) >= 3:
                result['country_code'] = parts[0][:3] if parts[0] else ''
                result['surname'] = parts[1].replace('<', ' ').strip() if parts[1] else ''
                result['given_names'] = parts[2].replace('<', ' ').strip() if parts[2] else ''
        
        # Parse line 2: PASSPORT_NUMBER<CHECK<NATIONALITY<DOB<CHECK<SEX<EXPIRY<CHECK<PERSONAL_NUMBER<CHECK
        if len(line2) >= 44:
            result['passport_number'] = line2[:9].replace('<', '').strip()
            result['nationality'] = line2[10:13]
            
            # Date of birth (YYMMDD)
            dob = line2[13:19]
            if dob.isdigit():
                year = '19' + dob[:2] if int(dob[:2]) > 50 else '20' + dob[:2]
                result['date_of_birth'] = f"{year}-{dob[2:4]}-{dob[4:6]}"
            
            # Sex
            sex = line2[20]
            result['sex'] = 'Male' if sex == 'M' else 'Female' if sex == 'F' else 'Unknown'
            
            # Expiry date (YYMMDD)
            expiry = line2[21:27]
            if expiry.isdigit():
                year = '19' + expiry[:2] if int(expiry[:2]) > 50 else '20' + expiry[:2]
                result['expiry_date'] = f"{year}-{expiry[2:4]}-{expiry[4:6]}"
            
            # Personal number
            result['personal_number'] = line2[28:42].replace('<', '').strip()
        
        return result
    
    def parse_id_card_mrz(self, lines: list) -> dict:
        """Parse ID card MRZ data."""
        if len(lines) < 3:
            return {'error': 'Insufficient MRZ lines for ID card'}
        
        result = {
            'document_type': 'id_card',
            'country_code': '',
            'document_number': '',
            'date_of_birth': '',
            'sex': '',
            'expiry_date': '',
            'nationality': '',
            'surname': '',
            'given_names': '',
            'raw_lines': lines
        }
        
        # Parse the lines based on ID card format
        line1 = lines[0].upper().replace(' ', '').replace('-', '')
        line2 = lines[1].upper().replace(' ', '').replace('-', '')
        line3 = lines[2].upper().replace(' ', '').replace('-', '')
        
        # Line 1: I<COUNTRY<DOCUMENT_NUMBER<CHECK
        if line1.startswith('I'):
            country_and_doc = line1[1:]  # Remove 'I'
            parts = country_and_doc.split('<')
            if len(parts) >= 2:
                result['country_code'] = parts[0][:3] if parts[0] else ''
                result['document_number'] = parts[1].replace('<', '').strip() if parts[1] else ''
        
        # Line 2: DOB<CHECK<SEX<EXPIRY<CHECK<NATIONALITY<FILLER<CHECK
        if len(line2) >= 30:
            # Date of birth
            dob = line2[:6]
            if dob.isdigit():
                year = '19' + dob[:2] if int(dob[:2]) > 50 else '20' + dob[:2]
                result['date_of_birth'] = f"{year}-{dob[2:4]}-{dob[4:6]}"
            
            # Sex
            sex = line2[7] if len(line2) > 7 else ''
            result['sex'] = 'Male' if sex == 'M' else 'Female' if sex == 'F' else 'Unknown'
            
            # Expiry date
            expiry = line2[8:14] if len(line2) > 13 else ''
            if expiry.isdigit():
                year = '19' + expiry[:2] if int(expiry[:2]) > 50 else '20' + expiry[:2]
                result['expiry_date'] = f"{year}-{expiry[2:4]}-{expiry[4:6]}"
            
            # Nationality
            result['nationality'] = line2[15:18] if len(line2) > 17 else ''
        
        # Line 3: SURNAME<<GIVEN_NAMES<<<<<<<<<<<<<<<<
        name_parts = line3.split('<')
        if len(name_parts) >= 2:
            result['surname'] = name_parts[0].strip()
            result['given_names'] = name_parts[1].replace('<', ' ').strip() if name_parts[1] else ''
        
        return result
    
    def parse_mrz(self, text: str) -> dict:
        """Parse MRZ text and extract structured data."""
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        
        if not lines:
            return {'error': 'No MRZ text found'}
        
        # Clean lines - remove extra spaces and normalize
        cleaned_lines = []
        for line in lines:
            # Remove spaces and normalize characters
            cleaned = re.sub(r'\s+', '', line)
            # Replace common OCR mistakes
            cleaned = cleaned.replace('0', 'O').replace('1', 'I').replace('8', 'B')
            if len(cleaned) > 20:  # MRZ lines are typically long
                cleaned_lines.append(cleaned)
        
        if not cleaned_lines:
            return {'error': 'No valid MRZ lines found'}
        
        # Detect document type
        doc_type = self.detect_mrz_type(cleaned_lines)
        
        if doc_type == 'passport':
            return self.parse_passport_mrz(cleaned_lines)
        elif doc_type == 'id_card':
            return self.parse_id_card_mrz(cleaned_lines)
        elif doc_type == 'visa':
            return {'document_type': 'visa', 'raw_lines': cleaned_lines, 'note': 'Visa parsing not fully implemented'}
        else:
            return {
                'document_type': 'unknown',
                'raw_lines': cleaned_lines,
                'error': 'Unable to determine document type'
            }


def preprocess_mrz_image(image_input: Union[str, Path, Image.Image]) -> Image.Image:
    """
    Specialized preprocessing for MRZ regions.
    
    Args:
        image_input: Image path or PIL Image object
        
    Returns:
        Preprocessed PIL Image optimized for MRZ OCR
    """
    # Load image
    if isinstance(image_input, (str, Path)):
        image = load_image(image_input)
    else:
        image = image_input
    
    cv_image = pil_to_cv2(image)
    
    # Convert to grayscale
    if len(cv_image.shape) == 3:
        gray = to_grayscale(cv_image)
    else:
        gray = cv_image.copy()
    
    # Resize for better OCR (MRZ text is usually small)
    height, width = gray.shape
    if width < 1200:  # Scale up if image is small
        scale = 1200 / width
        new_width = int(width * scale)
        new_height = int(height * scale)
        gray = cv2.resize(gray, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    
    # Apply threshold to get binary image
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Apply morphological operations to clean up the image
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 1))
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    
    return Image.fromarray(cleaned, mode='L')


def detect_mrz_region(image_input: Union[str, Path, Image.Image]) -> Optional[Image.Image]:
    """
    Detect and extract MRZ region from document image.
    
    Args:
        image_input: Image path or PIL Image object
        
    Returns:
        Cropped MRZ region as PIL Image, or None if not found
    """
    # Load image
    if isinstance(image_input, (str, Path)):
        image = load_image(image_input)
    else:
        image = image_input
    
    cv_image = pil_to_cv2(image)
    gray = to_grayscale(cv_image) if len(cv_image.shape) == 3 else cv_image.copy()
    
    # Apply threshold
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Look for rectangular regions that might contain MRZ
    height, width = gray.shape
    mrz_candidates = []
    
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        
        # MRZ is typically in the bottom 30% of the document
        # and spans most of the width
        if (y > height * 0.7 and  # Bottom region
            w > width * 0.6 and   # Wide enough
            h > 20 and h < height * 0.2):  # Appropriate height
            
            mrz_candidates.append((x, y, w, h))
    
    if mrz_candidates:
        # Take the largest candidate (most likely to be MRZ)
        x, y, w, h = max(mrz_candidates, key=lambda rect: rect[2] * rect[3])
        
        # Add some padding
        padding = 10
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(width - x, w + 2 * padding)
        h = min(height - y, h + 2 * padding)
        
        # Crop the MRZ region
        mrz_region = gray[y:y+h, x:x+w]
        return Image.fromarray(mrz_region, mode='L')
    
    # If no MRZ region detected, return bottom 25% of image
    bottom_region = gray[int(height * 0.75):, :]
    return Image.fromarray(bottom_region, mode='L')


def extract_mrz_text(image_input: Union[str, Path, Image.Image],
                     auto_detect_region: bool = True) -> dict:
    """
    Extract and parse MRZ text from document image.
    
    Args:
        image_input: Image path or PIL Image object
        auto_detect_region: Whether to automatically detect MRZ region
        
    Returns:
        Dictionary containing parsed MRZ data
    """
    try:
        # Auto-detect MRZ region if requested
        if auto_detect_region:
            mrz_image = detect_mrz_region(image_input)
        else:
            mrz_image = image_input
        
        # Preprocess the MRZ image
        processed_image = preprocess_mrz_image(mrz_image)
        
        # OCR configuration optimized for MRZ
        mrz_config = '--psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789<'
        
        # Extract text
        raw_text = pytesseract.image_to_string(processed_image, config=mrz_config)
        
        # Parse MRZ data
        parser = MRZParser()
        parsed_data = parser.parse_mrz(raw_text)
        
        # Add OCR metadata
        parsed_data['raw_ocr_text'] = raw_text
        parsed_data['processing_method'] = 'MRZ-optimized OCR'
        
        return parsed_data
        
    except Exception as e:
        return {
            'error': f'MRZ extraction failed: {str(e)}',
            'document_type': 'unknown'
        }


def extract_raw_mrz_text(image_input: Union[str, Path, Image.Image],
                         auto_detect_region: bool = True) -> str:
    """
    Extract raw MRZ text from document image without parsing.
    
    Args:
        image_input: Image path or PIL Image object
        auto_detect_region: Whether to automatically detect MRZ region
        
    Returns:
        Raw MRZ text as string
    """
    try:
        # Auto-detect MRZ region if requested
        if auto_detect_region:
            mrz_image = detect_mrz_region(image_input)
        else:
            mrz_image = image_input
        
        # Preprocess the MRZ image
        processed_image = preprocess_mrz_image(mrz_image)
        
        # OCR configuration optimized for MRZ
        mrz_config = '--psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789<'
        
        # Extract text
        raw_text = pytesseract.image_to_string(processed_image, config=mrz_config)
        return raw_text.strip()
        
    except Exception as e:
        return f'MRZ extraction failed: {str(e)}'


if __name__ == '__main__':
    # Test MRZ extraction
    import sys
    
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        print(f"Processing MRZ from: {image_path}")
        
        result = extract_mrz_text(image_path)
        
        if 'error' in result:
            print(f"❌ Error: {result['error']}")
        else:
            print(f"\n📄 Document Type: {result.get('document_type', 'Unknown')}")
            
            if result['document_type'] == 'passport':
                print(f"🌍 Country: {result.get('country_code', 'N/A')}")
                print(f"📘 Passport Number: {result.get('passport_number', 'N/A')}")
                print(f"👤 Name: {result.get('given_names', '')} {result.get('surname', '')}")
                print(f"🎂 Date of Birth: {result.get('date_of_birth', 'N/A')}")
                print(f"⚥ Sex: {result.get('sex', 'N/A')}")
                print(f"⏳ Expiry Date: {result.get('expiry_date', 'N/A')}")
                print(f"🏛️ Nationality: {result.get('nationality', 'N/A')}")
                
            elif result['document_type'] == 'id_card':
                print(f"🌍 Country: {result.get('country_code', 'N/A')}")
                print(f"🆔 Document Number: {result.get('document_number', 'N/A')}")
                print(f"👤 Name: {result.get('given_names', '')} {result.get('surname', '')}")
                print(f"🎂 Date of Birth: {result.get('date_of_birth', 'N/A')}")
                print(f"⚥ Sex: {result.get('sex', 'N/A')}")
                print(f"⏳ Expiry Date: {result.get('expiry_date', 'N/A')}")
                print(f"🏛️ Nationality: {result.get('nationality', 'N/A')}")
            
            print(f"\n📝 Raw OCR Text:")
            print(result.get('raw_ocr_text', ''))
    else:
        print("Usage: python mrz_ocr.py <image_path>")
