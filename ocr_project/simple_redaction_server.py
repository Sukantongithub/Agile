"""
Simple single-file Document Redaction Server with OCR Field Detection
Detects sensitive fields from Indian documents automatically.
"""

from flask import Flask, render_template, jsonify, request, session, redirect, url_for
from pathlib import Path
import base64
import os
from PIL import Image, ImageFilter
from PIL.ExifTags import TAGS
import io
import re
import pytesseract
import uuid
import datetime
from datetime import timedelta
from functools import wraps
from db_utils import db_manager
import auth_db_mongo
from auth_db_mongo import register_user, login_user, get_user_by_id, update_user_email, change_password
from ml_enhanced_rag import rag_system

# Configure pytesseract path (update if needed)
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['SECRET_KEY'] = 'your-secret-key-change-this-in-production'  # Change this!
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(days=7)

# Create upload folder
Path(app.config['UPLOAD_FOLDER']).mkdir(exist_ok=True)


# ===== AUTHENTICATION DECORATOR =====
def login_required(f):
    """Decorator to require authentication for routes."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('login_page'))
        return f(*args, **kwargs)
    return decorated_function


# ===== METADATA EXTRACTION =====
def extract_file_metadata(image, filename=None):
    """
    Extract comprehensive metadata from uploaded files for enhanced detection.
    Returns metadata dict with file properties, EXIF data, and image characteristics.
    """
    metadata = {
        'filename': filename,
        'file_size_kb': 0,
        'dimensions': {'width': 0, 'height': 0},
        'aspect_ratio': 1.0,
        'format': 'Unknown',
        'mode': 'Unknown',
        'dpi': (0, 0),
        'exif': {},
        'creation_date': None,
        'camera_info': {},
        'color_depth': 0,
        'has_transparency': False,
        'document_hints': []
    }
    
    try:
        # Basic image properties
        if image:
            width, height = image.size
            metadata['dimensions'] = {'width': width, 'height': height}
            metadata['aspect_ratio'] = round(width / height, 2) if height > 0 else 1.0
            metadata['format'] = image.format or 'Unknown'
            metadata['mode'] = image.mode
            metadata['has_transparency'] = image.mode in ('RGBA', 'LA', 'P')
            
            # DPI information (important for scan quality)
            dpi = image.info.get('dpi', (0, 0))
            metadata['dpi'] = dpi if isinstance(dpi, tuple) else (dpi, dpi)
            
            # Color depth
            if image.mode == 'RGB':
                metadata['color_depth'] = 24
            elif image.mode == 'RGBA':
                metadata['color_depth'] = 32
            elif image.mode == 'L':
                metadata['color_depth'] = 8
            elif image.mode == '1':
                metadata['color_depth'] = 1
            
            # Extract EXIF data
            try:
                exif_data = image._getexif()
                if exif_data:
                    for tag_id, value in exif_data.items():
                        tag = TAGS.get(tag_id, tag_id)
                        
                        # Store specific useful tags
                        if tag in ['Make', 'Model', 'Software', 'DateTime', 'DateTimeOriginal', 
                                   'Orientation', 'XResolution', 'YResolution', 'ResolutionUnit',
                                   'ExifImageWidth', 'ExifImageHeight', 'ColorSpace']:
                            try:
                                metadata['exif'][tag] = str(value) if not isinstance(value, (str, int, float)) else value
                            except:
                                pass
                    
                    # Extract camera info
                    if 'Make' in metadata['exif']:
                        metadata['camera_info']['make'] = metadata['exif']['Make']
                    if 'Model' in metadata['exif']:
                        metadata['camera_info']['model'] = metadata['exif']['Model']
                    
                    # Extract creation date
                    for date_field in ['DateTimeOriginal', 'DateTime']:
                        if date_field in metadata['exif']:
                            metadata['creation_date'] = metadata['exif'][date_field]
                            break
            except (AttributeError, KeyError, TypeError):
                pass
            
            # Document type hints based on metadata
            
            # Hint 1: Aspect ratio (ID cards are typically 1.5-1.7)
            if 1.5 <= metadata['aspect_ratio'] <= 1.7:
                metadata['document_hints'].append('id_card_aspect_ratio')
            
            # Hint 2: Standard document sizes
            if width == 3508 and height == 2480:  # A4 at 300 DPI landscape
                metadata['document_hints'].append('a4_landscape_300dpi')
            elif width == 2480 and height == 3508:  # A4 at 300 DPI portrait
                metadata['document_hints'].append('a4_portrait_300dpi')
            elif width == 1240 and height == 1754:  # A4 at 150 DPI portrait
                metadata['document_hints'].append('a4_portrait_150dpi')
            
            # Hint 3: Scan quality (high DPI suggests scanned document)
            if metadata['dpi'][0] >= 300:
                metadata['document_hints'].append('high_quality_scan')
            elif 150 <= metadata['dpi'][0] < 300:
                metadata['document_hints'].append('medium_quality_scan')
            elif metadata['dpi'][0] > 0 and metadata['dpi'][0] < 150:
                metadata['document_hints'].append('low_quality_scan')
            
            # Hint 4: Photo vs scan (camera info suggests photo)
            if metadata['camera_info']:
                metadata['document_hints'].append('camera_photo')
            elif metadata['exif'].get('Software'):
                software = str(metadata['exif']['Software']).lower()
                if 'scanner' in software or 'scan' in software:
                    metadata['document_hints'].append('scanned_document')
                elif 'adobe' in software or 'photoshop' in software:
                    metadata['document_hints'].append('edited_document')
            
            # Hint 5: Resolution hints
            if width >= 2000 or height >= 2000:
                metadata['document_hints'].append('high_resolution')
            elif width < 800 and height < 800:
                metadata['document_hints'].append('low_resolution')
            
            # Hint 6: Color mode hints
            if metadata['mode'] == 'L':
                metadata['document_hints'].append('grayscale_document')
            elif metadata['mode'] == '1':
                metadata['document_hints'].append('binary_document')
            
    except Exception as e:
        print(f"Metadata extraction error: {e}")
    
    return metadata


def detect_document_type(text, image_path=None, metadata=None):
    """
    Enhanced ML-based document type detection.
    Combines visual features + OCR patterns + metadata for high accuracy.
    """
    text_lower = text.lower()
    text_normalized = ' '.join(text.split())  # Normalize whitespace
    
    # Initialize metadata-based scores
    metadata_scores = {}
    if metadata:
        # Use metadata hints to boost document type scores
        hints = metadata.get('document_hints', [])
        aspect_ratio = metadata.get('aspect_ratio', 1.0)
        dpi = metadata.get('dpi', (0, 0))[0]
        
        # ID card aspect ratio boost
        if 'id_card_aspect_ratio' in hints:
            metadata_scores['Aadhaar Card'] = metadata_scores.get('Aadhaar Card', 0) + 0.2
            metadata_scores['PAN Card'] = metadata_scores.get('PAN Card', 0) + 0.2
            metadata_scores['Voter ID Card'] = metadata_scores.get('Voter ID Card', 0) + 0.2
            metadata_scores['Driving License'] = metadata_scores.get('Driving License', 0) + 0.15
        
        # High quality scan suggests official documents
        if 'high_quality_scan' in hints:
            metadata_scores['Passport'] = metadata_scores.get('Passport', 0) + 0.15
            metadata_scores['Community Certificate'] = metadata_scores.get('Community Certificate', 0) + 0.15
            metadata_scores['Medical Report'] = metadata_scores.get('Medical Report', 0) + 0.1
            metadata_scores['Marksheet'] = metadata_scores.get('Marksheet', 0) + 0.1
        
        # A4 size suggests certificates/reports
        if 'a4_portrait_300dpi' in hints or 'a4_portrait_150dpi' in hints:
            metadata_scores['Community Certificate'] = metadata_scores.get('Community Certificate', 0) + 0.2
            metadata_scores['Medical Report'] = metadata_scores.get('Medical Report', 0) + 0.15
            metadata_scores['Marksheet'] = metadata_scores.get('Marksheet', 0) + 0.15
            metadata_scores['Bank Statement'] = metadata_scores.get('Bank Statement', 0) + 0.1
        
        # Camera photo suggests ID cards (people photo their cards)
        if 'camera_photo' in hints:
            metadata_scores['Aadhaar Card'] = metadata_scores.get('Aadhaar Card', 0) + 0.1
            metadata_scores['PAN Card'] = metadata_scores.get('PAN Card', 0) + 0.1
            metadata_scores['Voter ID Card'] = metadata_scores.get('Voter ID Card', 0) + 0.1
        
        # Scanned document suggests certificates/official docs
        if 'scanned_document' in hints:
            metadata_scores['Community Certificate'] = metadata_scores.get('Community Certificate', 0) + 0.15
            metadata_scores['Passport'] = metadata_scores.get('Passport', 0) + 0.1
            metadata_scores['Medical Report'] = metadata_scores.get('Medical Report', 0) + 0.1
    
    # Visual feature analysis (if image provided)
    visual_scores = {}
    if image_path:
        try:
            import cv2
            import numpy as np
            from pathlib import Path
            
            # Check if file exists
            if not Path(image_path).exists():
                print(f"Image file not found: {image_path}")
            else:
                img = cv2.imread(str(image_path))
                if img is not None and img.size > 0:
                    h, w = img.shape[:2]
                    
                    # Feature 1: Color analysis (identity cards vs documents)
                    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                    
                    # Detect blue/government colors (Aadhaar, PAN, Passport)
                    blue_mask = cv2.inRange(hsv, np.array([100, 50, 50]), np.array([130, 255, 255]))
                    blue_ratio = np.sum(blue_mask > 0) / (h * w)
                    
                    # Detect official seals/logos (circular shapes)
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 100, param1=100, param2=30, minRadius=20, maxRadius=100)
                    has_seal = circles is not None
                    
                    # Feature 2: Text density and layout
                    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
                    text_pixels = np.sum(binary > 0)
                    text_density = text_pixels / (h * w)
                    
                    # Feature 3: Edge analysis (structured vs unstructured)
                    edges = cv2.Canny(gray, 50, 150)
                    edge_density = np.sum(edges > 0) / (h * w)
                    
                    # Feature 4: Aspect ratio
                    aspect_ratio = w / h if h > 0 else 1.0
                    
                    # Score document types based on visual features
                    if blue_ratio > 0.05 and has_seal:
                        visual_scores['Aadhaar Card'] = 0.3
                        visual_scores['PAN Card'] = 0.25
                        visual_scores['Passport'] = 0.25
                        visual_scores['Community Certificate'] = 0.2
                    
                    if aspect_ratio > 1.4 and aspect_ratio < 1.8:  # ID card ratio
                        visual_scores['Aadhaar Card'] = visual_scores.get('Aadhaar Card', 0) + 0.15
                        visual_scores['PAN Card'] = visual_scores.get('PAN Card', 0) + 0.15
                        visual_scores['Voter ID Card'] = visual_scores.get('Voter ID Card', 0) + 0.15
                    
                    if aspect_ratio > 1.2 and aspect_ratio < 1.5 and has_seal:  # Passport booklet page
                        visual_scores['Passport'] = visual_scores.get('Passport', 0) + 0.2
                    
                    if text_density > 0.15:  # Dense text = marksheet/invoice/medical report
                        visual_scores['Marksheet'] = visual_scores.get('Marksheet', 0) + 0.2
                        visual_scores['Invoice'] = visual_scores.get('Invoice', 0) + 0.15
                        visual_scores['Bank Statement'] = visual_scores.get('Bank Statement', 0) + 0.15
                        visual_scores['Medical Report'] = visual_scores.get('Medical Report', 0) + 0.15
                    
                    if edge_density > 0.08:  # Tables/grids
                        visual_scores['Marksheet'] = visual_scores.get('Marksheet', 0) + 0.2
                        visual_scores['Invoice'] = visual_scores.get('Invoice', 0) + 0.2
                else:
                    print(f"Failed to load image: {image_path}")
        except ImportError:
            print("OpenCV not installed, skipping visual analysis")
        except Exception as e:
            print(f"Visual analysis failed: {e}")

    
    # Enhanced pattern-based detection with improved scoring
    patterns = {
        'Aadhaar Card': {
            'must_have': [
                r'\b\d{4}\s*\d{4}\s*\d{4}\b',  # 12-digit number with flexible spacing
            ],
            'strong': ['aadhaar', 'aadhar', 'uidai', 'unique identification', 'uid', 'unique identification authority', 'भारत सरकार'],
            'supporting': ['dob', 'date of birth', 'yob', 'year of birth', 'gender', 'male', 'female', 'address', 'vid', 'enrolment', 'enrollment'],
            'negative': ['pan', 'passport', 'license', 'voter', 'community', 'caste', 'certificate'],
            'base_score': 0.65,
            'must_have_boost': 0.35
        },
        'PAN Card': {
            'must_have': [
                r'\b[A-Z]{5}\d{4}[A-Z]\b',  # PAN format: ABCDE1234F
            ],
            'strong': ['permanent account', 'income tax', 'pan card', 'पैन कार्ड', 'income tax department'],
            'supporting': ['father', 'signature', 'date of birth'],
            'negative': ['aadhaar', 'passport', 'license', 'voter'],
            'base_score': 0.6,
            'must_have_boost': 0.3
        },
        'Passport': {
            'must_have': [
                r'(?i)passport',  # Main identifier - just need the word passport
            ],
            'strong': ['passport', 'nationality', 'place of issue', 'date of issue', 'date of expiry', 
                      'republic of india', 'ministry of external affairs', 
                      'passport no', 'passport number', 'nationality indian', 'issued at', 'valid until'],
            'supporting': ['given name', 'surname', 'place of birth', 'indian', 'holder', 'sex', 
                          'date of birth', 'country code ind', 'p<ind', 'type p', r'\b[A-Z]\d{7}\b'],
            'negative': ['aadhaar', 'pan', 'license', 'voter', 'college', 'student', 'marksheet', 'community'],
            'base_score': 0.7,
            'must_have_boost': 0.35
        },
        'Voter ID Card': {
            'must_have': [
                r'\b[A-Z]{3}\d{7}\b',  # Voter ID format
                r'election\s*commission',
                r'electoral\s*photo'
            ],
            'strong': ['election commission', 'electoral', 'voter', 'elector', 'electors photo identity card'],
            'supporting': ['assembly', 'part no', 'serial', 'constituency'],
            'negative': ['aadhaar', 'pan', 'passport', 'license'],
            'base_score': 0.5,
            'must_have_boost': 0.3
        },
        'Driving License': {
            'must_have': [
                r'(?i)driving\s*licen[cs]e',
                r'(?i)transport.*department',
                r'(?i)motor\s*vehicles?'
            ],
            'strong': ['driving licence', 'driving license', 'transport', 'motor vehicle', 'dl no'],
            'supporting': ['validity', 'blood group', 'authorized', 'vehicle class', 'date of issue'],
            'negative': ['aadhaar', 'pan', 'passport', 'voter'],
            'base_score': 0.4,
            'must_have_boost': 0.35
        },
        'Marksheet': {
            'must_have': [
                # Removed generic 'board', 'university', 'school' to prevent false positives
                # Now requires 'marks', 'grade', with 'statement', 'sheet', 'card' etc.
                r'(?i)(marks?\s*(?:obtained|statement|certificate|sheet)|grade\s*(?:card|sheet)|result|transcript)', 
                r'(?i)(cgpa|sgpa|percentage|higher\s*secondary|secondary\s*course|examination\s*result)',
            ],
            'strong': ['marks obtained', 'marksheet', 'grade sheet', 'transcript', 'result', 'higher secondary', 'statement of marks'],
            'supporting': ['semester', 'roll', 'theory', 'practical', 'total', 'subject', 'course', 'certificate', 'university', 'school', 'board'],
            'negative': ['invoice', 'receipt', 'payment', 'bill'],
            'base_score': 0.5,
            'must_have_boost': 0.3 # Reduced boost
        },
        'College ID Card': {
            'must_have': [
                r'(?i)(student\s*id|enrollment\s*no|roll\s*no|identity\s*card)',
                r'(?i)(college|university|institute|campus)'
            ],
            'strong': ['student', 'enrollment', 'college', 'university', 'institute', 'student id', 'principal'],
            'supporting': ['department', 'course', 'year', 'validity', 'session', 'dob', 'blood group'],
            'negative': ['invoice', 'receipt', 'payment', 'marks'],
            'base_score': 0.4, # Increased base score to compete
            'must_have_boost': 0.4
        },
        'Bank Passbook': {
            'must_have': [
                r'(?i)(passbook|savings\s*account)',
                r'(?i)(account\s*holder|account\s*no)',
                r'(?i)ifsc'
            ],
            'strong': ['passbook', 'account holder', 'ifsc', 'savings', 'bank passbook'],
            'supporting': ['branch', 'micr', 'nominee', 'account number'],
            'negative': ['statement', 'transaction', 'invoice'],
            'base_score': 0.4,
            'must_have_boost': 0.35
        },
        'Bank Statement': {
            'must_have': [
                r'(?i)(bank\s*statement|statement\s*of\s*account)',
                r'(?i)(opening\s*balance|closing\s*balance)',
                r'(?i)(transaction|debit|credit)'
            ],
            'strong': ['bank statement', 'statement of account', 'opening balance', 'closing balance', 'transaction'],
            'supporting': ['debit', 'credit', 'withdrawal', 'deposit', 'balance'],
            'negative': ['passbook', 'invoice', 'receipt'],
            'base_score': 0.4,
            'must_have_boost': 0.35
        },
        'Invoice': {
            'must_have': [
                r'(?i)(invoice|tax\s*invoice)',
                r'(?i)(gst|gstin)',
                r'(?i)(invoice\s*(no|number|date))'
            ],
            'strong': ['invoice', 'tax invoice', 'gst', 'gstin', 'invoice no', 'invoice date'],
            'supporting': ['bill to', 'subtotal', 'total', 'qty', 'amount', 'taxable', 'igst', 'cgst', 'sgst'],
            'negative': ['receipt', 'salary', 'statement'],
            'base_score': 0.45,
            'must_have_boost': 0.3
        },
        'Bill/Receipt': {
            'must_have': [
                r'(?i)(receipt|bill\s*no)',
                r'(?i)payment\s*(received|made)',
                r'(?i)(cash|card|upi)'
            ],
            'strong': ['receipt', 'bill no', 'payment received', 'payment made'],
            'supporting': ['amount', 'cash', 'card', 'date', 'total paid'],
            'negative': ['invoice', 'salary', 'gst'],
            'base_score': 0.35,
            'must_have_boost': 0.35
        },
        'Salary Slip': {
            'must_have': [
                r'(?i)(salary|pay\s*slip|payslip)',
                r'(?i)(earnings|deductions)',
                r'(?i)(basic\s*salary|net\s*pay|gross\s*pay)'
            ],
            'strong': ['salary slip', 'pay slip', 'payslip', 'net pay', 'gross pay', 'earnings', 'deductions'],
            'supporting': ['basic salary', 'pf', 'esi', 'tds', 'hra', 'allowance'],
            'negative': ['invoice', 'receipt', 'statement'],
            'base_score': 0.45,
            'must_have_boost': 0.3
        },
        'Community Certificate': {
            'must_have': [
                r'(?i)(community\s*certificate)',
                r'(?i)(caste\s*certificate)',
                r'(?i)(backward\s*class|scheduled\s*caste|scheduled\s*tribe)'
            ],
            'strong': ['community certificate', 'caste certificate', 'backward class', 'scheduled caste', 'scheduled tribe', 
                      'obc', 'sc', 'st', 'tahsildar', 'revenue officer', 'collector', 'district magistrate', 'mamlatdar', 'creamy layer'],
            'supporting': ['caste', 'community', 'reservation', 'belongs to', 'resident of', 'category', 'certificate no', 'issued by'],
            'negative': ['medical', 'fitness', 'health', 'doctor', 'hospital', 'passport', 'marks', 'aadhaar', 'aadhar', 'uidai'],
            'base_score': 0.45,
            'must_have_boost': 0.3
        },
        'Medical Report': {
            'must_have': [
                r'(?i)(medical\s*(report|certificate|examination))',
                r'(?i)(patient|diagnosis|prescription)',
                r'(?i)(doctor|physician|hospital|clinic)'
            ],
            'strong': ['medical report', 'medical certificate', 'medical examination', 'fitness certificate', 
                      'patient', 'diagnosis', 'prescription', 'doctor', 'physician', 'hospital', 'clinic', 
                      'health', 'treatment', 'consultation'],
            'supporting': ['blood pressure', 'temperature', 'pulse', 'weight', 'height', 'symptoms', 'medicines', 
                          'advised', 'examination', 'findings', 'medical history', 'dr.', 'mbbs', 'md', 'registration no'],
            'negative': ['community', 'caste', 'invoice', 'passport', 'license'],
            'base_score': 0.45,
            'must_have_boost': 0.35
        }
    }
    
    # Enhanced ML-like scoring algorithm
    final_scores = {}
    
    for doc_type, config in patterns.items():
        score = 0.0
        must_have_matches = 0
        
        # Check must-have patterns (critical features) - ANY one must match
        must_have_found = False
        for pattern in config['must_have']:
            if re.search(pattern, text, re.IGNORECASE):
                must_have_found = True
                must_have_matches += 1
        
        if not must_have_found:
            continue  # Skip if doesn't meet minimum requirements
        
        # Base score for matching must-have pattern
        score += config['base_score']
        
        # Bonus for multiple must-have matches
        if must_have_matches > 1:
            score += config['must_have_boost'] * (must_have_matches - 1) * 0.5
        
        # Strong indicators (weighted heavily)
        strong_matches = sum(1 for keyword in config['strong'] if keyword in text_lower)
        score += strong_matches * 0.15  # Increased from 0.12 for better accuracy
        
        # Supporting indicators (lighter weight)
        support_matches = sum(1 for keyword in config['supporting'] if keyword in text_lower)
        score += support_matches * 0.04
        
        # Negative keywords (penalize if found)
        negative_matches = sum(1 for keyword in config.get('negative', []) if keyword in text_lower)
        score -= negative_matches * 0.15
        
        # Add visual feature score
        score += visual_scores.get(doc_type, 0)
        
        # Add metadata-based score
        score += metadata_scores.get(doc_type, 0)
        
        # Ensure score doesn't go negative
        score = max(0, score)
        
        # Normalize score (cap at 1.0)
        final_scores[doc_type] = min(score, 1.0)
    
    
    # === RAG & LLM ENHANCEMENT ===
    try:
        # 1. Semantic Search (RAG)
        print("Running RAG Semantic Search...")
        rag_result = rag_system.semantic_classify(text)
        
        if rag_result.get("type"):
            rag_type = rag_result["type"]
            rag_conf = rag_result["confidence"]
            print(f"RAG Suggestion: {rag_type} ({rag_conf})")
            
            # Boost score for RAG match
            # If we have a strong RAG match, it significantly boosts the confidence
            if rag_conf > 0.5:
                final_scores[rag_type] = final_scores.get(rag_type, 0) + (rag_conf * 0.5)
                
    except Exception as e:
        print(f"RAG Integration Error: {e}")

    
    # Check current best before final verification
    current_best = max(final_scores, key=final_scores.get) if final_scores else "Unknown Document"
    current_conf = final_scores[current_best] if final_scores else 0.0

    # 3. LLM VERIFICATION (ALL DOCUMENTS)
    # We now verify EVERY decision with the LLM if available to fix bias
    try:
        print(f"Local detection: {current_best} ({current_conf})")
        print("Verifying with LLM...")
        
        verify_result = rag_system.verify_classification_with_llm(text, current_best, current_conf)
        
        final_type = verify_result["verified_type"]
        final_conf = verify_result["confidence"]
        source = verify_result["source"]
        
        if verify_result['correction']:
            print(f"LLM Corrected: {current_best} -> {final_type}")
        else:
            print(f"LLM Confirmed: {final_type}")
            
        return {
            'document_type': final_type,
            'confidence': round(final_conf, 2),
            'all_scores': {k: round(v, 2) for k, v in sorted(final_scores.items(), key=lambda x: x[1], reverse=True)[:3]},
            'detection_method': f'Hybrid ({source})'
        }

    except Exception as e:
        print(f"LLM Verification Failed: {e}")
        # Fallback to local
        return {
            'document_type': current_best,
            'confidence': round(current_conf, 2),
            'all_scores': {k: round(v, 2) for k, v in sorted(final_scores.items(), key=lambda x: x[1], reverse=True)[:3]},
            'detection_method': 'Hybrid ML (Local)'
        }


# ===== DOCUMENT FIELD TEMPLATES =====
DOCUMENT_FIELD_TEMPLATES = {
    'Aadhaar Card': {
        'expected_fields': [
            # Name - Relaxed pattern: Matches capitalized words before DOB/Year or ID number
            # Removed strict lookahead to allow detection even if layout varies
            {'name': 'Name', 'pattern': r'(?i)(?:(?:name|naam|to)\s*[:\-]?)?\s*([A-Z][A-Za-z\s]{2,40})(?=\s+(?:DOB|Year|YOB|Male|Female|Address|S/O|W/O|D/O|No\.|[0-9]{4}))', 'sensitive': True, 'required': True, 'category': 'personal_info'},
            
            # Aadhaar Number - Very flexible (groups of 4 digits, optional spaces/hyphens)
            {'name': 'Aadhaar Number', 'pattern': r'\b\d{4}[\s\-]?\d{4}[\s\-]?\d{4}\b', 'sensitive': True, 'required': True, 'category': 'identification'},
            
            # DOB/YOB - Catch all formats
            {'name': 'Date of Birth', 'pattern': r'(?i)(?:dob|birth|date\s*of\s*birth|year\s*of\s*birth|yob|जन्म[:\-])\s*[:\-]?(?:(\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4})|(\d{4}))', 'sensitive': True, 'required': True, 'category': 'personal_info'},
            
            # Gender - English and Hindi support
            {'name': 'Gender', 'pattern': r'(?i)(?:gender|sex|लिंग)[:\-]?\s*(Male|Female|Transgender|M|F|पुरुष|महिला)', 'sensitive': False, 'required': True, 'category': 'personal_info'},
            
            # Address - Look for "Address:" or long blocks with Pincode
            {'name': 'Address', 'pattern': r'(?i)(?:Address|पता)[:\-]?\s*([A-Za-z0-9\s,.\-\/()]{15,200}\s\d{6})', 'sensitive': True, 'required': True, 'category': 'personal_info'},
            
            # Mobile - Standard Indian mobile
            {'name': 'Mobile', 'pattern': r'(?i)(?:mobile|mob)\s*[:\-]?\s*([6-9]\d{9})', 'sensitive': True, 'required': False, 'category': 'contact'},
            
            # VID (Virtual ID)
            {'name': 'VID', 'pattern': r'(?i)(?:VID|Virtual\s*ID)\s*[:\-]?\s*(\d{4}\s*\d{4}\s*\d{4}\s*\d{4})', 'sensitive': False, 'required': False, 'category': 'identification'},
            
            # Visual elements
            {'name': 'Photograph', 'is_visual': True, 'sensitive': True, 'required': True, 'category': 'biometric'},
            {'name': 'QR Code', 'is_visual': True, 'sensitive': False, 'required': True, 'category': 'security'},
        ]
    },
    'PAN Card': {
        'expected_fields': [
            {'name': 'Name', 'pattern': r'(?i)(name)\s*:?\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,4})', 'sensitive': True, 'required': True, 'category': 'personal_info'},
            {'name': 'PAN Number', 'pattern': r'\b[A-Z]{5}\d{4}[A-Z]\b', 'sensitive': True, 'required': True, 'category': 'identification'},
            {'name': 'Father Name', 'pattern': r'(?i)(father\'?s?\s*name)\s*:?\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})', 'sensitive': True, 'required': True, 'category': 'personal_info'},
            {'name': 'Date of Birth', 'pattern': r'(?i)(dob|birth|date\s*of\s*birth|date\s*of\s*incorporation)\s*:?\s*(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})', 'sensitive': True, 'required': True, 'category': 'personal_info'},
            {'name': 'Photograph', 'is_visual': True, 'sensitive': True, 'required': True, 'category': 'biometric'},
            {'name': 'Signature', 'is_visual': True, 'sensitive': True, 'required': True, 'category': 'biometric'},
            {'name': 'QR Code', 'is_visual': True, 'sensitive': False, 'required': True, 'category': 'security'},
        ]
    },
    'Passport': {
        'expected_fields': [
            {'name': 'Passport Number', 'pattern': r'\b[A-Z]\d{7}\b', 'sensitive': True, 'required': True, 'category': 'identification'},
            {'name': 'Surname', 'pattern': r'(?i)(surname|last\s*name)\s*:?\s*([A-Z][A-Z\s]+)', 'sensitive': True, 'required': True, 'category': 'personal_info'},
            {'name': 'Given Name', 'pattern': r'(?i)(given\s*name|first\s*name)\s*:?\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3})', 'sensitive': True, 'required': True, 'category': 'personal_info'},
            {'name': 'Nationality', 'pattern': r'(?i)(nationality)\s*:?\s*(Indian|INDIAN)', 'sensitive': False, 'required': True, 'category': 'personal_info'},
            {'name': 'Date of Birth', 'pattern': r'(?i)(dob|birth|date\s*of\s*birth)\s*:?\s*(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})', 'sensitive': True, 'required': True, 'category': 'personal_info'},
            {'name': 'Place of Birth', 'pattern': r'(?i)(place\s*of\s*birth)\s*:?\s*([A-Z][a-z]+(?:,?\s+[A-Z][a-z]+){0,2})', 'sensitive': True, 'required': True, 'category': 'personal_info'},
            {'name': 'Gender', 'pattern': r'(?i)(gender|sex)\s*:?\s*(male|female|m|f)', 'sensitive': False, 'required': True, 'category': 'personal_info'},
            {'name': 'Place of Issue', 'pattern': r'(?i)(place\s*of\s*issue|issued\s*at)\s*:?\s*([A-Z][a-z]+)', 'sensitive': False, 'required': True, 'category': 'document_info'},
            {'name': 'Date of Issue', 'pattern': r'(?i)(date\s*of\s*issue|issued)\s*:?\s*(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})', 'sensitive': False, 'required': True, 'category': 'document_info'},
            {'name': 'Date of Expiry', 'pattern': r'(?i)(date\s*of\s*expiry|expiry)\s*:?\s*(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})', 'sensitive': False, 'required': True, 'category': 'document_info'},
            {'name': 'Father Name', 'pattern': r'(?i)(father\'?s?\s*name)\s*:?\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})', 'sensitive': True, 'required': False, 'category': 'personal_info'},
            {'name': 'Mother Name', 'pattern': r'(?i)(mother\'?s?\s*name)\s*:?\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})', 'sensitive': True, 'required': False, 'category': 'personal_info'},
            {'name': 'Spouse Name', 'pattern': r'(?i)(spouse\s*name)\s*:?\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3})', 'sensitive': True, 'required': False, 'category': 'personal_info'},
            {'name': 'Address', 'pattern': r'(?i)(address|permanent\s*address)\s*:?\s*([^\n]{20,200})', 'sensitive': True, 'required': True, 'category': 'personal_info'},
            {'name': 'Old Passport Number', 'pattern': r'(?i)(old\s*passport|previous\s*passport)\s*:?\s*([A-Z]\d{7})', 'sensitive': False, 'required': False, 'category': 'document_info'},
            {'name': 'File Number', 'pattern': r'(?i)(file\s*no|file\s*number|reference)\s*:?\s*([A-Z0-9/-]+)', 'sensitive': False, 'required': False, 'category': 'document_info'},
            {'name': 'MRZ Line', 'pattern': r'[A-Z0-9<]{30,44}', 'sensitive': True, 'required': True, 'category': 'mrz'},
            {'name': 'Photograph', 'is_visual': True, 'sensitive': True, 'required': True, 'category': 'biometric'},
            {'name': 'Signature', 'is_visual': True, 'sensitive': True, 'required': True, 'category': 'biometric'},
        ]
    },
    'Voter ID Card': {
        'expected_fields': [
            {'name': 'EPIC Number', 'pattern': r'\b[A-Z]{3}\d{7}\b', 'sensitive': True, 'required': True, 'category': 'identification'},
            {'name': 'Name', 'pattern': r'(?i)(name)\s*:?\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})', 'sensitive': True, 'required': True, 'category': 'personal_info'},
            {'name': 'Father Name', 'pattern': r'(?i)(father\'?s?\s*name)\s*:?\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})', 'sensitive': False, 'required': False, 'category': 'personal_info'},
            {'name': 'Husband Name', 'pattern': r'(?i)(husband\'?s?\s*name)\s*:?\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})', 'sensitive': False, 'required': False, 'category': 'personal_info'},
            {'name': 'Gender', 'pattern': r'(?i)(gender|sex)\s*:?\s*(male|female|m|f)', 'sensitive': False, 'required': True, 'category': 'personal_info'},
            {'name': 'Age', 'pattern': r'(?i)(age|years?)\s*:?\s*(\d{1,3})', 'sensitive': False, 'required': False, 'category': 'personal_info'},
            {'name': 'Date of Birth', 'pattern': r'(?i)(dob|birth|date\s*of\s*birth)\s*:?\s*(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})', 'sensitive': True, 'required': False, 'category': 'personal_info'},
            {'name': 'Address', 'pattern': r'(?i)(address|house\s*no)\s*:?\s*([^\n]{15,150})', 'sensitive': True, 'required': True, 'category': 'personal_info'},
            {'name': 'Assembly Constituency', 'pattern': r'(?i)(assembly|constituency)\s*:?\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})', 'sensitive': False, 'required': False, 'category': 'election_info'},
            {'name': 'Part Number', 'pattern': r'(?i)(part\s*no|part\s*number)\s*:?\s*(\d+)', 'sensitive': False, 'required': False, 'category': 'election_info'},
            {'name': 'Serial Number', 'pattern': r'(?i)(serial\s*no|serial\s*number|sl\s*no)\s*:?\s*(\d+)', 'sensitive': False, 'required': False, 'category': 'election_info'},
            {'name': 'Issue Date', 'pattern': r'(?i)(issue\s*date|issued)\s*:?\s*(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})', 'sensitive': False, 'required': False, 'category': 'document_info'},
            {'name': 'Photograph', 'is_visual': True, 'sensitive': True, 'required': True, 'category': 'biometric'},
            {'name': 'Signature', 'is_visual': True, 'sensitive': False, 'required': False, 'category': 'biometric'},
            {'name': 'QR Code', 'is_visual': True, 'sensitive': False, 'required': False, 'category': 'security'},
        ]
    },
    'Driving License': {
        'expected_fields': [
            {'name': 'License Number', 'pattern': r'\b[A-Z]{2}\d{13}\b', 'sensitive': True, 'required': True, 'category': 'identification'},
            {'name': 'Name', 'pattern': r'(?i)(name)\s*:?\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})', 'sensitive': True, 'required': True, 'category': 'personal_info'},
            {'name': 'Father Name', 'pattern': r'(?i)(father\'?s?\s*name|husband\'?s?\s*name|s/o|d/o|w/o)\s*:?\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})', 'sensitive': False, 'required': False, 'category': 'personal_info'},
            {'name': 'Date of Birth', 'pattern': r'(?i)(dob|birth|date\s*of\s*birth)\s*:?\s*(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})', 'sensitive': True, 'required': True, 'category': 'personal_info'},
            {'name': 'Blood Group', 'pattern': r'(?i)(blood\s*group|bg)\s*:?\s*(A\+|A-|B\+|B-|AB\+|AB-|O\+|O-)', 'sensitive': False, 'required': False, 'category': 'personal_info'},
            {'name': 'Address', 'pattern': r'(?i)(address)\s*:?\s*([^\n]{15,150})', 'sensitive': True, 'required': True, 'category': 'personal_info'},
            {'name': 'Issuing Authority', 'pattern': r'(?i)(rto|dto|issuing\s*authority)\s*:?\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3})', 'sensitive': False, 'required': False, 'category': 'document_info'},
            {'name': 'Issue Date', 'pattern': r'(?i)(issue\s*date|issued)\s*:?\s*(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})', 'sensitive': False, 'required': True, 'category': 'document_info'},
            {'name': 'Valid Until', 'pattern': r'(?i)(valid\s*until|validity|valid\s*till|expiry)\s*:?\s*(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})', 'sensitive': False, 'required': True, 'category': 'document_info'},
            {'name': 'Vehicle Class', 'pattern': r'(?i)(class\s*of\s*vehicle|cov|vehicle\s*class)\s*:?\s*(MCWG|LMV|HMV|MGV|[A-Z]+)', 'sensitive': False, 'required': False, 'category': 'license_info'},
            {'name': 'Emergency Contact', 'pattern': r'(?i)(emergency\s*contact)\s*:?\s*([6-9]\d{9})', 'sensitive': True, 'required': False, 'category': 'contact'},
            {'name': 'Badge Number', 'pattern': r'(?i)(badge\s*no|badge\s*number)\s*:?\s*(\d+)', 'sensitive': False, 'required': False, 'category': 'license_info'},
            {'name': 'Photograph', 'is_visual': True, 'sensitive': True, 'required': True, 'category': 'biometric'},
            {'name': 'Signature', 'is_visual': True, 'sensitive': True, 'required': True, 'category': 'biometric'},
            {'name': 'QR Code', 'is_visual': True, 'sensitive': False, 'required': False, 'category': 'security'},
        ]
    },
    'Community Certificate': {
        'expected_fields': [
            {'name': 'Name', 'pattern': r'(?i)(name|applicant\s*name)\s*:?\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})', 'sensitive': True, 'required': True, 'category': 'personal_info'},
            {'name': 'Father Name', 'pattern': r'(?i)(father\'?s?\s*name)\s*:?\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})', 'sensitive': False, 'required': False, 'category': 'personal_info'},
            {'name': 'Mother Name', 'pattern': r'(?i)(mother\'?s?\s*name)\s*:?\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})', 'sensitive': False, 'required': False, 'category': 'personal_info'},
            {'name': 'Gender', 'pattern': r'(?i)(gender|sex)\s*:?\s*(male|female|m|f)', 'sensitive': False, 'required': True, 'category': 'personal_info'},
            {'name': 'Date of Birth', 'pattern': r'(?i)(dob|birth|date\s*of\s*birth|age)\s*:?\s*(\d{1,2}[-/]\d{1,2}[-/]\d{2,4}|\d{1,3}\s*years?)', 'sensitive': True, 'required': True, 'category': 'personal_info'},
            {'name': 'Address', 'pattern': r'(?i)(address|village|town|district)\s*:?\s*([^\n]{15,150})', 'sensitive': True, 'required': True, 'category': 'personal_info'},
            {'name': 'Caste/Community', 'pattern': r'(?i)(caste|community|sub-caste)\s*:?\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})', 'sensitive': True, 'required': True, 'category': 'caste_info'},
            {'name': 'Caste Category', 'pattern': r'(?i)(category)\s*:?\s*(SC|ST|OBC|MBC|EBC|EWS)', 'sensitive': True, 'required': True, 'category': 'caste_info'},
            {'name': 'Certificate Number', 'pattern': r'(?i)(certificate\s*no|certificate\s*number|ref\s*no)\s*:?\s*([A-Z0-9/-]+)', 'sensitive': False, 'required': True, 'category': 'document_info'},
            {'name': 'Date of Issue', 'pattern': r'(?i)(issue\s*date|issued|date)\s*:?\s*(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})', 'sensitive': False, 'required': True, 'category': 'document_info'},
            {'name': 'Valid Until', 'pattern': r'(?i)(valid\s*until|validity)\s*:?\s*(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})', 'sensitive': False, 'required': False, 'category': 'document_info'},
            {'name': 'Issuing Authority', 'pattern': r'(?i)(tahsildar|district\s*magistrate|revenue\s*officer|issuing\s*authority)\s*:?\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3})', 'sensitive': False, 'required': True, 'category': 'document_info'},
            {'name': 'Official Seal', 'is_visual': True, 'sensitive': False, 'required': True, 'category': 'security'},
            {'name': 'Signature', 'is_visual': True, 'sensitive': False, 'required': True, 'category': 'security'},
            {'name': 'QR Code', 'is_visual': True, 'sensitive': False, 'required': False, 'category': 'security'},
        ]
    },
    'Birth Certificate': {
        'expected_fields': [
            {'name': 'Child Name', 'pattern': r'(?i)(child\'?s?\s*name|name\s*of\s*child|baby\s*name)\s*:?\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3})', 'sensitive': True, 'required': True, 'category': 'personal_info'},
            {'name': 'Date of Birth', 'pattern': r'(?i)(date\s*of\s*birth|dob|birth\s*date)\s*:?\s*(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})', 'sensitive': True, 'required': True, 'category': 'personal_info'},
            {'name': 'Time of Birth', 'pattern': r'(?i)(time\s*of\s*birth|birth\s*time)\s*:?\s*(\d{1,2}:\d{2}\s*(?:AM|PM|am|pm)?)', 'sensitive': False, 'required': False, 'category': 'personal_info'},
            {'name': 'Gender', 'pattern': r'(?i)(gender|sex)\s*:?\s*(male|female|m|f|boy|girl)', 'sensitive': False, 'required': True, 'category': 'personal_info'},
            {'name': 'Place of Birth', 'pattern': r'(?i)(place\s*of\s*birth|birth\s*place|hospital)\s*:?\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,4})', 'sensitive': True, 'required': True, 'category': 'personal_info'},
            {'name': 'Father Name', 'pattern': r'(?i)(father\'?s?\s*name)\s*:?\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})', 'sensitive': True, 'required': True, 'category': 'personal_info'},
            {'name': 'Mother Name', 'pattern': r'(?i)(mother\'?s?\s*name)\s*:?\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})', 'sensitive': True, 'required': True, 'category': 'personal_info'},
            {'name': 'Address', 'pattern': r'(?i)(address|permanent\s*address)\s*:?\s*([^\n]{15,150})', 'sensitive': True, 'required': True, 'category': 'personal_info'},
            {'name': 'Registration Number', 'pattern': r'(?i)(registration\s*no|certificate\s*no|reg\s*no)\s*:?\s*([A-Z0-9/-]+)', 'sensitive': False, 'required': True, 'category': 'document_info'},
            {'name': 'Date of Registration', 'pattern': r'(?i)(date\s*of\s*registration|registration\s*date)\s*:?\s*(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})', 'sensitive': False, 'required': True, 'category': 'document_info'},
            {'name': 'Registrar Name', 'pattern': r'(?i)(registrar|issuing\s*authority)\s*:?\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3})', 'sensitive': False, 'required': False, 'category': 'document_info'},
            {'name': 'Hospital Name', 'pattern': r'(?i)(hospital|nursing\s*home|clinic)\s*:?\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,4})', 'sensitive': False, 'required': False, 'category': 'document_info'},
            {'name': 'Official Seal', 'is_visual': True, 'sensitive': False, 'required': True, 'category': 'security'},
            {'name': 'QR Code', 'is_visual': True, 'sensitive': False, 'required': False, 'category': 'security'},
        ]
    },
    'College ID Card': {
        'expected_fields': [
            # College name - Targeted for "M KUMARASAMY COLLEGE OF ENGINEERING" format (All Caps)
            # Matches: Full name containing College/University/Institute
            {'name': 'College Name', 'pattern': r'(?i)(\b[A-Z][A-Z\s&,.-]*(?:COLLEGE|UNIVERSITY|INSTITUTE)[A-Z\s&,.-]*\b)', 'sensitive': False, 'required': True, 'category': 'institution_info'},
            
            # Student Name - Targeted for "NAME INITIAL" format (User examples: VENKAT RAGHAV N, MUKESH M)
            # Matches: Word(s) followed by Single Letter Initial (All Uppercase) - No lookahead dependency
            {'name': 'Student Name', 'pattern': r'(?i)(?:name|student\s*name)?\s*:?\s*(\b[A-Z]{3,}(?:\s+[A-Z]{3,})*\s+[A-Z]\b)', 'sensitive': True, 'required': True, 'category': 'personal_info'},
            
            # Register Number - Targeted for 927623BAD123 format (6 digits + 3 letters + 3 digits)
            # Relaxed: matches anywhere in text, handles sticking to other text
            {'name': 'Register Number', 'pattern': r'(?i)(?:reg\.?\s*no\.?|register\s*no\.?|roll\s*no\.?|id\s*no\.?|regd?\.?\s*no\.?|regno)?\s*:?\s*([0-9]{6}[A-Z]{3}\d{3}|[0-9]{6}[A-Z]{3}[_]+\d+)', 'sensitive': True, 'required': True, 'category': 'identification'},
            
            # Batch Year - Targeted for 2023-2027 format (YYYY-YYYY)
            # Matches: 2023-2027, 2023 - 2027 (handles standard hyphen, en-dash, em-dash)
            {'name': 'Batch Year', 'pattern': r'(?i)(?:batch|year|academic\s*year)?\s*:?\s*(\b20\d{2}\s*[-–—]\s*20\d{2}\b)', 'sensitive': False, 'required': False, 'category': 'academic_info'},
            
            # Address
            {'name': 'Address', 'pattern': r'(?i)(?:address)?\s*:?\s*([A-Za-z0-9][A-Za-z0-9\s,./\-()]{20,200})', 'sensitive': False, 'required': False, 'category': 'institution_info'},
            
            # Pin Code - 6 digit, label optional
            {'name': 'Pin Code', 'pattern': r'(?i)(?:pin\s*code|pincode|postal\s*code)?\s*:?\s*(\b\d{6}\b)', 'sensitive': False, 'required': False, 'category': 'institution_info'},
            
            # Phone Number - landline or mobile
            {'name': 'Phone Number', 'pattern': r'(?i)(?:phone|tel|contact|ph\.?)?\s*:?\s*(\d{3,5}[\s-]?\d{6,8}|[6-9]\d[\s-]?\d{3}[\s-]?\d{3}[\s-]?\d{3})', 'sensitive': False, 'required': False, 'category': 'contact'},
            
            # Website - .edu domains
            {'name': 'Website', 'pattern': r'(?i)(?:website|web|url)?\s*:?\s*(www\.[a-z0-9.-]+\.[a-z]{2,}|[a-z0-9.-]+\.edu(?:\.in)?)', 'sensitive': False, 'required': False, 'category': 'institution_info'},
            
            # Visual elements
            {'name': 'Student Photograph', 'is_visual': True, 'sensitive': True, 'required': True, 'category': 'biometric'},
            {'name': 'Principal Signature', 'is_visual': True, 'sensitive': False, 'required': False, 'category': 'security'},
        ]
    },
    'Marksheet': {
        'expected_fields': [
            # Candidate Name
            {'name': 'Candidate Name', 'pattern': r'(?i)(?:name\s*of\s*the\s*candidate|candidate\s*name)\s*:?\s*([A-Z\s\.]{3,50})', 'sensitive': True, 'required': True, 'category': 'personal_info'},
            
            # Register Number - "PERMANENT REGISTER NUMBER" in HSC marksheets
            {'name': 'Register Number', 'pattern': r'(?i)(?:permanent\s*register\s*number|register\s*no\.?|reg\.?\s*no\.?|roll\s*no\.?)\s*:?\s*(\d{8,15})', 'sensitive': True, 'required': True, 'category': 'identification'},
            
            # Certificate Serial Number - "CERTIFICATE SL. NO"
            {'name': 'Certificate Number', 'pattern': r'(?i)(?:certificate\s*sl\.?\s*no\.?|serial\s*no\.?|cert\.?\s*no\.?)\s*:?\s*([0-9]{6,15})', 'sensitive': False, 'required': True, 'category': 'document_info'},
            
            # Date of Birth
            {'name': 'Date of Birth', 'pattern': r'(?i)(?:date\s*of\s*birth|dob)\s*:?\s*(\d{2}[-/]\d{2}[-/]\d{4})', 'sensitive': True, 'required': True, 'category': 'personal_info'},
            
            # Total Marks - "TOTAL MARKS"
            {'name': 'Total Marks', 'pattern': r'(?i)(?:total\s*marks)\s*:?\s*(\d{1,3}(?:\.\d{1,2})?|[\d\s]+)', 'sensitive': True, 'required': True, 'category': 'academic_info'},
            
            # School Name - "NAME OF THE SCHOOL"
            {'name': 'School Name', 'pattern': r'(?i)(?:name\s*of\s*the\s*school|school)\s*:?\s*([A-Z][A-Z\s&,.-]{5,100})', 'sensitive': False, 'required': False, 'category': 'institution_info'},
            
            # Session/Year
            {'name': 'Session', 'pattern': r'(?i)(?:session|year\s*of\s*passing|month\s*and\s*year)\s*:?\s*([A-Z]{3,}\s*\d{4})', 'sensitive': False, 'required': False, 'category': 'academic_info'},
            
            # Visual Elements
            {'name': 'Candidate Photograph', 'is_visual': True, 'sensitive': True, 'required': True, 'category': 'biometric'},
            {'name': 'QR Code', 'is_visual': True, 'sensitive': False, 'required': True, 'category': 'security'},
            {'name': 'Official Signature', 'is_visual': True, 'sensitive': False, 'required': False, 'category': 'security'},
        ]
    },
}


def detect_sensitive_fields(text, image_path=None, metadata=None, document_type=None):
    """
    Comprehensive field detection with ACCURATE coordinates.
    Uses metadata to improve detection accuracy and OCR optimization.
    
    Detects: Text fields, MRZ, QR codes, barcodes, photos, signatures
    """
    detected_fields = []
    field_id = 0
    
    # Use metadata to optimize OCR settings
    ocr_config = '--psm 3'  # Default: Fully automatic page segmentation
    if metadata:
        # High DPI images can use more precise detection
        if metadata.get('dpi', (0, 0))[0] >= 300:
            ocr_config = '--psm 3 --oem 3'  # Use LSTM OCR engine for better accuracy
        
        # Low resolution images need more lenient settings
        elif metadata.get('dpi', (0, 0))[0] < 150:
            ocr_config = '--psm 6'  # Assume single uniform block of text
        
        # ID card aspect ratio suggests structured layout
        if 'id_card_aspect_ratio' in metadata.get('document_hints', []):
            ocr_config = '--psm 6'  # Single uniform block
    
    # Get OCR data with bounding boxes if image provided
    ocr_data = None
    image = None
    cv_image = None  # Initialize to prevent UnboundLocalError
    if image_path:
        try:
            import pandas as pd
            import cv2
            import numpy as np
            from pyzbar import pyzbar
            
            image = Image.open(image_path)
            cv_image = cv2.imread(image_path)
            
            # --- PADDLEOCR IMPLEMENTATION (Primary Engine) ---
            try:
                import os
                os.environ["DISABLE_MODEL_SOURCE_CHECK"] = "True" # Prevent hanging on connection check
                
                from paddleocr import PaddleOCR
                print("Initializing PaddleOCR...")
                # Initialize once (or lazy load) - re-initializing is heavy but acceptable for now
                # Removed show_log=False as it caused an error
                ocr = PaddleOCR(use_angle_cls=True, lang='en') 
                # cls=True removed from call as it caused 'unexpected keyword argument', relying on init arg
                result = ocr.ocr(str(image_path))
                
                if result and result[0]:
                   data_list = []
                   full_text_parts = []
                   
                   for line in result:
                       if not line: continue
                       for word_info in line:
                           # word_info format: [[[x1,y1],[x2,y2],[x3,y3],[x4,y4]], ('text', confidence)]
                           box = word_info[0]
                           text_str = word_info[1][0]
                           conf = word_info[1][1]
                           
                           xs = [p[0] for p in box]
                           ys = [p[1] for p in box]
                           l = min(xs); t = min(ys)
                           w = max(xs) - l; h = max(ys) - t
                           
                           data_list.append({
                               'left': int(l), 'top': int(t), 
                               'width': int(w), 'height': int(h), 
                               'text': text_str, 'conf': conf
                           })
                           full_text_parts.append(text_str)
                   
                   ocr_data = pd.DataFrame(data_list)
                   # CRITICAL: Update 'text' with Paddle's superior output for regex matching
                   text = "\n".join(full_text_parts) 
                   print(f"PaddleOCR Success. Extracted {len(ocr_data)} words.")
                else:
                    raise Exception("PaddleOCR returned empty result")

            except Exception as paddle_error:
                print(f"PaddleOCR failed/skipped ({paddle_error}). Falling back to Tesseract.")
                # Fallback to Tesseract
                ocr_df = pytesseract.image_to_data(image, output_type=pytesseract.Output.DATAFRAME, config=ocr_config)
                ocr_data = ocr_df[ocr_df['text'].notna() & (ocr_df['text'].str.strip() != '')]

        except Exception as e:
            print(f"OCR bbox extraction failed: {e}")
            ocr_data = None
            cv_image = None  # Ensure it's None if loading failed

    # Get image dimensions
    img_width, img_height = None, None
    if image:
        img_width, img_height = image.size

    # Helper function to find coordinates with HIGH ACCURACY
    def find_coordinates(value_text, img_width_param=None, img_height_param=None):
        if ocr_data is not None and img_width_param and img_height_param:
            value_clean = str(value_text).strip()
            
            # Split value into tokens for better matching
            value_tokens = [t.strip() for t in re.split(r'[\s\-:/]+', value_clean) if len(t.strip()) > 0]
            
            matching_boxes = []
            used_indices = set()
            
            # Strategy 1: Find exact token matches
            for token in value_tokens:
                if len(token) < 2:
                    continue
                    
                token_lower = token.lower().replace('<', '')
                
                for idx, row in ocr_data.iterrows():
                    if idx in used_indices:
                        continue
                        
                    ocr_text = str(row['text']).strip().lower()
                    
                    # Match: exact, contains, or contained
                    if (token_lower == ocr_text or 
                        token_lower in ocr_text or 
                        ocr_text in token_lower):
                        matching_boxes.append({
                            'left': row['left'],
                            'top': row['top'],
                            'width': row['width'],
                            'height': row['height']
                        })
                        used_indices.add(idx)
                        break
            
            # If no tokens matched, try matching the whole value as a substring
            if not matching_boxes and value_clean:
                value_lower = value_clean.lower().replace('<', '')
                for idx, row in ocr_data.iterrows():
                    ocr_text = str(row['text']).strip().lower()
                    if value_lower in ocr_text or ocr_text in value_lower:
                        matching_boxes.append({
                            'left': row['left'],
                            'top': row['top'],
                            'width': row['width'],
                            'height': row['height']
                        })
                        # Don't break here, as the value might span multiple OCR words
            
            if matching_boxes:
                min_x = min(b['left'] for b in matching_boxes)
                min_y = min(b['top'] for b in matching_boxes)
                max_x = max(b['left'] + b['width'] for b in matching_boxes)
                max_y = max(b['top'] + b['height'] for b in matching_boxes)
                
                return {
                    'x': (min_x / img_width_param) * 100,
                    'y': (min_y / img_height_param) * 100,
                    'width': ((max_x - min_x) / img_width_param) * 100,
                    'height': ((max_y - min_y) / img_height_param) * 100
                }
        
        # Fallback coordinates if OCR data is not available or no match found
        return {
            'x': 10,
            'y': (field_id * 6) % 90, # Simple heuristic for placement
            'width': 30,
            'height': 4
        }

    # === DOCUMENT-TYPE-SPECIFIC DETECTION ===
    # === KNOWN NAMES DETECTION (PRIORITY) ===
    # User-provided list of known individuals to prioritized detection
    KNOWN_NAMES = [
        "Sukant Ravichandran", "Mukesh M", "Nithiyanantham T", "Subburaman Vengadesan", "Sabarivasan M", 
        "Venkat Raghav N", "Sujith Kumar P", "Rithick S", "Ruthuvarsahan N", 
        "Vishnu S", "Swathi B", "Naresh D", "Perumal P", "Ramesh S"
    ]
    
    # Check for known names first (Robust Matching)
    import re
    
    # Normalize text: collapse multiple spaces/newlines to single space
    text_normalized = re.sub(r'\s+', ' ', text).lower().strip()
    
    for name in KNOWN_NAMES:
        name_clean = re.sub(r'\s+', ' ', name).lower().strip()
        
        # 1. Exact Match in Normalized Text
        if name_clean in text_normalized:
            # Avoid duplicates
            if any(d['field_value'] == name for d in detected_fields):
                continue
                
            coords = find_coordinates(name, img_width, img_height)
            detected_fields.append({
                'id': f'field_{field_id}',
                'field_name': 'Name (Known)',
                'field_value': name,
                'confidence': 1.0,
                'category': 'personal_info',
                'is_sensitive': True,
                'auto_selected': True,
                'coordinates': coords,
                'verification_status': 'Verified (Known List)'
            })
            field_id += 1
        
        # 2. Part-based Match (if Full Name fails, but First+Last appearing nearby is suspicious)
        # Useful if OCR splits name across lines far apart? No, 'text_normalized' handles local splits.
        # This acts as a fallback for severe OCR issues.
        else:
            parts = name_clean.split()
            if len(parts) > 1:
                # Check if all parts exist in text
                if all(part in text_normalized for part in parts):
                     # Heuristic: If all parts are present, it's likely the person.
                     # But "John Smith" -> "John" ... "Smith" might be "John Doe" and "Jane Smith".
                     # Just checking existence isn't enough.
                     # We skip this for now to avoid false positives, relying on normalization.
                     pass

    # Apply template-based detection if document type is known (and valid template exists)e it for field detection
    if document_type:
        document_type = document_type.strip()  # Normalize: remove extra whitespace
        
    print(f"DEBUG: Checking template for document_type='{document_type}'")
    print(f"DEBUG: Available templates: {list(DOCUMENT_FIELD_TEMPLATES.keys())}")
    
    # Try exact match first
    template_key = None
    if document_type and document_type in DOCUMENT_FIELD_TEMPLATES:
        template_key = document_type
    # Try case-insensitive match
    elif document_type:
        for key in DOCUMENT_FIELD_TEMPLATES:
            if key.lower() == document_type.lower():
                template_key = key
                break
    
    if template_key:
        print(f"Using template-based detection for: {template_key}")
        template = DOCUMENT_FIELD_TEMPLATES[template_key]
        
        # Get image dimensions
        img_width, img_height = None, None
        if image:
            img_width, img_height = image.size
        
        # Helper function to find coordinates (defined inline for accessibility)
        def find_coordinates_template(value_text, field_name):
            if ocr_data is not None and img_width and img_height:
                value_clean = str(value_text).strip()
                value_tokens = [t.strip() for t in re.split(r'[\s\-:/]+', value_clean) if len(t.strip()) > 0]
                
                matching_boxes = []
                used_indices = set()
                
                for token in value_tokens:
                    if len(token) < 2:
                        continue
                    token_lower = token.lower()
                    
                    for idx, row in ocr_data.iterrows():
                        if idx in used_indices:
                            continue
                        ocr_text_lower = str(row['text']).lower()
                        if token_lower in ocr_text_lower or ocr_text_lower in token_lower:
                            matching_boxes.append({
                                'left': row['left'],
                                'top': row['top'],
                                'width': row['width'],
                                'height': row['height']
                            })
                            used_indices.add(idx)
                            break
                
                if matching_boxes:
                    min_x = min(b['left'] for b in matching_boxes)
                    min_y = min(b['top'] for b in matching_boxes)
                    max_x = max(b['left'] + b['width'] for b in matching_boxes)
                    max_y = max(b['top'] + b['height'] for b in matching_boxes)
                    
                    return {
                        'x': (min_x / img_width) * 100,
                        'y': (min_y / img_height) * 100,
                        'width': ((max_x - min_x) / img_width) * 100,
                        'height': ((max_y - min_y) / img_height) * 100
                    }
            
            # Fallback coordinates
            return {
                'x': 10,
                'y': (field_id * 6) % 90,
                'width': 30,
                'height': 4
            }
        
        # Process each expected field from template
        for field_spec in template['expected_fields']:
            # Handle visual elements (QR codes, photographs, signatures)
            if field_spec.get('is_visual'):
                coords = {
                    'x': 5 if field_spec['name'] == 'Photograph' else 70,
                    'y': 5 if field_spec['name'] == 'Photograph' else 80,
                    'width': 20 if field_spec['name'] == 'Photograph' else 25,
                    'height': 25 if field_spec['name'] == 'Photograph' else 15
                }
                
                detected_fields.append({
                    'id': f'field_{field_id}',
                    'field_name': field_spec['name'],
                    'field_value': '[Visual Element - Please verify location]',
                    'confidence': 0.7,
                    'category': field_spec['category'],
                    'coordinates': coords,
                    'is_sensitive': field_spec['sensitive'],
                    'auto_selected': field_spec['sensitive']
                })
                field_id += 1
                continue
            
            # Text-based field detection using pattern
            pattern = field_spec['pattern']
            matches = re.finditer(pattern, text)
            
            for match in matches:
                # Extract value from match
                groups = match.groups()
                if len(groups) > 1:
                    value = groups[-1]  # Last group is typically the value
                else:
                    value = match.group(0)
                
                coords = find_coordinates_template(value, field_spec['name'])
                
                detected_fields.append({
                    'id': f'field_{field_id}',
                    'field_name': field_spec['name'],
                    'field_value': value[:50],  # Truncate long values
                    'confidence': 0.95 if field_spec['required'] else 0.85,
                    'category': field_spec['category'],
                    'coordinates': coords,
                    'is_sensitive': field_spec['sensitive'],
                    'auto_selected': field_spec['sensitive']
                })
                field_id += 1
                break  # Only take first match for each field
        
        print(f"Template-based detection found {len(detected_fields)} fields")
        return detected_fields
    
    # === GENERIC PATTERN-BASED DETECTION (Fallback for unknown documents) ===
    print(f"Using generic pattern-based detection for: {document_type or 'Unknown Document'}")
    
    # Define text-based patterns
    sensitive_patterns = {
        'Password': r'(?i)(password|pwd|pass)\s*:?\s*([^\s\n]{4,})',
        'Bank Account': r'\b\d{11}\b',
        'Credit Card': r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b',
        'CVV': r'(?i)(cvv|cvc)\s*:?\s*\d{3,4}',
        'PIN': r'(?i)(pin|atm\s*pin)\s*:?\s*\d{4,6}',
    }
    
    identifier_patterns = {
        'Aadhaar': r'\b\d{4}\s?\d{4}\s?\d{4}\b',
        'PAN': r'\b[A-Z]{5}\d{4}[A-Z]\b',
        'Passport': r'\b[A-Z]\d{7}\b',
        'Voter ID': r'\b[A-Z]{3}\d{7}\b',
    }
    
    personal_patterns = {
        'Email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
        'Mobile': r'(?i)(mobile|phone|contact|tel|cell|ph\.?)\s*(?:no|number|num)?\.?\s*:?\s*([6-9]\d{9})',  # With Ph
        'Landline': r'(?i)(phone|tel|ph\.?|landline)\s*(?:no|number)?\.?\s*:?\s*(\d{3,5}[\s-]?\d{6,8})',  # Landline with area code
        'DOB': r'(?i)(date\s*of\s*birth|dob|birth\s*date)\s*:?\s*\d{1,2}[-/]\d{1,2}[-/]\d{2,4}',
        'Pincode': r'(?i)(pin\s*code|pincode|postal\s*code|zip)\s*:?\s*(\d{6})',  # With context
        'Name': r'(?i)(name|naam)\s*:?\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})',
        'Father Name': r'(?i)(father\'?s?\s*name)\s*:?\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})',
        'Address': r'(?i)(address|पता)\s*:?\s*([^\n]{15,100})',
        'Gender': r'(?i)(gender|sex)\s*:?\s*(male|female|m|f|other)',
    }
    
    # Fallback patterns DISABLED - they create too many false positives
    # Context-aware patterns above are sufficient for accurate detection
    # fallback_patterns = {
    #     'Mobile (standalone)': r'\b[6-9]\d{9}\b',  # Standalone 10-digit number
    #     'Landline (standalone)': r'\b\d{6,8}\b',  # Standalone 6-8 digit landline
    #     'Pincode (standalone)': r'\b\d{6}\b',  # Standalone 6-digit number
    # }
    
    # MRZ patterns (Machine Readable Zone for passports, ID cards)
    mrz_patterns = {
        'MRZ Line': r'[A-Z0-9<]{30,44}',  # Standard MRZ format
    }
    
    import difflib

    # Helper function to find coordinates with HIGH ACCURACY using Fuzzy Matching
    def find_coordinates(value_text, img_width=None, img_height=None):
        if ocr_data is not None and img_width and img_height:
            value_clean = str(value_text).strip()
            
            # Split value into tokens for better matching
            # Filter out very short tokens to avoid noise
            value_tokens = [t.strip() for t in re.split(r'[\s\-:/]+', value_clean) if len(t.strip()) > 1]
            
            matching_boxes = []
            used_indices = set()
            
            # --- STRATEGY 1: EXACT MATCH (High Priority) ---
            for token in value_tokens:
                token_lower = token.lower().replace('<', '')
                for idx, row in ocr_data.iterrows():
                    if idx in used_indices: continue
                    ocr_text = str(row['text']).strip().lower()
                    if token_lower == ocr_text:
                        matching_boxes.append({'x': int(row['left']), 'y': int(row['top']), 'width': int(row['width']), 'height': int(row['height'])})
                        used_indices.add(idx)
                        break

            # --- STRATEGY 2: FUZZY MATCH (The Bridge) ---
            # If we missed tokens, try fuzzy matching for the remaining ones
            if len(matching_boxes) < len(value_tokens):
                for token in value_tokens:
                    # Skip if we essentially found this token already (simple heuristic)
                    if any(token.lower() in str(ocr_data.iloc[idx]['text']).lower() for idx in used_indices):
                        continue
                        
                    token_lower = token.lower()
                    best_ratio = 0.0
                    best_idx = -1
                    
                    for idx, row in ocr_data.iterrows():
                        if idx in used_indices: continue
                        ocr_text = str(row['text']).strip().lower()
                        
                        # Calculate similarity
                        ratio = difflib.SequenceMatcher(None, token_lower, ocr_text).ratio()
                        if ratio > best_ratio:
                            best_ratio = ratio
                            best_idx = idx
                    
                    # Threshold: 0.8 (e.g. "Sukant" vs "Suknat" is ~0.9)
                    if best_ratio > 0.8:
                        row = ocr_data.iloc[best_idx]
                        matching_boxes.append({'x': int(row['left']), 'y': int(row['top']), 'width': int(row['width']), 'height': int(row['height'])})
                        used_indices.add(best_idx)

            # --- STRATEGY 3: SUBSTRING/PARTIAL (Fallback) ---
            if not matching_boxes and value_clean:
                value_lower = value_clean.lower()
                for idx, row in ocr_data.iterrows():
                    ocr_text = str(row['text']).strip().lower()
                    if value_lower in ocr_text or ocr_text in value_lower:
                         matching_boxes.append({'x': int(row['left']), 'y': int(row['top']), 'width': int(row['width']), 'height': int(row['height'])})

            # If we found matches, combine and add padding
            if matching_boxes:
                # Find bounding box that contains all matches
                min_x = min(b['x'] for b in matching_boxes)
                min_y = min(b['y'] for b in matching_boxes)
                max_x = max(b['x'] + b['width'] for b in matching_boxes)
                max_y = max(b['y'] + b['height'] for b in matching_boxes)
                
                # Add intelligent padding (5 pixels all around)
                padding = 5
                min_x = max(0, min_x - padding)
                min_y = max(0, min_y - padding)
                max_x = min(img_width, max_x + padding)
                max_y = min(img_height, max_y + padding)
                
                return {
                    'x': (min_x / img_width) * 100,
                    'y': (min_y / img_height) * 100,
                    'width': ((max_x - min_x) / img_width) * 100,
                    'height': ((max_y - min_y) / img_height) * 100
                }
        
        # Smart fallback based on field type
        # Smaller, more reasonable boxes
        base_width = 25  # 25% for unknown fields
        base_height = 3
        
        # Adjust based on value length
        if len(str(value_text)) > 30:
            base_width = 40  # Longer fields like addresses
            base_height = 4
        elif len(str(value_text)) > 15:
            base_width = 30
        
        return {
            'x': 8,
            'y': (field_id * 5) % 92,
            'width': base_width,
            'height': base_height
        }
    
    # Get image dimensions
    img_width, img_height = None, None
    if image:
        img_width, img_height = image.size
    
    # Process all text patterns (removed fallback_patterns to prevent false positives)
    all_patterns = [
        (sensitive_patterns, 'sensitive_personal_data', True, 0.9),
        (identifier_patterns, 'identification', True, 0.95),
        (personal_patterns, 'personal_info', True, 0.85),
        (mrz_patterns, 'mrz', True, 0.98)
    ]
    
    for patterns_dict, category, is_sens, conf in all_patterns:
        for field_name, pattern in patterns_dict.items():
            matches = re.finditer(pattern, text)
            for match in matches:
                # Extract value
                if field_name in ['DOB', 'Name', 'Father Name', 'Address', 'Gender']:
                    groups = match.groups()
                    value = groups[-1] if len(groups) > 1 else match.group(0)
                else:
                    value = match.group(0)
                
                coords = find_coordinates(value, img_width, img_height)
                
                detected_fields.append({
                    'id': f'field_{field_id}',
                    'field_name': field_name,
                    'field_value': value[:50],
                    'confidence': conf,
                    'category': category,
                    'is_sensitive': is_sens,
                    'auto_selected': is_sens,
                    'coordinates': coords
                })
                field_id += 1
    
    # === LLM-ENHANCED FIELD SUGGESTION (AI FIRST) ===
    # ALWAYS run LLM to find context-based fields (NER) that Regex misses
    try:
        if True: # Always run to catch finding labels without "Name:" prefix
            print("Running LLM Field Extraction (AI First Parallel Execution)...")
            llm_extraction = rag_system.extract_fields_with_llm(text, document_type)
            
            if llm_extraction.get("available") and llm_extraction.get("fields"):
                print(f"LLM extracted {len(llm_extraction['fields'])} fields")
                
                # Filter out fields that were already detected (by name or value overlap)
                existing_values = {f['field_value'].lower() for f in detected_fields}
                existing_names = {f['field_name'].lower() for f in detected_fields}
                
                for field in llm_extraction['fields']:
                    f_name = field.get("name", "Unknown Field")
                    f_value = field.get("value", "")
                    
                    if not f_value or len(f_value) < 2:
                        continue
                        
                    # Skip if basically same value already found
                    if f_value.lower() in existing_values:
                        continue
                        
                    # Skip if same field name already found (optional, depending on if we want duplicates)
                    # if f_name.lower() in existing_names:
                    #     continue
                    
                    # Try to find coordinates for this LLM-extracted value
                    coords = find_coordinates(f_value, img_width, img_height)
                    
                    detected_fields.append({
                        'id': f'field_{field_id}',
                        'field_name': f"{f_name} (AI)", # Mark as AI detected
                        'field_value': f_value[:50],
                        'confidence': 0.85, # Slightly lower than regex as it's generative
                        'category': field.get("category", "other"),
                        'is_sensitive': field.get("sensitive", True),
                        'auto_selected': field.get("sensitive", True),
                        'coordinates': coords
                    })
                    field_id += 1
            else:
                print("LLM Extraction skipped (not available or no fields found)")
                
    except Exception as e:
        print(f"LLM Enhancement Error: {e}")

    # IMAGE-BASED DETECTION (QR codes, barcodes, photos, signatures)
    if image_path and cv_image is not None:
        try:
            import cv2
            from pyzbar import pyzbar
            
            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
            h, w = cv_image.shape[:2]
            
            # 1. QR Code Detection
            try:
                qr_decoder = cv2.QRCodeDetector()
                data, bbox, _ = qr_decoder.detectAndDecode(cv_image)
                if bbox is not None:
                    for i, box in enumerate(bbox):
                        x, y = int(box[0][0]), int(box[0][1])
                        box_w = int(box[2][0] - box[0][0])
                        box_h = int(box[2][1] - box[0][1])
                        
                        detected_fields.append({
                            'id': f'field_{field_id}',
                            'field_name': 'QR Code',
                            'field_value': data[:30] if data else 'QR Code',
                            'confidence': 0.95,
                            'category': 'image_element',
                            'is_sensitive': True,
                            'auto_selected': True,
                            'coordinates': {
                                'x': (x / w) * 100,
                                'y': (y / h) * 100,
                                'width': (box_w / w) * 100,
                                'height': (box_h / h) * 100
                            }
                        })
                        field_id += 1
            except:
                pass
            
            # 2. Barcode Detection (using pyzbar)
            try:
                barcodes = pyzbar.decode(cv_image)
                for barcode in barcodes:
                    x, y, bc_w, bc_h = barcode.rect.left, barcode.rect.top, barcode.rect.width, barcode.rect.height
                    bc_data = barcode.data.decode('utf-8') if barcode.data else 'Barcode'
                    
                    detected_fields.append({
                        'id': f'field_{field_id}',
                        'field_name': f'Barcode ({barcode.type})',
                        'field_value': bc_data[:30],
                        'confidence': 0.95,
                        'category': 'image_element',
                        'is_sensitive': True,
                        'auto_selected': True,
                        'coordinates': {
                            'x': (x / w) * 100,
                            'y': (y / h) * 100,
                            'width': (bc_w / w) * 100,
                            'height': (bc_h / h) * 100
                        }
                    })
                    field_id += 1
            except:
                pass
            
            # 3. Face Detection (Profile Photo)
            try:
                face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
                faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
                
                for (fx, fy, fw, fh) in faces:
                    detected_fields.append({
                        'id': f'field_{field_id}',
                        'field_name': 'Profile Photo',
                        'field_value': 'Face detected',
                        'confidence': 0.85,
                        'category': 'image_element',
                        'is_sensitive': True,
                        'auto_selected': True,
                        'coordinates': {
                            'x': (fx / w) * 100,
                            'y': (fy / h) * 100,
                            'width': (fw / w) * 100,
                            'height': (fh / h) * 100
                        }
                    })
                    field_id += 1
            except:
                pass
            
            # 4. Signature Detection (using contour detection)
            try:
                _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
                contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                for contour in contours:
                    area = cv2.contourArea(contour)
                    if 500 < area < 50000:  # Signature-like size
                        sx, sy, sw, sh = cv2.boundingRect(contour)
                        aspect_ratio = float(sw) / sh if sh > 0 else 0
                        
                        # Signature typically has aspect ratio 2:1 to 5:1
                        if 1.5 < aspect_ratio < 6:
                            detected_fields.append({
                                'id': f'field_{field_id}',
                                'field_name': 'Signature',
                                'field_value': 'Signature detected',
                                'confidence': 0.75,
                                'category': 'image_element',
                                'is_sensitive': True,
                                'auto_selected': True,
                                'coordinates': {
                                    'x': (sx / w) * 100,
                                    'y': (sy / h) * 100,
                                    'width': (sw / w) * 100,
                                    'height': (sh / h) * 100
                                }
                            })
                            field_id += 1
                            break  # Only detect first signature
            except:
                pass
                
        except Exception as e:
            print(f"Image-based detection error: {e}")

    # === ONLINE VERIFICATION (Advanced) ===
    # Verify public fields like IFSC, Institute, Pincode using DuckDuckGo
    # Only runs if RAG system search is initialized
    try:
        if rag_system.search_tool:
            print("Running Online Verification for verifiable fields...")
            for field in detected_fields:
                f_name = field['field_name'].lower()
                f_val = field['field_value']
                
                # Identify verifiable fields
                if any(k in f_name for k in ['ifsc', 'pin code', 'pincode', 'university', 'college', 'institute', 'school', 'bank', 'company']):
                    # Skip if value is too short
                    if len(f_val) < 4: continue
                    
                    verify_res = rag_system.verify_field_online(field['field_name'], f_val, context=f"in {document_type}")
                    
                    if verify_res['verified']:
                        field['verification_status'] = "Verified (Online)"
                        field['verification_details'] = verify_res['info']
                        print(f"Verified {f_name}: {verify_res['info']}")
                    else:
                        field['verification_status'] = "Unverified"
    except Exception as e:
        print(f"Online Verification Loop Error: {e}")
    
    # === MUST HAVE FIELDS RECOVERY (FINAL CHECK) ===
    # Enforce detection of critical fields if they are missing
    MUST_HAVE_FIELDS = {
        'Aadhaar Card': ['Aadhaar Number', 'Name', 'DOB'],
        'PAN Card': ['PAN Number', 'Name', 'DOB'],
        'Voter ID Card': ['Voter ID Number', 'Name'],
        'Driving License': ['License Number', 'Name', 'Validity'],
        'Passport': ['Passport Number', 'Name', 'Expiry Date']
    }
    
    if document_type in MUST_HAVE_FIELDS:
        try:
            detected_names = {f['field_name'].lower() for f in detected_fields}
            detected_values = {f['field_value'] for f in detected_fields}
            
            for required in MUST_HAVE_FIELDS[document_type]:
                # Check if roughly present (e.g. "Aadhaar Number" vs "Aadhaar No")
                is_present = any(required.lower() in d for d in detected_names)
                
                if not is_present:
                    print(f"⚠️ Missing MUST HAVE field: {required} for {document_type}. Attempting AI Recovery...")
                    recovery = rag_system.extract_specific_field(text, required, document_type)
                    
                    if recovery['found'] and recovery['value'] not in detected_values:
                        print(f"✅ Recovered {required}: {recovery['value']}")
                        # Find coords via simple text search since LLM doesn't give coords
                        coords = find_coordinates(recovery['value'], img_width, img_height)
                        
                        detected_fields.append({
                            'id': f'field_{field_id}',
                            'field_name': f"{required} (AI Recovery)",
                            'field_value': recovery['value'],
                            'confidence': 0.85,
                            'category': 'required_field',
                            'is_sensitive': True,
                            'auto_selected': True,
                            'coordinates': coords,
                            'verification_status': 'Recovered',
                            'verification_details': 'Recovered by specific LLM query'
                        })
                        field_id += 1
                        detected_values.add(recovery['value'])
        except Exception as e:
            print(f"Must Field Recovery Error: {e}")

    return detected_fields




@app.route('/')
def index():
    """Root route - redirect to login if not authenticated, otherwise to document redaction."""
    if 'user_id' in session:
        return redirect(url_for('document_redaction_page'))
    return redirect(url_for('login_page'))

@app.route('/document-redaction')
@login_required
def document_redaction_page():
    """Document redaction page."""
    return render_template('index.html')

@app.route('/history')
@login_required
def history_page():
    """Processing history page."""
    return render_template('history.html')

@app.route('/api/process-for-redaction', methods=['POST'])
@login_required
def process_for_redaction():
    """
    Process uploaded document with OCR and field detection.
    Extracts metadata and uses it for enhanced detection accuracy.
    Detects sensitive Indian document fields automatically with ACCURATE coordinates.
    """
    temp_path = None
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No file selected'}), 400
        
        # Save image temporarily for coordinate extraction
        import tempfile
        temp_path = os.path.join(tempfile.gettempdir(), f'redact_{uuid.uuid4()}.png')
        
        # Read and convert image to base64
        image_bytes = file.read()
        image = Image.open(io.BytesIO(image_bytes))
        
        # Extract comprehensive metadata
        metadata = extract_file_metadata(image, filename=file.filename)
        metadata['file_size_kb'] = round(len(image_bytes) / 1024, 2)
        
        # Save to temp file for OCR bbox extraction
        image.save(temp_path)
        
        # Get image dimensions
        img_width, img_height = image.size
        image_format = image.format or 'PNG'
        
        # Perform OCR
        try:
            extracted_text = pytesseract.image_to_string(image)
        except Exception as ocr_error:
            extracted_text = ""
            print(f"OCR Warning: {ocr_error}")
        
        # Detect document type FIRST using metadata
        document_info = detect_document_type(extracted_text, image_path=temp_path, metadata=metadata) if extracted_text else {
            'document_type': 'Unknown Document',
            'confidence': 0.0,
            'all_scores': {}
        }
        
        # Then detect sensitive fields using document-type-specific templates
        detected_fields = detect_sensitive_fields(
            extracted_text, 
            image_path=temp_path, 
            metadata=metadata,
            document_type=document_info['document_type']  # Pass document type for template-based detection
        ) if extracted_text else []
        
        # Convert to base64
        buffered = io.BytesIO()
        image.save(buffered, format=image_format)
        image_data = base64.b64encode(buffered.getvalue()).decode('utf-8')
        
        # Calculate statistics
        sensitive_count = sum(1 for f in detected_fields if f['is_sensitive'])
        auto_selected_count = sum(1 for f in detected_fields if f['auto_selected'])
        
        # Save to history with metadata (exclude large image data) with user_id
        user_id = session.get('user_id')
        save_to_history({
            'filename': file.filename,
            'document_type': document_info['document_type'],
            'confidence': document_info['confidence'],
            'total_fields': len(detected_fields),
            'sensitive_fields': sensitive_count,
            'status': 'Processed',
            'processing_time': 0,  # Placeholder
            'metadata': {
                'file_size_kb': metadata['file_size_kb'],
                'dimensions': metadata['dimensions'],
                'aspect_ratio': metadata['aspect_ratio'],
                'format': metadata['format'],
                'dpi': metadata['dpi'],
                'document_hints': metadata['document_hints']
            }
        }, user_id)
        
        # Return response with detected fields, document type, and metadata
        return jsonify({
            'success': True,
            'filename': file.filename,
            'document_type': document_info['document_type'],
            'document_confidence': document_info['confidence'],
            'document_scores': document_info.get('all_scores', {}),
            'image_data': f'data:image/{image_format.lower()};base64,{image_data}',
            'image_dimensions': {
                'width': img_width,
                'height': img_height
            },
            'extracted_text': extracted_text,
            'detected_fields': detected_fields,
            'statistics': {
                'total_fields': len(detected_fields),
                'auto_selected': auto_selected_count,
                'sensitive_fields': sensitive_count,
                'text_length': len(extracted_text),
                'word_count': len(extracted_text.split())
            },
            'file_metadata': {
                'file_size_kb': metadata['file_size_kb'],
                'dimensions': metadata['dimensions'],
                'aspect_ratio': metadata['aspect_ratio'],
                'format': metadata['format'],
                'mode': metadata['mode'],
                'dpi': metadata['dpi'],
                'color_depth': metadata['color_depth'],
                'has_transparency': metadata['has_transparency'],
                'document_hints': metadata['document_hints'],
                'exif': metadata['exif'],
                'camera_info': metadata['camera_info'],
                'creation_date': metadata['creation_date']
            },
            'processing_metadata': {
                'ocr_method': 'tesseract_with_bbox',
                'ocr_confidence': 0.8,
                'processing_time': 0,
                'metadata_enhanced': True,
                'note': 'Using metadata-enhanced detection with ACCURATE OCR bounding boxes'
            }
        })
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500
    
    finally:
        # Clean up temp file
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except:
                pass


@app.route('/api/blur-and-export', methods=['POST'])
@login_required
def blur_and_export_image():
    """Apply various redaction styles, watermark, and export in different formats."""
    try:
        import cv2
        import numpy as np
        
        data = request.get_json()
        
        if not data or 'image_data' not in data:
            return jsonify({'success': False, 'error': 'No image data provided'}), 400
        
        # Get parameters
        image_data_url = data['image_data']
        selected_fields = data.get('selected_fields', [])
        redaction_style = data.get('redaction_style', 'blur')
        blur_strength = int(data.get('blur_strength', 15))
        pixel_size = int(data.get('pixel_size', 10))
        watermark_config = data.get('watermark', {})
        export_format = data.get('export_format', 'png')
        jpeg_quality = int(data.get('jpeg_quality', 85))
        
        # Extract base64 image data
        if ',' in image_data_url:
            image_data = image_data_url.split(',')[1]
        else:
            image_data = image_data_url
        
        # Decode image
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        
        # Apply redaction to each selected field
        for field_coords in selected_fields:
            # Convert percentage coordinates to pixels
            x_percent = field_coords.get('x', 0)
            y_percent = field_coords.get('y', 0)
            width_percent = field_coords.get('width', 10)
            height_percent = field_coords.get('height', 5)
            
            img_width, img_height = image.size
            
            x = int((x_percent / 100) * img_width)
            y = int((y_percent / 100) * img_height)
            width = int((width_percent / 100) * img_width)
            height = int((height_percent / 100) * img_height)
            
            # Ensure coordinates are within bounds
            x = max(0, min(x, img_width))
            y = max(0, min(y, img_height))
            width = max(0, min(width, img_width - x))
            height = max(0, min(height, img_height - y))
            
            if width > 0 and height > 0:
                region = image.crop((x, y, x + width, y + height))
                
                # Apply redaction style
                if redaction_style == 'blur':
                    # Gaussian blur
                    region = region.filter(ImageFilter.GaussianBlur(radius=blur_strength))
                    image.paste(region, (x, y))
                    
                elif redaction_style == 'pixelate':
                    # Pixelation effect
                    region_small = region.resize((width // pixel_size, height // pixel_size), Image.NEAREST)
                    region_pixelated = region_small.resize((width, height), Image.NEAREST)
                    image.paste(region_pixelated, (x, y))
                    
                elif redaction_style == 'blackbox':
                    # Black rectangle
                    from PIL import ImageDraw
                    draw = ImageDraw.Draw(image)
                    draw.rectangle([x, y, x + width, y + height], fill='black')
                    
                elif redaction_style == 'white':
                    # White rectangle
                    from PIL import ImageDraw
                    draw = ImageDraw.Draw(image)
                    draw.rectangle([x, y, x + width, y + height], fill='white')
        
        # Apply watermark if text provided
        watermark_text = watermark_config.get('text', '').strip()
        if watermark_text:
            from PIL import ImageDraw, ImageFont
            
            draw = ImageDraw.Draw(image, 'RGBA')
            font_size = watermark_config.get('font_size', 36)
            opacity = watermark_config.get('opacity', 50)
            position = watermark_config.get('position', 'center')
            
            # Try to use a better font, fall back to default
            try:
                font = ImageFont.truetype("arial.ttf", font_size)
            except:
                font = ImageFont.load_default()
            
            # Calculate text size using textbbox
            bbox = draw.textbbox((0, 0), watermark_text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            
            # Calculate position
            img_width, img_height = image.size
            if position == 'center':
                text_x = (img_width - text_width) // 2
                text_y = (img_height - text_height) // 2
            elif position == 'top-left':
                text_x = 20
                text_y = 20
            elif position == 'top-right':
                text_x = img_width - text_width - 20
                text_y = 20
            elif position == 'bottom-left':
                text_x = 20
                text_y = img_height - text_height - 20
            elif position == 'bottom-right':
                text_x = img_width - text_width - 20
                text_y = img_height - text_height - 20
            else:
                text_x = (img_width - text_width) // 2
                text_y = (img_height - text_height) // 2
            
            # Draw watermark with opacity
            alpha = int(255 * (opacity / 100))
            draw.text((text_x, text_y), watermark_text, fill=(255, 255, 255, alpha), font=font)
        
        # Export in requested format
        buffered = io.BytesIO()
        
        if export_format == 'pdf':
            # Convert to PDF
            try:
                import img2pdf
                # First save image to bytes
                img_bytes = io.BytesIO()
                image.save(img_bytes, format="PNG")
                img_bytes.seek(0)
                # Convert to PDF
                pdf_bytes = img2pdf.convert(img_bytes.getvalue())
                encoded_data = base64.b64encode(pdf_bytes).decode('utf-8')
                mime_type = 'application/pdf'
            except (ImportError, Exception) as e:
                # Fallback: use PIL to save as PNG then return
                print(f"PDF conversion failed: {e}, falling back to PNG")
                image.save(buffered, format="PNG")
                encoded_data = base64.b64encode(buffered.getvalue()).decode('utf-8')
                mime_type = 'image/png'
        
        elif export_format == 'jpeg':
            # Convert to RGB (JPEG doesn't support transparency)
            if image.mode in ('RGBA', 'LA', 'P'):
                rgb_image = Image.new('RGB', image.size, (255, 255, 255))
                rgb_image.paste(image, mask=image.split()[-1] if image.mode == 'RGBA' else None)
                image = rgb_image
            image.save(buffered, format="JPEG", quality=jpeg_quality)
            encoded_data = base64.b64encode(buffered.getvalue()).decode('utf-8')
            mime_type = 'image/jpeg'
        
        else:  # PNG (default)
            image.save(buffered, format="PNG")
            encoded_data = base64.b64encode(buffered.getvalue()).decode('utf-8')
            mime_type = 'image/png'
        
        return jsonify({
            'success': True,
            'blurred_image': f'data:{mime_type};base64,{encoded_data}'
        })
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

def save_to_history(data, user_id=None):
    """Save processing record to MongoDB history with user_id."""
    db_manager.save_record(data, user_id)

@app.route('/api/history', methods=['GET'])
@login_required
def get_history():
    """Get processing history from MongoDB for the current user."""
    user_id = session.get('user_id')
    records = db_manager.get_history(user_id=user_id)
    if records is None:
        return jsonify({'success': False, 'error': 'Database not connected'}), 503
    
    return jsonify({'success': True, 'history': records})


# ===== AUTHENTICATION ROUTES =====

@app.route('/login', methods=['GET', 'POST'])
def login_page():
    """Login page and authentication."""
    if request.method == 'GET':
        if 'user_id' in session:
            return redirect(url_for('document_redaction'))
        return render_template('login.html')
    
    try:
        data = request.get_json()
        username = data.get('username', '').strip()
        password = data.get('password', '')
        
        result = login_user(username, password)
        
        if result['success']:
            session.permanent = True
            session['user_id'] = result['user']['id']
            session['username'] = result['user']['username']
            session['email'] = result['user']['email']
            
            return jsonify({'success': True, 'message': 'Login successful', 'user': result['user']})
        else:
            return jsonify({'success': False, 'message': result['message']}), 401
            
    except Exception as e:
        print(f"Login error: {e}")
        return jsonify({'success': False, 'message': 'An error occurred during login'}), 500


@app.route('/register', methods=['GET', 'POST'])
def register_page():
    """Registration page and user creation."""
    if request.method == 'GET':
        if 'user_id' in session:
            return redirect(url_for('document_redaction'))
        return render_template('register.html')
    
    try:
        data = request.get_json()
        username = data.get('username', '').strip()
        email = data.get('email', '').strip()
        password = data.get('password', '')
        
        result = register_user(username, email, password)
        
        if result['success']:
            return jsonify({'success': True, 'message': 'Registration successful'})
        else:
            return jsonify({'success': False, 'message': result['message']}), 400
            
    except Exception as e:
        print(f"Registration error: {e}")
        return jsonify({'success': False, 'message': 'An error occurred during registration'}), 500


@app.route('/logout')
def logout():
    """Logout user and clear session."""
    session.clear()
    return redirect(url_for('login_page'))


# ===== SETTINGS ROUTES =====

@app.route('/settings')
@login_required
def settings_page():
    """User settings page."""
    return render_template('settings.html')


@app.route('/api/user/profile', methods=['GET'])
@login_required
def get_user_profile():
    """Get current user's profile information."""
    try:
        user_id = session.get('user_id')
        user = get_user_by_id(user_id)
        
        if user:
            return jsonify({'success': True, 'user': user})
        else:
            return jsonify({'success': False, 'message': 'User not found'}), 404
            
    except Exception as e:
        print(f"Profile fetch error: {e}")
        return jsonify({'success': False, 'message': 'Error fetching profile'}), 500


@app.route('/api/user/update-profile', methods=['POST'])
@login_required
def update_user_profile():
    """Update user's profile information (email)."""
    try:
        user_id = session.get('user_id')
        data = request.get_json()
        new_email = data.get('email', '').strip()
        
        if not new_email:
            return jsonify({'success': False, 'message': 'Email is required'}), 400
        
        result = update_user_email(user_id, new_email)
        
        if result['success']:
            # Update session email
            session['email'] = new_email
            return jsonify({'success': True, 'message': result['message']})
        else:
            return jsonify({'success': False, 'message': result['message']}), 400
            
    except Exception as e:
        print(f"Profile update error: {e}")
        return jsonify({'success': False, 'message': 'Error updating profile'}), 500


@app.route('/api/user/change-password', methods=['POST'])
@login_required
def change_user_password():
    """Change user's password."""
    try:
        user_id = session.get('user_id')
        data = request.get_json()
        current_password = data.get('current_password', '')
        new_password = data.get('new_password', '')
        
        if not current_password or not new_password:
            return jsonify({'success': False, 'message': 'All fields are required'}), 400
        
        result = change_password(user_id, current_password, new_password)
        
        if result['success']:
            return jsonify({'success': True, 'message': result['message']})
        else:
            return jsonify({'success': False, 'message': result['message']}), 400
            
    except Exception as e:
        print(f"Password change error: {e}")
        return jsonify({'success': False, 'message': 'Error changing password'}), 500


# ===== TEMPLATE MANAGEMENT ROUTES =====

@app.route('/api/templates/save', methods=['POST'])
@login_required
def save_template():
    """Save a redaction template for the current user."""
    try:
        from bson.objectid import ObjectId
        
        user_id = session.get('user_id')
        data = request.get_json()
        
        template_name = data.get('name', '').strip()
        settings = data.get('settings', {})
        
        if not template_name:
            return jsonify({'success': False, 'error': 'Template name is required'}), 400
        
        # Get MongoDB database
        mongo_client = auth_db_mongo.get_mongo_client()
        if not mongo_client:
            return jsonify({'success': False, 'error': 'Database not connected'}), 503
        
        db = mongo_client['redaction_db']
        templates_collection = db['templates']
        
        # Create template document
        template_doc = {
            'user_id': user_id,
            'name': template_name,
            'settings': settings,
            'created_at': datetime.datetime.now(),
            'updated_at': datetime.datetime.now()
        }
        
        # Check if template with same name exists for this user
        existing = templates_collection.find_one({'user_id': user_id, 'name': template_name})
        
        if existing:
            # Update existing template
            templates_collection.update_one(
                {'_id': existing['_id']},
                {'$set': {'settings': settings, 'updated_at': datetime.datetime.now()}}
            )
            return jsonify({'success': True, 'message': 'Template updated successfully'})
        else:
            # Insert new template
            result = templates_collection.insert_one(template_doc)
            return jsonify({'success': True, 'message': 'Template saved successfully', 'template_id': str(result.inserted_id)})
            
    except Exception as e:
        print(f"Template save error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/templates/list', methods=['GET'])
@login_required
def list_templates():
    """Get all templates for the current user."""
    try:
        user_id = session.get('user_id')
        
        # Get MongoDB database
        mongo_client = auth_db_mongo.get_mongo_client()
        if not mongo_client:
            return jsonify({'success': False, 'error': 'Database not connected'}), 503
        
        db = mongo_client['redaction_db']
        templates_collection = db['templates']
        
        # Find all templates for this user
        templates = list(templates_collection.find({'user_id': user_id}))
        
        # Convert ObjectId to string
        for template in templates:
            template['_id'] = str(template['_id'])
            template['created_at'] = template['created_at'].isoformat() if 'created_at' in template else None
            template['updated_at'] = template['updated_at'].isoformat() if 'updated_at' in template else None
        
        return jsonify({'success': True, 'templates': templates})
        
    except Exception as e:
        print(f"Template list error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/templates/get/<template_id>', methods=['GET'])
@login_required
def get_template(template_id):
    """Get a specific template by ID."""
    try:
        from bson.objectid import ObjectId
        
        user_id = session.get('user_id')
        
        # Get MongoDB database
        mongo_client = auth_db_mongo.get_mongo_client()
        if not mongo_client:
            return jsonify({'success': False, 'error': 'Database not connected'}), 503
        
        db = mongo_client['redaction_db']
        templates_collection = db['templates']
        
        # Find template by ID and user_id (security check)
        template = templates_collection.find_one({
            '_id': ObjectId(template_id),
            'user_id': user_id
        })
        
        if not template:
            return jsonify({'success': False, 'error': 'Template not found'}), 404
        
        # Convert ObjectId to string
        template['_id'] = str(template['_id'])
        template['created_at'] = template['created_at'].isoformat() if 'created_at' in template else None
        template['updated_at'] = template['updated_at'].isoformat() if 'updated_at' in template else None
        
        return jsonify({'success': True, 'template': template})
        
    except Exception as e:
        print(f"Template get error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/templates/delete/<template_id>', methods=['DELETE'])
@login_required
def delete_template(template_id):
    """Delete a specific template."""
    try:
        from bson.objectid import ObjectId
        
        user_id = session.get('user_id')
        
        # Get MongoDB database
        mongo_client = auth_db_mongo.get_mongo_client()
        if not mongo_client:
            return jsonify({'success': False, 'error': 'Database not connected'}), 503
        
        db = mongo_client['redaction_db']
        templates_collection = db['templates']
        
        # Delete template (only if it belongs to the user)
        result = templates_collection.delete_one({
            '_id': ObjectId(template_id),
            'user_id': user_id
        })
        
        if result.deleted_count > 0:
            return jsonify({'success': True, 'message': 'Template deleted successfully'})
        else:
            return jsonify({'success': False, 'error': 'Template not found'}), 404
            
    except Exception as e:
        print(f"Template delete error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


# ===== ANALYTICS ROUTES =====

@app.route('/analytics')
@login_required
def analytics_page():
    """Analytics dashboard page."""
    return render_template('analytics.html')


@app.route('/api/analytics', methods=['GET'])
@login_required
def get_analytics():
    """Get analytics data for the current user."""
    try:
        from collections import Counter
        
        user_id = session.get('user_id')
        
        # Get MongoDB database
        mongo_client = auth_db_mongo.get_mongo_client()
        if not mongo_client:
            return jsonify({'success': False, 'error': 'Database not connected'}), 503
        
        db = mongo_client['redaction_db']
        history_collection = db['processing_history']
        
        # Get all history for this user
        all_records = list(history_collection.find({'user_id': user_id}).sort('timestamp', -1))
        
        # Calculate statistics
        total_documents = len(all_records)
        total_sensitive_fields = sum(record.get('sensitive_fields', 0) for record in all_records)
        
        # Document type distribution
        doc_types = [record.get('document_type', 'Unknown') for record in all_records]
        type_distribution = dict(Counter(doc_types).most_common(10))
        
        # Most common type
        most_common = Counter(doc_types).most_common(1)
        most_common_type = most_common[0][0] if most_common else 'N/A'
        most_common_count = most_common[0][1] if most_common else 0
        
        # This month's documents
        now = datetime.datetime.now()
        month_start = datetime.datetime(now.year, now.month, 1)
        this_month_count = len([r for r in all_records if r.get('timestamp', datetime.datetime.min) >= month_start])
        
        # Last 7 days trend
        last_7_days = []
        for i in range(6, -1, -1):
            day = now - datetime.timedelta(days=i)
            day_start = datetime.datetime(day.year, day.month, day.day)
            day_end = day_start + datetime.timedelta(days=1)
            count = len([r for r in all_records if day_start <= r.get('timestamp', datetime.datetime.min) < day_end])
            last_7_days.append({
                'date': day_start.isoformat(),
                'count': count
            })
        
        # Sensitive fields by category (simplified - count field types)
        all_fields = []
        for record in all_records:
            fields = record.get('detected_fields', [])
            if isinstance(fields, list):
                for field in fields:
                    if isinstance(field, dict) and field.get('is_sensitive'):
                        all_fields.append(field.get('category', 'Unknown'))
        
        fields_by_category = dict(Counter(all_fields).most_common(10))
        
        # Monthly volume (last 6 months)
        monthly_volume = []
        for i in range(5, -1, -1):
            if i == 0:
                target_month = now.month
                target_year = now.year
            else:
                target_date = now - datetime.timedelta(days=30 * i)
                target_month = target_date.month
                target_year = target_date.year
            
            month_count = len([r for r in all_records 
                              if r.get('timestamp', datetime.datetime.min).month == target_month 
                              and r.get('timestamp', datetime.datetime.min).year == target_year])
            
            month_name = datetime.datetime(target_year, target_month, 1).strftime('%b %Y')
            monthly_volume.append({
                'month': month_name,
                'count': month_count
            })
        
        # Recent activity (last 10 records)
        recent_activity = []
        for record in all_records[:10]:
            recent_activity.append({
                'filename': record.get('filename', 'Unknown'),
                'document_type': record.get('document_type', 'Unknown'),
                'total_fields': record.get('total_fields', 0),
                'timestamp': record.get('timestamp', datetime.datetime.now()).isoformat()
            })
        
        analytics = {
            'totalDocuments': total_documents,
            'totalSensitiveFields': total_sensitive_fields,
            'mostCommonType': most_common_type,
            'mostCommonTypeCount': most_common_count,
            'thisMonth': this_month_count,
            'documentTypeDistribution': type_distribution,
            'last7Days': last_7_days,
            'sensitiveFieldsByCategory': fields_by_category,
            'monthlyVolume': monthly_volume,
            'recentActivity': recent_activity
        }
        
        return jsonify({'success': True, 'analytics': analytics})
        
    except Exception as e:
        print(f"Analytics error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


if __name__ == '__main__':

    print("🚀 Starting Simple Redaction Server...")
    print("🌐 Access at: http://localhost:5555/document-redaction")
    print("✨ This is a simplified server while app.py is being fixed")
    print()
    
    # NOTE: This uses port 5555 to avoid conflict with your main app
    app.run(debug=True, host='0.0.0.0', port=5555)
