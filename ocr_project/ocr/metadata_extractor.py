"""
Metadata extraction module for enhanced document type detection.
Extracts EXIF data, file properties, and generates document-specific hints.
"""

from PIL import Image
from PIL.ExifTags import TAGS
from typing import Dict, Any, Optional
import os
import re


def extract_file_metadata(image_path: str, filename: Optional[str] = None) -> Dict[str, Any]:
    """
    Extract comprehensive metadata from uploaded files for enhanced detection.
    
    Args:
        image_path: Path to the image file
        filename: Optional filename (if not provided, extracted from path)
        
    Returns:
        Dictionary with file properties, EXIF data, and document hints
    """
    metadata = {
        'filename': filename or os.path.basename(image_path),
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
        'document_hints': [],
        'filename_hints': []
    }
    
    try:
        # Extract filename-based hints (case-insensitive)
        if metadata['filename']:
            filename_lower = metadata['filename'].lower()
            
            # Government IDs
            if any(keyword in filename_lower for keyword in ['aadhaar', 'aadhar', 'uid']):
                metadata['filename_hints'].append('aadhaar_card')
            if any(keyword in filename_lower for keyword in ['pan', 'pan_card', 'pancard']):
                metadata['filename_hints'].append('pan_card')
            if 'passport' in filename_lower:
                metadata['filename_hints'].append('passport')
            if any(keyword in filename_lower for keyword in ['voter', 'epic', 'voter_id']):
                metadata['filename_hints'].append('voter_id')
            if any(keyword in filename_lower for keyword in ['driving', 'licence', 'license', 'dl']):
                metadata['filename_hints'].append('driving_licence')
            if 'ration' in filename_lower:
                metadata['filename_hints'].append('ration_card')
            
            # Certificates
            if 'birth' in filename_lower and 'certificate' in filename_lower:
                metadata['filename_hints'].append('birth_certificate')
            if 'marriage' in filename_lower and 'certificate' in filename_lower:
                metadata['filename_hints'].append('marriage_certificate')
            if 'mark' in filename_lower and 'sheet' in filename_lower:
                metadata['filename_hints'].append('mark_sheet')
            if 'certificate' in filename_lower:
                metadata['filename_hints'].append('certificate')
                
            # Utility Bills
            if 'electricity' in filename_lower or 'electric' in filename_lower:
                metadata['filename_hints'].append('electricity_bill')
            if 'water' in filename_lower and 'bill' in filename_lower:
                metadata['filename_hints'].append('water_bill')
            if any(keyword in filename_lower for keyword in ['telephone', 'mobile', 'phone']) and 'bill' in filename_lower:
                metadata['filename_hints'].append('telephone_bill')
            if 'gas' in filename_lower and 'bill' in filename_lower:
                metadata['filename_hints'].append('gas_bill')
                
            # Financial
            if 'bank' in filename_lower and ('statement' in filename_lower or 'passbook' in filename_lower):
                metadata['filename_hints'].append('bank_statement')
            if 'cheque' in filename_lower or 'check' in filename_lower:
                metadata['filename_hints'].append('cheque')
            if 'gst' in filename_lower:
                metadata['filename_hints'].append('gst_certificate')
            
            # Other
            if 'visa' in filename_lower:
                metadata['filename_hints'].append('visa')
            if 'rent' in filename_lower and 'agreement' in filename_lower:
                metadata['filename_hints'].append('rent_agreement')
        
        # Open image and extract properties
        with Image.open(image_path) as image:
            # Get file size
            metadata['file_size_kb'] = os.path.getsize(image_path) / 1024
            
            # Basic image properties
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
            
            # Generate document type hints based on metadata
            
            # Hint 1: Aspect ratio
            # ID cards (Aadhaar, PAN, etc.) are typically 1.5-1.7
            if 1.5 <= metadata['aspect_ratio'] <= 1.7:
                metadata['document_hints'].append('id_card_aspect_ratio')
            # Cheques are typically wider (2.0-2.5)
            elif 2.0 <= metadata['aspect_ratio'] <= 2.5:
                metadata['document_hints'].append('cheque_aspect_ratio')
            # A4 portrait is approximately 1.414
            elif 1.35 <= metadata['aspect_ratio'] <= 1.45:
                metadata['document_hints'].append('a4_portrait_ratio')
            # A4 landscape is approximately 0.707
            elif 0.65 <= metadata['aspect_ratio'] <= 0.75:
                metadata['document_hints'].append('a4_landscape_ratio')
            
            # Hint 2: Standard document sizes
            if width == 3508 and height == 2480:  # A4 at 300 DPI landscape
                metadata['document_hints'].append('a4_landscape_300dpi')
            elif width == 2480 and height == 3508:  # A4 at 300 DPI portrait
                metadata['document_hints'].append('a4_portrait_300dpi')
            elif width == 1240 and height == 1754:  # A4 at 150 DPI portrait
                metadata['document_hints'].append('a4_portrait_150dpi')
            elif width == 1754 and height == 1240:  # A4 at 150 DPI landscape
                metadata['document_hints'].append('a4_landscape_150dpi')
            
            # Hint 3: Scan quality based on DPI
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
        print(f"⚠️ Metadata extraction error: {e}")
    
    return metadata


def calculate_metadata_score(metadata: Dict[str, Any], document_type: str) -> float:
    """
    Calculate confidence score based on metadata hints for a specific document type.
    
    Args:
        metadata: Metadata dictionary from extract_file_metadata
        document_type: Target document type to score against
        
    Returns:
        Confidence score between 0.0 and 1.0
    """
    score = 0.0
    max_score = 0.0
    
    # Filename hints (40% weight)
    max_score += 0.4
    if document_type in metadata.get('filename_hints', []):
        score += 0.4
    
    # Document hints (30% weight)
    max_score += 0.3
    document_hint_map = {
        'aadhaar_card': ['id_card_aspect_ratio', 'high_quality_scan'],
        'pan_card': ['id_card_aspect_ratio', 'high_quality_scan'],
        'passport': ['a4_portrait_ratio', 'high_quality_scan'],
        'voter_id': ['id_card_aspect_ratio'],
        'driving_licence': ['id_card_aspect_ratio'],
        'ration_card': ['id_card_aspect_ratio'],
        'birth_certificate': ['a4_portrait_ratio', 'scanned_document'],
        'marriage_certificate': ['a4_portrait_ratio', 'scanned_document'],
        'electricity_bill': ['a4_portrait_ratio', 'scanned_document'],
        'water_bill': ['a4_portrait_ratio', 'scanned_document'],
        'gas_bill': ['a4_portrait_ratio', 'scanned_document'],
        'telephone_bill': ['a4_portrait_ratio', 'scanned_document'],
        'bank_statement': ['a4_portrait_ratio', 'scanned_document'],
        'cheque': ['cheque_aspect_ratio'],
        'gst_certificate': ['a4_portrait_ratio', 'scanned_document'],
        'mark_sheet': ['a4_portrait_ratio', 'scanned_document'],
        'certificate': ['a4_portrait_ratio', 'scanned_document']
    }
    
    expected_hints = document_hint_map.get(document_type, [])
    if expected_hints:
        hint_matches = sum(1 for hint in expected_hints if hint in metadata.get('document_hints', []))
        score += 0.3 * (hint_matches / len(expected_hints))
    
    # Quality indicators (30% weight)
    max_score += 0.3
    quality_score = 0.0
    
    # High resolution is good for most documents
    if 'high_resolution' in metadata.get('document_hints', []):
        quality_score += 0.15
    if 'high_quality_scan' in metadata.get('document_hints', []) or 'medium_quality_scan' in metadata.get('document_hints', []):
        quality_score += 0.15
    
    score += quality_score
    
    # Normalize to 0-1 range
    return min(score, 1.0)
