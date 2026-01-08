"""
High Confidence Document Upload Processor
Handles document type detection with confidence filtering for accurate document classification.
"""

import os
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple, Any
from werkzeug.utils import secure_filename
import uuid

from ocr import extract_text, DocumentProcessor
from ocr.certificate_ocr import extract_certificate_text, preprocess_certificate
from ocr.mrz_ocr import extract_mrz_text, MRZParser, extract_raw_mrz_text
from ocr.rag_field_suggestion import RAGFieldSuggestionEngine
from ocr.deep_learning_ocr import extract_text_deep_learning, get_deep_learning_ocr_info
from document_history_db import db_manager


class HighConfidenceDocumentProcessor:
    """
    Advanced document processor that focuses on high confidence document type detection.
    Only returns document types that meet high confidence thresholds for accuracy.
    """
    
    def __init__(self, high_confidence_threshold: float = 0.7):
        """
        Initialize the high confidence document processor.
        
        Args:
            high_confidence_threshold: Minimum confidence score for document types (default 0.7)
        """
        self.doc_processor = DocumentProcessor()
        self.rag_engine = RAGFieldSuggestionEngine()
        self.high_confidence_threshold = high_confidence_threshold
        self.allowed_extensions = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff', 'webp'}
    
    def is_allowed_file(self, filename: str) -> bool:
        """Check if file extension is allowed."""
        return '.' in filename and \
               filename.rsplit('.', 1)[1].lower() in self.allowed_extensions
    
    def detect_high_confidence_document_types(self, image_path: str) -> List[Tuple[str, float]]:
        """
        Detect document types with high confidence only.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            List of (document_type, confidence_score) tuples for high confidence types only
        """
        try:
            # Quick OCR scan to detect document type
            quick_text = extract_text(image_path, config='--psm 6')
            
            # Get high confidence document types
            high_confidence_types = self.doc_processor.detector.get_high_confidence_document_types(
                quick_text, 
                min_confidence=self.high_confidence_threshold
            )
            
            return high_confidence_types
            
        except Exception as e:
            print(f"Document type detection error: {e}")
            return [('general', 0.5)]
    
    def process_document_upload(self, file, upload_folder: str) -> Dict[str, Any]:
        """
        Process uploaded document with high confidence document type detection.
        
        Args:
            file: Uploaded file object
            upload_folder: Directory to save uploaded files
            
        Returns:
            Dictionary with processing results including only high confidence document types
        """
        try:
            # Validate file
            if not file or file.filename == '':
                return {'error': 'No file provided or selected', 'status': 'error'}
            
            if not self.is_allowed_file(file.filename):
                return {
                    'error': f'File type not allowed. Supported: {", ".join(self.allowed_extensions)}',
                    'status': 'error'
                }
            
            # Generate unique filename and save file
            filename = secure_filename(file.filename)
            unique_filename = f"{uuid.uuid4()}_{filename}"
            
            # Create upload folder if needed
            Path(upload_folder).mkdir(exist_ok=True)
            file_path = Path(upload_folder) / unique_filename
            
            file.save(str(file_path))
            
            # Detect high confidence document types
            high_confidence_types = self.detect_high_confidence_document_types(str(file_path))
            
            # Get the best document type for processing
            best_doc_type = high_confidence_types[0][0] if high_confidence_types else 'general'
            best_confidence = high_confidence_types[0][1] if high_confidence_types else 0.5
            
            # Process based on detected document type
            result = self._process_by_document_type(str(file_path), best_doc_type, filename)
            
            # Add high confidence information
            result.update({
                'high_confidence_types': high_confidence_types,
                'total_high_confidence_types': len(high_confidence_types),
                'confidence_threshold': self.high_confidence_threshold,
                'high_confidence_filtering': True
            })
            
            # Clean up uploaded file
            file_path.unlink()
            
            return result
            
        except Exception as e:
            # Clean up uploaded file on error
            if 'file_path' in locals() and file_path.exists():
                file_path.unlink()
            return {'error': f'Document processing failed: {str(e)}', 'status': 'error'}
    
    def _process_by_document_type(self, file_path: str, doc_type: str, filename: str) -> Dict[str, Any]:
        """
        Process document based on detected high confidence document type.
        
        Args:
            file_path: Path to the image file
            doc_type: Detected document type
            filename: Original filename
            
        Returns:
            Dictionary with processing results
        """
        try:
            # Specialized processing for specific document types
            if doc_type == 'mrz':
                return self._process_mrz_document(file_path, filename)
            elif doc_type == 'certificate':
                return self._process_certificate_document(file_path, filename)
            else:
                return self._process_general_document(file_path, doc_type, filename)
                
        except Exception as e:
            return {'error': f'Document type processing failed: {str(e)}', 'status': 'error'}
    
    def _process_mrz_document(self, file_path: str, filename: str) -> Dict[str, Any]:
        """Process MRZ (Machine Readable Zone) documents."""
        try:
            # MRZ specific processing
            mrz_data = extract_mrz_text(file_path)
            
            if mrz_data and mrz_data.get('mrz_code'):
                # Parse MRZ for structured data
                parser = MRZParser()
                parsed_data = parser.parse_mrz(mrz_data['mrz_code'])
                
                # Generate RAG analysis
                rag_analysis = self.rag_engine.analyze_document_with_classification(
                    mrz_data.get('full_text', ''), 
                    top_k=8
                )
                
                # Save to database
                record_id = db_manager.save_document_record(
                    filename=filename,
                    document_type='mrz',
                    extracted_text=mrz_data.get('full_text', ''),
                    confidence=0.95,
                    processing_metadata={
                        'method_used': 'MRZ OCR (specialized)',
                        'high_confidence_filtering': True
                    },
                    structured_data=parsed_data,
                    rag_suggestions=rag_analysis['field_suggestions'],
                    document_classifications=rag_analysis['document_classifications']
                )
                
                return {
                    'success': True,
                    'extracted_text': mrz_data.get('full_text', ''),
                    'filename': filename,
                    'document_type': 'mrz',
                    'confidence': 0.95,
                    'method_used': 'MRZ OCR (specialized)',
                    'structured_data': parsed_data,
                    'mrz_data': mrz_data,
                    'document_classifications': rag_analysis['document_classifications'],
                    'rag_suggestions': rag_analysis['field_suggestions'],
                    'record_id': record_id,
                    'status': 'success'
                }
            else:
                # Fallback if MRZ detection fails
                return self._process_general_document(file_path, 'general', filename)
                
        except Exception as e:
            return {'error': f'MRZ processing failed: {str(e)}', 'status': 'error'}
    
    def _process_certificate_document(self, file_path: str, filename: str) -> Dict[str, Any]:
        """Process certificate documents."""
        try:
            # Certificate specific processing
            cert_text = extract_certificate_text(file_path)
            
            # Generate RAG analysis
            rag_analysis = self.rag_engine.analyze_document_with_classification(
                cert_text, 
                top_k=8
            )
            
            # Save to database
            record_id = db_manager.save_document_record(
                filename=filename,
                document_type='certificate',
                extracted_text=cert_text,
                confidence=0.85,
                processing_metadata={
                    'method_used': 'Certificate OCR (specialized)',
                    'high_confidence_filtering': True
                },
                structured_data={'certificate_text': cert_text},
                rag_suggestions=rag_analysis['field_suggestions'],
                document_classifications=rag_analysis['document_classifications']
            )
            
            return {
                'success': True,
                'extracted_text': cert_text,
                'filename': filename,
                'document_type': 'certificate',
                'confidence': 0.85,
                'method_used': 'Certificate OCR (specialized)',
                'structured_data': {'certificate_text': cert_text},
                'document_classifications': rag_analysis['document_classifications'],
                'rag_suggestions': rag_analysis['field_suggestions'],
                'record_id': record_id,
                'status': 'success'
            }
            
        except Exception as e:
            return {'error': f'Certificate processing failed: {str(e)}', 'status': 'error'}
    
    def _process_general_document(self, file_path: str, doc_type: str, filename: str) -> Dict[str, Any]:
        """Process general documents with deep learning OCR."""
        try:
            # Try deep learning OCR first, fallback to traditional OCR
            try:
                dl_result = extract_text_deep_learning(file_path, engine='auto')
                if 'error' not in dl_result and dl_result.get('text'):
                    raw_text = dl_result['text']
                    dl_confidence = dl_result.get('confidence', 0.0)
                    processing_method = f"Deep Learning OCR ({dl_result.get('engine', 'Neural Network')})"
                    dl_metadata = dl_result.get('metadata', {})
                    text_blocks = dl_result.get('text_blocks', [])
                    layout_analysis = dl_result.get('layout_analysis', {})
                else:
                    # Fallback to traditional OCR
                    raw_text = extract_text(file_path)
                    dl_confidence = 0.7
                    processing_method = "Traditional OCR (Tesseract) - Deep Learning Fallback"
                    dl_metadata = {'fallback_reason': dl_result.get('error', 'Unknown')}
                    text_blocks = []
                    layout_analysis = {}
            except Exception as e:
                # Final fallback to traditional OCR
                raw_text = extract_text(file_path)
                dl_confidence = 0.7
                processing_method = "Traditional OCR (Tesseract) - Exception Fallback"
                dl_metadata = {'fallback_reason': str(e)}
                text_blocks = []
                layout_analysis = {}
            
            # Process with comprehensive document processor
            processed_data = self.doc_processor.process_document(file_path, raw_text)
            
            # Override confidence with deep learning confidence if available
            if dl_confidence > processed_data.get('confidence', 0):
                processed_data['confidence'] = dl_confidence
            
            # Generate enhanced RAG analysis with document classification
            try:
                rag_analysis = self.rag_engine.analyze_document_with_classification(
                    raw_text, 
                    top_k=8
                )
            except Exception as e:
                print(f"RAG processing error: {e}")
                rag_analysis = {
                    "document_classifications": [],
                    "field_suggestions": [],
                    "analysis_summary": {
                        "total_classifications": 0,
                        "best_document_type": "UNKNOWN",
                        "best_confidence": "0.000",
                        "total_field_suggestions": 0,
                        "high_confidence_fields": 0,
                        "high_confidence_doc_types": 0
                    }
                }
            
            # Save to document history
            record_id = db_manager.save_document_record(
                filename=filename,
                document_type=processed_data['document_type'],
                extracted_text=raw_text.strip(),
                confidence=processed_data.get('confidence', 0.0),
                processing_metadata={
                    'method_used': processing_method,
                    'processed_at': processed_data.get('processed_at'),
                    'confidence_score': f"{processed_data.get('confidence', 0.0):.2f}",
                    'rag_enabled': True,
                    'suggestions_count': rag_analysis['analysis_summary']['total_field_suggestions'],
                    'document_classification': {
                        'best_type': rag_analysis['analysis_summary']['best_document_type'],
                        'confidence': rag_analysis['analysis_summary']['best_confidence'],
                        'total_classifications': rag_analysis['analysis_summary']['total_classifications'],
                        'high_confidence_types': rag_analysis['analysis_summary'].get('high_confidence_doc_types', 0)
                    },
                    'deep_learning_metadata': dl_metadata,
                    'text_blocks_count': len(text_blocks),
                    'layout_analysis': layout_analysis,
                    'high_confidence_filtering': True
                },
                structured_data=processed_data.get('structured_data', {}),
                rag_suggestions=rag_analysis['field_suggestions'],
                document_classifications=rag_analysis['document_classifications']
            )
            
            return {
                'success': True,
                'extracted_text': raw_text.strip(),
                'filename': filename,
                'document_type': processed_data['document_type'],
                'confidence': processed_data.get('confidence', 0.0),
                'method_used': processing_method,
                'structured_data': processed_data.get('structured_data', {}),
                'document_classifications': rag_analysis['document_classifications'],
                'rag_suggestions': rag_analysis['field_suggestions'],
                'deep_learning_info': {
                    'engine': dl_metadata.get('engine', 'Traditional'),
                    'text_blocks': len(text_blocks),
                    'layout_structure': layout_analysis.get('structure', 'unknown'),
                    'confidence_distribution': layout_analysis.get('confidence_distribution', {}),
                    'regions': layout_analysis.get('regions', [])
                },
                'processing_metadata': {
                    'processed_at': processed_data.get('processed_at'),
                    'confidence_score': f"{processed_data.get('confidence', 0.0):.2f}",
                    'rag_enabled': True,
                    'suggestions_count': rag_analysis['analysis_summary']['total_field_suggestions'],
                    'document_classification': {
                        'best_type': rag_analysis['analysis_summary']['best_document_type'],
                        'confidence': rag_analysis['analysis_summary']['best_confidence'],
                        'total_classifications': rag_analysis['analysis_summary']['total_classifications'],
                        'high_confidence_types': rag_analysis['analysis_summary'].get('high_confidence_doc_types', 0)
                    },
                    'deep_learning_enabled': True,
                    'high_confidence_filtering': True
                },
                'record_id': record_id,
                'status': 'success'
            }
            
        except Exception as e:
            return {'error': f'General document processing failed: {str(e)}', 'status': 'error'}


# Example usage
if __name__ == "__main__":
    processor = HighConfidenceDocumentProcessor(high_confidence_threshold=0.7)
    print("High Confidence Document Processor initialized")
    print(f"Confidence threshold: {processor.high_confidence_threshold}")
    print(f"Allowed file types: {processor.allowed_extensions}")
