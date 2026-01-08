# Document Redaction API Endpoint
# ADD THIS CODE TO app.py BEFORE if __name__ == '__main__':

@app.route('/document-redaction')
def document_redaction_page():
    """Document redaction page for the redaction workflow."""
    return render_template('document_redaction.html')

@app.route('/api/process-for-redaction', methods=['POST'])
def process_for_redaction():
    """
    Process document for redaction workflow.
    Returns: OCR text + ML-detected fields with coordinates + original image
    Optimized single endpoint for the complete redaction flow.
    """
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'success': False, 'error': f'File type not allowed. Supported: {", ".join(ALLOWED_EXTENSIONS)}'}), 400
        
        # Generate unique filename
        filename = secure_filename(file.filename)
        unique_filename = f"{uuid.uuid4()}_{filename}"
        
        # Create upload folder if needed
        create_upload_folder()
        
        # Save uploaded file
        file_path = Path(app.config['UPLOAD_FOLDER']) / unique_filename
        file.save(file_path)
        
        try:
            # Get image dimensions for coordinate conversion
            from PIL import Image
            with Image.open(file_path) as img:
                img_width, img_height = img.size
                image_format = img.format
            
            # Step 1: OCR Extraction (fast mode for redaction)
            logger.info("🔍 Step 1: OCR extraction for redaction...")
            ocr_result = quick_ocr_extract(str(file_path))
            extracted_text = ocr_result['text']
            
            if not extracted_text or not extracted_text.strip():
                file_path.unlink()
                return jsonify({
                    'success': False,
                    'error': 'No text could be extracted from the document'
                })
            
            # Step 2: Field Detection with ML
            logger.info("🎯 Step 2: Detecting fields with ML model...")
            
            detected_fields = []
            if FIELD_DETECTION_AVAILABLE and field_extraction_pipeline:
                try:
                    # Extract fields using the pipeline
                    analysis = field_extraction_pipeline.extract_fields_from_text(
                        extracted_text,
                        document_image_path=str(file_path)
                    )
                    
                    # Process detected fields and add coordinates
                    for idx, field in enumerate(analysis.extracted_fields):
                        field_data = {
                            'id': f'field_{idx}',
                            'field_name': getattr(field, 'field_name', 'unknown'),
                            'field_value': getattr(field, 'field_value', ''),
                            'confidence': getattr(field, 'confidence', 0),
                            'category': getattr(field, 'category', 'other'),
                            'is_sensitive': False,
                            'auto_selected': False
                        }
                        
                        # Try to get location from field
                        location = getattr(field, 'location', {})
                        
                        if location and all(k in location for k in ['x', 'y', 'width', 'height']):
                            # Use existing coordinates (convert to percentages)
                            field_data['coordinates'] = {
                                'x': (location['x'] / img_width) * 100,
                                'y': (location['y'] / img_height) * 100,
                                'width': (location['width'] / img_width) * 100,
                                'height': (location['height'] / img_height) * 100
                            }
                        else:
                            # Fallback: estimate location from text search
                            field_value = field_data['field_value']
                            if field_value and len(field_value) > 2:
                                # Simple estimation (can be improved with OCR bounding boxes)
                                estimated_y = (idx * 5) % 80  # Distribute vertically
                                field_data['coordinates'] = {
                                    'x': 10,  # 10% from left
                                    'y': estimated_y,
                                    'width': min(len(field_value) * 2, 80),  # Width based on text length
                                    'height': 5  # 5% height
                                }
                        
                        # Determine if field is sensitive (for auto-selection)
                        sensitive_categories = ['personal_info', 'identification', 'financial', 'contact_info']
                        sensitive_names = ['name', 'address', 'phone', 'email', 'aadhar', 'pan', 'passport', 
                                          'account', 'ssn', 'tax', 'salary', 'income', 'dob', 'date_of_birth']
                        
                        field_name_lower = field_data['field_name'].lower()
                        field_category_lower = field_data['category'].lower()
                        
                        is_sensitive = (
                            field_category_lower in sensitive_categories or
                            any(sens in field_name_lower for sens in sensitive_names)
                        )
                        
                        field_data['is_sensitive'] = is_sensitive
                        
                        # Auto-select if confidence > 0.6 and sensitive
                        if is_sensitive and field_data['confidence'] > 0.6:
                            field_data['auto_selected'] = True
                        
                        detected_fields.append(field_data)
                    
                    logger.info(f"✅ Detected {len(detected_fields)} fields, {sum(1 for f in detected_fields if f['auto_selected'])} auto-selected")
                
                except Exception as e:
                    logger.warning(f"Field detection failed: {e}, continuing with empty fields")
                    detected_fields = []
            
            # Step 3: Convert image to base64 for frontend display
            with open(file_path, 'rb') as f:
                image_data = base64.b64encode(f.read()).decode('utf-8')
            
            # Clean up uploaded file
            file_path.unlink()
            
            # Return complete redaction data
            return jsonify({
                'success': True,
                'filename': filename,
                'image_data': f'data:image/{image_format.lower()};base64,{image_data}',
                'image_dimensions': {
                    'width': img_width,
                    'height': img_height
                },
                'extracted_text': extracted_text,
                'detected_fields': detected_fields,
                'statistics': {
                    'total_fields': len(detected_fields),
                    'auto_selected': sum(1 for f in detected_fields if f['auto_selected']),
                    'sensitive_fields': sum(1 for f in detected_fields if f['is_sensitive']),
                    'text_length': len(extracted_text),
                    'word_count': len(extracted_text.split())
                },
                'processing_metadata': {
                    'ocr_method': ocr_result.get('method', 'fast_mode'),
                    'ocr_confidence': ocr_result.get('confidence', 0.7),
                    'processing_time': ocr_result.get('processing_time', 0)
                }
            })
        
        except Exception as e:
            # Clean up on error
            if file_path.exists():
                file_path.unlink()
            logger.error(f"Redaction processing error: {e}")
            return jsonify({
                'success': False,
                'error': f'Processing failed: {str(e)}'
            }), 500
    
    except Exception as e:
        logger.error(f"Redaction upload error: {e}")
        return jsonify({
            'success': False,
            'error': f'Upload failed: {str(e)}'
        }), 500
