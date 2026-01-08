# Raw OCR Text Extraction Project

A Python project for extracting text from images using **Raw OCR** (no preprocessing) - optimized for certificates and clean documents.

## Features

- **Smart Document Detection**: Automatically detects certificates and applies optimized raw OCR
- **Raw OCR Processing**: No preprocessing - works best for high-quality certificates and documents
- **Structured Data Extraction**: Extracts recipient names, dates, certificate types from certificates
- **Web Interface**: Beautiful drag-and-drop interface with real-time processing
- **Command-line Interface**: Easy command-line usage with options
- **Support for Multiple Formats**: PNG, JPG, TIFF, BMP, GIF, WEBP

## Why Raw OCR?

For certificates and high-quality documents, raw OCR often works better than preprocessing because:
- ✅ **Preserves original text quality** 
- ✅ **No image degradation** from processing
- ✅ **Better for clean, professional documents**
- ✅ **Faster processing** without preprocessing steps

## Prerequisites

### Windows Setup
1. **Install Tesseract OCR:**
   ```powershell
   # Option 1: Using winget (recommended)
   winget install --id UB-Mannheim.TesseractOCR
   
   # Option 2: Manual download from:
   # https://github.com/UB-Mannheim/tesseract/wiki
   ```

2. **Add Tesseract to PATH permanently:**
   ```powershell
   # Add to system PATH (requires admin or restart)
   $env:PATH += ";C:\Program Files\Tesseract-OCR"
   [Environment]::SetEnvironmentVariable("Path", $env:PATH, [EnvironmentVariableTarget]::User)
   
   # For current session only:
   $env:PATH += ";C:\Program Files\Tesseract-OCR"
   ```

3. **Verify Tesseract installation:**
   ```powershell
   tesseract --version
   ```

## Installation

1. **Create virtual environment:**
   ```powershell
   python -m venv venv
   .\venv\Scripts\Activate.ps1
   ```

2. **Install dependencies:**
   ```powershell
   pip install -r requirements.txt
   ```

## Usage

### Web Interface (Recommended)
1. **Quick Start:**
   ```powershell
   # Option 1: Use the startup script (recommended)
   .\start_web_app.ps1
   
   # Option 2: Manual startup
   $env:PATH += ";C:\Program Files\Tesseract-OCR"
   python app.py
   ```

2. **Open your browser and go to:**
   ```
   http://localhost:5000
   ```

3. **Upload and extract:**
   - Drag and drop any image file onto the upload area
   - Or click "Choose File" to browse for images
   - **Raw OCR extraction** happens automatically (no preprocessing)
   - **Certificate detection** shows structured information
   - Copy the results with one click

### Command Line Interface
```powershell
# Basic extraction
python -m ocr.ocr path/to/image.jpg

# With preprocessing
python -m ocr.ocr path/to/image.jpg --preprocess

# Save output to file
python -m ocr.ocr path/to/image.jpg --output "extracted_text.txt"

# Use different language (if available)
python -m ocr.ocr path/to/image.jpg --lang deu --preprocess
```

### Python API
```python
from ocr import extract_text
from ocr.certificate_ocr import extract_certificate_structure

# Raw OCR text extraction (recommended)
text = extract_text("path/to/image.jpg")
print(text)

# For certificates - extract structured data
text = extract_text("path/to/certificate.jpg")
structured_data = extract_certificate_structure(text)
print(f"Recipient: {structured_data['recipient_name']}")
print(f"Certificate Type: {structured_data['certificate_type']}")
print(f"Date: {structured_data['date']}")
```

## Certificate OCR Enhancements

The system automatically detects certificates and applies specialized processing for maximum accuracy:

### 🎯 **Smart Detection**
- Automatically identifies certificates, diplomas, and awards
- Uses different processing pipelines based on document type

### 🔍 **Enhanced Accuracy**  
- **Multiple OCR configurations**: Tests 4 different Tesseract settings
- **High-DPI scaling**: 2-3x upscaling for better text recognition
- **CLAHE enhancement**: Advanced contrast enhancement
- **Morphological operations**: Text area enhancement and noise reduction
- **Adaptive thresholding**: Multiple threshold methods combined

### 📊 **Structured Extraction**
Automatically extracts and organizes:
- 📜 **Certificate Type**: "Certificate of Appreciation", "Achievement Award", etc.
- 👤 **Recipient Name**: The person receiving the certificate  
- 📅 **Date**: Certificate issue/award date
- ✍️ **Signature Line**: Authority signatures and titles
- 🏢 **Organization**: Issuing organization (when present)

### 💯 **Confidence Scoring**
- Provides accuracy confidence scores (0-100%)
- Compares multiple processing methods
- Reports the best result with highest confidence

## Testing

```powershell
# Run the comprehensive test suite
pytest tests/ -v

# Test certificate-specific functionality  
python test_certificate_ocr.py

# Test with your own certificate
python -m ocr.certificate_ocr path/to/your/certificate.jpg
```

## Project Structure

```
ocr_project/
├── ocr/
│   ├── __init__.py
│   ├── ocr.py           # Main OCR functionality
│   └── preprocess.py    # Image preprocessing
├── tests/
│   ├── test_ocr.py
│   └── test_preprocess.py
├── requirements.txt
└── README.md
```

## Troubleshooting

- **Tesseract not found**: Ensure Tesseract is installed and in PATH
- **Poor OCR accuracy**: Try preprocessing options or higher DPI images
- **Memory issues**: Use smaller images or batch processing for large datasets