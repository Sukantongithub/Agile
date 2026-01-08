"""OCR text extraction using Tesseract"""

import pytesseract
from PIL import Image
import argparse
import sys
from pathlib import Path
from typing import Union, Optional

from .preprocess import preprocess_image


def extract_text(image_input: Union[str, Path, Image.Image], 
                lang: str = 'eng',
                config: str = '--psm 6') -> str:
    """
    Extract text from an image using Tesseract OCR.
    
    Args:
        image_input: Path to image file or PIL Image object
        lang: Language code for OCR (default: 'eng')
        config: Tesseract configuration string (default: '--psm 6')
    
    Returns:
        Extracted text as string
    
    Raises:
        FileNotFoundError: If image file doesn't exist
        Exception: If Tesseract is not installed or configured properly
    """
    try:
        if isinstance(image_input, (str, Path)):
            image = Image.open(image_input)
        else:
            image = image_input
        
        # Extract text using pytesseract
        text = pytesseract.image_to_string(image, lang=lang, config=config)
        return text.strip()
        
    except FileNotFoundError:
        raise FileNotFoundError(f"Image file not found: {image_input}")
    except pytesseract.TesseractNotFoundError:
        raise Exception(
            "Tesseract not found. Please install Tesseract OCR and add it to your PATH.\n"
            "Windows: Download from https://github.com/UB-Mannheim/tesseract/wiki"
        )
    except Exception as e:
        raise Exception(f"OCR extraction failed: {str(e)}")


def extract_text_from_file(image_path: Union[str, Path], 
                          preprocess: bool = False,
                          lang: str = 'eng',
                          config: str = '--psm 6') -> str:
    """
    Extract text from an image file with optional preprocessing.
    
    Args:
        image_path: Path to image file
        preprocess: Whether to apply preprocessing (default: False)
        lang: Language code for OCR (default: 'eng')
        config: Tesseract configuration string (default: '--psm 6')
    
    Returns:
        Extracted text as string
    """
    if preprocess:
        processed_image = preprocess_image(image_path)
        return extract_text(processed_image, lang=lang, config=config)
    else:
        return extract_text(image_path, lang=lang, config=config)


def get_available_languages() -> list:
    """Get list of available Tesseract languages."""
    try:
        return pytesseract.get_languages(config='')
    except Exception:
        return ['eng']  # fallback to English only


def main():
    """Command line interface for OCR text extraction."""
    parser = argparse.ArgumentParser(description='Extract text from images using OCR')
    parser.add_argument('image_path', help='Path to image file')
    parser.add_argument('--preprocess', action='store_true', 
                       help='Apply preprocessing to improve OCR accuracy')
    parser.add_argument('--lang', default='eng',
                       help='Language code for OCR (default: eng)')
    parser.add_argument('--config', default='--psm 6',
                       help='Tesseract configuration (default: --psm 6)')
    parser.add_argument('--output', '-o',
                       help='Output file path (default: print to console)')
    
    args = parser.parse_args()
    
    try:
        # Check if image file exists
        image_path = Path(args.image_path)
        if not image_path.exists():
            print(f"Error: Image file not found: {image_path}", file=sys.stderr)
            sys.exit(1)
        
        # Extract text
        print(f"Processing: {image_path}")
        if args.preprocess:
            print("Applying preprocessing...")
        
        text = extract_text_from_file(
            image_path,
            preprocess=args.preprocess,
            lang=args.lang,
            config=args.config
        )
        
        # Output results
        if args.output:
            output_path = Path(args.output)
            output_path.write_text(text, encoding='utf-8')
            print(f"Text saved to: {output_path}")
        else:
            print("\n" + "="*50)
            print("EXTRACTED TEXT:")
            print("="*50)
            print(text)
            print("="*50)
            
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()