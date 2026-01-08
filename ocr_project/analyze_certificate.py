#!/usr/bin/env python3
"""
Certificate OCR Analyzer - Command line tool for certificate text extraction
Usage: python analyze_certificate.py <image_path>
"""

import sys
import argparse
from pathlib import Path

# Add current directory to path so we can import our modules
sys.path.insert(0, str(Path(__file__).parent))

from ocr.certificate_ocr import extract_certificate_text, preprocess_certificate
from ocr import extract_text, load_image

def analyze_certificate(image_path: str, show_comparison: bool = True, save_processed: bool = False):
    """Analyze a certificate image and show detailed results."""
    
    print(f"🔍 Analyzing certificate: {image_path}")
    print("=" * 60)
    
    # Check if file exists
    if not Path(image_path).exists():
        print(f"❌ Error: File not found: {image_path}")
        return False
    
    try:
        # Extract using certificate-specific method
        result = extract_certificate_text(image_path)
        
        print(f"📊 Certificate OCR Results")
        print("-" * 30)
        print(f"Best Method: {result.get('best_source', 'unknown').title()} OCR")
        print(f"Confidence: {result['best_confidence']:.1f}%")
        
        print(f"\n📝 Extracted Text:")
        print("-" * 30)
        print(result['best_text'])
        
        # Show structured data
        if result['structured_data']:
            print(f"\n🔍 Structured Information:")
            print("-" * 30)
            
            for key, value in result['structured_data'].items():
                if value and key != 'other_text' and isinstance(value, str):
                    display_key = key.replace('_', ' ').title()
                    print(f"{display_key}: {value}")
        
        # Show comparison if requested
        if show_comparison:
            print(f"\n⚖️ Method Comparison:")
            print("-" * 30)
            
            raw_results = {k: v for k, v in result['all_results'].items() if k.startswith('raw_')}
            processed_results = {k: v for k, v in result['all_results'].items() if k.startswith('processed_')}
            
            if raw_results:
                best_raw = max(raw_results.values(), key=lambda x: x.get('confidence', 0))
                print(f"Raw OCR Best: {best_raw['confidence']:.1f}% confidence")
                
            if processed_results:
                best_processed = max(processed_results.values(), key=lambda x: x.get('confidence', 0))
                print(f"Processed OCR Best: {best_processed['confidence']:.1f}% confidence")
        
        # Save processed image if requested
        if save_processed:
            processed_img = preprocess_certificate(image_path)
            processed_path = Path(image_path).with_suffix('.processed.png')
            processed_img.save(processed_path)
            print(f"\n💾 Processed image saved: {processed_path}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error processing certificate: {str(e)}")
        return False

def main():
    parser = argparse.ArgumentParser(
        description="Analyze certificate images with specialized OCR",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python analyze_certificate.py certificate.jpg
  python analyze_certificate.py diploma.png --no-comparison
  python analyze_certificate.py award.tiff --save-processed
        """
    )
    
    parser.add_argument('image_path', help='Path to certificate image file')
    parser.add_argument('--no-comparison', action='store_true', 
                       help='Skip showing method comparison')
    parser.add_argument('--save-processed', action='store_true',
                       help='Save the processed image to disk')
    
    args = parser.parse_args()
    
    print("🎓 Certificate OCR Analyzer")
    print("=" * 60)
    
    success = analyze_certificate(
        args.image_path, 
        show_comparison=not args.no_comparison,
        save_processed=args.save_processed
    )
    
    if success:
        print(f"\n✅ Analysis complete!")
    else:
        print(f"\n❌ Analysis failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()