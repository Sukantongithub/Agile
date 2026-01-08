
import sys
import pandas as pd
import difflib

print("Testing Dependencies...")
try:
    import paddleocr
    print("PaddleOCR Imported Successfully!")
except ImportError:
    print("PaddleOCR Import Failed.")

try:
    import difflib
    print("Difflib Imported Successfully!")
except ImportError:
    print("Difflib Import Failed.")

# Mock OCR Data to test Fuzzy Matching
data = {
    'left': [100, 200, 300],
    'top': [100, 100, 100],
    'width': [50, 50, 50],
    'height': [20, 20, 20],
    'text': ['Sukant', 'Ravchandran', 'Student'] # Notice typo in Ravichandran
}
ocr_data = pd.DataFrame(data)

print("\nTesting Fuzzy Coordinate Matching...")

def find_coordinates(value_text):
    # Simplified version of the logic I implemented
    value_clean = str(value_text).strip()
    value_tokens = value_clean.split()
    
    matching_boxes = []
    used_indices = set()
    
    # Fuzzy Match
    for token in value_tokens:
        token_lower = token.lower()
        best_ratio = 0.0
        best_idx = -1
        
        for idx, row in ocr_data.iterrows():
            if idx in used_indices: continue
            ocr_text = str(row['text']).strip().lower()
            
            ratio = difflib.SequenceMatcher(None, token_lower, ocr_text).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                best_idx = idx
        
        print(f"Token '{token}' best match: '{ocr_data.iloc[best_idx]['text']}' (Score: {best_ratio:.2f})")
        
        if best_ratio > 0.8:
            matching_boxes.append('Found')

    return len(matching_boxes)

# Test with typo
count = find_coordinates("Sukant Ravichandran")
if count == 2:
    print("✅ PASS: Fuzzy matching handled the typo and found both parts.")
else:
    print(f"❌ FAIL: Expected 2 matches, found {count}")
