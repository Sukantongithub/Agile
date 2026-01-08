
from ml_enhanced_rag import rag_system
import os

print("Testing Must Field Recovery...")

if not rag_system.llm:
    print("LLM Unavailable. Skipping recovery test.")
else:
    # Scenario: Regex failed to pick up Aadhaar number, but it is in the text.
    # We call extract_specific_field directly to simulate the recovery step.
    
    aadhaar_text = """
    GOVERNMENT OF INDIA
    Name: Rahul Kumar
    DOB: 01/01/1990
    Address: 123 Street...
    1234 5678 9012
    """
    
    print("\nTest: Recovering Missing Aadhaar Number")
    res = rag_system.extract_specific_field(aadhaar_text, "Aadhaar Number", "Aadhaar Card")
    print(f"Result: {res}")
    
    if res['found'] and "1234 5678 9012" in res['value']:
        print("PASS: Correctly recovered Aadhaar Number")
    else:
        print("FAIL: Failed to recover critical field")
