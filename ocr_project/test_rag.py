
from ml_enhanced_rag import rag_system
import time
import os

print("Testing RAG System (Classification + Extraction)...")

# Allow time for init if it was running in background (though imports are blocking usually)
if not rag_system.is_initialized:
    print("RAG System not initialized (possibly missing dependencies or data).")
    exit(1)

# Test 1: Voter ID text Classification
voter_text = """
ELECTION COMMISSION OF INDIA
ELECTORAL PHOTO IDENTITY CARD
Name: JOHN DOE
Father's Name: JANE DOE
EPIC No: ABC1234567
Timestamp: 2023-10-10
"""
print(f"\nScanning Voter ID text for classification...")
result = rag_system.semantic_classify(voter_text)
print(f"Result: {result}")

# Test 2: Field Extraction (This requires OpenAI key, if not present, it should return available=False)
print(f"\nTesting LLM Field Extraction...")
extract_res = rag_system.extract_fields_with_llm(voter_text, "Voter ID Card")
print(f"Extraction Result: {extract_res}")

if not extract_res['available']:
    print("LLM Extraction correctly unavailable (Check OPENAI_API_KEY).")
else:
    print(f"LLM Extraction Success! Found {len(extract_res['fields'])} fields.")
    for f in extract_res['fields']:
        print(f" - {f.get('name')}: {f.get('value')}")
