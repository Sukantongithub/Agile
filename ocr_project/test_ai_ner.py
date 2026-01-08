
from ml_enhanced_rag import rag_system
import json

print("Testing AI-First NER Extraction...")

# Simulate the user's problematic document text (No labels)
doc_text = """
TO WHOM IT MAY CONCERN

This is to certify that Venkat Raghav N (Reg No: 998877) acts as a Student.
Date of Birth is 12/05/2001.
He is studying in Batch 2023-2027.
"""

print(f"Input Text:\n{doc_text}\n")

if rag_system.llm:
    print("Running Extraction...")
    res = rag_system.extract_fields_with_llm(doc_text)
    
    print("\nResult:")
    print(json.dumps(res, indent=2))
    
    fields = res.get('fields', [])
    names = [f['field_value'] for f in fields if f['field_name'] == 'Name']
    
    if "Venkat Raghav N" in names:
        print("\n✅ PASS: Contextual NER detected the name correctly without 'Name:' label.")
    else:
        print("\n❌ FAIL: Name not detected.")
else:
    print("LLM Unavailable. Cannot test.")
