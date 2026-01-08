
from ml_enhanced_rag import rag_system
import os

print("Testing Known Name Detection...")

# Simulate a document text
doc_text = """
CONFIDENTIAL REPORT DATED 12/12/2024
This certifies that Venkat Raghav N has completed the course.
ID: 998877
"""

print(f"Document Text:\n{doc_text}")

# 1. Test LLM Extraction Awareness
if rag_system.llm:
    print("\nTest 1: LLM Extraction with Known Name awareness")
    res = rag_system.extract_fields_with_llm(doc_text)
    print(f"LLM Result: {res}")
    
    names_found = [f['field_value'] for f in res.get('fields', []) if f['field_name'] == 'Name']
    if any("Venkat" in n for n in names_found):
        print("PASS: LLM detected known name 'Venkat Raghav N'")
    else:
        print("FAIL: LLM missed the known name")
else:
    print("LLM unavailable, skipping Test 1")

# 2. Test Local Detection Logic (Simulation)
print("\nTest 2: Local Detection Logic Matching")
KNOWN_NAMES = [
    "Sukant R", "Mukesh M", "Nithiyanantham T", "Subburaman V", "Sabarivasan M", 
    "Venkat Raghav N", "Sujith Kumar P", "Rithick S", "Ruthuvarsahan N", 
    "Vishnu S", "Swathi B", "Naresh D", "Perumal P", "Ramesh S"
]

detected = []
for name in KNOWN_NAMES:
    if name.lower() in doc_text.lower():
        detected.append(name)
        print(f"MATCH: Found '{name}' in text")

if "Venkat Raghav N" in detected:
    print("PASS: Local logic correctly matched the name")
else:
    print("FAIL: Local logic failed to match")
