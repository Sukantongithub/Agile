
from ml_enhanced_rag import rag_system
import os

print("Testing RAG System Verification Logic...")

# Mocking LLM response for testing without API key if needed
# But better to test the method signature and logic flow

if not rag_system.llm:
    print("WARNING: OpenAI API Key not found. Testing in 'Local (LLM Unavailable)' mode.")
    
    # Test 1: Verification without LLM
    print("\nTest 1: Verification without LLM")
    res = rag_system.verify_classification_with_llm("Some text", "Marksheet", 0.4)
    print(f"Result: {res}")
    
    if res['source'] == "Local (LLM Unavailable)" and not res['correction']:
        print("PASS: Correctly handled missing LLM")
    else:
        print("FAIL: Unexpected behavior without LLM")

else:
    print("OpenAI API Key found. Testing real LLM verification.")
    
    # Test 2: Confirming correct classification
    print("\nTest 2: Confirming correct match")
    voter_text = "ELECTION COMMISSION OF INDIA... VOTER ID... Name: John"
    res = rag_system.verify_classification_with_llm(voter_text, "Voter ID Card", 0.8)
    print(f"Result: {res}")
    
    if not res['correction'] and res['verified_type'] == "Voter ID Card":
         print("PASS: Correctly confirmed Voter ID")
    else:
         print("FAIL: LLM should have confirmed")

    # Test 3: Correcting wrong classification (The Bias Fix)
    print("\nTest 3: Correcting wrong match (Bias Fix)")
    # This text looks like a Birth Certificate but was "detected" as Marksheet
    text_ambiguous = "CERTIFICATE OF BIRTH... Name: Baby Doe... Father: John Doe... Date of Birth: 2023... Registrar of Births"
    res_correct = rag_system.verify_classification_with_llm(text_ambiguous, "Marksheet", 0.5)
    print(f"Result: {res_correct}")
    
    if res_correct['correction'] and "Birth" in res_correct['verified_type']:
        print("PASS: LLM correctly fixed Marksheet -> Birth Certificate")
    else:
        print(f"FAIL: LLM failed to correct. Got: {res_correct['verified_type']}")

