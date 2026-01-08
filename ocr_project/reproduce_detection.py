
import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ml_enhanced_rag import rag_system

print("Checking RAG initialization...")
if rag_system.is_initialized:
    print("RAG System Initialized Successfully.")
else:
    print("RAG System parameters:")
    print(f"Vector Store: {rag_system.vector_store is not None}")
    print(f"Embeddings: {rag_system.embeddings is not None}")
    print("RAG System NOT Initialized.")

print(f"LLM Available: {rag_system.llm is not None}")

# Test Classification
test_text = """
ELECTION COMMISSION OF INDIA
ELECTORAL PHOTO IDENTITY CARD
Name: JOHN DOE
EPIC No: ABC1234567
"""
print(f"\nTesting Classification for:\n{test_text.strip()}")
result = rag_system.semantic_classify(test_text)
print(f"Result: {result}")

# Test LLM Verification
if rag_system.llm:
    print("\nTesting LLM Verification...")
    verify_res = rag_system.verify_classification_with_llm(test_text, "Voter ID Card", 0.6)
    print(f"Verification Result: {verify_res}")
else:
    print("\nSkipping LLM Verification (No LLM).")
