
from ml_enhanced_rag import rag_system
import os
import time

print("Testing Online Verification...")

# Check if search tool is initialized
if not rag_system.search_tool:
    print("Search Tool NOT initialized. Check if duckduckgo-search is installed.")
    try:
        from langchain_community.tools import DuckDuckGoSearchRun
        print("DuckDuckGoSearchRun import successful.")
        tool = DuckDuckGoSearchRun()
        print("DuckDuckGoSearchRun instantiation successful.")
    except Exception as e:
        print(f"Debug: Import/Init failed: {e}")
    # We can't proceed with real test if tool is missing
else:
    print("Search Tool Initialized.")
    
    # Test 1: Verify Valid IFSC
    print("\nTest 1: Verifying Valid IFSC (SBIN0040014)")
    # Using a known IFSC
    res = rag_system.verify_field_online("IFSC Code", "SBIN0040014", context="State Bank of India")
    print(f"Result: {res}")
    
    if res['verified']:
        print("PASS: Verified valid IFSC")
    else:
        print("FAIL: Could not verify valid IFSC (could be network or LLM interpretation)")
        
    # Test 2: Verify Fake University
    print("\nTest 2: Verifying Fake University (Hogwarts College of Wizardry)")
    res_fake = rag_system.verify_field_online("University", "Hogwarts College of Wizardry", context="in India")
    print(f"Result: {res_fake}")
    
    if not res_fake['verified']:
         print("PASS: Correctly rejected fake university")
    else:
         print("FAIL: Verification incorrectly accepted fake university")

