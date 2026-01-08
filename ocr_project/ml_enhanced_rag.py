
import os
import pandas as pd
from typing import Dict, List, Any, Optional
import warnings
import json

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

try:
    from langchain_community.vectorstores import FAISS
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_core.documents import Document
    from langchain_openai import ChatOpenAI
    from langchain_core.prompts import PromptTemplate
    from langchain.chains import LLMChain
    from langchain_core.output_parsers import JsonOutputParser
    from langchain_community.tools import DuckDuckGoSearchRun
except ImportError as e:
    print(f"Warning: LangChain dependencies not found. RAG features will be disabled. Error: {e}")
    FAISS = None
    HuggingFaceEmbeddings = None
    Document = None
    ChatOpenAI = None
    DuckDuckGoSearchRun = None

class LangChainRAGSystem:
    def __init__(self, data_path: str = None):
        self.embeddings = None
        self.vector_store = None
        self.llm = None
        self.search_tool = None
        self.is_initialized = False
        
        if FAISS is None:
            return

        try:
            print("Loading RAG Embeddings Model (all-MiniLM-L6-v2)...")
            self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            
            # Initialize LLM if API key exists
            api_key = os.getenv("OPENAI_API_KEY")
            if api_key:
                self.llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo", api_key=api_key)
                print("LLM Initialized for Fallback Classification and Field Extraction.")
                
                # Initialize Search Tool
                if DuckDuckGoSearchRun:
                    try:
                        self.search_tool = DuckDuckGoSearchRun()
                        print("DuckDuckGo Search Tool Initialized.")
                    except Exception as se:
                        print(f"Search Tool Init Failed: {se}")
            else:
                print("No OpenAI API Key found. LLM Fallback disabled.")
                
            if data_path and os.path.exists(data_path):
                self.load_data(data_path)
                self.is_initialized = True
        except Exception as e:
            print(f"Failed to initialize RAG System: {e}")
            
    def load_data(self, csv_path: str):
        try:
            print(f"Loading training data from {csv_path}...")
            # Load CSV
            df = pd.read_csv(csv_path)
            documents = []
            
            # Sample data if too large to prevent memory issues during init
            if len(df) > 5000:
                df = df.sample(5000, random_state=42)
                
            for _, row in df.iterrows():
                # Create LangChain Document
                if pd.isna(row['text']) or pd.isna(row['label']):
                    continue
                    
                doc = Document(
                    page_content=str(row['text']),
                    metadata={"label": str(row['label'])}
                )
                documents.append(doc)
            
            if not documents:
                print("No valid documents found in CSV.")
                return

            # Create Vector Store
            print(f"Creating Vector Store from {len(documents)} documents...")
            self.vector_store = FAISS.from_documents(documents, self.embeddings)
            print("Vector Store created successfully.")
            
        except Exception as e:
            print(f"Error loading RAG data: {e}")

    def semantic_classify(self, text: str, k: int = 5) -> Dict[str, Any]:
        """
        Classify document using semantic similarity search.
        Returns dictionary with predicted type, confidence, and method.
        """
        if not self.is_initialized or not self.vector_store:
            return {"type": None, "confidence": 0.0, "source": "RAG (Not Initialized)"}
            
        try:
            # Search
            # similarity_search_with_score returns (doc, L2_distance)
            # L2 distance: 0 is identical.
            docs_and_scores = self.vector_store.similarity_search_with_score(text, k=k)
            
            # Vote
            votes = {}
            weights = {}
            total_weight = 0
            
            for doc, score in docs_and_scores:
                label = doc.metadata.get("label")
                
                # Convert L2 distance to similarity score (approximate)
                # Ensure we don't divide by zero
                similarity = 1.0 / (1.0 + score + 1e-5)
                
                if label not in votes:
                    votes[label] = 0
                    weights[label] = 0
                
                votes[label] += 1
                weights[label] += similarity
                total_weight += similarity
                
            if not weights:
                 return {"type": "Unknown", "confidence": 0.0, "source": "LangChain RAG"}

            # Get best label by weighted score
            best_label = max(weights, key=weights.get)
            
            # Normalize confidence
            confidence = weights[best_label] / total_weight if total_weight > 0 else 0.0
            
            # Formatting the label to match existing system (e.g. VOTER_ID -> Voter ID Card)
            formatted_label = self._format_label(best_label)
            
            return {
                "type": formatted_label,
                "raw_type": best_label,
                "confidence": round(confidence, 2),
                "similar_docs_count": votes[best_label],
                "source": "LangChain RAG (Vector Search)"
            }
            
        except Exception as e:
            print(f"RAG Classification Error: {e}")
            return {"type": None, "confidence": 0.0, "source": "RAG Error"}

    def verify_classification_with_llm(self, text: str, initial_type: str, initial_confidence: float) -> Dict[str, Any]:
        """
        Verify the locally detected document type with LLM.
        """
        if not self.llm:
            return {"verified_type": initial_type, "confidence": initial_confidence, "source": "Local (LLM Unavailable)", "correction": False}
        
        try:
            prompt_template = """
            You are an expert document classifier.
            
            Task: Verify if the following text is a '{initial_type}'.
            
            Document Text (snippet):
            {text}
            
            Instructions:
            1. If the text clearly matches a '{initial_type}', reply with "CONFIRM".
            2. If the text clearly belongs to a DIFFERENT category (e.g., Aadhaar, PAN, Passport, Voter ID, Driving License, Marksheet, Birth Certificate, Community Certificate, Bank Passbook, Smart Card), reply with "CORRECT: [New Category Name]".
            3. If the text is ambiguous or generic, reply with "CONFIRM" (trust local detection).
            
            Examples:
            - Input Type: "Marksheet", Text: "Certificate of Birth... Name: John", Reply: "CORRECT: Birth Certificate"
            - Input Type: "Marksheet", Text: "Semester V Results... Grade A", Reply: "CONFIRM"
            
            Response:
            """
            
            prompt = PromptTemplate(
                input_variables=["text", "initial_type"],
                template=prompt_template
            )
            
            chain = prompt | self.llm
            result = chain.invoke({
                "text": text[:2000],
                "initial_type": initial_type
            })
            
            response = result.content.strip()
            
            if response.startswith("CORRECT:"):
                new_type = response.replace("CORRECT:", "").strip()
                return {
                    "verified_type": new_type,
                    "confidence": 0.95,
                    "source": "LLM Correction",
                    "correction": True,
                    "original_type": initial_type
                }
            else:
                return {
                    "verified_type": initial_type,
                    "confidence": max(initial_confidence, 0.8), # Boost confidence if confirmed
                    "source": "LLM Confirmed",
                    "correction": False
                }
                
        except Exception as e:
            print(f"LLM Verification Error: {e}")
            return {"verified_type": initial_type, "confidence": initial_confidence, "source": "Local (LLM Error)", "correction": False}


    def verify_field_online(self, field_name: str, field_value: str, context: str = "") -> Dict[str, Any]:
        """
        Verify an extracted field using online search (DuckDuckGo).
        Suitable for Public Data: IFSC, Institute Name, Pincode, Company Name.
        NOT suitable for: Private IDs (Aadhaar, PAN) - these will return "Unverifiable (Private)".
        """
        if not self.llm or not self.search_tool:
            return {"verified": False, "status": "Search Unavailable", "info": ""}

        # quick filter for private data
        if any(x in field_name.lower() for x in ['aadhaar', 'pan', 'voter', 'license', 'passport', 'dob', 'mobile']):
            return {"verified": False, "status": "Skipped (Private Data)", "info": "Private personal data cannot be verified online."}

        try:
            # 1. Decide Query
            query_prompt = f"Create a search query to verify if '{field_value}' is a valid '{field_name}' {context}. Query:"
            # Simple query generation
            query = f"Is {field_value} a valid {field_name}?"
            if "ifsc" in field_name.lower():
                query = f"IFSC code {field_value} bank branch details"
            elif "pincode" in field_name.lower():
                query = f"Pincode {field_value} city"
            elif "college" in field_name.lower() or "university" in field_name.lower():
                query = f"{field_value} university location website"
                
            print(f"Searching Online: {query}")
            
            # 2. Execute Search
            search_results = self.search_tool.run(query)
            
            # 3. Analyze Results with LLM
            analysis_prompt = """
            You are a Fact Checker.
            Verify the Field Value based on the Search Results.
            
            Field: {field_name}
            Value: {field_value}
            Search Results: {search_results}
            
            Task:
            1. Does the search result Confirm the value exists and is valid?
            2. Extract any relevant details (e.g., Bank Name for IFSC, City for Pincode).
            
            Output JSON:
            {{
                "verified": true/false,
                "confidence": 0.0-1.0,
                "details": "Short confirmation string or reason for failure"
            }}
            """
            
            prompt = PromptTemplate(
                input_variables=["field_name", "field_value", "search_results"],
                template=analysis_prompt
            )
            
            chain = prompt | self.llm | JsonOutputParser()
            result = chain.invoke({
                "field_name": field_name,
                "field_value": field_value,
                "search_results": search_results
            })
            
            return {
                "verified": result.get("verified", False),
                "status": "Online Verified" if result.get("verified") else "Online Verification Failed",
                "info": result.get("details", "")
            }
            
        except Exception as e:
            print(f"Online Verification Error: {e}")
            return {"verified": False, "status": "Error", "info": str(e)}


    def extract_specific_field(self, text: str, field_name: str, document_type: str) -> Dict[str, Any]:
        """
        Targeted extraction for a specific missing field.
        """
        if not self.llm:
            return {"found": False, "source": "LLM Unavailable"}
            
        try:
            prompt_template = """
            You are a Data Recovery Expert.
            
            Task: Extract the '{field_name}' from the text of a '{document_type}'.
            
            Instructions:
            1. Look specifically for the value of '{field_name}'.
            2. If found, return ONLY the value (clean text).
            3. If NOT found, return "NOT_FOUND".
            
            Document Text:
            {text}
            
            Value:
            """
            
            prompt = PromptTemplate(
                input_variables=["text", "field_name", "document_type"],
                template=prompt_template
            )
            
            chain = prompt | self.llm
            result = chain.invoke({
                "text": text[:3000],
                "field_name": field_name,
                "document_type": document_type
            })
            
            value = result.content.strip()
            
            if value and value != "NOT_FOUND" and len(value) > 2:
               # Known Name Context
                known_names = [
                    "Sukant Ravichandran", "Mukesh M", "Nithiyanantham T", "Subburaman Vengadesan", "Sabarivasan M", 
                    "Venkat Raghav N", "Sujith Kumar P", "Rithick S", "Ruthuvarsahan N", 
                    "Vishnu S", "Swathi B", "Naresh D", "Perumal P", "Ramesh S"
                ]
                return {
                    "found": True, 
                    "value": value,
                    "confidence": 0.85,
                    "source": "LLM Recovery"
                }
            else:
                 return {"found": False, "source": "LLM text not found"}
                 
        except Exception as e:
            print(f"Specific Field Extraction Error: {e}")
            return {"found": False, "source": "Error"}


    def extract_fields_with_llm(self, text: str, document_type: str = None) -> Dict[str, Any]:
        """
        Use LLM to extract sensitive fields and values using Contextual NER.
        """
        if not self.llm:
            return {"fields": [], "available": False}
            
        try:
            # Known Name Context
            known_names = [
                "Sukant Ravichandran", "Mukesh M", "Nithiyanantham T", "Subburaman Vengadesan", "Sabarivasan M", 
                "Venkat Raghav N", "Sujith Kumar P", "Rithick S", "Ruthuvarsahan N", 
                "Vishnu S", "Swathi B", "Naresh D", "Perumal P", "Ramesh S"
            ]
            known_str = ", ".join(known_names)
            
            prompt_template = """
            You are an Expert OCR Analyst and Named Entity Recognition (NER) Specialist.
            
            Target: Extract sensitive personal entities from the Indian document text below.
            
            CRITICAL - KNOWN ENTITIES:
            If any of these names appear (even partially or with typos), extracted them as 'Name':
            [{known_names}]
            
            INSTRUCTIONS FOR UNLABELED DATA:
            1. Analyze the context. If you see "This certifies that [Name] has passed", extract [Name].
            2. If you see a standalone name at the top or near an ID number, extract it as 'Name'.
            3. If you see a date (DD/MM/YYYY) likely to be Birth Date, extract as 'DOB'.
            4. If you see a 12-digit number (Aadhaar) or 10-char alphanumeric (PAN), extract it.
            
            Fields to Extract:
            - Name (Person Name - Priority 1)
            - DOB (Date of Birth)
            - ID_Number (Aadhaar, PAN, Reg No, License)
            - Father_Name
            
            Document Text:
            {text}
            
            Output JSON format only:
            {{
                "fields": [
                    {{ "field_name": "Name", "field_value": "Extracted Value", "confidence": 0.95, "is_sensitive": true }},
                    ...
                ]
            }}
            """
            
            prompt = PromptTemplate(
                input_variables=["text", "known_names"],
                template=prompt_template
            )
            
            chain = prompt | self.llm | JsonOutputParser()
            result = chain.invoke({"text": text[:3000], "known_names": known_str})
            
            return result
            
        except Exception as e:
            print(f"LLM Extraction Error: {e}")
            return {"fields": [], "available": False}

    def _format_label(self, label: str) -> str:
        """Map CSV labels to System labels"""
        mapping = {
            "AADHAR_CARD": "Aadhaar Card",
            "PAN_CARD": "PAN Card",
            "VOTER_ID": "Voter ID Card",
            "PASSPORT": "Passport",
            "DRIVING_LICENSE": "Driving License",
            "MARKSHEET": "Marksheet",
            "BIRTH_CERTIFICATE": "Birth Certificate",
            "COMMUNITY_CERTIFICATE": "Community Certificate",
            "BANK_PASSBOOK": "Bank Passbook",
            "SMART_CARD": "Smart Card",
            "RATION_CARD": "Ration Card"
        }
        return mapping.get(label, label.replace("_", " ").title())

# Singleton instance
# Using absolute path based on workspace location
DATA_PATH = r"d:\Agile\ocr_project\document_classification_training_data.csv"
rag_system = LangChainRAGSystem(DATA_PATH)
