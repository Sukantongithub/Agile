"""
RAG (Retrieval-Augmented Generation) System for OCR Field Suggestion
Provides intelligent field suggestions based on document content and patterns.
"""

import json
import numpy as np
import re
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from pathlib import Path
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class FieldPattern:
    """Represents a field pattern in the knowledge base."""
    field_name: str
    field_type: str
    field_category: str  # New field for categorization
    document_type: str
    patterns: List[str]
    keywords: List[str]
    examples: List[str]
    confidence_weight: float
    description: str

@dataclass
class FieldSuggestion:
    """Represents a suggested field extraction."""
    field_name: str
    field_type: str
    field_category: str  # New field for categorization
    suggested_value: str
    confidence: float
    source_text: str
    pattern_matched: str
    reasoning: str

@dataclass
class DocumentClassification:
    """Represents a document type classification result."""
    document_type: str
    confidence: float
    keywords_found: List[str]
    patterns_matched: List[str]
    reasoning: str

class DocumentFieldKnowledgeBase:
    """Knowledge base containing field patterns for different document types."""
    
    # Field Categories
    FIELD_CATEGORIES = {
        'NAME': 'name',
        'ADDRESS': 'address', 
        'PHONE_NUMBER': 'phone_number',
        'AADHAR_NUMBER': 'aadhar_number',
        'ID_NUMBER': 'id_number',
        'OTHER': 'other'
    }
    
    # Document Types with their classification patterns
    DOCUMENT_TYPES = {
        'AADHAR_CARD': {
            'keywords': ['aadhar', 'aadhaar', 'uid', 'unique identification', 'government of india'],
            'patterns': [
                r'\b\d{4}\s+\d{4}\s+\d{4}\b',  # Aadhar number pattern
                r'UNIQUE\s+IDENTIFICATION\s+AUTHORITY',
                r'GOVERNMENT\s+OF\s+INDIA',
                r'DATE\s+OF\s+BIRTH',
                r'AADHAR\s+NUMBER'
            ],
            'required_fields': ['aadhar_number', 'full_name', 'date_of_birth'],
            'confidence_weight': 0.95
        },
        'VOTER_ID': {
            'keywords': ['voter', 'election', 'electoral', 'epic', 'voter id'],
            'patterns': [
                r'ELECTION\s+COMMISSION\s+OF\s+INDIA',
                r'ELECTORAL\s+PHOTO\s+IDENTITY\s+CARD',
                r'VOTER\s+ID',
                r'EPIC\s+NO',
                r'\b[A-Z]{3}\d{7}\b'  # Voter ID pattern
            ],
            'required_fields': ['voter_id', 'full_name', 'father_name'],
            'confidence_weight': 0.9
        },
        'PAN_CARD': {
            'keywords': ['pan', 'income tax', 'permanent account number', 'tax'],
            'patterns': [
                r'INCOME\s+TAX\s+DEPARTMENT',
                r'PERMANENT\s+ACCOUNT\s+NUMBER',
                r'PAN\s+CARD',
                r'\b[A-Z]{5}\d{4}[A-Z]\b'  # PAN number pattern
            ],
            'required_fields': ['pan_number', 'full_name', 'father_name'],
            'confidence_weight': 0.9
        },
        'DRIVING_LICENSE': {
            'keywords': ['driving', 'license', 'transport', 'motor', 'vehicle'],
            'patterns': [
                r'DRIVING\s+LICENSE',
                r'TRANSPORT\s+DEPARTMENT',
                r'MOTOR\s+VEHICLE',
                r'LICENSE\s+TO\s+DRIVE',
                r'\b[A-Z]{2}\d{13}\b'  # Driving license pattern
            ],
            'required_fields': ['driving_license', 'full_name', 'address'],
            'confidence_weight': 0.9
        },
        'PASSPORT': {
            'keywords': ['passport', 'republic of india', 'ministry of external affairs'],
            'patterns': [
                r'PASSPORT',
                r'REPUBLIC\s+OF\s+INDIA',
                r'MINISTRY\s+OF\s+EXTERNAL\s+AFFAIRS',
                r'PASSPORT\s+NO',
                r'\b[A-Z]\d{7}\b'  # Passport number pattern
            ],
            'required_fields': ['passport_number', 'full_name', 'date_of_birth'],
            'confidence_weight': 0.95
        },
        'MARKSHEET': {
            'keywords': ['marksheet', 'marks', 'grade', 'examination', 'board', 'university', 'result'],
            'patterns': [
                r'MARK\s*SHEET',
                r'MARKS\s+OBTAINED',
                r'EXAMINATION\s+RESULT',
                r'BOARD\s+OF\s+EDUCATION',
                r'UNIVERSITY',
                r'GRADE\s+CARD',
                r'PERCENTAGE',
                r'CGPA',
                r'TOTAL\s+MARKS'
            ],
            'required_fields': ['full_name', 'marks', 'percentage'],
            'confidence_weight': 0.85
        },
        'RATION_CARD': {
            'keywords': ['ration', 'food', 'civil supplies', 'bpl', 'apl'],
            'patterns': [
                r'RATION\s+CARD',
                r'FOOD\s+CARD',
                r'CIVIL\s+SUPPLIES',
                r'BPL\s+CARD',
                r'APL\s+CARD',
                r'FOOD\s+SECURITY',
                r'FAMILY\s+CARD'
            ],
            'required_fields': ['full_name', 'address', 'family_size'],
            'confidence_weight': 0.8
        },
        'BANK_PASSBOOK': {
            'keywords': ['bank', 'account', 'balance', 'passbook', 'savings', 'current'],
            'patterns': [
                r'BANK\s+PASSBOOK',
                r'SAVINGS\s+ACCOUNT',
                r'CURRENT\s+ACCOUNT',
                r'ACCOUNT\s+NUMBER',
                r'BALANCE',
                r'TRANSACTION',
                r'IFSC\s+CODE'
            ],
            'required_fields': ['account_number', 'full_name', 'balance'],
            'confidence_weight': 0.85
        },
        'BIRTH_CERTIFICATE': {
            'keywords': ['birth', 'certificate', 'registrar', 'municipality', 'corporation'],
            'patterns': [
                r'BIRTH\s+CERTIFICATE',
                r'CERTIFICATE\s+OF\s+BIRTH',
                r'REGISTRAR\s+OF\s+BIRTHS',
                r'MUNICIPAL\s+CORPORATION',
                r'DATE\s+OF\s+BIRTH',
                r'PLACE\s+OF\s+BIRTH'
            ],
            'required_fields': ['full_name', 'date_of_birth', 'place_of_birth'],
            'confidence_weight': 0.9
        },
        'COMMUNITY_CERTIFICATE': {
            'keywords': ['community', 'caste', 'obc', 'sc', 'st', 'certificate'],
            'patterns': [
                r'COMMUNITY\s+CERTIFICATE',
                r'CASTE\s+CERTIFICATE',
                r'OBC\s+CERTIFICATE',
                r'SC\s+CERTIFICATE',
                r'ST\s+CERTIFICATE',
                r'BACKWARD\s+CLASS',
                r'SCHEDULED\s+CASTE',
                r'SCHEDULED\s+TRIBE'
            ],
            'required_fields': ['full_name', 'community', 'father_name'],
            'confidence_weight': 0.85
        },
        'SMART_CARD': {
            'keywords': ['smart card', 'chip card', 'health card', 'employee card'],
            'patterns': [
                r'SMART\s+CARD',
                r'CHIP\s+CARD',
                r'HEALTH\s+CARD',
                r'EMPLOYEE\s+CARD',
                r'ID\s+CARD'
            ],
            'required_fields': ['full_name', 'id_number'],
            'confidence_weight': 0.7
        }
    }
    
    def __init__(self):
        self.field_patterns = self._initialize_knowledge_base()
        self.document_classifier = self._initialize_document_classifier()
        
    def _initialize_knowledge_base(self) -> List[FieldPattern]:
        """Initialize the knowledge base with comprehensive field patterns."""
        patterns = []
        
        # NAME CATEGORY - Various types of names
        patterns.extend([
            FieldPattern(
                field_name="full_name",
                field_type="text",
                field_category=self.FIELD_CATEGORIES['NAME'],
                document_type="general",
                patterns=[
                    r"NAME\s*:?\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)",
                    r"FULL\s*NAME\s*:?\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)",
                    r"APPLICANT\s*:?\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)",
                    r"CUSTOMER\s*NAME\s*:?\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)",
                    r"HOLDER\s*NAME\s*:?\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)"
                ],
                keywords=["name", "full", "applicant", "customer", "holder", "first", "last"],
                examples=["Name: Rajesh Kumar", "FULL NAME: Priya Sharma", "Customer Name: Arjun Singh"],
                confidence_weight=0.9,
                description="Person's complete name"
            ),
            FieldPattern(
                field_name="father_name",
                field_type="text",
                field_category=self.FIELD_CATEGORIES['NAME'],
                document_type="government",
                patterns=[
                    r"FATHER\s*NAME\s*:?\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)",
                    r"FATHER\s*:?\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)",
                    r"S/O\s*:?\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)",
                    r"SON\s*OF\s*:?\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)"
                ],
                keywords=["father", "s/o", "son of", "parent"],
                examples=["Father Name: Ramesh Kumar", "S/O: Suresh Gupta", "Father: Mahesh Sharma"],
                confidence_weight=0.85,
                description="Father's name as mentioned in official documents"
            ),
            FieldPattern(
                field_name="spouse_name", 
                field_type="text",
                field_category=self.FIELD_CATEGORIES['NAME'],
                document_type="general",
                patterns=[
                    r"SPOUSE\s*NAME\s*:?\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)",
                    r"HUSBAND\s*NAME\s*:?\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)",
                    r"WIFE\s*NAME\s*:?\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)",
                    r"W/O\s*:?\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)",
                    r"D/O\s*:?\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)"
                ],
                keywords=["spouse", "husband", "wife", "w/o", "d/o"],
                examples=["Spouse Name: Sunita Devi", "W/O: Rajesh Kumar", "Husband Name: Amit Patel"],
                confidence_weight=0.8,
                description="Spouse or partner name"
            )
        ])
        
        # ADDRESS CATEGORY - Various address components
        patterns.extend([
            FieldPattern(
                field_name="full_address",
                field_type="text",
                field_category=self.FIELD_CATEGORIES['ADDRESS'],
                document_type="general",
                patterns=[
                    r"ADDRESS\s*:?\s*([A-Z0-9][A-Za-z0-9\s,\.\-/]+(?:INDIA)?)",
                    r"PERMANENT\s*ADDRESS\s*:?\s*([A-Z0-9][A-Za-z0-9\s,\.\-/]+)",
                    r"RESIDENCE\s*:?\s*([A-Z0-9][A-Za-z0-9\s,\.\-/]+)",
                    r"HOME\s*ADDRESS\s*:?\s*([A-Z0-9][A-Za-z0-9\s,\.\-/]+)"
                ],
                keywords=["address", "permanent", "residence", "home", "location"],
                examples=["Address: 123 Main Street, Mumbai 400001", "Permanent Address: Sector 15, Gurgaon"],
                confidence_weight=0.85,
                description="Complete residential or business address"
            ),
            FieldPattern(
                field_name="pincode",
                field_type="identifier",
                field_category=self.FIELD_CATEGORIES['ADDRESS'],
                document_type="general",
                patterns=[
                    r"PIN\s*CODE\s*:?\s*(\d{6})",
                    r"PINCODE\s*:?\s*(\d{6})",
                    r"PIN\s*:?\s*(\d{6})",
                    r"POSTAL\s*CODE\s*:?\s*(\d{6})",
                    r"(\d{6})\s*INDIA?"
                ],
                keywords=["pin", "pincode", "postal", "code", "zip"],
                examples=["PIN CODE: 400001", "Pincode: 110001", "PIN: 560001"],
                confidence_weight=0.95,
                description="Indian postal PIN code (6 digits)"
            ),
            FieldPattern(
                field_name="state",
                field_type="text",
                field_category=self.FIELD_CATEGORIES['ADDRESS'],
                document_type="general",
                patterns=[
                    r"STATE\s*:?\s*(ANDHRA PRADESH|ARUNACHAL PRADESH|ASSAM|BIHAR|CHHATTISGARH|GOA|GUJARAT|HARYANA|HIMACHAL PRADESH|JHARKHAND|KARNATAKA|KERALA|MADHYA PRADESH|MAHARASHTRA|MANIPUR|MEGHALAYA|MIZORAM|NAGALAND|ODISHA|PUNJAB|RAJASTHAN|SIKKIM|TAMIL NADU|TELANGANA|TRIPURA|UTTAR PRADESH|UTTARAKHAND|WEST BENGAL|DELHI)",
                    r"(MUMBAI|BANGALORE|CHENNAI|KOLKATA|HYDERABAD|PUNE|AHMEDABAD|SURAT|JAIPUR|LUCKNOW|KANPUR|NAGPUR|INDORE|BHOPAL|VISAKHAPATNAM|PATNA|VADODARA|GHAZIABAD|LUDHIANA|AGRA|NASHIK|FARIDABAD|MEERUT|RAJKOT|KALYAN|VASAI|VARANASI|SRINAGAR|AURANGABAD|DHANBAD|AMRITSAR|NAVI MUMBAI|ALLAHABAD|RANCHI|HOWRAH|COIMBATORE|JABALPUR|GWALIOR|VIJAYAWADA|JODHPUR|MADURAI|RAIPUR|KOTA|GUWAHATI|CHANDIGARH|SOLAPUR|HUBLI|TIRUCHIRAPPALLI|BAREILLY|MYSORE|TIRUNELVELI|SALEM|MIRA BHAYANDAR|JALANDHAR|BHUBANESWAR|ALIGARH|MORADABAD|GORAKHPUR|JABALPUR|AMRAVATI|MANGALORE|TIRUVANANTHAPURAM|MALEGAON|GAYA|JALGAON|UDAIPUR|MAHESHTALA|DAVANAGERE|KOZHIKODE|KURNOOL|RAJPUR SONARPUR|RAJAHMUNDRY|BOKARO|SOUTH DUMDUM|BELLARY|PATIALA|GOPALPUR|AGARTALA|BHAGALPUR|MUZAFFARNAGAR|BHATPARA|PANIHATI|LATUR|DHULE|ROHTAK|KORBA|BHILWARA|BERHAMPUR|MUZAFFARPUR|AHMEDNAGAR|MATHURA|KOLLAM|AVADI|KADAPA|KAMARHATI|SAMBALPUR|BILASPUR|SHAHJAHANPUR|SATARA|BIJAPUR|RAMPUR|SHIVAMOGGA|CHANDRAPUR|JUNAGADH|THRISSUR|ALWAR|BARDHAMAN|KULTI|KAKINADA|NIZAMABAD|PARBHANI|TUMKUR|KHAMMAM|OZHUKARAI|BIHAR SHARIF|PANIPAT|DARBHANGA|BALLY|AIZAWL|DEWAS|ICHALKARANJI|KARNAL|BATHINDA|JALNA|ELURU|KIRARI SULEMAN NAGAR|BARASAT|PURNIA|SATNA|MAUAJI|ULHASNAGAR|MIRA-BHAYANDAR|AMBERNATH|TIRUNELVELI|BHARATPUR|BEGUSARAI|NEW DELHI|GANDHIDHAM|BARANAGAR|TIRUVOTTIYUR|PUDUCHERRY|SILIGURI|THANE|NOIDA|FARIDABAD|GHAZIABAD|HOWRAH)",
                    r"STATE\s*:?\s*([A-Z][A-Z\s]+)",
                ],
                keywords=["state", "maharashtra", "karnataka", "gujarat", "tamil nadu", "delhi", "punjab", "rajasthan"],
                examples=["State: Maharashtra", "KARNATAKA", "State: Tamil Nadu"],
                confidence_weight=0.8,
                description="Indian state or union territory"
            )
        ])
        
        # PHONE NUMBER CATEGORY - Indian phone number patterns
        patterns.extend([
            FieldPattern(
                field_name="mobile_number",
                field_type="phone",
                field_category=self.FIELD_CATEGORIES['PHONE_NUMBER'],
                document_type="general",
                patterns=[
                    r"MOBILE\s*:?\s*(\+91\s*)?(\d{10})",
                    r"PHONE\s*:?\s*(\+91\s*)?(\d{10})",
                    r"TEL\s*:?\s*(\+91\s*)?(\d{10})",
                    r"CONTACT\s*:?\s*(\+91\s*)?(\d{10})",
                    r"(\+91[\s\-]?[6-9]\d{9})",
                    r"([6-9]\d{9})"
                ],
                keywords=["mobile", "phone", "tel", "contact", "number", "+91", "cell"],
                examples=["Mobile: +91 9876543210", "Phone: 9123456789", "Contact: +91-8765432109"],
                confidence_weight=0.95,
                description="Indian mobile phone number (10 digits starting with 6-9)"
            ),
            FieldPattern(
                field_name="landline_number",
                field_type="phone",
                field_category=self.FIELD_CATEGORIES['PHONE_NUMBER'],
                document_type="general",
                patterns=[
                    r"LANDLINE\s*:?\s*(\+91\s*)?(\d{2,4}\s*\d{6,8})",
                    r"TEL\s*:?\s*(\+91\s*)?(\d{2,4}[\s\-]\d{6,8})",
                    r"PHONE\s*:?\s*(\+91\s*)?(\d{2,4}[\s\-]\d{6,8})",
                    r"(\d{2,4}[\s\-]\d{6,8})"
                ],
                keywords=["landline", "tel", "phone", "office", "home"],
                examples=["Landline: 011-12345678", "Tel: 022 98765432", "Phone: 080-87654321"],
                confidence_weight=0.8,
                description="Indian landline number with area code"
            )
        ])
        
        # AADHAR NUMBER CATEGORY - Aadhar specific patterns
        patterns.extend([
            FieldPattern(
                field_name="aadhar_number",
                field_type="identifier",
                field_category=self.FIELD_CATEGORIES['AADHAR_NUMBER'],
                document_type="government",
                patterns=[
                    r"AADHAR\s*NO\.?\s*:?\s*(\d{4}\s*\d{4}\s*\d{4})",
                    r"AADHAAR\s*NO\.?\s*:?\s*(\d{4}\s*\d{4}\s*\d{4})",
                    r"AADHAR\s*NUMBER\s*:?\s*(\d{4}\s*\d{4}\s*\d{4})",
                    r"AADHAAR\s*NUMBER\s*:?\s*(\d{4}\s*\d{4}\s*\d{4})",
                    r"UID\s*:?\s*(\d{4}\s*\d{4}\s*\d{4})",
                    r"(\d{4}\s\d{4}\s\d{4})",
                    r"(\d{12})"
                ],
                keywords=["aadhar", "aadhaar", "uid", "unique", "identification", "12 digit"],
                examples=["Aadhar No.: 1234 5678 9012", "AADHAAR NUMBER: 987654321098", "UID: 1111 2222 3333"],
                confidence_weight=0.98,
                description="12-digit Aadhar/Aadhaar unique identification number"
            )
        ])
        
        # ID NUMBER CATEGORY - Various Indian identification numbers
        patterns.extend([
            FieldPattern(
                field_name="pan_number",
                field_type="identifier",
                field_category=self.FIELD_CATEGORIES['ID_NUMBER'],
                document_type="financial",
                patterns=[
                    r"PAN\s*NO\.?\s*:?\s*([A-Z]{5}\d{4}[A-Z])",
                    r"PAN\s*NUMBER\s*:?\s*([A-Z]{5}\d{4}[A-Z])",
                    r"PAN\s*:?\s*([A-Z]{5}\d{4}[A-Z])",
                    r"([A-Z]{5}\d{4}[A-Z])"
                ],
                keywords=["pan", "permanent", "account", "number", "income", "tax"],
                examples=["PAN No.: ABCDE1234F", "PAN NUMBER: XYXYX9999Z", "PAN: PQRST5678U"],
                confidence_weight=0.95,
                description="Permanent Account Number (PAN) - 10 character alphanumeric"
            ),
            FieldPattern(
                field_name="driving_license",
                field_type="identifier",
                field_category=self.FIELD_CATEGORIES['ID_NUMBER'],
                document_type="government",
                patterns=[
                    r"DRIVING\s*LICENSE\s*NO\.?\s*:?\s*([A-Z]{2}\d{2}\s?\d{11})",
                    r"DL\s*NO\.?\s*:?\s*([A-Z]{2}\d{2}\s?\d{11})",
                    r"LICENSE\s*NUMBER\s*:?\s*([A-Z]{2}\d{2}\s?\d{11})",
                    r"([A-Z]{2}\d{2}\s?\d{11})"
                ],
                keywords=["driving", "license", "dl", "vehicle", "transport"],
                examples=["Driving License No.: MH1420110012345", "DL NO: KA0520150067890", "License: TN0312200098765"],
                confidence_weight=0.9,
                description="Driving License Number (state code + 13 digits)"
            ),
            FieldPattern(
                field_name="voter_id",
                field_type="identifier", 
                field_category=self.FIELD_CATEGORIES['ID_NUMBER'],
                document_type="government",
                patterns=[
                    r"VOTER\s*ID\s*:?\s*([A-Z]{3}\d{7})",
                    r"EPIC\s*NO\.?\s*:?\s*([A-Z]{3}\d{7})",
                    r"ELECTORAL\s*ROLL\s*:?\s*([A-Z]{3}\d{7})",
                    r"([A-Z]{3}\d{7})"
                ],
                keywords=["voter", "id", "epic", "electoral", "roll", "election"],
                examples=["Voter ID: ABC1234567", "EPIC NO: XYZ9876543", "Electoral Roll: PQR5554433"],
                confidence_weight=0.85,
                description="Voter ID/EPIC Number (3 letters + 7 digits)"
            ),
            FieldPattern(
                field_name="passport_number",
                field_type="identifier",
                field_category=self.FIELD_CATEGORIES['ID_NUMBER'],
                document_type="government",
                patterns=[
                    r"PASSPORT\s*NO\.?\s*:?\s*([A-Z]\d{7})",
                    r"PASSPORT\s*NUMBER\s*:?\s*([A-Z]\d{7})",
                    r"PASSPORT\s*:?\s*([A-Z]\d{7})",
                    r"([A-Z]\d{7})"
                ],
                keywords=["passport", "number", "travel", "document", "international"],
                examples=["Passport No.: A1234567", "PASSPORT NUMBER: Z9876543", "Passport: M5554433"],
                confidence_weight=0.9,
                description="Indian Passport Number (1 letter + 7 digits)"
            )
        ])
        
        # OTHER CATEGORY - Existing patterns that don't fit above categories
        # OTHER CATEGORY - Financial, document, and other field types
        patterns.extend([
            FieldPattern(
                field_name="invoice_number",
                field_type="identifier",
                field_category=self.FIELD_CATEGORIES['OTHER'],
                document_type="invoice",
                patterns=[
                    r"INVOICE\s*#?\s*:?\s*([A-Z0-9\-]+)",
                    r"INV\s*#?\s*:?\s*([A-Z0-9\-]+)",
                    r"BILL\s*#?\s*:?\s*([A-Z0-9\-]+)",
                    r"NUMBER\s*:?\s*([A-Z0-9\-]+)"
                ],
                keywords=["invoice", "number", "bill", "inv", "#"],
                examples=["INV-12345", "BILL-2024-001", "Invoice Number: A123"],
                confidence_weight=0.9,
                description="Unique identifier for invoice documents"
            ),
            FieldPattern(
                field_name="total_amount",
                field_type="currency",
                field_category=self.FIELD_CATEGORIES['OTHER'],
                document_type="invoice",
                patterns=[
                    r"TOTAL\s*:?\s*₹?\s*([\d,]+\.?\d*)",
                    r"AMOUNT\s*DUE\s*:?\s*₹?\s*([\d,]+\.?\d*)",
                    r"GRAND\s*TOTAL\s*:?\s*₹?\s*([\d,]+\.?\d*)",
                    r"BALANCE\s*:?\s*₹?\s*([\d,]+\.?\d*)",
                    r"TOTAL\s*:?\s*\$?\s*([\d,]+\.?\d*)",
                    r"AMOUNT\s*DUE\s*:?\s*\$?\s*([\d,]+\.?\d*)"
                ],
                keywords=["total", "amount", "due", "grand", "balance", "₹", "$", "rupees"],
                examples=["Total: ₹1,234.56", "Amount Due $500.00", "GRAND TOTAL: ₹2,500.00"],
                confidence_weight=0.95,
                description="Final amount to be paid on invoice"
            ),
            FieldPattern(
                field_name="email_address",
                field_type="email",
                field_category=self.FIELD_CATEGORIES['OTHER'],
                document_type="general",
                patterns=[
                    r"EMAIL\s*:?\s*([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})",
                    r"E-MAIL\s*:?\s*([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})",
                    r"([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})"
                ],
                keywords=["email", "e-mail", "@", ".com", ".org", ".in"],
                examples=["Email: john@example.com", "john.doe@company.org", "contact@business.co.in"],
                confidence_weight=0.95,
                description="Email address"
            ),
            FieldPattern(
                field_name="date_of_birth",
                field_type="date",
                field_category=self.FIELD_CATEGORIES['OTHER'],
                document_type="general",
                patterns=[
                    r"DOB\s*:?\s*(\d{1,2}/\d{1,2}/\d{4})",
                    r"BIRTH\s*DATE\s*:?\s*(\d{1,2}/\d{1,2}/\d{4})",
                    r"BORN\s*:?\s*(\d{1,2}/\d{1,2}/\d{4})",
                    r"DATE\s*OF\s*BIRTH\s*:?\s*(\d{1,2}/\d{1,2}/\d{4})",
                    r"(\d{1,2}-\d{1,2}-\d{4})"
                ],
                keywords=["dob", "birth", "born", "date"],
                examples=["DOB: 01/15/1980", "Birth Date: 12/25/1990", "Born: 03/10/1985"],
                confidence_weight=0.95,
                description="Date of birth"
            ),
            FieldPattern(
                field_name="account_number",
                field_type="identifier",
                field_category=self.FIELD_CATEGORIES['OTHER'],
                document_type="financial",
                patterns=[
                    r"ACCOUNT\s*#?\s*:?\s*(\d+)",
                    r"ACCT\s*#?\s*:?\s*(\d+)",
                    r"A/C\s*:?\s*(\d+)",
                    r"BANK\s*ACCOUNT\s*:?\s*(\d+)"
                ],
                keywords=["account", "acct", "a/c", "number", "#", "bank"],
                examples=["Account #123456789", "ACCT: 987654321", "A/C 555666777"],
                confidence_weight=0.9,
                description="Bank account number"
            )
        ])
        
        return patterns
    
    def _initialize_document_classifier(self) -> Pipeline:
        """Initialize advanced ML-based document classifier with comprehensive training data."""
        training_texts = []
        training_labels = []
        
        # Enhanced training data with realistic document samples
        training_samples = {
            'AADHAR_CARD': [
                "government of india unique identification authority aadhaar card name date birth address mobile",
                "भारत सरकार आधार कार्ड unique identification uid number government india",
                "uidai aadhaar number twelve digit unique identity document indian citizen",
                "aadhar card government issued identity proof uid verification document",
                "unique identification authority india aadhaar enrollment number biometric"
            ],
            'VOTER_ID': [
                "election commission india electoral photo identity card epic voter",
                "भारत निर्वाचन आयोग voter id card election commission epic number",
                "electoral roll voter registration card polling station constituency",
                "election identity card voter epic number assembly constituency part",
                "निर्वाचक फोटो पहचान पत्र election commission voter identity document"
            ],
            'PAN_CARD': [
                "income tax department permanent account number pan card india",
                "आयकर विभाग pan card permanent account number tax department",
                "pan number income tax identification permanent account alphanumeric",
                "tax identification number permanent account pan india revenue",
                "income tax department govt india permanent account number card"
            ],
            'DRIVING_LICENSE': [
                "driving license transport department motor vehicle license drive",
                "ड्राइविंग लाइसेंस transport department vehicle license motor driving",
                "license drive motor vehicle transport authority state government",
                "driving permit vehicle license transport department road authority",
                "motor vehicle act driving license transport commissioner state"
            ],
            'PASSPORT': [
                "passport republic india ministry external affairs travel document",
                "पासपोर्ट भारत गणराज्य passport republic india travel document",
                "ministry external affairs passport india travel international document",
                "passport travel document republic india mea issued passport",
                "indian passport ministry external affairs travel document republic"
            ],
            'MARKSHEET': [
                "marksheet marks obtained examination result grade percentage cgpa",
                "अंक तालिका marksheet examination result university college marks",
                "examination result marksheet grade card university board marks",
                "marks certificate examination marksheet grade percentage result",
                "academic transcript marksheet examination university college result"
            ],
            'RATION_CARD': [
                "ration card food security civil supplies public distribution system",
                "राशन कार्ड food card civil supplies ration distribution family",
                "public distribution system ration card food security family card",
                "food security card ration civil supplies below poverty line",
                "family ration card civil supplies food distribution government"
            ],
            'BANK_PASSBOOK': [
                "bank passbook savings account current account bank statement balance",
                "बैंक पासबुक savings account bank passbook account number ifsc",
                "bank account passbook savings current account transaction balance",
                "passbook bank account holder savings current transaction record",
                "bank statement passbook account balance transaction savings current"
            ],
            'BIRTH_CERTIFICATE': [
                "birth certificate registrar births deaths municipal corporation",
                "जन्म प्रमाण पत्र birth certificate municipal corporation registrar",
                "certificate birth registrar births deaths municipal authority",
                "birth registration certificate municipal corporation birth record",
                "registrar births deaths birth certificate municipal government"
            ],
            'COMMUNITY_CERTIFICATE': [
                "community certificate caste certificate obc sc st backward class",
                "जाति प्रमाण पत्र community certificate caste obc sc st",
                "caste certificate community backward class scheduled caste tribe",
                "community caste certificate obc sc st backward class revenue",
                "scheduled caste tribe community certificate backward class obc"
            ],
            'SMART_CARD': [
                "smart card chip card employee card health card identification",
                "स्मार्ट कार्ड smart card chip card digital identity employee",
                "chip card smart card employee identification digital card",
                "smart card technology chip based card employee health card",
                "digital smart card chip card employee identification health"
            ]
        }
        
        # Generate comprehensive training data
        for doc_type, samples in training_samples.items():
            for sample in samples:
                training_texts.append(sample)
                training_labels.append(doc_type)
        
        # Add keyword-based training samples
        for doc_type, doc_info in self.DOCUMENT_TYPES.items():
            # Create contextual keyword combinations
            keywords = doc_info['keywords']
            for i in range(len(keywords)):
                for j in range(i+1, min(i+4, len(keywords))):
                    combined_text = ' '.join(keywords[i:j+1])
                    training_texts.append(combined_text)
                    training_labels.append(doc_type)
        
        # Add pattern-based training samples
        for doc_type, doc_info in self.DOCUMENT_TYPES.items():
            for pattern in doc_info['patterns']:
                # Convert regex patterns to descriptive text
                pattern_descriptions = {
                    r'\b\d{4}\s+\d{4}\s+\d{4}\b': 'twelve digit number format aadhaar uid',
                    r'\b[A-Z]{3}\d{7}\b': 'three letters seven digits voter epic format',
                    r'\b[A-Z]{5}\d{4}[A-Z]\b': 'five letters four digits one letter pan format',
                    r'\b[A-Z]{2}\d{13}\b': 'two letters thirteen digits driving license format',
                    r'\b[A-Z]\d{7}\b': 'one letter seven digits passport format'
                }
                
                for regex, description in pattern_descriptions.items():
                    if regex in pattern:
                        training_texts.append(description)
                        training_labels.append(doc_type)
        
        # Enhanced ML pipeline with better feature extraction
        classifier = Pipeline([
            ('tfidf', TfidfVectorizer(
                max_features=2000,
                lowercase=True,
                ngram_range=(1, 3),  # Include trigrams for better context
                min_df=1,
                max_df=0.95,
                stop_words=None,  # Keep all words for domain-specific terms
                sublinear_tf=True
            )),
            ('nb', MultinomialNB(alpha=0.5))  # Reduced alpha for less smoothing
        ])
        
        # Train the classifier if we have data
        if training_texts and training_labels:
            classifier.fit(training_texts, training_labels)
            
        return classifier
    
    def retrain_classifier_with_new_data(self, new_training_data: Dict[str, List[str]]) -> None:
        """
        Retrain the classifier with new training data.
        
        Args:
            new_training_data: Dictionary with document_type as key and list of sample texts as values
        """
        # Combine existing training data with new data
        all_texts = []
        all_labels = []
        
        # Add existing training data
        for doc_type, doc_info in self.DOCUMENT_TYPES.items():
            keywords = doc_info['keywords']
            for i in range(len(keywords)):
                for j in range(i+1, min(i+4, len(keywords))):
                    combined_text = ' '.join(keywords[i:j+1])
                    all_texts.append(combined_text)
                    all_labels.append(doc_type)
        
        # Add new training data
        for doc_type, samples in new_training_data.items():
            for sample in samples:
                all_texts.append(sample.lower())
                all_labels.append(doc_type)
        
        # Retrain the classifier
        if all_texts and all_labels:
            self.document_classifier.fit(all_texts, all_labels)
            logger.info(f"Classifier retrained with {len(all_texts)} samples")
    
    def save_trained_model(self, model_path: str = "models/rag_document_classifier.pkl") -> None:
        """Save the trained RAG classifier model."""
        from pathlib import Path
        import pickle
        
        Path(model_path).parent.mkdir(exist_ok=True)
        
        model_data = {
            'classifier': self.document_classifier,
            'document_types': self.DOCUMENT_TYPES,
            'field_categories': self.FIELD_CATEGORIES,
            'training_timestamp': datetime.now(),
            'model_version': '2.0'
        }
        
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"RAG model saved to: {model_path}")
    
    def load_trained_model(self, model_path: str = "models/rag_document_classifier.pkl") -> None:
        """Load a pre-trained RAG classifier model."""
        import pickle
        
        try:
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            self.document_classifier = model_data['classifier']
            logger.info(f"RAG model loaded from: {model_path}")
            logger.info(f"Model version: {model_data.get('model_version', 'unknown')}")
        except FileNotFoundError:
            logger.warning(f"Model file not found: {model_path}. Using default classifier.")
        except Exception as e:
            logger.error(f"Error loading model: {e}. Using default classifier.")
    
    def classify_document(self, text: str) -> List[DocumentClassification]:
        """
        Advanced document classification using hybrid ML and rule-based approach.
        
        Args:
            text: Extracted text from the document
            
        Returns:
            List of document classifications sorted by confidence
        """
        text_lower = text.lower()
        classifications = []
        
        # Preprocess text for better feature extraction
        processed_text = self._preprocess_text_for_classification(text_lower)
        
        # Rule-based classification using patterns and keywords
        rule_based_scores = {}
        for doc_type, doc_info in self.DOCUMENT_TYPES.items():
            confidence = 0.0
            keywords_found = []
            patterns_matched = []
            
            # Enhanced keyword matching with fuzzy matching
            keyword_score = 0
            for keyword in doc_info['keywords']:
                if keyword.lower() in text_lower:
                    keywords_found.append(keyword)
                    keyword_score += 1
                # Fuzzy matching for common OCR errors
                elif self._fuzzy_match_keyword(keyword.lower(), text_lower):
                    keywords_found.append(keyword + " (fuzzy)")
                    keyword_score += 0.8
            
            # Normalize keyword score
            if doc_info['keywords']:
                keyword_confidence = keyword_score / len(doc_info['keywords'])
                confidence += keyword_confidence * 0.6  # 60% weight for keywords
            
            # Enhanced pattern matching
            pattern_score = 0
            for pattern in doc_info['patterns']:
                try:
                    matches = re.findall(pattern, text, re.IGNORECASE)
                    if matches:
                        patterns_matched.extend([pattern] * len(matches))
                        pattern_score += len(matches)
                except re.error:
                    continue
            
            # Normalize pattern score
            if doc_info['patterns']:
                pattern_confidence = min(1.0, pattern_score / len(doc_info['patterns']))
                confidence += pattern_confidence * 0.4  # 40% weight for patterns
            
            # Apply document-specific confidence weight
            confidence *= doc_info['confidence_weight']
            rule_based_scores[doc_type] = confidence
            
            # Store classification if above threshold
            if confidence > 0.05:  # Lower threshold for more candidates
                reasoning = f"Found {len(keywords_found)} keywords and {len(patterns_matched)} pattern matches"
                classifications.append(
                    DocumentClassification(
                        document_type=doc_type,
                        confidence=confidence,
                        keywords_found=keywords_found,
                        patterns_matched=patterns_matched,
                        reasoning=reasoning
                    )
                )
        
        # Enhanced ML-based classification
        try:
            ml_predictions = self.document_classifier.predict_proba([processed_text])
            ml_classes = self.document_classifier.classes_
            
            # Get confidence scores for all classes
            ml_scores = {}
            for i, prob in enumerate(ml_predictions[0]):
                doc_type = ml_classes[i]
                ml_scores[doc_type] = prob
            
            # Combine ML and rule-based scores using weighted fusion
            for doc_type in self.DOCUMENT_TYPES.keys():
                rule_score = rule_based_scores.get(doc_type, 0.0)
                ml_score = ml_scores.get(doc_type, 0.0)
                
                # Weighted fusion: 70% rule-based, 30% ML
                combined_score = (rule_score * 0.7) + (ml_score * 0.3)
                
                # Update existing classification or create new one
                existing = next((c for c in classifications if c.document_type == doc_type), None)
                if existing:
                    existing.confidence = min(0.95, combined_score)
                    existing.reasoning += f" + ML boost ({ml_score:.3f})"
                elif ml_score > 0.15:  # Add ML-only classifications with high confidence
                    classifications.append(
                        DocumentClassification(
                            document_type=doc_type,
                            confidence=combined_score,
                            keywords_found=[],
                            patterns_matched=[],
                            reasoning=f"ML-based classification (confidence: {ml_score:.3f})"
                        )
                    )
            
        except Exception as e:
            logger.warning(f"ML classification failed: {e}")
        
        # Advanced post-processing and ranking
        classifications = self._post_process_classifications(classifications, text_lower)
        
        # Sort by confidence and return top results
        classifications.sort(key=lambda x: x.confidence, reverse=True)
        return classifications[:5]  # Return top 5 classifications
    
    def _preprocess_text_for_classification(self, text: str) -> str:
        """Preprocess text for better ML classification."""
        # Remove extra whitespace and normalize
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Expand common abbreviations
        abbreviations = {
            'govt': 'government',
            'id': 'identity',
            'dob': 'date of birth',
            'a/c': 'account',
            'no': 'number',
            'dept': 'department'
        }
        
        for abbr, full in abbreviations.items():
            text = re.sub(r'\b' + abbr + r'\b', full, text, flags=re.IGNORECASE)
        
        return text
    
    def _fuzzy_match_keyword(self, keyword: str, text: str, threshold: float = 0.8) -> bool:
        """Fuzzy matching for OCR errors and variations."""
        # Simple character substitution for common OCR errors
        ocr_substitutions = {
            '0': 'o', '1': 'l', '5': 's', '8': 'b',
            'o': '0', 'l': '1', 's': '5', 'b': '8'
        }
        
        # Generate variations of the keyword
        variations = [keyword]
        for original, replacement in ocr_substitutions.items():
            if original in keyword:
                variations.append(keyword.replace(original, replacement))
        
        # Check if any variation exists in text
        for variation in variations:
            if variation in text:
                return True
        
        return False
    
    def _post_process_classifications(self, classifications: List[DocumentClassification], text: str) -> List[DocumentClassification]:
        """Post-process classifications for better accuracy."""
        # Remove duplicates
        seen_types = set()
        unique_classifications = []
        for cls in classifications:
            if cls.document_type not in seen_types:
                seen_types.add(cls.document_type)
                unique_classifications.append(cls)
        
        # Apply context-based boosting
        for cls in unique_classifications:
            # Boost confidence for documents with multiple strong indicators
            if len(cls.keywords_found) >= 3 and len(cls.patterns_matched) >= 2:
                cls.confidence = min(0.95, cls.confidence * 1.15)
                cls.reasoning += " (Strong indicators boost)"
            
            # Apply document-specific logic
            if cls.document_type == 'AADHAR_CARD':
                # Look for 12-digit number pattern
                if re.search(r'\b\d{4}\s*\d{4}\s*\d{4}\b', text):
                    cls.confidence = min(0.95, cls.confidence * 1.2)
                    cls.reasoning += " (Aadhaar number pattern detected)"
            
            elif cls.document_type == 'PAN_CARD':
                # Look for PAN format
                if re.search(r'\b[A-Z]{5}\d{4}[A-Z]\b', text):
                    cls.confidence = min(0.95, cls.confidence * 1.25)
                    cls.reasoning += " (PAN number format detected)"
        
        return unique_classifications

class RAGFieldSuggestionEngine:
    """RAG-based field suggestion engine for intelligent OCR field extraction."""
    
    def __init__(self, knowledge_base_path: Optional[str] = None):
        self.knowledge_base = DocumentFieldKnowledgeBase()
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 3),
            lowercase=True
        )
        self.field_vectors = None
        self.field_index = []
        self.cache_path = knowledge_base_path or "rag_cache.pkl"
        
        # Initialize or load the vectorized knowledge base
        self._initialize_vectors()
    
    def _initialize_vectors(self):
        """Initialize TF-IDF vectors for all field patterns."""
        try:
            # Try to load cached vectors
            if Path(self.cache_path).exists():
                with open(self.cache_path, 'rb') as f:
                    cache_data = pickle.load(f)
                    self.field_vectors = cache_data['vectors']
                    self.field_index = cache_data['index'] 
                    self.vectorizer = cache_data['vectorizer']
                logger.info("Loaded cached RAG vectors")
                return
        except Exception as e:
            logger.warning(f"Could not load cached vectors: {e}")
        
        # Build vectors from scratch
        logger.info("Building RAG field vectors...")
        
        # Create text corpus from all field patterns
        corpus = []
        self.field_index = []
        
        for pattern in self.knowledge_base.field_patterns:
            # Combine all pattern information into searchable text
            text_content = f"{pattern.field_name} {pattern.field_type} {pattern.document_type} "
            text_content += " ".join(pattern.keywords) + " "
            text_content += " ".join(pattern.examples) + " "
            text_content += pattern.description
            
            corpus.append(text_content)
            self.field_index.append(pattern)
        
        # Fit vectorizer and transform corpus
        self.field_vectors = self.vectorizer.fit_transform(corpus)
        
        # Cache the results
        try:
            cache_data = {
                'vectors': self.field_vectors,
                'index': self.field_index,
                'vectorizer': self.vectorizer
            }
            with open(self.cache_path, 'wb') as f:
                pickle.dump(cache_data, f)
            logger.info("Cached RAG vectors for future use")
        except Exception as e:
            logger.warning(f"Could not cache vectors: {e}")
    
    def suggest_fields(self, text: str, document_type: str = None, top_k: int = 10) -> List[FieldSuggestion]:
        """
        Suggest relevant fields for extraction based on document text.
        
        Args:
            text: OCR extracted text from document
            document_type: Optional document type hint
            top_k: Number of top suggestions to return
            
        Returns:
            List of field suggestions with confidence scores
        """
        suggestions = []
        
        # Vector similarity search
        query_vector = self.vectorizer.transform([text])
        similarities = cosine_similarity(query_vector, self.field_vectors)[0]
        
        # Get top similar field patterns
        top_indices = np.argsort(similarities)[::-1][:top_k * 2]  # Get more than needed for filtering
        
        for idx in top_indices:
            pattern = self.field_index[idx]
            similarity_score = similarities[idx]
            
            # Apply document type filter if specified
            if document_type and pattern.document_type != document_type and pattern.document_type != 'general':
                continue
                
            # Try to extract field value using pattern matching
            extracted_value, match_confidence = self._extract_field_value(text, pattern)
            
            if extracted_value:
                # Calculate overall confidence
                overall_confidence = (
                    similarity_score * 0.3 +  # RAG similarity
                    match_confidence * 0.4 +  # Pattern match quality
                    pattern.confidence_weight * 0.3  # Pattern reliability
                )
                
                suggestion = FieldSuggestion(
                    field_name=pattern.field_name,
                    field_type=pattern.field_type,
                    field_category=pattern.field_category,
                    suggested_value=extracted_value,
                    confidence=overall_confidence,
                    source_text=text,
                    pattern_matched=pattern.patterns[0] if pattern.patterns else "",
                    reasoning=f"Matched {pattern.description} using RAG similarity {similarity_score:.3f}"
                )
                
                suggestions.append(suggestion)
        
        # Sort by confidence and remove duplicates
        suggestions = sorted(suggestions, key=lambda x: x.confidence, reverse=True)
        seen_fields = set()
        unique_suggestions = []
        
        for suggestion in suggestions:
            if suggestion.field_name not in seen_fields:
                seen_fields.add(suggestion.field_name)
                unique_suggestions.append(suggestion)
                if len(unique_suggestions) >= top_k:
                    break
        
        return unique_suggestions
    
    def _extract_field_value(self, text: str, pattern: FieldPattern) -> Tuple[Optional[str], float]:
        """
        Extract field value using the pattern's regex patterns.
        
        Args:
            text: Text to search in
            pattern: Field pattern to match
            
        Returns:
            Tuple of (extracted_value, confidence)
        """
        for regex_pattern in pattern.patterns:
            try:
                matches = re.finditer(regex_pattern, text, re.IGNORECASE | re.MULTILINE)
                for match in matches:
                    if match.groups():
                        # Extract the captured group
                        extracted_value = match.group(1).strip()
                        
                        # Calculate match confidence based on pattern specificity
                        confidence = 0.8  # Base confidence
                        
                        # Boost confidence for exact keyword matches
                        for keyword in pattern.keywords:
                            if keyword.lower() in text.lower():
                                confidence += 0.05
                        
                        # Boost confidence for field type validation
                        if self._validate_field_type(extracted_value, pattern.field_type):
                            confidence += 0.15
                        
                        return extracted_value, min(confidence, 1.0)
                        
            except re.error as e:
                logger.warning(f"Regex error in pattern {regex_pattern}: {e}")
                continue
        
        return None, 0.0
    
    def _validate_field_type(self, value: str, field_type: str) -> bool:
        """Validate if extracted value matches expected field type."""
        if field_type == "currency":
            return bool(re.match(r'^\$?[\d,]+\.?\d*$', value))
        elif field_type == "date":
            return bool(re.match(r'^\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4}$', value))
        elif field_type == "email":
            return bool(re.match(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', value))
        elif field_type == "phone":
            return bool(re.match(r'^\(?\d{3}\)?[\s\-]?\d{3}[\s\-]?\d{4}$', value))
        elif field_type == "identifier":
            return bool(re.match(r'^[A-Z0-9\-]+$', value))
        else:  # text or other types
            return len(value.strip()) > 0
    
    def add_field_pattern(self, pattern: FieldPattern):
        """Add a new field pattern to the knowledge base and update vectors."""
        self.knowledge_base.field_patterns.append(pattern)
        self._initialize_vectors()  # Rebuild vectors
        logger.info(f"Added new field pattern: {pattern.field_name}")
    
    def get_field_suggestions_summary(self, suggestions: List[FieldSuggestion]) -> Dict[str, Any]:
        """Generate a summary of field suggestions for display."""
        if not suggestions:
            return {"message": "No field suggestions found", "suggestions": []}
        
        summary = {
            "total_suggestions": len(suggestions),
            "avg_confidence": np.mean([s.confidence for s in suggestions]),
            "suggestions": []
        }
        
        for suggestion in suggestions:
            summary["suggestions"].append({
                "field_name": suggestion.field_name,
                "field_type": suggestion.field_type,
                "field_category": suggestion.field_category,
                "value": suggestion.suggested_value,
                "confidence": f"{suggestion.confidence:.3f}",
                "reasoning": suggestion.reasoning
            })
        
        return summary
    
    def analyze_document_with_classification(self, text: str, top_k: int = 10) -> Dict[str, Any]:
        """
        Analyze document with both type classification and field suggestions.
        
        Args:
            text: OCR extracted text from document
            top_k: Number of top field suggestions to return
            
        Returns:
            Dictionary containing document classifications and field suggestions
        """
        # Classify document type
        document_classifications = self.knowledge_base.classify_document(text)
        
        # Filter document classifications to only show high confidence ones (>= 0.7)
        high_confidence_classifications = [
            cls for cls in document_classifications 
            if cls.confidence >= 0.7
        ]
        
        # Get field suggestions
        field_suggestions = self.suggest_fields(text, top_k=top_k)
        
        # Enhanced analysis based on document type (use high confidence classifications)
        enhanced_suggestions = []
        if high_confidence_classifications:
            best_doc_type = high_confidence_classifications[0].document_type
            
            # Filter and prioritize suggestions based on document type
            for suggestion in field_suggestions:
                # Check if this field is relevant for the detected document type
                doc_info = self.knowledge_base.DOCUMENT_TYPES.get(best_doc_type, {})
                required_fields = doc_info.get('required_fields', [])
                
                # Boost confidence for required fields
                if suggestion.field_name in required_fields:
                    suggestion.confidence = min(0.95, suggestion.confidence * 1.2)
                    suggestion.reasoning += f" (Required for {best_doc_type})"
                
                enhanced_suggestions.append(suggestion)
        else:
            enhanced_suggestions = field_suggestions
        
        # Sort by confidence again after boosting
        enhanced_suggestions.sort(key=lambda x: x.confidence, reverse=True)
        
        # Filter document classifications to only show high confidence ones (>= 0.7)
        high_confidence_classifications = [
            cls for cls in document_classifications 
            if cls.confidence >= 0.7
        ]
        
        return {
            "document_classifications": [
                {
                    "document_type": cls.document_type,
                    "confidence": f"{cls.confidence:.3f}",
                    "keywords_found": cls.keywords_found,
                    "patterns_matched": len(cls.patterns_matched),
                    "reasoning": cls.reasoning
                } for cls in high_confidence_classifications
            ],
            "field_suggestions": [
                {
                    "field_name": suggestion.field_name,
                    "field_type": suggestion.field_type,
                    "field_category": suggestion.field_category,
                    "suggested_value": suggestion.suggested_value,
                    "confidence": f"{suggestion.confidence:.3f}",
                    "reasoning": suggestion.reasoning
                } for suggestion in enhanced_suggestions[:top_k]
            ],
            "analysis_summary": {
                "total_classifications": len(high_confidence_classifications),
                "best_document_type": high_confidence_classifications[0].document_type if high_confidence_classifications else "UNKNOWN",
                "best_confidence": f"{high_confidence_classifications[0].confidence:.3f}" if high_confidence_classifications else "0.000",
                "total_field_suggestions": len(enhanced_suggestions),
                "high_confidence_fields": len([s for s in enhanced_suggestions if s.confidence > 0.7]),
                "high_confidence_doc_types": len(high_confidence_classifications)
            }
        }

# Example usage and testing
if __name__ == "__main__":
    # Initialize RAG engine
    rag_engine = RAGFieldSuggestionEngine()
    
    # Test with sample invoice text
    sample_invoice = """
    RAJESH KUMAR
    S/O: RAM PRASAD
    Address: 123 MG Road, Bangalore 560001
    Mobile: +91 9876543210
    Email: rajesh.kumar@email.com
    Aadhar No.: 1234 5678 9012
    PAN: ABCDE1234F
    
    ACME CORPORATION
    456 Business St
    Delhi, India 110001
    
    INVOICE #INV-2024-001
    
    Bill To:
    Priya Sharma
    789 Customer Ave
    Mumbai 400001
    Phone: 9123456789
    
    Total Amount: ₹1,234.56
    Due Date: 12/15/2024
    """
    
    print("🤖 RAG Field Suggestion Engine Test")
    print("=" * 50)
    print(f"Sample Text:\n{sample_invoice}")
    print("-" * 50)
    
    suggestions = rag_engine.suggest_fields(sample_invoice, "invoice")
    summary = rag_engine.get_field_suggestions_summary(suggestions)
    
    print(f"Found {summary['total_suggestions']} suggestions:")
    print(f"Average confidence: {summary['avg_confidence']:.3f}")
    print()
    
    for i, suggestion in enumerate(summary['suggestions'], 1):
        print(f"{i}. {suggestion['field_name']} ({suggestion['field_type']}) - Category: {suggestion['field_category']}")
        print(f"   Value: {suggestion['value']}")
        print(f"   Confidence: {suggestion['confidence']}")
        print(f"   Reasoning: {suggestion['reasoning']}")
        print()