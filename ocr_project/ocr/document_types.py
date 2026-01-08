"""
Advanced document type detection and processing for OCR system.
Supports comprehensive Indian document types with metadata integration.
"""

import re
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime


class DocumentTypeDetector:
    """Advanced document type detection with confidence scoring and metadata integration."""
    
    def __init__(self):
        self.document_patterns = {
            # === GOVERNMENT IDs ===
            'aadhaar_card': {
                'keywords': [
                    'AADHAAR', 'AADHAR', 'UID', 'UNIQUE IDENTIFICATION', 'UIDAI',
                    'भारत सरकार', 'GOVERNMENT OF INDIA', 'ENROLLMENT', 'VID',
                    'VIRTUAL ID', 'PERMANENT ACCOUNT', 'DOB', 'MALE', 'FEMALE'
                ],
                'patterns': [
                    r'\d{4}\s*\d{4}\s*\d{4}',  # 12-digit Aadhaar number
                    r'DOB\s*:?\s*\d{2}/\d{2}/\d{4}',
                    r'ENROLLMENT\s*NO',
                    r'VID\s*:?\s*\d{16}'
                ],
                'min_keywords': 1,
                'confidence_threshold': 0.3
            },
            'pan_card': {
                'keywords': [
                    'PAN', 'PERMANENT ACCOUNT NUMBER', 'INCOME TAX DEPARTMENT',
                    'GOVT OF INDIA', 'SIGNATURE', 'FATHER', 'NAME', 'DATE OF BIRTH'
                ],
                'patterns': [
                    r'[A-Z]{5}\d{4}[A-Z]',  # PAN format: 5 letters, 4 digits, 1 letter
                    r'PERMANENT\s+ACCOUNT\s+NUMBER',
                    r'INCOME\s+TAX'
                ],
                'min_keywords': 1,
                'confidence_threshold': 0.3
            },
            'passport': {
                'keywords': [
                    'PASSPORT', 'REPUBLIC OF INDIA', 'P<IND', 'NATIONALITY',
                    'PASSPORT NO', 'PLACE OF BIRTH', 'PLACE OF ISSUE',
                    'DATE OF ISSUE', 'DATE OF EXPIRY', 'SURNAME', 'GIVEN NAME'
                ],
                'patterns': [
                    r'P<IND[A-Z<]+',  # MRZ line
                    r'[A-Z]\d{7}',  # Passport number
                    r'PASSPORT\s*NO',
                    r'DATE\s*OF\s*ISSUE',
                    r'DATE\s*OF\s*EXPIRY'
                ],
                'min_keywords': 2,
                'confidence_threshold': 0.5
            },
            'voter_id': {
                'keywords': [
                    'ELECTION COMMISSION', 'ELECTOR', 'PHOTO IDENTITY CARD',
                    'EPIC', 'VOTER', 'ASSEMBLY CONSTITUENCY', 'PART NO',
                    'SERIAL NO', 'NAME', 'AGE', 'SEX', 'FATHER', 'HUSBAND'
                ],
                'patterns': [
                    r'[A-Z]{3}\d{7}',  # Voter ID format
                    r'ELECTION\s+COMMISSION',
                    r'ELECTOR.*PHOTO',
                    r'PART\s*NO',
                    r'SERIAL\s*NO'
                ],
                'min_keywords': 2,
                'confidence_threshold': 0.5
            },
            'driving_licence': {
                'keywords': [
                    'DRIVING LICENCE', 'DRIVING LICENSE', 'DL', 'LICENSE TO DRIVE',
                    'TRANSPORT', 'VALID TILL', 'VALID FROM', 'DATE OF ISSUE',
                    'BLOOD GROUP', 'COV', 'MCWG', 'LMV', 'VEHICLE CLASS'
                ],
                'patterns': [
                    r'[A-Z]{2}\d{13}',  # DL number format (varies by state)
                    r'DL\s*NO',
                    r'VALID\s*TILL',
                    r'VALID\s*FROM',
                    r'COV\s*:?',
                    r'MCWG|LMV|HMV'
                ],
                'min_keywords': 2,
                'confidence_threshold': 0.5
            },
            'ration_card': {
                'keywords': [
                    'RATION CARD', 'FOOD', 'CIVIL SUPPLIES', 'CONSUMER',
                    'CARD NUMBER', 'PRIORITY', 'AAY', 'BPL', 'APL',
                    'HEAD OF FAMILY', 'MEMBER'
                ],
                'patterns': [
                    r'RATION\s*CARD',
                    r'FOOD.*CIVIL\s*SUPPLIES',
                    r'CARD\s*NO',
                    r'AAY|BPL|APL'
                ],
                'min_keywords': 2,
                'confidence_threshold': 0.5
            },
            
            # === CERTIFICATES ===
            'birth_certificate': {
                'keywords': [
                    'BIRTH CERTIFICATE', 'CERTIFICATE OF BIRTH', 'DATE OF BIRTH',
                    'PLACE OF BIRTH', 'NAME OF CHILD', 'FATHER NAME', 'MOTHER NAME',
                    'REGISTRATION NUMBER', 'REGISTRAR', 'MUNICIPAL', 'CORPORATION'
                ],
                'patterns': [
                    r'BIRTH\s*CERTIFICATE',
                    r'DATE\s*OF\s*BIRTH',
                    r'PLACE\s*OF\s*BIRTH',
                    r'REGISTRATION\s*NO',
                    r'REGISTRAR'
                ],
                'min_keywords': 2,
                'confidence_threshold': 0.5
            },
            'marriage_certificate': {
                'keywords': [
                    'MARRIAGE CERTIFICATE', 'CERTIFICATE OF MARRIAGE', 'SOLEMNIZED',
                    'BRIDEGROOM', 'BRIDE', 'WITNESS', 'MARRIAGE REGISTRATION',
                    'DATE OF MARRIAGE', 'PLACE OF MARRIAGE', 'REGISTRAR'
                ],
                'patterns': [
                    r'MARRIAGE\s*CERTIFICATE',
                    r'SOLEMNIZED',
                    r'DATE\s*OF\s*MARRIAGE',
                    r'BRIDEGROOM|BRIDE',
                    r'WITNESS'
                ],
                'min_keywords': 2,
                'confidence_threshold': 0.5
            },
            'caste_certificate': {
                'keywords': [
                    'CASTE CERTIFICATE', 'SCHEDULED CASTE', 'SCHEDULED TRIBE',
                    'OTHER BACKWARD CLASS', 'OBC', 'SC', 'ST', 'COMMUNITY',
                    'TEHSILDAR', 'REVENUE', 'DISTRICT'
                ],
                'patterns': [
                    r'CASTE\s*CERTIFICATE',
                    r'SC|ST|OBC',
                    r'SCHEDULED\s*CASTE|SCHEDULED\s*TRIBE',
                    r'BACKWARD\s*CLASS',
                    r'TEHSILDAR'
                ],
                'min_keywords': 2,
                'confidence_threshold': 0.5
            },
            'character_certificate': {
                'keywords': [
                    'CHARACTER CERTIFICATE', 'GOOD CHARACTER', 'CONDUCT',
                    'BEHAVIOUR', 'PRINCIPAL', 'HEADMASTER', 'SCHOOL',
                    'COLLEGE', 'HEREBY CERTIFY', 'MORAL CHARACTER'
                ],
                'patterns': [
                    r'CHARACTER\s*CERTIFICATE',
                    r'GOOD\s*CHARACTER',
                    r'MORAL\s*CHARACTER',
                    r'HEREBY\s*CERTIFY'
                ],
                'min_keywords': 2,
                'confidence_threshold': 0.5
            },
            'migration_certificate': {
                'keywords': [
                    'MIGRATION CERTIFICATE', 'TRANSFER CERTIFICATE', 'TC',
                    'MIGRATED', 'UNIVERSITY', 'COLLEGE', 'BOARD',
                    'REGISTRATION NUMBER', 'ROLL NUMBER', 'YEAR'
                ],
                'patterns': [
                    r'MIGRATION\s*CERTIFICATE',
                    r'TRANSFER\s*CERTIFICATE',
                    r'REGISTRATION\s*NO',
                    r'ROLL\s*NO'
                ],
                'min_keywords': 2,
                'confidence_threshold': 0.5
            },
            'bonafide_certificate': {
                'keywords': [
                    'BONAFIDE CERTIFICATE', 'BONA FIDE', 'STUDENT',
                    'STUDYING', 'INSTITUTION', 'SCHOOL', 'COLLEGE',
                    'PRINCIPAL', 'ACADEMIC YEAR', 'CLASS', 'HEREBY CERTIFY'
                ],
                'patterns': [
                    r'BONAFIDE|BONA\s*FIDE',
                    r'HEREBY\s*CERTIFY',
                    r'ACADEMIC\s*YEAR',
                    r'STUDYING'
                ],
                'min_keywords': 2,
                'confidence_threshold': 0.5
            },
            
            # === UTILITY BILLS ===
            'electricity_bill': {
                'keywords': [
                    'ELECTRICITY', 'ELECTRIC', 'POWER', 'CONSUMER NUMBER',
                    'UNITS CONSUMED', 'KWH', 'BILLING PERIOD', 'DUE DATE',
                    'CURRENT READING', 'PREVIOUS READING', 'TARIFF', 'DISCOM'
                ],
                'patterns': [
                    r'ELECTRICITY|ELECTRIC',
                    r'CONSUMER\s*NO',
                    r'UNITS\s*CONSUMED',
                    r'KWH|UNITS',
                    r'BILLING\s*PERIOD',
                    r'CURRENT\s*READING'
                ],
                'min_keywords': 2,
                'confidence_threshold': 0.5
            },
            'water_bill': {
                'keywords': [
                    'WATER BILL', 'WATER SUPPLY', 'CONSUMER NUMBER',
                    'BILLING PERIOD', 'DUE DATE', 'AMOUNT DUE',
                    'WATER CHARGES', 'MUNICIPAL', 'CORPORATION'
                ],
                'patterns': [
                    r'WATER\s*BILL',
                    r'WATER\s*SUPPLY',
                    r'CONSUMER\s*NO',
                    r'BILLING\s*PERIOD',
                    r'WATER\s*CHARGES'
                ],
                'min_keywords': 2,
                'confidence_threshold': 0.5
            },
            'gas_bill': {
                'keywords': [
                    'GAS', 'LPG', 'PNG', 'CONSUMER NUMBER', 'CUSTOMER ID',
                    'BILLING PERIOD', 'DUE DATE', 'AMOUNT DUE',
                    'GAS CONNECTION', 'CUBIC METERS', 'SCM'
                ],
                'patterns': [
                    r'GAS',
                    r'LPG|PNG',
                    r'CONSUMER\s*NO',
                    r'BILLING\s*PERIOD',
                    r'CUBIC\s*METERS|SCM'
                ],
                'min_keywords': 2,
                'confidence_threshold': 0.5
            },
            'telephone_bill': {
                'keywords': [
                    'TELEPHONE', 'MOBILE', 'POSTPAID', 'BILL', 'CUSTOMER ID',
                    'BILLING PERIOD', 'DUE DATE', 'AMOUNT DUE',
                    'PHONE NUMBER', 'PLAN', 'AIRTEL', 'VODAFONE', 'JIO', 'BSNL'
                ],
                'patterns': [
                    r'TELEPHONE|MOBILE',
                    r'POSTPAID',
                    r'BILLING\s*PERIOD',
                    r'PHONE\s*NO',
                    r'\d{10}'  # 10-digit phone number
                ],
                'min_keywords': 2,
                'confidence_threshold': 0.5
            },
            
            # === FINANCIAL DOCUMENTS ===
            'bank_statement': {
                'keywords': [
                    'BANK STATEMENT', 'PASSBOOK', 'ACCOUNT STATEMENT',
                    'ACCOUNT NUMBER', 'IFSC', 'BALANCE', 'DEPOSIT',
                    'WITHDRAWAL', 'TRANSACTION', 'CREDIT', 'DEBIT', 'BRANCH'
                ],
                'patterns': [
                    r'BANK\s*STATEMENT',
                    r'ACCOUNT\s*NO',
                    r'IFSC',
                    r'BALANCE',
                    r'TRANSACTION',
                    r'\d{2}/\d{2}/\d{4}\s+[₹$]\s*[\d,]+\.?\d*'
                ],
                'min_keywords': 2,
                'confidence_threshold': 0.5
            },
            'cheque': {
                'keywords': [
                    'CHEQUE', 'CHECK', 'PAY', 'RUPEES', 'ACCOUNT PAYEE',
                    'BANK', 'IFSC', 'MICR', 'CHEQUE NUMBER', 'DATE'
                ],
                'patterns': [
                    r'CHEQUE|CHECK',
                    r'PAY',
                    r'RUPEES',
                    r'ACCOUNT\s*PAYEE',
                    r'MICR',
                    r'\d{6}'  # 6-digit cheque number
                ],
                'min_keywords': 2,
                'confidence_threshold': 0.5
            },
            'gst_certificate': {
                'keywords': [
                    'GST', 'GOODS AND SERVICES TAX', 'GSTIN', 'REGISTRATION',
                    'TAX PAYER', 'TAXPAYER', 'CERTIFICATE OF REGISTRATION',
                    'CENTRAL TAX', 'STATE TAX', 'VALID FROM'
                ],
                'patterns': [
                    r'GST',
                    r'GSTIN',
                    r'\d{2}[A-Z]{5}\d{4}[A-Z]\d[Z][A-Z\d]',  # GSTIN format
                    r'GOODS\s*AND\s*SERVICES\s*TAX',
                    r'CERTIFICATE\s*OF\s*REGISTRATION'
                ],
                'min_keywords': 2,
                'confidence_threshold': 0.5
            },
            
            # === EDUCATIONAL DOCUMENTS ===
            'mark_sheet': {
                'keywords': [
                    'MARK SHEET', 'MARKS', 'GRADE', 'PERCENTAGE', 'EXAMINATION',
                    'UNIVERSITY', 'BOARD', 'STUDENT', 'ROLL NUMBER',
                    'REGISTRATION NUMBER', 'SEMESTER', 'YEAR', 'SUBJECT',
                    'TOTAL', 'OBTAINED', 'MAXIMUM', 'CGPA', 'GPA'
                ],
                'patterns': [
                    r'MARK\s*SHEET',
                    r'EXAMINATION',
                    r'ROLL\s*NO',
                    r'REGISTRATION\s*NO',
                    r'SEMESTER|YEAR',
                    r'TOTAL\s*MARKS',
                    r'CGPA|GPA'
                ],
                'min_keywords': 2,
                'confidence_threshold': 0.5
            },
            'school_leaving_certificate': {
                'keywords': [
                    'SCHOOL LEAVING CERTIFICATE', 'LEAVING CERTIFICATE', 'SLC',
                    'STUDENT', 'STUDYING', 'LEFT', 'SCHOOL', 'PRINCIPAL',
                    'CLASS', 'CONDUCT', 'CHARACTER', 'DATE OF LEAVING'
                ],
                'patterns': [
                    r'LEAVING\s*CERTIFICATE',
                    r'SLC',
                    r'DATE\s*OF\s*LEAVING',
                    r'PRINCIPAL',
                    r'CONDUCT'
                ],
                'min_keywords': 2,
                'confidence_threshold': 0.5
            },
            'transfer_certificate': {
                'keywords': [
                    'TRANSFER CERTIFICATE', 'TC', 'STUDENT', 'TRANSFERRED',
                    'SCHOOL', 'PRINCIPAL', 'CLASS', 'ADMISSION NUMBER',
                    'DATE OF ADMISSION', 'DATE OF LEAVING', 'CONDUCT', 'CHARACTER'
                ],
                'patterns': [
                    r'TRANSFER\s*CERTIFICATE',
                    r'TC',
                    r'DATE\s*OF\s*LEAVING',
                    r'DATE\s*OF\s*ADMISSION',
                    r'ADMISSION\s*NO'
                ],
                'min_keywords': 2,
                'confidence_threshold': 0.5
            },
            
            # === OTHER DOCUMENTS ===
            'visa': {
                'keywords': [
                    'VISA', 'IMMIGRATION', 'EMBASSY', 'PASSPORT NUMBER',
                    'NATIONALITY', 'VALID FROM', 'VALID UNTIL', 'ENTRIES',
                    'SINGLE ENTRY', 'MULTIPLE ENTRY', 'DURATION OF STAY'
                ],
                'patterns': [
                    r'VISA',
                    r'IMMIGRATION',
                    r'EMBASSY',
                    r'PASSPORT\s*NO',
                    r'VALID\s*FROM',
                    r'VALID\s*UNTIL'
                ],
                'min_keywords': 2,
                'confidence_threshold': 0.5
            },
            'rent_agreement': {
                'keywords': [
                    'RENT AGREEMENT', 'RENTAL AGREEMENT', 'LEASE AGREEMENT',
                    'LESSOR', 'LESSEE', 'TENANT', 'LANDLORD', 'RENT',
                    'MONTHLY RENT', 'SECURITY DEPOSIT', 'PREMISES', 'WITNESSETH'
                ],
                'patterns': [
                    r'RENT\s*AGREEMENT',
                    r'RENTAL\s*AGREEMENT',
                    r'LEASE',
                    r'LESSOR|LESSEE',
                    r'LANDLORD|TENANT',
                    r'MONTHLY\s*RENT'
                ],
                'min_keywords': 2,
                'confidence_threshold': 0.5
            },
            'npr_details': {
                'keywords': [
                    'NPR', 'NATIONAL POPULATION REGISTER', 'POPULATION',
                    'CENSUS', 'HOUSEHOLD', 'SCHEDULE', 'ENUMERATION',
                    'RESIDENT', 'PLACE OF BIRTH', 'NATIONALITY'
                ],
                'patterns': [
                    r'NPR',
                    r'NATIONAL\s*POPULATION\s*REGISTER',
                    r'CENSUS',
                    r'ENUMERATION'
                ],
                'min_keywords': 2,
                'confidence_threshold': 0.5
            },
            
            # === EXISTING DOCUMENT TYPES (keep for backward compatibility) ===
            'invoice': {
                'keywords': [
                    'INVOICE', 'BILL TO', 'SHIP TO', 'SUBTOTAL', 'TAX', 'TOTAL',
                    'AMOUNT DUE', 'INVOICE NUMBER', 'DATE DUE', 'PAYMENT TERMS',
                    'QTY', 'QUANTITY', 'PRICE', 'DESCRIPTION', 'ITEM'
                ],
                'patterns': [
                    r'INVOICE\s*#?\s*\d+',
                    r'TOTAL\s*\$?\s*[\d,]+\.?\d*',
                    r'DUE\s*DATE',
                    r'BILL\s*TO:'
                ],
                'min_keywords': 1,
                'confidence_threshold': 0.3
            },
            'receipt': {
                'keywords': [
                    'RECEIPT', 'THANK YOU', 'CHANGE', 'CASH', 'CREDIT',
                    'CARD', 'SUBTOTAL', 'TAX', 'TOTAL', 'STORE', 'LOCATION',
                    'TRANSACTION', 'TIME', 'DATE'
                ],
                'patterns': [
                    r'\$\s*\d+\.\d{2}',
                    r'TOTAL\s*\$?\s*\d+\.\d{2}',
                    r'CHANGE\s*\$?\s*\d+\.\d{2}',
                    r'\d{2}/\d{2}/\d{4}'
                ],
                'min_keywords': 1,
                'confidence_threshold': 0.3
            },
            'form': {
                'keywords': [
                    'APPLICATION', 'FORM', 'NAME:', 'ADDRESS:', 'PHONE:',
                    'EMAIL:', 'SIGNATURE', 'DATE:', 'PLEASE PRINT',
                    'FIRST NAME', 'LAST NAME', 'ZIP CODE', 'STATE'
                ],
                'patterns': [
                    r'NAME:\s*[_\s]*',
                    r'ADDRESS:\s*[_\s]*',
                    r'PHONE:\s*[_\s]*',
                    r'EMAIL:\s*[_\s]*'
                ],
                'min_keywords': 2,
                'confidence_threshold': 0.5
            },
            'business_card': {
                'keywords': [
                    'CEO', 'MANAGER', 'DIRECTOR', 'PRESIDENT', 'VP',
                    'EMAIL', 'PHONE', 'MOBILE', 'OFFICE', 'FAX',
                    'WWW', 'HTTP', 'COM', 'ORG', 'NET'
                ],
                'patterns': [
                    r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}',
                    r'\(\d{3}\)\s*\d{3}-\d{4}',
                    r'\d{3}-\d{3}-\d{4}',
                    r'www\.[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
                ],
                'min_keywords': 1,
                'confidence_threshold': 0.5
            },
            'medical': {
                'keywords': [
                    'PATIENT', 'DOCTOR', 'PHYSICIAN', 'CLINIC', 'HOSPITAL',
                    'PRESCRIPTION', 'DIAGNOSIS', 'TREATMENT', 'MEDICAL',
                    'CHART', 'DOB', 'ALLERGIES', 'MEDICATION'
                ],
                'patterns': [
                    r'DOB:\s*\d{2}/\d{2}/\d{4}',
                    r'PATIENT\s*ID:\s*\d+',
                    r'DR\.\s*[A-Z][a-z]+',
                    r'RX\s*#?\d+'
                ],
                'min_keywords': 1,
                'confidence_threshold': 0.5
            },
            'legal': {
                'keywords': [
                    'COURT', 'LEGAL', 'CONTRACT', 'AGREEMENT', 'WHEREAS',
                    'THEREFORE', 'PARTY', 'WITNESS', 'NOTARY', 'SWORN',
                    'AFFIDAVIT', 'PLAINTIFF', 'DEFENDANT', 'ATTORNEY'
                ],
                'patterns': [
                    r'WHEREAS\s+',
                    r'THEREFORE\s+',
                    r'IN\s+WITNESS\s+WHEREOF',
                    r'STATE\s+OF\s+[A-Z]+'
                ],
                'min_keywords': 1,
                'confidence_threshold': 0.5
            },
            'academic': {
                'keywords': [
                    'UNIVERSITY', 'COLLEGE', 'SCHOOL', 'STUDENT', 'GRADE',
                    'TRANSCRIPT', 'COURSE', 'CREDIT', 'GPA', 'SEMESTER',
                    'DEGREE', 'BACHELOR', 'MASTER', 'GRADUATION'
                ],
                'patterns': [
                    r'GPA:\s*\d+\.\d+',
                    r'STUDENT\s*ID:\s*\d+',
                    r'CREDIT\s*HOURS?:\s*\d+',
                    r'SEMESTER:\s*[A-Z]+'
                ],
                'min_keywords': 1,
                'confidence_threshold': 0.5
            },
            'financial': {
                'keywords': [
                    'BANK', 'STATEMENT', 'ACCOUNT', 'BALANCE', 'DEPOSIT',
                    'WITHDRAWAL', 'TRANSACTION', 'ROUTING', 'CHECK',
                    'PAYMENT', 'INTEREST', 'FEE', 'CREDIT', 'DEBIT'
                ],
                'patterns': [
                    r'ACCOUNT\s*#?\s*\d+',
                    r'BALANCE:\s*\$?\s*[\d,]+\.\d{2}',
                    r'ROUTING\s*#?\s*\d{9}',
                    r'\d{2}/\d{2}/\d{4}\s+[\d,]+\.\d{2}'
                ],
                'min_keywords': 1,
                'confidence_threshold': 0.5
            },
            'government': {
                'keywords': [
                    'GOVERNMENT', 'DEPARTMENT', 'BUREAU', 'AGENCY', 'FEDERAL',
                    'STATE', 'COUNTY', 'MUNICIPAL', 'LICENSE', 'PERMIT',
                    'REGISTRATION', 'OFFICIAL', 'SEAL', 'AUTHORIZED'
                ],
                'patterns': [
                    r'LICENSE\s*#?\s*[A-Z0-9]+',
                    r'PERMIT\s*#?\s*\d+',
                    r'REGISTRATION\s*#?\s*[A-Z0-9]+',
                    r'EXPIRES?:\s*\d{2}/\d{2}/\d{4}'
                ],
                'min_keywords': 1,
                'confidence_threshold': 0.5
            },
            'mrz': {
                'keywords': ['P<', 'I<', 'V<', '<<<', '<<'],
                'patterns': [
                    r'P<[A-Z]{3}[A-Z<]+',
                    r'I<[A-Z]{3}[A-Z<]+',
                    r'V<[A-Z]{3}[A-Z<]+',
                    r'[A-Z0-9<]{30,}'
                ],
                'min_keywords': 1,
                'confidence_threshold': 0.8
            },
            'certificate': {
                'keywords': [
                    'CERTIFICATE', 'APPRECIATION', 'ACHIEVEMENT', 'AWARD', 'RECOGNITION',
                    'PRESENTED TO', 'HEREBY CERTIFY', 'COMPLETION', 'DIPLOMA', 'HONOR'
                ],
                'patterns': [
                    r'CERTIFICATE\s+OF\s+[A-Z]+',
                    r'PRESENTED\s+TO\s*:?',
                    r'THIS\s+CERTIFIES\s+THAT',
                    r'IN\s+RECOGNITION\s+OF'
                ],
                'min_keywords': 1,
                'confidence_threshold': 0.5
            }
        }
    
    def detect_document_type(
        self, 
        text: str, 
        metadata: Optional[Dict[str, Any]] = None,
        min_confidence: float = 0.7
    ) -> Tuple[str, float]:
        """
        Detect document type with confidence score using both text and metadata.
        
        Args:
            text: OCR extracted text
            metadata: Optional metadata dictionary from metadata_extractor
            min_confidence: Minimum confidence threshold (default 0.7 for high confidence)
            
        Returns:
            Tuple of (document_type, confidence_score)
        """
        # Convert text to uppercase for case-insensitive matching
        text_upper = text.upper()
        scores = {}
        
        # Calculate text-based scores for all document types
        for doc_type, config in self.document_patterns.items():
            text_score = self._calculate_text_score(text_upper, config)
            
            # If metadata is provided, combine with metadata score
            if metadata:
                metadata_score = self._calculate_metadata_score(metadata, doc_type)
                # Combined score: 60% text, 40% metadata
                combined_score = (0.6 * text_score) + (0.4 * metadata_score)
            else:
                combined_score = text_score
            
            # Only include if it meets the document type's threshold
            if combined_score >= config['confidence_threshold']:
                scores[doc_type] = combined_score
        
        # Filter for high confidence results
        high_confidence_scores = {k: v for k, v in scores.items() if v >= min_confidence}
        
        if high_confidence_scores:
            best_type = max(high_confidence_scores.keys(), key=lambda k: high_confidence_scores[k])
            return best_type, high_confidence_scores[best_type]
        elif scores:
            # If no high confidence match but there are some matches, return the best one
            best_type = max(scores.keys(), key=lambda k: scores[k])
            return best_type, scores[best_type]
        else:
            return 'general', 0.5
    
    def get_high_confidence_document_types(
        self, 
        text: str, 
        metadata: Optional[Dict[str, Any]] = None,
        min_confidence: float = 0.7
    ) -> List[Tuple[str, float]]:
        """
        Get all document types that meet the high confidence threshold.
        
        Args:
            text: OCR extracted text
            metadata: Optional metadata dictionary
            min_confidence: Minimum confidence threshold (default 0.7)
            
        Returns:
            List of (document_type, confidence_score) tuples sorted by confidence
        """
        text_upper = text.upper()
        high_confidence_types = []
        
        for doc_type, config in self.document_patterns.items():
            text_score = self._calculate_text_score(text_upper, config)
            
            if metadata:
                metadata_score = self._calculate_metadata_score(metadata, doc_type)
                combined_score = (0.6 * text_score) + (0.4 * metadata_score)
            else:
                combined_score = text_score
            
            if combined_score >= max(config['confidence_threshold'], min_confidence):
                high_confidence_types.append((doc_type, combined_score))
        
        # Sort by confidence score (highest first)
        high_confidence_types.sort(key=lambda x: x[1], reverse=True)
        return high_confidence_types
    
    def _calculate_text_score(self, text: str, config: Dict) -> float:
        """Calculate confidence score based on text patterns (case-insensitive)."""
        keyword_matches = sum(1 for keyword in config['keywords'] if keyword in text)
        pattern_matches = sum(1 for pattern in config['patterns'] if re.search(pattern, text, re.IGNORECASE))
        
        # Keyword score (0-0.6)
        keyword_score = min(keyword_matches / len(config['keywords']) * 0.6, 0.6)
        
        # Pattern score (0-0.2)
        pattern_score = min(pattern_matches / len(config['patterns']) * 0.2, 0.2) if config['patterns'] else 0
        
        total_score = keyword_score + pattern_score
        
        # Boost for meeting minimum requirements
        if keyword_matches >= config['min_keywords']:
            total_score += 0.2
            
        # Cap at 1.0
        return min(total_score, 1.0)
    
    def _calculate_metadata_score(self, metadata: Dict[str, Any], doc_type: str) -> float:
        """Calculate confidence score based on metadata hints."""
        score = 0.0
        
        # Filename hints (50% weight)
        if doc_type in metadata.get('filename_hints', []):
            score += 0.5
        
        # Document hints (30% weight)
        document_hint_map = {
            'aadhaar_card': ['id_card_aspect_ratio', 'high_quality_scan'],
            'pan_card': ['id_card_aspect_ratio', 'high_quality_scan'],
            'passport': ['a4_portrait_ratio', 'high_quality_scan'],
            'voter_id': ['id_card_aspect_ratio'],
            'driving_licence': ['id_card_aspect_ratio'],
            'ration_card': ['id_card_aspect_ratio'],
            'birth_certificate': ['a4_portrait_ratio', 'scanned_document'],
            'marriage_certificate': ['a4_portrait_ratio', 'scanned_document'],
            'caste_certificate': ['a4_portrait_ratio', 'scanned_document'],
            'character_certificate': ['a4_portrait_ratio', 'scanned_document'],
            'migration_certificate': ['a4_portrait_ratio', 'scanned_document'],
            'bonafide_certificate': ['a4_portrait_ratio', 'scanned_document'],
            'electricity_bill': ['a4_portrait_ratio', 'scanned_document'],
            'water_bill': ['a4_portrait_ratio', 'scanned_document'],
            'gas_bill': ['a4_portrait_ratio', 'scanned_document'],
            'telephone_bill': ['a4_portrait_ratio', 'scanned_document'],
            'bank_statement': ['a4_portrait_ratio', 'scanned_document'],
            'cheque': ['cheque_aspect_ratio'],
            'gst_certificate': ['a4_portrait_ratio', 'scanned_document'],
            'mark_sheet': ['a4_portrait_ratio', 'scanned_document'],
            'school_leaving_certificate': ['a4_portrait_ratio', 'scanned_document'],
            'transfer_certificate': ['a4_portrait_ratio', 'scanned_document'],
            'visa': ['a4_portrait_ratio', 'scanned_document'],
            'rent_agreement': ['a4_portrait_ratio', 'scanned_document'],
            'certificate': ['a4_portrait_ratio', 'scanned_document']
        }
        
        expected_hints = document_hint_map.get(doc_type, [])
        if expected_hints:
            hint_matches = sum(1 for hint in expected_hints if hint in metadata.get('document_hints', []))
            score += 0.3 * (hint_matches / len(expected_hints))
        
        # Quality indicators (20% weight)
        quality_score = 0.0
        if 'high_resolution' in metadata.get('document_hints', []):
            quality_score += 0.1
        if 'high_quality_scan' in metadata.get('document_hints', []) or 'medium_quality_scan' in metadata.get('document_hints', []):
            quality_score += 0.1
        
        score += quality_score
        
        # Cap at 1.0
        return min(score, 1.0)


# Keep the DocumentProcessor class for backward compatibility
class DocumentProcessor:
    """Process different document types with specialized extraction."""
    
    def __init__(self):
        self.detector = DocumentTypeDetector()
    
    def process_document(
        self, 
        image_path: str, 
        raw_text: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Process document based on its detected type.
        
        Args:
            image_path: Path to the image file
            raw_text: Raw OCR text
            metadata: Optional metadata dictionary
            
        Returns:
            Dictionary with processed document data
        """
        doc_type, confidence = self.detector.detect_document_type(raw_text, metadata)
        
        result = {
            'document_type': doc_type,
            'confidence': confidence,
            'raw_text': raw_text,
            'processed_at': datetime.now().isoformat()
        }
        
        # Add basic structured data (you can expand this for each new document type)
        result['structured_data'] = self._extract_basic_data(raw_text, doc_type)
        
        return result
    
    def _extract_basic_data(self, text: str, doc_type: str) -> Dict[str, Any]:
        """Extract basic structured data from text."""
        data = {}
        
        # Extract dates
        dates = re.findall(r'\d{1,2}/\d{1,2}/\d{4}', text)
        if dates:
            data['dates_found'] = dates
        
        # Extract emails
        emails = re.findall(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', text)
        if emails:
            data['emails_found'] = emails
        
        # Extract phone numbers
        phones = re.findall(r'(\(?\d{3}\)?[\s\-]?\d{3}[\s\-]?\d{4})', text)
        if phones:
            data['phones_found'] = phones
        
        # Extract numbers that might be IDs
        if doc_type in ['aadhaar_card']:
            aadhaar_match = re.search(r'\d{4}\s*\d{4}\s*\d{4}', text)
            if aadhaar_match:
                data['aadhaar_number'] = aadhaar_match.group(0)
        
        if doc_type in ['pan_card']:
            pan_match = re.search(r'[A-Z]{5}\d{4}[A-Z]', text)
            if pan_match:
                data['pan_number'] = pan_match.group(0)
        
        return data