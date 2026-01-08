"""
MongoDB Database Manager for Document Processing History
Handles storage and retrieval of document processing records
"""

from pymongo import MongoClient
from datetime import datetime, timedelta
import json
from typing import Dict, List, Any, Optional
import os
from bson import ObjectId

class DocumentHistoryDB:
    """MongoDB database manager for document processing history."""
    
    def __init__(self, 
                 connection_string: str = "mongodb://localhost:27017/",
                 database_name: str = "ocr_document_history"):
        """
        Initialize MongoDB connection.
        
        Args:
            connection_string: MongoDB connection string
            database_name: Name of the database to use
        """
        try:
            self.client = MongoClient(connection_string)
            self.db = self.client[database_name]
            self.collection = self.db.document_records
            
            # Create indexes for better query performance
            self.collection.create_index([("processed_at", -1)])  # Most recent first
            self.collection.create_index([("filename", 1)])
            self.collection.create_index([("document_type", 1)])
            
            print(f"✅ Connected to MongoDB database: {database_name}")
            
        except Exception as e:
            print(f"❌ Failed to connect to MongoDB: {e}")
            # Fallback to local JSON storage
            self.client = None
            self.db = None
            self.collection = None
            self.json_storage_path = "document_history.json"
            print("⚠️ Using local JSON file storage as fallback")
    
    def save_document_record(self, 
                           filename: str,
                           document_type: str,
                           extracted_text: str,
                           confidence: float = 0.0,
                           processing_metadata: Dict = None,
                           structured_data: Dict = None,
                           rag_suggestions: List[Dict] = None,
                           document_classifications: List[Dict] = None) -> str:
        """
        Save a document processing record.
        
        Args:
            filename: Name of the processed file
            document_type: Detected document type
            extracted_text: OCR extracted text
            confidence: Processing confidence score
            processing_metadata: Additional processing metadata
            structured_data: Structured data extracted from document
            rag_suggestions: RAG field suggestions
            document_classifications: Document classification results
            
        Returns:
            Record ID (ObjectId or UUID)
        """
        
        record = {
            "filename": filename,
            "document_type": document_type,
            "extracted_text": extracted_text,
            "confidence": confidence,
            "processing_metadata": processing_metadata or {},
            "structured_data": structured_data or {},
            "rag_suggestions": rag_suggestions or [],
            "document_classifications": document_classifications or [],
            "processed_at": datetime.utcnow(),
            "text_length": len(extracted_text),
            "suggestions_count": len(rag_suggestions) if rag_suggestions else 0,
            "status": "completed"
        }
        
        try:
            if self.collection is not None:
                # MongoDB storage
                result = self.collection.insert_one(record)
                record_id = str(result.inserted_id)
                print(f"📝 Document record saved to MongoDB: {record_id}")
                
            else:
                # JSON fallback storage
                record_id = str(ObjectId())
                record["_id"] = record_id
                record["processed_at"] = record["processed_at"].isoformat()
                
                # Load existing records
                existing_records = self._load_json_records()
                existing_records.append(record)
                
                # Save updated records
                self._save_json_records(existing_records)
                print(f"📝 Document record saved to JSON: {record_id}")
            
            return record_id
            
        except Exception as e:
            print(f"❌ Failed to save document record: {e}")
            return None
    
    def get_document_history(self, 
                           limit: int = 50,
                           document_type: str = None,
                           days_back: int = 30) -> List[Dict]:
        """
        Retrieve document processing history.
        
        Args:
            limit: Maximum number of records to return
            document_type: Filter by document type (optional)
            days_back: Number of days to look back
            
        Returns:
            List of document records
        """
        try:
            # Calculate date threshold
            date_threshold = datetime.utcnow() - timedelta(days=days_back)
            
            if self.collection is not None:
                # MongoDB query
                query = {"processed_at": {"$gte": date_threshold}}
                if document_type:
                    query["document_type"] = document_type
                
                cursor = self.collection.find(query).sort("processed_at", -1).limit(limit)
                records = list(cursor)
                
                # Convert ObjectId to string for JSON serialization
                for record in records:
                    record["_id"] = str(record["_id"])
                    
            else:
                # JSON fallback
                all_records = self._load_json_records()
                
                # Filter by date and document type
                filtered_records = []
                for record in all_records:
                    record_date = datetime.fromisoformat(record["processed_at"])
                    if record_date >= date_threshold:
                        if not document_type or record["document_type"] == document_type:
                            filtered_records.append(record)
                
                # Sort by date (most recent first) and limit
                filtered_records.sort(key=lambda x: x["processed_at"], reverse=True)
                records = filtered_records[:limit]
            
            print(f"📚 Retrieved {len(records)} document records")
            return records
            
        except Exception as e:
            print(f"❌ Failed to retrieve document history: {e}")
            return []
    
    def get_document_by_id(self, record_id: str) -> Dict:
        """Get a specific document record by ID."""
        try:
            if self.collection is not None:
                record = self.collection.find_one({"_id": ObjectId(record_id)})
                if record:
                    record["_id"] = str(record["_id"])
                    return record
            else:
                all_records = self._load_json_records()
                for record in all_records:
                    if record.get("_id") == record_id:
                        return record
            
            return None
            
        except Exception as e:
            print(f"❌ Failed to retrieve document by ID: {e}")
            return None
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get processing statistics."""
        try:
            if self.collection is not None:
                # MongoDB aggregation
                pipeline = [
                    {"$group": {
                        "_id": "$document_type",
                        "count": {"$sum": 1},
                        "avg_confidence": {"$avg": "$confidence"},
                        "total_suggestions": {"$sum": "$suggestions_count"}
                    }},
                    {"$sort": {"count": -1}}
                ]
                
                type_stats = list(self.collection.aggregate(pipeline))
                total_docs = self.collection.count_documents({})
                
                recent_docs = self.collection.count_documents({
                    "processed_at": {"$gte": datetime.utcnow() - timedelta(days=7)}
                })
                
            else:
                # JSON fallback
                all_records = self._load_json_records()
                total_docs = len(all_records)
                
                # Group by document type
                type_counts = {}
                type_confidences = {}
                type_suggestions = {}
                
                recent_threshold = datetime.utcnow() - timedelta(days=7)
                recent_docs = 0
                
                for record in all_records:
                    doc_type = record["document_type"]
                    
                    if doc_type not in type_counts:
                        type_counts[doc_type] = 0
                        type_confidences[doc_type] = []
                        type_suggestions[doc_type] = 0
                    
                    type_counts[doc_type] += 1
                    type_confidences[doc_type].append(record.get("confidence", 0))
                    type_suggestions[doc_type] += record.get("suggestions_count", 0)
                    
                    # Check if recent
                    record_date = datetime.fromisoformat(record["processed_at"])
                    if record_date >= recent_threshold:
                        recent_docs += 1
                
                # Build type stats
                type_stats = []
                for doc_type in type_counts:
                    avg_conf = sum(type_confidences[doc_type]) / len(type_confidences[doc_type])
                    type_stats.append({
                        "_id": doc_type,
                        "count": type_counts[doc_type],
                        "avg_confidence": avg_conf,
                        "total_suggestions": type_suggestions[doc_type]
                    })
                
                type_stats.sort(key=lambda x: x["count"], reverse=True)
            
            return {
                "total_documents": total_docs,
                "recent_documents": recent_docs,
                "document_types": type_stats,
                "database_type": "MongoDB" if self.collection else "JSON File"
            }
            
        except Exception as e:
            print(f"❌ Failed to get statistics: {e}")
            return {"total_documents": 0, "recent_documents": 0, "document_types": []}
    
    def delete_old_records(self, days_old: int = 90) -> int:
        """Delete records older than specified days."""
        try:
            date_threshold = datetime.utcnow() - timedelta(days=days_old)
            
            if self.collection is not None:
                result = self.collection.delete_many({"processed_at": {"$lt": date_threshold}})
                deleted_count = result.deleted_count
            else:
                all_records = self._load_json_records()
                remaining_records = []
                deleted_count = 0
                
                for record in all_records:
                    record_date = datetime.fromisoformat(record["processed_at"])
                    if record_date >= date_threshold:
                        remaining_records.append(record)
                    else:
                        deleted_count += 1
                
                self._save_json_records(remaining_records)
            
            print(f"🗑️ Deleted {deleted_count} old records")
            return deleted_count
            
        except Exception as e:
            print(f"❌ Failed to delete old records: {e}")
            return 0
    
    def _load_json_records(self) -> List[Dict]:
        """Load records from JSON file."""
        try:
            if os.path.exists(self.json_storage_path):
                with open(self.json_storage_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            return []
        except Exception as e:
            print(f"❌ Failed to load JSON records: {e}")
            return []
    
    def _save_json_records(self, records: List[Dict]):
        """Save records to JSON file."""
        try:
            with open(self.json_storage_path, 'w', encoding='utf-8') as f:
                json.dump(records, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"❌ Failed to save JSON records: {e}")
    
    def close_connection(self):
        """Close MongoDB connection."""
        if self.client:
            self.client.close()
            print("🔌 MongoDB connection closed")

# Global database instance
db_manager = DocumentHistoryDB()

def cleanup_db():
    """Cleanup function for application shutdown."""
    db_manager.close_connection()