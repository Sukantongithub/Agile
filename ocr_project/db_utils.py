import datetime
from pymongo import MongoClient
import os

# MongoDB Configuration
MONGO_URI = os.getenv('MONGO_URI', 'mongodb://localhost:27017/')
DB_NAME = 'redaction_db'
COLLECTION_NAME = 'processing_history'

class DBManager:
    def __init__(self):
        self.client = None
        self.db = None
        self.collection = None
        self.connect()

    def connect(self):
        try:
            self.client = MongoClient(MONGO_URI)
            self.db = self.client[DB_NAME]
            self.collection = self.db[COLLECTION_NAME]
            print("✅ Connected to MongoDB successfully!")
        except Exception as e:
            print(f"❌ Failed to connect to MongoDB: {e}")
            self.client = None
            self.collection = None

    def save_record(self, data, user_id=None):
        """Save processing record to MongoDB history with user_id."""
        if self.collection is not None:
            try:
                # Add timestamp if not present
                if 'timestamp' not in data:
                    data['timestamp'] = datetime.datetime.utcnow()
                
                # Add user_id if provided
                if user_id:
                    data['user_id'] = user_id
                
                self.collection.insert_one(data)
                print(f"✅ Saved to history: {data.get('filename')} (user: {user_id})")
                return True
            except Exception as e:
                print(f"❌ Failed to save history: {e}")
                return False
        return False

    def get_history(self, user_id=None, limit=50):
        """Get processing history from MongoDB, optionally filtered by user_id."""
        if self.collection is None:
            return None
        
        try:
            # Build query filter
            query = {}
            if user_id:
                query['user_id'] = user_id
            
            # Get last N records, sorted by timestamp desc
            records = list(self.collection.find(query, {'_id': 0}).sort('timestamp', -1).limit(limit))
            return records
        except Exception as e:
            print(f"❌ Failed to fetch history: {e}")
            return []

# Global instance
db_manager = DBManager()
