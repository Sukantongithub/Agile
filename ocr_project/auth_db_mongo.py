"""User authentication database module using MongoDB."""

from pymongo import MongoClient
import os
from datetime import datetime
from werkzeug.security import generate_password_hash, check_password_hash

# MongoDB Configuration
MONGO_URI = os.getenv('MONGO_URI', 'mongodb://localhost:27017/')
DB_NAME = 'redaction_db'
USERS_COLLECTION = 'users'

class MongoAuthManager:
    def __init__(self):
        self.client = None
        self.db = None
        self.users_collection = None
        self.connect()

    def connect(self):
        """Initialize the MongoDB connection and create indexes."""
        try:
            self.client = MongoClient(MONGO_URI)
            self.db = self.client[DB_NAME]
            self.users_collection = self.db[USERS_COLLECTION]
            
            # Create unique indexes for username and email
            self.users_collection.create_index('username', unique=True)
            self.users_collection.create_index('email', unique=True)
            
            print(f"✅ Connected to MongoDB authentication (database: {DB_NAME})")
        except Exception as e:
            print(f"❌ Failed to connect to MongoDB for authentication: {e}")
            self.client = None
            self.users_collection = None

    def register_user(self, username, email, password):
        """
        Register a new user in MongoDB.
        
        Args:
            username: Unique username
            email: Unique email address
            password: Plain text password (will be hashed)
        
        Returns:
            dict: {'success': bool, 'message': str, 'user_id': str (if success)}
        """
        try:
            # Validate inputs
            if not username or not email or not password:
                return {'success': False, 'message': 'All fields are required'}
            
            if len(username) < 3:
                return {'success': False, 'message': 'Username must be at least 3 characters'}
            
            if len(password) < 6:
                return {'success': False, 'message': 'Password must be at least 6 characters'}
            
            if '@' not in email or '.' not in email:
                return {'success': False, 'message': 'Invalid email format'}
            
            if self.users_collection is None:
                return {'success': False, 'message': 'Database not connected'}
            
            # Check if username already exists
            if self.users_collection.find_one({'username': username}):
                return {'success': False, 'message': 'Username already exists'}
            
            # Check if email already exists
            if self.users_collection.find_one({'email': email}):
                return {'success': False, 'message': 'Email already registered'}
            
            # Hash password
            password_hash = generate_password_hash(password, method='pbkdf2:sha256')
            
            # Insert user document
            user_doc = {
                'username': username,
                'email': email,
                'password_hash': password_hash,
                'created_at': datetime.utcnow(),
                'last_login': None
            }
            
            result = self.users_collection.insert_one(user_doc)
            
            return {
                'success': True,
                'message': 'Registration successful',
                'user_id': str(result.inserted_id)
            }
                
        except Exception as e:
            print(f"Registration error: {e}")
            return {'success': False, 'message': f'Error: {str(e)}'}

    def login_user(self, username_or_email, password):
        """
        Validate user login credentials.
        
        Args:
            username_or_email: Username or email address
            password: Plain text password
        
        Returns:
            dict: {'success': bool, 'message': str, 'user': dict (if success)}
        """
        try:
            if not username_or_email or not password:
                return {'success': False, 'message': 'All fields are required'}
            
            if self.users_collection is None:
                return {'success': False, 'message': 'Database not connected'}
            
            # Check if input is email or username
            if '@' in username_or_email:
                user = self.users_collection.find_one({'email': username_or_email})
            else:
                user = self.users_collection.find_one({'username': username_or_email})
            
            if not user:
                return {'success': False, 'message': 'Invalid credentials'}
            
            # Verify password
            if not check_password_hash(user['password_hash'], password):
                return {'success': False, 'message': 'Invalid credentials'}
            
            # Update last login
            self.users_collection.update_one(
                {'_id': user['_id']},
                {'$set': {'last_login': datetime.utcnow()}}
            )
            
            return {
                'success': True,
                'message': 'Login successful',
                'user': {
                    'id': str(user['_id']),
                    'username': user['username'],
                    'email': user['email'],
                    'created_at': user.get('created_at')
                }
            }
            
        except Exception as e:
            print(f"Login error: {e}")
            return {'success': False, 'message': f'Error: {str(e)}'}

    def get_user_by_id(self, user_id):
        """Get user information by ID."""
        try:
            if self.users_collection is None:
                return None
            
            from bson.objectid import ObjectId
            user = self.users_collection.find_one({'_id': ObjectId(user_id)})
            
            if user:
                return {
                    'id': str(user['_id']),
                    'username': user['username'],
                    'email': user['email'],
                    'created_at': user.get('created_at')
                }
            return None
            
        except Exception as e:
            print(f"Error getting user: {e}")
            return None

    def update_user_email(self, user_id, new_email):
        """Update user's email address."""
        try:
            if self.users_collection is None:
                return {'success': False, 'message': 'Database not connected'}
            
            # Validate email
            if not new_email or '@' not in new_email or '.' not in new_email:
                return {'success': False, 'message': 'Invalid email format'}
            
            from bson.objectid import ObjectId
            
            # Check if email is already in use by another user
            existing_user = self.users_collection.find_one({'email': new_email})
            if existing_user and str(existing_user['_id']) != user_id:
                return {'success': False, 'message': 'Email already in use'}
            
            # Update email
            result = self.users_collection.update_one(
                {'_id': ObjectId(user_id)},
                {'$set': {'email': new_email}}
            )
            
            if result.modified_count > 0:
                return {'success': True, 'message': 'Email updated successfully'}
            else:
                return {'success': False, 'message': 'No changes made'}
            
        except Exception as e:
            print(f"Error updating email: {e}")
            return {'success': False, 'message': f'Error: {str(e)}'}

    def change_password(self, user_id, current_password, new_password):
        """Change user's password."""
        try:
            if self.users_collection is None:
                return {'success': False, 'message': 'Database not connected'}
            
            # Validate new password
            if not new_password or len(new_password) < 6:
                return {'success': False, 'message': 'New password must be at least 6 characters'}
            
            from bson.objectid import ObjectId
            
            # Get user
            user = self.users_collection.find_one({'_id': ObjectId(user_id)})
            if not user:
                return {'success': False, 'message': 'User not found'}
            
            # Verify current password
            if not check_password_hash(user['password_hash'], current_password):
                return {'success': False, 'message': 'Current password is incorrect'}
            
            # Hash new password
            new_password_hash = generate_password_hash(new_password, method='pbkdf2:sha256')
            
            # Update password
            self.users_collection.update_one(
                {'_id': ObjectId(user_id)},
                {'$set': {'password_hash': new_password_hash}}
            )
            
            return {'success': True, 'message': 'Password changed successfully'}
            
        except Exception as e:
            print(f"Error changing password: {e}")
            return {'success': False, 'message': f'Error: {str(e)}'}

# Global instance
mongo_auth_manager = MongoAuthManager()

def register_user(username, email, password):
    """Wrapper function for registration."""
    return mongo_auth_manager.register_user(username, email, password)

def login_user(username_or_email, password):
    """Wrapper function for login."""
    return mongo_auth_manager.login_user(username_or_email, password)

def get_user_by_id(user_id):
    """Wrapper function to get user by ID."""
    return mongo_auth_manager.get_user_by_id(user_id)

def update_user_email(user_id, new_email):
    """Wrapper function to update user email."""
    return mongo_auth_manager.update_user_email(user_id, new_email)

def change_password(user_id, current_password, new_password):
    """Wrapper function to change password."""
    return mongo_auth_manager.change_password(user_id, current_password, new_password)

def get_mongo_client():
    """Wrapper to get the raw MongoDB client."""
    return mongo_auth_manager.client
