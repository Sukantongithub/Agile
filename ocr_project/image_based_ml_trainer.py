"""
Image-Based ML Training System for Document Classification
Uses actual document images from ml_training/train_img folder to train ML models
"""

import os
import sys
import cv2
import numpy as np
import pytesseract
from PIL import Image, ImageEnhance
import json
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import re
import warnings
warnings.filterwarnings('ignore')

class ImageBasedDocumentTrainer:
    """Advanced ML trainer that uses actual document images."""
    
    def __init__(self, training_images_dir: str = "ml_training/train_img"):
        """Initialize the image-based trainer."""
        self.training_images_dir = Path(training_images_dir)
        self.models_dir = Path("models")
        self.models_dir.mkdir(exist_ok=True)
        
        # Document type mapping based on image filename patterns
        self.document_type_mapping = {
            'aadhar': 'AADHAR_CARD',
            'pan': 'PAN_CARD', 
            'voter': 'VOTER_ID',
            'driving': 'DRIVING_LICENSE',
            'passport': 'PASSPORT',
            'marksheet': 'MARKSHEET',
            'ration': 'RATION_CARD',
            'bank': 'BANK_PASSBOOK',
            'birth': 'BIRTH_CERTIFICATE',
            'community': 'COMMUNITY_CERTIFICATE',
            'smart': 'SMART_CARD',
            'college': 'COLLEGE_ID',
            'exam': 'EXAM_RECEIPT',
            'medical': 'MEDICAL_REPORT',
            'test': 'TEST_DOCUMENT',
            'unknown': 'UNKNOWN_DOCUMENT'
        }
        
        self.best_model = None
        self.training_data = None
        
        # Verify training directory exists
        if not self.training_images_dir.exists():
            raise FileNotFoundError(f"Training images directory not found: {self.training_images_dir}")
    
    def preprocess_image(self, image_path: str) -> Tuple[np.ndarray, str]:
        """
        Preprocess image for better OCR results.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Tuple of (processed_image, extracted_text)
        """
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not load image: {image_path}")
            
            # Convert to RGB (PIL format)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(image_rgb)
            
            # Enhance image quality
            enhancer = ImageEnhance.Contrast(pil_image)
            enhanced_image = enhancer.enhance(2.0)  # Increase contrast
            
            enhancer = ImageEnhance.Sharpness(enhanced_image)
            sharpened_image = enhancer.enhance(1.5)  # Increase sharpness
            
            # Convert back to OpenCV format for further processing
            opencv_image = cv2.cvtColor(np.array(sharpened_image), cv2.COLOR_RGB2BGR)
            
            # Convert to grayscale
            gray = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2GRAY)
            
            # Apply noise reduction
            denoised = cv2.fastNlMeansDenoising(gray)
            
            # Apply adaptive thresholding for better text extraction
            thresh = cv2.adaptiveThreshold(
                denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
            )
            
            # Extract text using Tesseract OCR
            custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789.,/:-() '
            
            try:
                extracted_text = pytesseract.image_to_string(
                    thresh, config=custom_config, lang='eng+hin'
                ).strip()
            except:
                # Fallback to basic OCR if advanced config fails
                extracted_text = pytesseract.image_to_string(thresh).strip()
            
            # Clean extracted text
            cleaned_text = self._clean_extracted_text(extracted_text)
            
            return thresh, cleaned_text
            
        except Exception as e:
            print(f"Error processing image {image_path}: {e}")
            return None, ""
    
    def _clean_extracted_text(self, text: str) -> str:
        """Clean and normalize extracted text."""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters that might interfere
        text = re.sub(r'[^A-Za-z0-9\s.,/:-]', '', text)
        
        # Convert to lowercase for consistency
        text = text.lower().strip()
        
        return text
    
    def determine_document_type(self, filename: str) -> str:
        """
        Determine document type from filename.
        
        Args:
            filename: Name of the image file
            
        Returns:
            Document type classification
        """
        filename_lower = filename.lower()
        
        for keyword, doc_type in self.document_type_mapping.items():
            if keyword in filename_lower:
                return doc_type
        
        return 'UNKNOWN_DOCUMENT'
    
    def load_training_data_from_images(self) -> Dict[str, Any]:
        """
        Load and process all training images.
        
        Returns:
            Dictionary containing processed training data
        """
        print("🔄 Loading and Processing Training Images...")
        print("=" * 50)
        
        training_data = {
            'texts': [],
            'labels': [],
            'file_info': []
        }
        
        # Get all image files
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        image_files = [f for f in self.training_images_dir.iterdir() 
                      if f.suffix.lower() in image_extensions]
        
        if not image_files:
            raise ValueError(f"No image files found in {self.training_images_dir}")
        
        print(f"📁 Found {len(image_files)} training images")
        
        processed_count = 0
        failed_count = 0
        
        for image_file in image_files:
            print(f"\n📷 Processing: {image_file.name}")
            
            try:
                # Preprocess image and extract text
                processed_image, extracted_text = self.preprocess_image(str(image_file))
                
                if not extracted_text:
                    print(f"⚠️ No text extracted from {image_file.name}")
                    failed_count += 1
                    continue
                
                # Determine document type
                doc_type = self.determine_document_type(image_file.name)
                
                # Store training data
                training_data['texts'].append(extracted_text)
                training_data['labels'].append(doc_type)
                training_data['file_info'].append({
                    'filename': image_file.name,
                    'document_type': doc_type,
                    'text_length': len(extracted_text),
                    'text_preview': extracted_text[:100] + '...' if len(extracted_text) > 100 else extracted_text
                })
                
                print(f"✅ Extracted: {len(extracted_text)} characters")
                print(f"🏷️ Classified as: {doc_type}")
                print(f"📝 Preview: {extracted_text[:60]}...")
                
                processed_count += 1
                
            except Exception as e:
                print(f"❌ Failed to process {image_file.name}: {e}")
                failed_count += 1
        
        print(f"\n📊 Processing Summary:")
        print(f"✅ Successfully processed: {processed_count} images")
        print(f"❌ Failed to process: {failed_count} images")
        
        if processed_count == 0:
            raise ValueError("No images were successfully processed!")
        
        self.training_data = training_data
        return training_data
    
    def train_image_based_models(self) -> Dict[str, Any]:
        """
        Train ML models using image-extracted data.
        
        Returns:
            Training results and model performance
        """
        if not self.training_data:
            self.load_training_data_from_images()
        
        print("\n🤖 Training Image-Based ML Models...")
        print("=" * 45)
        
        texts = self.training_data['texts']
        labels = self.training_data['labels']
        
        if len(set(labels)) < 2:
            print("⚠️ Warning: Only one document type found. Training basic classifier.")
        
        # Define models with optimized parameters for small datasets
        models = {
            'text_naive_bayes': Pipeline([
                ('tfidf', TfidfVectorizer(
                    max_features=min(500, len(texts) * 10),  # Adaptive feature count
                    ngram_range=(1, 2),
                    min_df=1,
                    max_df=0.8,
                    lowercase=True,
                    strip_accents='ascii'
                )),
                ('classifier', MultinomialNB(alpha=0.5))
            ]),
            'text_random_forest': Pipeline([
                ('tfidf', TfidfVectorizer(
                    max_features=min(300, len(texts) * 5),
                    ngram_range=(1, 2),
                    min_df=1,
                    max_df=0.8
                )),
                ('classifier', RandomForestClassifier(
                    n_estimators=min(50, len(texts) * 5),  # Adaptive tree count
                    max_depth=5,
                    random_state=42
                ))
            ])
        }
        
        # Train models
        results = {}
        
        for model_name, model in models.items():
            print(f"\n🔧 Training {model_name.replace('_', ' ').title()}...")
            
            try:
                # Train model
                model.fit(texts, labels)
                
                # Make predictions for evaluation
                if len(set(labels)) > 1:
                    predictions = model.predict(texts)
                    accuracy = accuracy_score(labels, predictions)
                    
                    # Cross-validation if we have enough samples
                    if len(texts) >= 3:
                        cv_scores = cross_val_score(model, texts, labels, cv=min(3, len(texts)))
                        cv_mean = cv_scores.mean()
                        cv_std = cv_scores.std()
                    else:
                        cv_mean = accuracy
                        cv_std = 0.0
                else:
                    accuracy = 1.0  # Single class, perfect accuracy
                    cv_mean = 1.0
                    cv_std = 0.0
                    predictions = labels
                
                results[model_name] = {
                    'model': model,
                    'accuracy': accuracy,
                    'cv_mean': cv_mean,
                    'cv_std': cv_std,
                    'predictions': predictions
                }
                
                print(f"✅ Accuracy: {accuracy:.3f}")
                print(f"✅ Cross-validation: {cv_mean:.3f} (±{cv_std:.3f})")
                
            except Exception as e:
                print(f"❌ Training failed for {model_name}: {e}")
                continue
        
        # Select best model
        if results:
            best_model_name = max(results.keys(), key=lambda k: results[k]['cv_mean'])
            self.best_model = results[best_model_name]['model']
            
            print(f"\n🏆 Best Model: {best_model_name.replace('_', ' ').title()}")
            print(f"🎯 Best Performance: {results[best_model_name]['cv_mean']:.3f}")
        
        return results
    
    def save_image_based_model(self, model_path: str = None) -> str:
        """Save the trained image-based model."""
        if model_path is None:
            model_path = self.models_dir / "image_based_document_classifier.pkl"
        
        if not self.best_model:
            raise ValueError("No trained model available to save!")
        
        model_data = {
            'model': self.best_model,
            'document_types': list(set(self.training_data['labels'])),
            'training_data_info': {
                'num_images': len(self.training_data['texts']),
                'file_info': self.training_data['file_info']
            },
            'training_timestamp': datetime.now(),
            'model_type': 'image_based_classifier'
        }
        
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"💾 Image-based model saved to: {model_path}")
        return str(model_path)
    
    def load_image_based_model(self, model_path: str = None) -> None:
        """Load a trained image-based model."""
        if model_path is None:
            model_path = self.models_dir / "image_based_document_classifier.pkl"
        
        try:
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            self.best_model = model_data['model']
            
            print(f"📥 Image-based model loaded from: {model_path}")
            print(f"📊 Trained on {model_data['training_data_info']['num_images']} images")
            print(f"📅 Training date: {model_data['training_timestamp']}")
            
        except FileNotFoundError:
            raise FileNotFoundError(f"Model file not found: {model_path}")
        except Exception as e:
            raise ValueError(f"Error loading model: {e}")
    
    def classify_new_image(self, image_path: str) -> Dict[str, Any]:
        """
        Classify a new document image.
        
        Args:
            image_path: Path to the image to classify
            
        Returns:
            Classification results
        """
        if not self.best_model:
            raise ValueError("No trained model available. Please train a model first.")
        
        print(f"\n🔍 Classifying new image: {Path(image_path).name}")
        
        # Preprocess image and extract text
        processed_image, extracted_text = self.preprocess_image(image_path)
        
        if not extracted_text:
            return {
                'error': 'No text could be extracted from the image',
                'confidence': 0.0,
                'predicted_type': 'UNKNOWN'
            }
        
        # Make prediction
        try:
            prediction = self.best_model.predict([extracted_text])[0]
            
            # Get probability if available
            try:
                probabilities = self.best_model.predict_proba([extracted_text])[0]
                confidence = probabilities.max()
                
                # Get all class probabilities
                classes = self.best_model.classes_
                all_probabilities = dict(zip(classes, probabilities))
            except:
                confidence = 1.0  # Fallback for models without probability
                all_probabilities = {prediction: 1.0}
            
            result = {
                'predicted_type': prediction,
                'confidence': confidence,
                'extracted_text': extracted_text,
                'text_length': len(extracted_text),
                'all_probabilities': all_probabilities,
                'image_processed': processed_image is not None
            }
            
            print(f"🎯 Prediction: {prediction}")
            print(f"📊 Confidence: {confidence:.3f}")
            print(f"📝 Text extracted: {len(extracted_text)} characters")
            
            return result
            
        except Exception as e:
            return {
                'error': f'Classification failed: {e}',
                'confidence': 0.0,
                'predicted_type': 'ERROR'
            }

def main():
    """Main function to demonstrate image-based training."""
    print("🖼️ Image-Based ML Document Classification Training")
    print("=" * 60)
    
    # Initialize trainer
    trainer = ImageBasedDocumentTrainer()
    
    try:
        # Load training data from images
        training_data = trainer.load_training_data_from_images()
        
        # Train models
        results = trainer.train_image_based_models()
        
        # Save the best model
        model_path = trainer.save_image_based_model()
        
        print(f"\n✅ Training Complete!")
        print(f"💾 Model saved to: {model_path}")
        
        # Test classification on training images
        print(f"\n🧪 Testing Classification on Training Images:")
        print("-" * 50)
        
        for info in training_data['file_info'][:3]:  # Test first 3 images
            image_path = trainer.training_images_dir / info['filename']
            if image_path.exists():
                result = trainer.classify_new_image(str(image_path))
                expected = info['document_type']
                predicted = result.get('predicted_type', 'ERROR')
                correct = "✅" if predicted == expected else "❌"
                
                print(f"\n{correct} {info['filename']}")
                print(f"   Expected: {expected.replace('_', ' ').title()}")
                print(f"   Predicted: {predicted.replace('_', ' ').title()}")
                print(f"   Confidence: {result.get('confidence', 0):.3f}")
    
    except Exception as e:
        print(f"❌ Training failed: {e}")
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)