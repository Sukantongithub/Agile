"""
Enhanced ML Training System for Document Classification
Trains models to identify Indian documents like Aadhar, PAN, Voter ID, etc.
"""

import numpy as np
import pandas as pd
import json
import pickle
from typing import Dict, List, Tuple, Any
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class DocumentClassifierTrainer:
    """Advanced ML trainer for Indian document classification."""
    
    def __init__(self):
        """Initialize the trainer with comprehensive training data."""
        self.document_types = [
            'AADHAR_CARD', 'PAN_CARD', 'VOTER_ID', 'DRIVING_LICENSE', 
            'PASSPORT', 'MARKSHEET', 'RATION_CARD', 'BANK_PASSBOOK',
            'BIRTH_CERTIFICATE', 'COMMUNITY_CERTIFICATE', 'SMART_CARD'
        ]
        
        self.training_data = self._generate_comprehensive_training_data()
        self.models = {}
        self.best_model = None
        self.vectorizer = None
        
    def _generate_comprehensive_training_data(self) -> Dict[str, List[str]]:
        """Generate comprehensive training data for all document types."""
        
        training_data = {
            'AADHAR_CARD': [
                # English samples
                "government of india unique identification authority aadhaar card uid number",
                "aadhaar card unique identification number government india uidai",
                "aadhar number twelve digit unique identity proof government issued",
                "unique identification authority india aadhaar card biometric verification",
                "uid aadhaar government india unique identification number card",
                "aadhaar enrollment number unique identity document indian citizen",
                "government india aadhaar card unique identification authority",
                "uidai issued aadhaar card unique identification number",
                "aadhar card government issued identity proof verification document",
                "unique identification aadhaar number government india citizen",
                
                # Hindi transliterated samples
                "भारत सरकार आधार कार्ड unique identification uid number",
                "आधार संख्या government issued identity proof aadhaar",
                "unique identification authority आधार कार्ड government india",
                "aadhaar card भारत सरकार unique identification number",
                
                # Mixed OCR scenarios
                "g0vernment 0f india aadhaar card unique identificati0n",
                "aadhar card g0v india uid numb3r unique identity",
                "unique identificati0n auth0rity aadhaar card india",
                
                # Additional patterns
                "aadhaar card address proof identity verification document",
                "uid number aadhaar card government india verification",
                "unique identification aadhaar biometric verification government"
            ],
            
            'PAN_CARD': [
                # English samples
                "income tax department government india permanent account number pan",
                "pan card permanent account number income tax department",
                "permanent account number card income tax identification",
                "income tax department pan card permanent account alphanumeric",
                "tax identification number permanent account pan india revenue",
                "pan number income tax department government india card",
                "permanent account number pan tax identification document",
                "income tax pan card permanent account number verification",
                "pan card tax identification permanent account government",
                "permanent account number income tax department india pan",
                
                # Hindi transliterated samples
                "आयकर विभाग pan card permanent account number",
                "permanent account number आयकर विभाग government india",
                "pan card income tax department भारत सरकार",
                
                # OCR variations
                "inc0me tax department permanent acc0unt number pan",
                "pan card perman3nt account numb3r income tax",
                
                # Additional patterns
                "pan card tax filing permanent account number verification",
                "income tax identification pan permanent account document"
            ],
            
            'VOTER_ID': [
                # English samples
                "election commission india electoral photo identity card epic",
                "voter id card election commission epic number constituency",
                "electoral photo identity card voter registration epic",
                "election commission voter identity card epic number",
                "voter id epic number election commission constituency",
                "electoral roll voter registration card polling station",
                "election identity card voter epic assembly constituency",
                "voter registration election commission epic identity card",
                "epic number voter id election commission india",
                "electoral photo identity voter card election commission",
                
                # Hindi transliterated samples
                "भारत निर्वाचन आयोग voter id card election commission",
                "निर्वाचक फोटो पहचान पत्र election commission epic",
                "voter id card भारत निर्वाचन आयोग epic number",
                
                # OCR variations
                "electi0n commission india v0ter id card epic",
                "v0ter id epic numb3r election c0mmission",
                
                # Additional patterns
                "voter id card polling booth election commission verification",
                "electoral identity card voter registration epic number"
            ],
            
            'DRIVING_LICENSE': [
                # English samples
                "driving license transport department motor vehicle authority",
                "transport department driving license vehicle permit state",
                "motor vehicle driving license transport authority government",
                "driving permit license transport department state government",
                "vehicle license driving permit transport commissioner",
                "driving license motor vehicle act transport department",
                "transport authority driving license vehicle permit state",
                "driving license state transport department motor vehicle",
                "vehicle driving license transport department government",
                "motor vehicle driving license transport authority permit",
                
                # Hindi transliterated samples
                "ड्राइविंग लाइसेंस transport department motor vehicle",
                "driving license परिवहन विभाग state government",
                "transport department ड्राइविंग लाइसेंस vehicle permit",
                
                # OCR variations
                "driving lic3nse transp0rt department m0tor vehicle",
                "transp0rt department driving licens3 vehicle permit"
            ],
            
            'PASSPORT': [
                # English samples
                "passport republic india ministry external affairs travel",
                "ministry external affairs passport republic india travel",
                "passport travel document republic india international",
                "republic india passport ministry external affairs issued",
                "travel document passport ministry external affairs india",
                "passport india republic ministry external affairs travel",
                "ministry external affairs travel passport republic india",
                "indian passport travel document republic india mea",
                "passport republic india travel document international",
                "travel passport ministry external affairs republic india",
                
                # Hindi transliterated samples
                "पासपोर्ट भारत गणराज्य ministry external affairs",
                "passport republic india विदेश मंत्रालय travel",
                "ministry external affairs पासपोर्ट travel document",
                
                # OCR variations
                "passp0rt republic india ministry external affairs",
                "ministry ext3rnal affairs passport republic india"
            ],
            
            'MARKSHEET': [
                # English samples
                "marksheet examination result grade percentage university",
                "marks obtained examination marksheet grade card university",
                "examination result marksheet university college marks",
                "grade card marksheet examination result percentage",
                "marksheet academic transcript university examination",
                "examination marksheet marks obtained grade percentage",
                "university marksheet examination result grade card",
                "marks certificate examination marksheet academic transcript",
                "marksheet grade percentage examination university result",
                "academic marksheet examination result university college",
                
                # Hindi transliterated samples
                "अंक तालिका marksheet examination result university",
                "marksheet परीक्षा परिणाम university grade card",
                "examination result अंक तालिका marks obtained",
                
                # OCR variations
                "marksh3et examination r3sult grade percentage",
                "examination marksh33t marks 0btained university"
            ],
            
            'RATION_CARD': [
                # English samples
                "ration card food security civil supplies distribution",
                "food card civil supplies ration public distribution",
                "ration card family card civil supplies food security",
                "civil supplies ration card food distribution system",
                "food security ration card civil supplies family",
                "public distribution ration card food security supplies",
                "ration card below poverty line civil supplies",
                "family ration card civil supplies food distribution",
                "food security card ration civil supplies government",
                "ration card food supplies civil distribution family",
                
                # Hindi transliterated samples
                "राशन कार्ड civil supplies food security family",
                "ration card नागरिक आपूर्ति food distribution",
                "food security राशन कार्ड civil supplies",
                
                # OCR variations
                "rati0n card f00d security civil supplies",
                "f00d card rati0n civil supplies distributi0n"
            ],
            
            'BANK_PASSBOOK': [
                # English samples
                "bank passbook savings account current account balance",
                "passbook bank account savings current transaction",
                "savings account passbook bank transaction balance",
                "bank account passbook holder savings current",
                "current account passbook bank savings transaction",
                "passbook savings current account bank balance",
                "bank passbook account number ifsc code balance",
                "account passbook bank savings current holder",
                "passbook bank statement account balance transaction",
                "savings passbook bank account current transaction",
                
                # Hindi transliterated samples
                "बैंक पासबुक savings account current account",
                "passbook बैंक खाता savings current transaction",
                "bank account पासबुक savings current balance",
                
                # OCR variations
                "bank passb00k savings acc0unt current",
                "passb00k bank acc0unt savings current transacti0n"
            ],
            
            'BIRTH_CERTIFICATE': [
                # English samples
                "birth certificate registrar births deaths municipal",
                "certificate birth registrar births municipal corporation",
                "birth registration certificate municipal authority",
                "registrar births deaths birth certificate municipal",
                "birth certificate municipal corporation registrar",
                "certificate birth municipal registrar births deaths",
                "birth record certificate registrar municipal corporation",
                "municipal birth certificate registrar births deaths",
                "birth certificate authority registrar municipal government",
                "registrar birth certificate municipal corporation authority",
                
                # Hindi transliterated samples
                "जन्म प्रमाण पत्र registrar births municipal",
                "birth certificate नगर निगम municipal corporation",
                "registrar births जन्म प्रमाण पत्र municipal",
                
                # OCR variations
                "birth c3rtificate registrar births municipal",
                "c3rtificate birth registrar births municipal"
            ],
            
            'COMMUNITY_CERTIFICATE': [
                # English samples
                "community certificate caste certificate backward class",
                "caste certificate scheduled caste tribe community",
                "community caste certificate obc sc st revenue",
                "scheduled caste community certificate backward class",
                "caste certificate community backward scheduled tribe",
                "community certificate scheduled caste tribe obc",
                "backward class community certificate caste scheduled",
                "caste community certificate scheduled backward class",
                "community certificate revenue department caste scheduled",
                "scheduled community certificate caste backward tribe",
                
                # Hindi transliterated samples
                "जाति प्रमाण पत्र community certificate caste",
                "community certificate जाति प्रमाण पत्र scheduled",
                "caste certificate समुदायिक प्रमाण पत्र backward",
                
                # OCR variations
                "c0mmunity certificate caste sch3duled backward",
                "caste c3rtificate community backward sch3duled"
            ],
            
            'SMART_CARD': [
                # English samples
                "smart card chip card employee identification digital",
                "chip card smart card employee health identification",
                "employee card smart chip card digital identification",
                "smart card technology chip based employee card",
                "digital smart card chip employee identification",
                "health card smart chip card employee digital",
                "smart card employee chip card identification",
                "chip based smart card employee digital identification",
                "employee smart card chip technology digital",
                "smart card digital chip employee identification",
                
                # Hindi transliterated samples
                "स्मार्ट कार्ड chip card employee identification",
                "smart card चिप कार्ड employee digital",
                "chip card स्मार्ट कार्ड digital identification",
                
                # OCR variations
                "smart card chip card empl0yee identificati0n",
                "chip card smart card 3mployee digital"
            ]
        }
        
        return training_data
        
    def prepare_training_data(self) -> Tuple[List[str], List[str]]:
        """Prepare training texts and labels."""
        texts = []
        labels = []
        
        for doc_type, samples in self.training_data.items():
            for sample in samples:
                texts.append(sample.lower())  # Normalize to lowercase
                labels.append(doc_type)
        
        return texts, labels
    
    def train_multiple_models(self) -> Dict[str, Any]:
        """Train multiple ML models and compare performance."""
        texts, labels = self.prepare_training_data()
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            texts, labels, test_size=0.2, random_state=42, stratify=labels
        )
        
        # Initialize models
        models = {
            'naive_bayes': Pipeline([
                ('tfidf', TfidfVectorizer(
                    max_features=2000,
                    ngram_range=(1, 3),
                    stop_words='english',
                    lowercase=True,
                    strip_accents='ascii'
                )),
                ('classifier', MultinomialNB(alpha=0.5))
            ]),
            'random_forest': Pipeline([
                ('tfidf', TfidfVectorizer(
                    max_features=1500,
                    ngram_range=(1, 2),
                    stop_words='english'
                )),
                ('classifier', RandomForestClassifier(
                    n_estimators=100,
                    random_state=42,
                    max_depth=20
                ))
            ]),
            'svm': Pipeline([
                ('tfidf', TfidfVectorizer(
                    max_features=1000,
                    ngram_range=(1, 2),
                    stop_words='english'
                )),
                ('classifier', SVC(
                    kernel='rbf',
                    probability=True,
                    random_state=42
                ))
            ])
        }
        
        results = {}
        
        print("🤖 Training Multiple ML Models for Document Classification...")
        print("=" * 60)
        
        for name, model in models.items():
            print(f"\n🔄 Training {name.replace('_', ' ').title()}...")
            
            # Train model
            model.fit(X_train, y_train)
            
            # Predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)
            
            # Metrics
            accuracy = accuracy_score(y_test, y_pred)
            cross_val_scores = cross_val_score(model, X_train, y_train, cv=5)
            
            results[name] = {
                'model': model,
                'accuracy': accuracy,
                'cross_val_mean': cross_val_scores.mean(),
                'cross_val_std': cross_val_scores.std(),
                'classification_report': classification_report(y_test, y_pred),
                'predictions': y_pred,
                'probabilities': y_pred_proba,
                'test_labels': y_test
            }
            
            print(f"✅ Accuracy: {accuracy:.4f}")
            print(f"✅ Cross-validation: {cross_val_scores.mean():.4f} (±{cross_val_scores.std():.4f})")
        
        # Find best model
        best_model_name = max(results.keys(), key=lambda k: results[k]['cross_val_mean'])
        self.best_model = results[best_model_name]['model']
        self.vectorizer = self.best_model.named_steps['tfidf']
        
        print(f"\n🏆 Best Model: {best_model_name.replace('_', ' ').title()}")
        print(f"🎯 Best Accuracy: {results[best_model_name]['accuracy']:.4f}")
        
        return results
    
    def hyperparameter_tuning(self) -> Dict[str, Any]:
        """Perform hyperparameter tuning for the best model."""
        texts, labels = self.prepare_training_data()
        
        print("\n🔧 Performing Hyperparameter Tuning...")
        print("=" * 50)
        
        # Define parameter grid for Naive Bayes (usually best performer)
        param_grid = {
            'tfidf__max_features': [1500, 2000, 2500],
            'tfidf__ngram_range': [(1, 2), (1, 3), (2, 3)],
            'classifier__alpha': [0.1, 0.5, 1.0, 2.0]
        }
        
        # Base pipeline
        base_model = Pipeline([
            ('tfidf', TfidfVectorizer(stop_words='english', lowercase=True)),
            ('classifier', MultinomialNB())
        ])
        
        # Grid search
        grid_search = GridSearchCV(
            base_model,
            param_grid,
            cv=5,
            scoring='accuracy',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(texts, labels)
        
        print(f"🎯 Best Parameters: {grid_search.best_params_}")
        print(f"🏆 Best Score: {grid_search.best_score_:.4f}")
        
        self.best_model = grid_search.best_estimator_
        self.vectorizer = self.best_model.named_steps['tfidf']
        
        return {
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_,
            'best_model': grid_search.best_estimator_
        }
    
    def evaluate_model_performance(self, results: Dict[str, Any]) -> None:
        """Create detailed performance evaluation."""
        print("\\n📊 Detailed Performance Analysis")
        print("=" * 40)
        
        for name, result in results.items():
            print(f"\\n📈 {name.replace('_', ' ').title()} Model:")
            print("-" * 30)
            print(result['classification_report'])
    
    def save_trained_model(self, model_path: str = "models/document_classifier.pkl") -> None:
        """Save the trained model."""
        Path(model_path).parent.mkdir(exist_ok=True)
        
        model_data = {
            'model': self.best_model,
            'vectorizer': self.vectorizer,
            'document_types': self.document_types,
            'training_timestamp': pd.Timestamp.now(),
            'model_info': {
                'model_type': type(self.best_model.named_steps['classifier']).__name__,
                'vectorizer_type': type(self.vectorizer).__name__,
                'feature_count': getattr(self.vectorizer, 'max_features', 'auto'),
                'ngram_range': getattr(self.vectorizer, 'ngram_range', (1, 1))
            }
        }
        
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"💾 Model saved to: {model_path}")
    
    def load_trained_model(self, model_path: str = "models/document_classifier.pkl") -> None:
        """Load a trained model."""
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.best_model = model_data['model']
        self.vectorizer = model_data['vectorizer']
        self.document_types = model_data['document_types']
        
        print(f"📥 Model loaded from: {model_path}")
        print(f"🤖 Model Type: {model_data['model_info']['model_type']}")
        print(f"📊 Features: {model_data['model_info']['feature_count']}")
    
    def predict_document_type(self, text: str) -> Dict[str, Any]:
        """Predict document type for given text."""
        if not self.best_model:
            raise ValueError("No trained model available. Please train a model first.")
        
        # Preprocess text
        processed_text = text.lower().strip()
        
        # Prediction
        prediction = self.best_model.predict([processed_text])[0]
        probabilities = self.best_model.predict_proba([processed_text])[0]
        
        # Get top 3 predictions
        top_indices = np.argsort(probabilities)[::-1][:3]
        top_predictions = [
            {
                'document_type': self.document_types[idx] if idx < len(self.document_types) 
                               else self.best_model.classes_[idx],
                'confidence': probabilities[idx],
                'confidence_percentage': f"{probabilities[idx] * 100:.2f}%"
            }
            for idx in top_indices
        ]
        
        return {
            'predicted_type': prediction,
            'confidence': probabilities.max(),
            'top_predictions': top_predictions,
            'all_probabilities': dict(zip(self.best_model.classes_, probabilities))
        }
    
    def create_training_report(self, results: Dict[str, Any]) -> str:
        """Generate comprehensive training report."""
        report = []
        report.append("📋 ML Document Classification Training Report")
        report.append("=" * 50)
        report.append(f"📅 Training Date: {pd.Timestamp.now()}")
        report.append(f"📚 Document Types: {len(self.document_types)}")
        report.append(f"🔢 Training Samples: {sum(len(samples) for samples in self.training_data.values())}")
        report.append("")
        
        report.append("🎯 Model Performance Summary:")
        report.append("-" * 30)
        
        for name, result in results.items():
            report.append(f"{name.replace('_', ' ').title()}:")
            report.append(f"  • Accuracy: {result['accuracy']:.4f}")
            report.append(f"  • Cross-validation: {result['cross_val_mean']:.4f} (±{result['cross_val_std']:.4f})")
            report.append("")
        
        # Best model info
        best_model_name = max(results.keys(), key=lambda k: results[k]['cross_val_mean'])
        report.append(f"🏆 Best Performing Model: {best_model_name.replace('_', ' ').title()}")
        report.append(f"🎯 Best Accuracy: {results[best_model_name]['accuracy']:.4f}")
        report.append("")
        
        # Document type coverage
        report.append("📄 Supported Document Types:")
        report.append("-" * 30)
        for i, doc_type in enumerate(self.document_types, 1):
            report.append(f"{i:2d}. {doc_type.replace('_', ' ').title()}")
        
        return "\\n".join(report)

def main():
    """Main training function."""
    print("🚀 Starting Enhanced ML Document Classification Training")
    print("=" * 60)
    
    # Initialize trainer
    trainer = DocumentClassifierTrainer()
    
    # Train multiple models
    results = trainer.train_multiple_models()
    
    # Perform hyperparameter tuning
    tuning_results = trainer.hyperparameter_tuning()
    
    # Evaluate performance
    trainer.evaluate_model_performance(results)
    
    # Save best model
    trainer.save_trained_model()
    
    # Generate training report
    report = trainer.create_training_report(results)
    print("\\n" + report)
    
    # Save training report
    with open("models/training_report.txt", "w", encoding='utf-8') as f:
        f.write(report)
    
    print("\\n✅ Training Complete!")
    print("💾 Model saved to: models/document_classifier.pkl")
    print("📊 Report saved to: models/training_report.txt")
    
    # Test with sample texts
    print("\\n🧪 Testing with Sample Documents:")
    print("-" * 40)
    
    test_samples = [
        ("aadhaar card government of india unique identification", "AADHAR_CARD"),
        ("pan card income tax department permanent account number", "PAN_CARD"),
        ("voter id election commission epic number", "VOTER_ID"),
        ("marksheet examination result university grade", "MARKSHEET"),
        ("bank passbook savings account balance", "BANK_PASSBOOK")
    ]
    
    for text, expected in test_samples:
        prediction = trainer.predict_document_type(text)
        correct = "✅" if prediction['predicted_type'] == expected else "❌"
        print(f"{correct} Text: {text[:40]}...")
        print(f"   Predicted: {prediction['predicted_type']} ({prediction['confidence']:.3f})")
        print(f"   Expected: {expected}")
        print()

if __name__ == "__main__":
    main()