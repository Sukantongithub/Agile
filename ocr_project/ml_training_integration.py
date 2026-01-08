"""
ML Training Integration for Document Classification System
Integrates advanced ML training with the existing RAG system
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from train_document_classifier import DocumentClassifierTrainer
from ocr.rag_field_suggestion import DocumentFieldKnowledgeBase
import json
import pickle
from pathlib import Path
from datetime import datetime
import numpy as np

class MLDocumentTrainingIntegration:
    """Integration class for ML training with RAG system."""
    
    def __init__(self):
        """Initialize the integration system."""
        self.trainer = DocumentClassifierTrainer()
        self.rag_kb = DocumentFieldKnowledgeBase()
        self.models_dir = Path("models")
        self.models_dir.mkdir(exist_ok=True)
        
    def train_comprehensive_models(self) -> dict:
        """Train both standalone and integrated models."""
        print("🔄 Starting Comprehensive ML Training Integration")
        print("=" * 60)
        
        # Step 1: Train standalone advanced classifier
        print("\\n📚 Step 1: Training Advanced Standalone Classifier")
        standalone_results = self.trainer.train_multiple_models()
        
        # Step 2: Perform hyperparameter tuning
        print("\\n🔧 Step 2: Hyperparameter Optimization")
        tuning_results = self.trainer.hyperparameter_tuning()
        
        # Step 3: Integrate with RAG system
        print("\\n🔗 Step 3: Integrating with RAG System")
        self._integrate_with_rag_system()
        
        # Step 4: Save all models
        print("\\n💾 Step 4: Saving Trained Models")
        self._save_all_models(standalone_results, tuning_results)
        
        # Step 5: Performance comparison
        print("\\n📊 Step 5: Performance Analysis")
        comparison_results = self._compare_model_performance()
        
        return {
            'standalone_results': standalone_results,
            'tuning_results': tuning_results,
            'comparison_results': comparison_results,
            'training_timestamp': datetime.now()
        }
    
    def _integrate_with_rag_system(self):
        """Integrate trained model with RAG system."""
        # Get training data from advanced trainer
        texts, labels = self.trainer.prepare_training_data()
        
        # Create new training data dictionary for RAG system
        rag_training_data = {}
        for text, label in zip(texts, labels):
            if label not in rag_training_data:
                rag_training_data[label] = []
            rag_training_data[label].append(text)
        
        # Retrain RAG classifier with enhanced data
        self.rag_kb.retrain_classifier_with_new_data(rag_training_data)
        
        print("✅ RAG system updated with enhanced training data")
    
    def _save_all_models(self, standalone_results: dict, tuning_results: dict):
        """Save all trained models."""
        # Save standalone advanced model
        self.trainer.save_trained_model("models/advanced_document_classifier.pkl")
        
        # Save RAG integrated model
        self.rag_kb.save_trained_model("models/rag_integrated_classifier.pkl")
        
        # Save training metadata
        metadata = {
            'training_date': datetime.now().isoformat(),
            'standalone_accuracy': max(r['accuracy'] for r in standalone_results.values()),
            'best_hyperparams': tuning_results['best_params'],
            'document_types_count': len(self.trainer.document_types),
            'training_samples_count': sum(len(samples) for samples in self.trainer.training_data.values())
        }
        
        with open("models/training_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        
        print("💾 All models and metadata saved successfully")
    
    def _compare_model_performance(self) -> dict:
        """Compare performance of different models."""
        print("🔍 Comparing Model Performance...")
        
        # Test samples for comparison
        test_samples = [
            ("aadhaar card government of india unique identification authority uid number", "AADHAR_CARD"),
            ("pan card income tax department permanent account number alphanumeric", "PAN_CARD"),
            ("voter id card election commission epic number constituency polling", "VOTER_ID"),
            ("driving license transport department motor vehicle authority state permit", "DRIVING_LICENSE"),
            ("passport republic india ministry external affairs travel document", "PASSPORT"),
            ("marksheet examination result university grade percentage cgpa marks", "MARKSHEET"),
            ("ration card food security civil supplies public distribution family", "RATION_CARD"),
            ("bank passbook savings account current account balance transaction", "BANK_PASSBOOK"),
            ("birth certificate registrar births deaths municipal corporation", "BIRTH_CERTIFICATE"),
            ("community certificate caste scheduled backward class obc sc st", "COMMUNITY_CERTIFICATE"),
            ("smart card chip card employee identification digital health card", "SMART_CARD")
        ]
        
        # Test standalone model
        standalone_correct = 0
        for text, expected in test_samples:
            prediction = self.trainer.predict_document_type(text)
            if prediction['predicted_type'] == expected:
                standalone_correct += 1
        
        # Test RAG integrated model  
        rag_correct = 0
        for text, expected in test_samples:
            classifications = self.rag_kb.classify_document(text)
            if classifications and classifications[0].document_type == expected:
                rag_correct += 1
        
        standalone_accuracy = standalone_correct / len(test_samples)
        rag_accuracy = rag_correct / len(test_samples)
        
        comparison = {
            'standalone_model': {
                'accuracy': standalone_accuracy,
                'correct_predictions': standalone_correct,
                'total_samples': len(test_samples)
            },
            'rag_integrated_model': {
                'accuracy': rag_accuracy,
                'correct_predictions': rag_correct,
                'total_samples': len(test_samples)
            }
        }
        
        print(f"📈 Standalone Model Accuracy: {standalone_accuracy:.3f}")
        print(f"📈 RAG Integrated Model Accuracy: {rag_accuracy:.3f}")
        
        return comparison
    
    def test_document_classification(self, test_text: str) -> dict:
        """Test document classification with both models."""
        print(f"\\n🧪 Testing Classification for: {test_text[:50]}...")
        
        # Test standalone model
        standalone_result = self.trainer.predict_document_type(test_text)
        
        # Test RAG model
        rag_results = self.rag_kb.classify_document(test_text)
        
        return {
            'text': test_text,
            'standalone_prediction': {
                'type': standalone_result['predicted_type'],
                'confidence': standalone_result['confidence'],
                'top_predictions': standalone_result['top_predictions']
            },
            'rag_prediction': {
                'type': rag_results[0].document_type if rag_results else 'UNKNOWN',
                'confidence': rag_results[0].confidence if rag_results else 0.0,
                'keywords_found': rag_results[0].keywords_found if rag_results else [],
                'reasoning': rag_results[0].reasoning if rag_results else 'No classification found'
            }
        }
    
    def generate_training_summary(self, training_results: dict) -> str:
        """Generate comprehensive training summary."""
        summary_lines = [
            "🎯 ML Document Classification Training Summary",
            "=" * 55,
            f"📅 Training Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"📚 Document Types Trained: {len(self.trainer.document_types)}",
            "",
            "📋 Supported Document Types:",
            "-" * 30
        ]
        
        for i, doc_type in enumerate(self.trainer.document_types, 1):
            summary_lines.append(f"{i:2d}. {doc_type.replace('_', ' ').title()}")
        
        summary_lines.extend([
            "",
            "🤖 Model Performance:",
            "-" * 20,
            f"Best Standalone Accuracy: {max(r['accuracy'] for r in training_results['standalone_results'].values()):.4f}",
            f"Cross-validation Score: {training_results['tuning_results']['best_score']:.4f}",
            "",
            "🔧 Best Hyperparameters:",
            "-" * 25
        ])
        
        for param, value in training_results['tuning_results']['best_params'].items():
            summary_lines.append(f"  • {param}: {value}")
        
        summary_lines.extend([
            "",
            "📊 Model Comparison:",
            "-" * 20,
            f"Standalone Model: {training_results['comparison_results']['standalone_model']['accuracy']:.3f} accuracy",
            f"RAG Integrated: {training_results['comparison_results']['rag_integrated_model']['accuracy']:.3f} accuracy",
            "",
            "💾 Saved Models:",
            "-" * 15,
            "  • models/advanced_document_classifier.pkl (Standalone)",
            "  • models/rag_integrated_classifier.pkl (RAG Integrated)",
            "  • models/training_metadata.json (Metadata)",
            "",
            "✅ Training Complete - Models Ready for Production!"
        ])
        
        return "\\n".join(summary_lines)

def main():
    """Main training integration function."""
    print("🚀 Starting ML Document Classification Training Integration")
    print("=" * 70)
    
    # Initialize integration system
    integration = MLDocumentTrainingIntegration()
    
    # Run comprehensive training
    training_results = integration.train_comprehensive_models()
    
    # Generate and save summary
    summary = integration.generate_training_summary(training_results)
    print("\\n" + summary)
    
    # Save summary to file
    with open("models/training_integration_summary.txt", "w") as f:
        f.write(summary)
    
    # Test with sample documents
    print("\\n🧪 Testing Document Classification Examples:")
    print("-" * 50)
    
    test_examples = [
        "aadhaar card unique identification authority government india uid number biometric",
        "voter id election commission epic number constituency electoral photo identity",
        "marksheet examination university result grade percentage cgpa academic transcript",
        "bank passbook savings account balance transaction ifsc code account number",
        "smart card chip card employee health identification digital technology"
    ]
    
    for text in test_examples:
        result = integration.test_document_classification(text)
        print(f"\\n📄 Text: {text[:40]}...")
        print(f"🤖 Standalone: {result['standalone_prediction']['type']} ({result['standalone_prediction']['confidence']:.3f})")
        print(f"🔗 RAG Model: {result['rag_prediction']['type']} ({result['rag_prediction']['confidence']:.3f})")
    
    print(f"\\n✅ Training Integration Complete!")
    print("💾 Summary saved to: models/training_integration_summary.txt")

if __name__ == "__main__":
    main()