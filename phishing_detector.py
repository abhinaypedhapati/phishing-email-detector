#!/usr/bin/env python3
"""
Phishing Email Detection Model
Machine Learning model to classify emails as Phishing or Safe
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.naive_bayes import MultinomialNB
import re
import joblib
import os

class PhishingEmailDetector:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
        self.classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        self.is_trained = False
    
    # Feature Extraction Functions
    def extract_features_from_text(self, text):
        """
        Extract handcrafted features from email text
        """
        if not isinstance(text, str):
            text = str(text)
        
        text_lower = text.lower()
        
        features = {
            # URL related features
            'has_url': 1 if re.search(r'https?://|www\.', text_lower) else 0,
            'url_count': len(re.findall(r'https?://[^\s]+', text_lower)),
            'has_shortened_url': 1 if re.search(r'bit\.ly|tinyurl|goo\.gl|ow\.ly', text_lower) else 0,
            
            # Suspicious keywords
            'urgent_words': 1 if re.search(r'urgent|immediately|asap|verify now|action required', text_lower) else 0,
            'account_words': 1 if re.search(r'account|password|verify|update|confirm|security', text_lower) else 0,
            'prize_words': 1 if re.search(r'winner|congratulations|prize|lottery|won', text_lower) else 0,
            'payment_words': 1 if re.search(r'payment|credit card|bank|transfer|invoice', text_lower) else 0,
            'personal_info': 1 if re.search(r'ssn|social security|tax id|driver license', text_lower) else 0,
            
            # Suspicious patterns
            'has_attachment': 1 if re.search(r'\.zip|\.exe|\.scr|\.bat', text_lower) else 0,
            'has_many_links': 1 if len(re.findall(r'https?://', text_lower)) > 3 else 0,
            'has_ip_address': 1 if re.search(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', text_lower) else 0,
            'has_many_exclamation': 1 if text_lower.count('!') > 3 else 0,
            
            # Email structure
            'subject_exists': 1 if 'subject:' in text_lower[:200] else 0,
            'from_exists': 1 if 'from:' in text_lower[:200] else 0,
            'reply_to_exists': 1 if 'reply-to:' in text_lower[:200] else 0,
            
            'text_length': min(len(text), 10000),
            'digit_ratio': sum(c.isdigit() for c in text) / max(len(text), 1),
            'special_char_ratio': sum(not c.isalnum() and not c.isspace() for c in text) / max(len(text), 1)
        }
        
        return features
    
    def create_feature_vector(self, text):
        """
        Convert text to feature vector for model
        """
        features = self.extract_features_from_text(text)
        return np.array(list(features.values()))
    
    def train_model(self, data_file=None):
        """
        Train the ML model on phishing email dataset
        """
        print("="*60)
        print("📧 PHISHING EMAIL DETECTION MODEL - TRAINING")
        print("="*60)
        
        # Load or create dataset
        if data_file and os.path.exists(data_file):
            df = pd.read_csv(data_file)
            print(f"✅ Loaded dataset: {len(df)} emails")
        else:
            print("⚠️ Dataset not found. Creating sample dataset...")
            df = self.create_sample_dataset()
            df.to_csv('phishing_emails_dataset.csv', index=False)
            print(f"✅ Created dataset: {len(df)} emails")
        
        # Split features and labels
        X_text = df['email_text'].values
        y = df['label'].values
        
        # Extract features
        X_features = np.array([self.create_feature_vector(text) for text in X_text])
        
        # Also use TF-IDF for text features
        X_tfidf = self.vectorizer.fit_transform(X_text).toarray()
        
        # Combine features
        X_combined = np.hstack([X_features, X_tfidf])
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X_combined, y, test_size=0.2, random_state=42
        )
        
        print(f"\n📊 Data Split:")
        print(f"   Training samples: {len(X_train)}")
        print(f"   Testing samples: {len(X_test)}")
        
        # Train classifier
        print("\n🔄 Training Random Forest Classifier...")
        self.classifier.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.classifier.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        
        print(f"\n📈 MODEL EVALUATION:")
        print(f"   Accuracy: {accuracy*100:.2f}%")
        print(f"\n   Confusion Matrix:")
        print(f"   ┌─────────────────────────────────────┐")
        print(f"   │              Predicted              │")
        print(f"   │         Safe     Phishing           │")
        print(f"   │ Safe    {cm[0,0]:5d}     {cm[0,1]:5d}        │")
        print(f"   │ Phish   {cm[1,0]:5d}     {cm[1,1]:5d}        │")
        print(f"   └─────────────────────────────────────┘")
        
        print(f"\n   Classification Report:")
        print(classification_report(y_test, y_pred, target_names=['Safe', 'Phishing']))
        
        self.is_trained = True
        
        # Save model
        joblib.dump(self.classifier, 'phishing_model.pkl')
        joblib.dump(self.vectorizer, 'vectorizer.pkl')
        print("\n✅ Model saved to 'phishing_model.pkl'")
        
        return accuracy, cm
    
    def predict_email(self, email_text):
        """
        Predict if an email is phishing or safe
        """
        if not self.is_trained:
            # Try to load trained model
            if os.path.exists('phishing_model.pkl'):
                self.classifier = joblib.load('phishing_model.pkl')
                self.vectorizer = joblib.load('vectorizer.pkl')
                self.is_trained = True
            else:
                raise Exception("Model not trained. Run train_model() first.")
        
        # Extract features
        features = self.create_feature_vector(email_text)
        tfidf = self.vectorizer.transform([email_text]).toarray()
        combined = np.hstack([features.reshape(1, -1), tfidf])
        
        # Predict
        prediction = self.classifier.predict(combined)[0]
        probability = self.classifier.predict_proba(combined)[0]
        
        result = {
            'is_phishing': bool(prediction == 1),
            'confidence': max(probability) * 100,
            'confidence_safe': probability[0] * 100,
            'confidence_phishing': probability[1] * 100,
            'features': self.extract_features_from_text(email_text)
        }
        
        return result
    
    def create_sample_dataset(self):
        """
        Create a sample dataset of phishing and legitimate emails
        """
        data = []
        
        # Phishing Emails (label = 1)
        phishing_emails = [
            # Banking phishing
            "URGENT: Your Bank Account Has Been Suspended! Dear Customer, Your account has been temporarily suspended due to unusual activity. Please verify your account immediately at https://fake-bank.com/verify or your account will be closed within 24 hours.",
            
            # PayPal phishing
            "PayPal Account Alert! We noticed a suspicious login from an unrecognized device. Click here to secure your account: http://bit.ly/paypal-secure. Failure to verify will result in account limitation.",
            
            # Lottery scam
            "CONGRATULATIONS! You have won $1,000,000 in our online lottery! Click here to claim your prize: https://tinyurl.com/claim-prize. Hurry! This offer expires in 24 hours.",
            
            # Password reset scam
            "Action Required: Reset Your Password. Someone tried to access your account. To keep your account safe, please reset your password now: http://fake-login-page.com/reset",
            
            # IRS tax scam
            "IRS - Final Notice! You have unpaid taxes. Pay immediately to avoid legal action. Click here: https://irs-gov-tax.com/payment. Your SSN has been flagged.",
            
            # Netflix account suspension
            "Netflix - Your account has been suspended! Update your payment information: http://netflix-account-update.com. Your streaming will stop in 2 days.",
            
            # Amazon security alert
            "Amazon Security Alert! A new device logged into your account. Verify your identity: https://amazon-verify-now.com. If not verified within 12 hours, your account will be locked.",
            
            # Job offer phishing
            "URGENT: Job Offer - Work from home! Earn $5000/week. Click here to apply: http://bit.ly/work-from-home. Limited positions available!",
            
            # Tech support scam
            "Your computer is infected with a virus! Call our support immediately: 1-888-555-0123 or click here to scan: https://fake-antivirus.com/scan",
            
            # Delivery notification phishing
            "Your package cannot be delivered. Please update your shipping address: https://fedex-delivery-update.com. Your package will be returned if not updated.",
            
            # Instagram verification scam
            "Instagram Verification! You've been selected for verification. Apply here: http://instagram-verify-now.com. Limited time offer!",
            
            # Crypto wallet scam
            "Your crypto wallet needs verification! Secure your funds: https://wallet-verify.com. Unverified wallets will be suspended.",
        ]
        
        for email in phishing_emails:
            data.append({'email_text': email, 'label': 1})
        
        # Safe/Legitimate Emails (label = 0)
        safe_emails = [
            # Work email
            "Subject: Team Meeting Tomorrow Hi Team, Just a reminder about our meeting tomorrow at 10 AM in Conference Room B. Please come prepared with your weekly updates. Best regards, John",
            
            # Newsletter
            "Subject: Weekly Newsletter Hello! Here's our weekly newsletter with updates. Check out our new blog post about cybersecurity best practices. Unsubscribe link at bottom.",
            
            # Welcome email
            "Welcome to our platform! Thank you for signing up. Please complete your profile to get started. Contact support if you have any questions.",
            
            # Order confirmation
            "Your order #ORD-12345 has been confirmed. Expected delivery: 3-5 business days. Track your order at our official website. Thank you for shopping with us!",
            
            # Meeting invitation
            "You're invited to a meeting. Date: Friday, 3 PM. Location: Zoom link in calendar. Please RSVP by Wednesday.",
            
            # Colleague message
            "Hey, can you review the document I shared with you on Google Drive? Let me know what you think. Thanks!",
            
            # Security update (legitimate)
            "Important Security Update: We've released a new security patch. Please update your app to the latest version. Visit our official website for details.",
            
            # Password change confirmation
            "Your password was successfully changed. If you did not make this change, please contact support immediately.",
            
            # Bill/Invoice
            "Your monthly invoice is ready. Amount: $49.99. Due date: 15th of month. View and pay on our secure customer portal.",
            
            # Event reminder
            "Reminder: Cybersecurity Conference starts tomorrow at 9 AM. Your ticket QR code is attached. We look forward to seeing you!",
            
            # Team lunch invite
            "Team Lunch! Join us for lunch on Friday at 12:30 PM at The Italian Place. Please let me know if you can make it.",
            
            # Progress report
            "Weekly Progress Report: Completed 5 tasks, 3 in progress. No blockers. Sprint review scheduled for Friday.",
        ]
        
        for email in safe_emails:
            data.append({'email_text': email, 'label': 0})
        
        # Add variations with slight modifications
        import random
        
        # Add more phishing variations
        for i in range(20):
            base = random.choice(phishing_emails)
            modified = base.replace("click here", "click this link")
            data.append({'email_text': modified, 'label': 1})
        
        # Add more safe variations
        for i in range(20):
            base = random.choice(safe_emails)
            data.append({'email_text': base, 'label': 0})
        
        df = pd.DataFrame(data)
        df = df.sample(frac=1).reset_index(drop=True)  # Shuffle
        
        return df
    
    def test_custom_email(self, email_text):
        """
        Test a custom email and print results
        """
        result = self.predict_email(email_text)
        
        print("\n" + "="*60)
        print("📧 EMAIL ANALYSIS RESULT")
        print("="*60)
        print(f"\n📝 Email Content Preview:")
        print(f"   {email_text[:200]}...")
        
        print(f"\n🎯 Prediction:")
        if result['is_phishing']:
            print(f"   🔴 PHISHING DETECTED! (Confidence: {result['confidence']:.1f}%)")
        else:
            print(f"   🟢 SAFE EMAIL (Confidence: {result['confidence']:.1f}%)")
        
        print(f"\n📊 Confidence Breakdown:")
        print(f"   Safe: {result['confidence_safe']:.1f}%")
        print(f"   Phishing: {result['confidence_phishing']:.1f}%")
        
        print(f"\n🔍 Features Detected:")
        features = result['features']
        print(f"   • Has URL: {'Yes' if features['has_url'] else 'No'}")
        print(f"   • URL Count: {features['url_count']}")
        print(f"   • Urgent Words: {'Yes' if features['urgent_words'] else 'No'}")
        print(f"   • Account Related: {'Yes' if features['account_words'] else 'No'}")
        print(f"   • Prize/Winning: {'Yes' if features['prize_words'] else 'No'}")
        print(f"   • Payment Related: {'Yes' if features['payment_words'] else 'No'}")
        print(f"   • Personal Info Request: {'Yes' if features['personal_info'] else 'No'}")
        print(f"   • Has Attachment: {'Yes' if features['has_attachment'] else 'No'}")
        
        return result


# ============================================================
# MAIN EXECUTION
# ============================================================

if __name__ == "__main__":
    # Create detector instance
    detector = PhishingEmailDetector()
    
    # Train the model
    detector.train_model()
    
    # Test with example emails
    print("\n" + "="*60)
    print("🧪 TESTING WITH EXAMPLES")
    print("="*60)
    
    # Test a phishing email
    phishing_test = "URGENT! Your bank account has been compromised! Click here to verify: https://fake-bank.com/verify"
    detector.test_custom_email(phishing_test)
    
    # Test a safe email
    safe_test = "Hi team, please find attached the weekly report. Let me know if you have any questions. Best, Manager"
    detector.test_custom_email(safe_test)
    
    print("\n" + "="*60)
    print("✅ Model training complete!")
    print("="*60)