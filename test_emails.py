#!/usr/bin/env python3
"""
Interactive Email Testing Script
Test your own emails against the trained model
"""

from phishing_detector import PhishingEmailDetector
import joblib
import os

def main():
    print("="*60)
    print("📧 PHISHING EMAIL DETECTOR - INTERACTIVE TEST")
    print("="*60)
    
    # Load model
    detector = PhishingEmailDetector()
    
    if os.path.exists('phishing_model.pkl'):
        detector.classifier = joblib.load('phishing_model.pkl')
        detector.vectorizer = joblib.load('vectorizer.pkl')
        detector.is_trained = True
        print("✅ Model loaded successfully!")
    else:
        print("⚠️ Model not found. Training new model...")
        detector.train_model()
    
    while True:
        print("\n" + "-"*60)
        print("OPTIONS:")
        print("1. Test a custom email")
        print("2. Test example emails")
        print("3. Exit")
        
        choice = input("\nEnter your choice (1-3): ").strip()
        
        if choice == '1':
            print("\n📝 Enter the email content (or type 'done' to finish):")
            lines = []
            while True:
                line = input()
                if line.lower() == 'done':
                    break
                lines.append(line)
            email_text = ' '.join(lines)
            
            if email_text:
                detector.test_custom_email(email_text)
            else:
                print("❌ No email content entered!")
        
        elif choice == '2':
            print("\n📋 Example Emails:")
            print("1. Phishing: 'URGENT! Your account will be closed! Click here: http://fake.com'")
            print("2. Phishing: 'You won $1,000,000! Claim now: https://bit.ly/claim'")
            print("3. Safe: 'Meeting tomorrow at 10 AM in Conference Room'")
            print("4. Safe: 'Your order #12345 has been shipped'")
            
            example = input("\nSelect example (1-4): ").strip()
            
            examples = {
                '1': "URGENT! Your PayPal account has been limited. Click here to verify: http://fake-paypal.com/verify",
                '2': "CONGRATULATIONS! You've won $1,000,000 in our lottery! Click here to claim: https://bit.ly/claim-prize",
                '3': "Hi team, meeting at 10 AM tomorrow to discuss the project. Please bring your updates.",
                '4': "Your order #ORD-12345 has been shipped and will arrive in 3-5 business days."
            }
            
            if example in examples:
                detector.test_custom_email(examples[example])
            else:
                print("❌ Invalid choice!")
        
        elif choice == '3':
            print("\n👋 Goodbye! Stay safe from phishing!")
            break
        
        else:
            print("❌ Invalid choice!")

if __name__ == "__main__":
    main()