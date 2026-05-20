#!/usr/bin/env python3
"""
Phishing Email Detection - Web Application
Flask web interface for the ML model
"""

from flask import Flask, render_template, request, jsonify
from phishing_detector import PhishingEmailDetector
import joblib
import os

app = Flask(__name__)

# Load model
detector = PhishingEmailDetector()
if os.path.exists('phishing_model.pkl'):
    detector.classifier = joblib.load('phishing_model.pkl')
    detector.vectorizer = joblib.load('vectorizer.pkl')
    detector.is_trained = True
    print("✅ Model loaded successfully!")
else:
    print("⚠️ Training new model...")
    detector.train_model()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/predict', methods=['POST'])
def predict():
    data = request.get_json()
    email_text = data.get('email', '')
    
    if not email_text:
        return jsonify({'error': 'No email content provided'}), 400
    
    result = detector.predict_email(email_text)
    
    return jsonify({
        'is_phishing': result['is_phishing'],
        'confidence': result['confidence'],
        'confidence_safe': result['confidence_safe'],
        'confidence_phishing': result['confidence_phishing'],
        'features': result['features']
    })

@app.route('/api/batch_predict', methods=['POST'])
def batch_predict():
    data = request.get_json()
    emails = data.get('emails', [])
    
    results = []
    for email in emails:
        result = detector.predict_email(email)
        results.append({
            'email': email[:100] + '...',
            'is_phishing': result['is_phishing'],
            'confidence': result['confidence']
        })
    
    return jsonify({'results': results})

if __name__ == '__main__':
    print("="*60)
    print("🔒 PHISHING EMAIL DETECTOR - WEB INTERFACE")
    print("="*60)
    print("Starting server at: http://localhost:5000")
    print("Press Ctrl+C to stop")
    print("="*60)
    app.run(debug=True, host='0.0.0.0', port=5000)