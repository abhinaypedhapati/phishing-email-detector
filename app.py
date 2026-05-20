#!/usr/bin/env python3
import sys
import traceback

print("Starting application...", file=sys.stderr)
print("Python version:", sys.version, file=sys.stderr)

try:
    from flask import Flask, render_template, request, jsonify
    print("Flask imported successfully", file=sys.stderr)
except Exception as e:
    print(f"Error importing Flask: {e}", file=sys.stderr)
    traceback.print_exc()
    sys.exit(1)

try:
    import re
    print("Re imported successfully", file=sys.stderr)
except Exception as e:
    print(f"Error importing re: {e}", file=sys.stderr)
    sys.exit(1)

app = Flask(__name__)
print("Flask app created", file=sys.stderr)

def detect_phishing(email_text):
    """Simple rule-based phishing detection"""
    if not isinstance(email_text, str):
        email_text = str(email_text)
    
    text_lower = email_text.lower()
    
    features = {
        'has_url': bool(re.search(r'https?://|www\.', text_lower)),
        'urgent_words': bool(re.search(r'urgent|immediately|asap|verify now|action required', text_lower)),
        'account_words': bool(re.search(r'account|password|verify|update|confirm|security', text_lower)),
        'prize_words': bool(re.search(r'winner|congratulations|prize|lottery|won', text_lower)),
        'payment_words': bool(re.search(r'payment|credit card|bank|transfer|invoice', text_lower)),
        'personal_info': bool(re.search(r'ssn|social security|tax id|driver license', text_lower))
    }
    
    phishing_score = sum(1 for v in features.values() if v)
    
    if phishing_score >= 3:
        is_phishing = True
        confidence = min(95, 70 + phishing_score * 5)
    elif phishing_score >= 2:
        is_phishing = True
        confidence = 60 + phishing_score * 5
    else:
        is_phishing = False
        confidence = max(85, 100 - phishing_score * 10)
    
    return {
        'is_phishing': is_phishing,
        'confidence': confidence,
        'features': features
    }

@app.route('/')
def index():
    try:
        print("Rendering index template...", file=sys.stderr)
        return render_template('index.html')
    except Exception as e:
        print(f"Error rendering template: {e}", file=sys.stderr)
        traceback.print_exc()
        return f"Error: {str(e)}", 500

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        email = data.get('email', '')
        if not email:
            return jsonify({'error': 'No email provided'}), 400
        result = detect_phishing(email)
        return jsonify(result)
    except Exception as e:
        print(f"Error in predict: {e}", file=sys.stderr)
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health():
    return jsonify({'status': 'ok', 'message': 'PhishShield is running!'})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"Starting server on port {port}...", file=sys.stderr)
    app.run(debug=False, host='0.0.0.0', port=port)