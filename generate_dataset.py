#!/usr/bin/env python3
"""
Generate a larger, more realistic dataset for training
"""

import pandas as pd
import random
import re

def generate_phishing_email():
    templates = [
        "URGENT: Your {company} account has been suspended! Click here to verify: {url}",
        "Dear Customer, We noticed suspicious activity on your account. Verify now: {url}",
        "CONGRATULATIONS! You've won {prize}! Claim your prize: {url}",
        "{company} Security Alert: Your password will expire. Update here: {url}",
        "Your account will be closed! Action required: {url}",
        "You have a pending refund of ${amount}. Claim it: {url}",
        "IRS Notice: You owe ${amount} in taxes. Pay immediately: {url}",
        "Your {company} subscription has expired. Renew now: {url}",
        "Someone logged into your account from {location}. Verify: {url}",
        "Payment failed! Update your billing info: {url}"
    ]
    
    companies = ["PayPal", "Netflix", "Amazon", "Bank of America", "Chase", "Wells Fargo", "Apple", "Microsoft", "Google", "Facebook"]
    urls = ["http://bit.ly/verify", "https://tinyurl.com/secure", "http://fake-login.com", "https://verify-account.net", "http://secure-update.org"]
    prizes = ["$1,000,000", "iPhone 15", "MacBook Pro", "$500 Gift Card", "All-Expenses Paid Trip"]
    locations = ["New York", "London", "Tokyo", "Unknown Device", "Russia", "China"]
    
    template = random.choice(templates)
    email = template.format(
        company=random.choice(companies),
        url=random.choice(urls),
        prize=random.choice(prizes),
        amount=random.randint(100, 5000),
        location=random.choice(locations)
    )
    
    # Add urgency indicators
    urgency_words = ["URGENT", "IMMEDIATE", "ASAP", "ACTION REQUIRED", "LAST CHANCE"]
    if random.random() > 0.5:
        email = random.choice(urgency_words) + "! " + email
    
    return email

def generate_safe_email():
    templates = [
        "Subject: Meeting Tomorrow\nHi team, Reminder about our meeting at {time} in {room}. Please come prepared.",
        "Your order #{order} has been confirmed. Delivery in {days} days.",
        "Welcome to our platform! Complete your profile to get started.",
        "Weekly Newsletter: Check out our latest blog post about {topic}.",
        "{sender} shared a document with you on Google Drive: {doc}",
        "Your password was successfully changed. If this wasn't you, contact support.",
        "Thank you for your purchase! Your receipt is attached.",
        "Project Update: {progress} tasks completed this week.",
        "Invoice #{invoice} is ready for payment. Due date: {date}",
        "Team Lunch on Friday at {time}. Please RSVP."
    ]
    
    topics = ["Cybersecurity", "Machine Learning", "Web Development", "Cloud Computing", "AI"]
    senders = ["John", "Sarah", "Mike", "Lisa", "David", "Anna"]
    docs = ["Q4_Report.pdf", "Project_Plan.docx", "Budget.xlsx", "Presentation.pptx"]
    progresses = ["5/8", "12/15", "3/4", "90%", "Almost complete"]
    
    template = random.choice(templates)
    email = template.format(
        time=f"{random.randint(9,17)}:00 AM",
        room=random.choice(["A", "B", "C", "Main Conference"]),
        order=random.randint(10000, 99999),
        days=random.randint(1,7),
        topic=random.choice(topics),
        sender=random.choice(senders),
        doc=random.choice(docs),
        progress=random.choice(progresses),
        invoice=random.randint(1000, 9999),
        date=f"{random.randint(1,28)}/{random.randint(1,12)}/2024"
    )
    
    return email

def generate_dataset(num_samples=2000):
    """
    Generate a balanced dataset
    """
    data = []
    
    # Generate phishing emails
    for _ in range(num_samples // 2):
        email = generate_phishing_email()
        data.append({'email_text': email, 'label': 1})
    
    # Generate safe emails
    for _ in range(num_samples // 2):
        email = generate_safe_email()
        data.append({'email_text': email, 'label': 0})
    
    # Shuffle
    random.shuffle(data)
    
    df = pd.DataFrame(data)
    df.to_csv('phishing_emails_dataset.csv', index=False)
    print(f"✅ Generated {len(df)} emails")
    print(f"   Phishing: {df['label'].sum()} emails")
    print(f"   Safe: {len(df) - df['label'].sum()} emails")
    
    return df

if __name__ == "__main__":
    generate_dataset(2000)