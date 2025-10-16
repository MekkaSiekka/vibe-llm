#!/usr/bin/env python3
"""Simple AI Detection CLI"""

import sys
import requests

def main():
    if len(sys.argv) < 2:
        print("Usage: python detect_simple.py \"Your text here\"")
        print("Example: python detect_simple.py \"Hello world\"")
        sys.exit(1)
    
    text = sys.argv[1]
    
    try:
        response = requests.post(
            'http://localhost:8000/detect/ai/simple',
            params={'text': text},
            timeout=30
        )
        
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            
            status = "🤖 AI Generated" if result.get('is_ai_generated', False) else "👤 Human Written"
            confidence = result.get('confidence', 0) * 100
            ai_prob = result.get('ai_probability', 0) * 100
            
            print(f"Result: {status}")
            print(f"Confidence: {confidence:.1f}%")
            print(f"AI Probability: {ai_prob:.1f}%")
            print(f"Model: {result.get('model', 'Unknown')}")
            print(f"Processing Time: {result.get('processing_time', 0):.3f}s")
        else:
            print(f"Error: {response.text}")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
