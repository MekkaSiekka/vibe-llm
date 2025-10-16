#!/usr/bin/env python3
"""
AI Detection CLI Tool

Simple command-line interface for testing AI detection with custom text.
"""

import argparse
import requests
import sys
import json
from typing import Optional


def detect_ai_text(text: str, url: str = "http://localhost:8000", detector: Optional[str] = None, verbose: bool = False) -> dict:
    """Detect if text is AI-generated using the API."""
    try:
        if verbose:
            print(f"🔍 Analyzing text: {text[:100]}{'...' if len(text) > 100 else ''}")
            print(f"📡 Using endpoint: {url}/detect/ai/simple")
        
        params = {"text": text}
        if detector:
            params["detector"] = detector
        
        response = requests.post(
            f"{url}/detect/ai/simple",
            params=params,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            return result
        else:
            return {"error": f"HTTP {response.status_code}: {response.text}"}
            
    except requests.exceptions.ConnectionError:
        return {"error": "Connection failed. Is the service running on the specified URL?"}
    except requests.exceptions.Timeout:
        return {"error": "Request timed out. The detection is taking too long."}
    except Exception as e:
        return {"error": f"Unexpected error: {str(e)}"}


def format_result(result: dict, verbose: bool = False) -> str:
    """Format the detection result for display."""
    if "error" in result:
        return f"❌ Error: {result['error']}"
    
    # Main result
    status = "🤖 AI Generated" if result.get('is_ai_generated', False) else "👤 Human Written"
    confidence = result.get('confidence', 0) * 100
    ai_prob = result.get('ai_probability', 0) * 100
    
    output = f"{status}\n"
    output += f"Confidence: {confidence:.1f}%\n"
    output += f"AI Probability: {ai_prob:.1f}%"
    
    if verbose:
        output += f"\nModel: {result.get('model', 'Unknown')}"
        output += f"\nProcessing Time: {result.get('processing_time', 0):.3f}s"
        output += f"\nMethod: {result.get('method', 'Unknown')}"
    
    return output


def main():
    parser = argparse.ArgumentParser(
        description="AI Detection CLI Tool - Test if text is AI-generated",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s "Hello world, this is a test"
  %(prog)s "AI has revolutionized industries" --verbose
  %(prog)s "My text here" --url http://localhost:8001
  %(prog)s "Test text" --detector roberta --json
        """
    )
    
    parser.add_argument("text", help="Text to analyze for AI generation")
    parser.add_argument("--url", default="http://localhost:8000", 
                       help="Base URL of the AI detection service (default: http://localhost:8000)")
    parser.add_argument("--detector", help="Specific detector to use")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Show detailed information")
    parser.add_argument("--json", action="store_true",
                       help="Output raw JSON response")
    parser.add_argument("--quiet", "-q", action="store_true",
                       help="Only show the result (AI/Human)")
    
    args = parser.parse_args()
    
    # Detect AI text
    result = detect_ai_text(args.text, args.url, args.detector, args.verbose)
    
    # Output results
    if args.json:
        print(json.dumps(result, indent=2))
    elif args.quiet:
        if "error" in result:
            print("ERROR")
            sys.exit(1)
        else:
            print("AI" if result.get('is_ai_generated', False) else "HUMAN")
    else:
        print(format_result(result, args.verbose))
    
    # Exit with appropriate code
    if "error" in result:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
