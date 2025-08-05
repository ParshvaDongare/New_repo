#!/usr/bin/env python3
"""
Simple test script to check Groq API connectivity
"""

import os
import sys
from dotenv import load_dotenv
from groq import Groq

# Load environment variables
load_dotenv()

def test_groq_keys():
    """Test all Groq API keys"""
    keys = []
    for i in range(1, 6):
        key = os.getenv(f'GROQ_API_KEYS_{i}')
        if key and key.strip():
            keys.append(key.strip())
    
    if not keys:
        print("❌ No Groq API keys found!")
        print("Please set GROQ_API_KEYS_1 through GROQ_API_KEYS_5 in your .env file")
        return False
    
    print(f"✅ Found {len(keys)} Groq API keys")
    
    # Test each key
    successful_keys = 0
    for i, key in enumerate(keys, 1):
        try:
            client = Groq(api_key=key, timeout=10.0, max_retries=0)
            
            response = client.chat.completions.create(
                model="llama3-8b-8192",
                messages=[
                    {"role": "user", "content": "Say 'Hello, this is a test!' and nothing else."}
                ],
                temperature=0,
                max_tokens=50
            )
            
            content = response.choices[0].message.content
            tokens = response.usage.total_tokens if hasattr(response, 'usage') else 0
            
            if content and content.strip():
                print(f"✅ Key {i}: SUCCESS - '{content.strip()}' ({tokens} tokens)")
                successful_keys += 1
            else:
                print(f"❌ Key {i}: EMPTY RESPONSE")
                
        except Exception as e:
            print(f"❌ Key {i}: ERROR - {e}")
    
    print(f"\n📊 Summary: {successful_keys}/{len(keys)} keys working")
    return successful_keys > 0

if __name__ == "__main__":
    print("🔍 Testing Groq API connectivity...")
    print("=" * 50)
    
    success = test_groq_keys()
    
    if success:
        print("\n✅ At least one Groq API key is working!")
        sys.exit(0)
    else:
        print("\n❌ No working Groq API keys found!")
        sys.exit(1)
