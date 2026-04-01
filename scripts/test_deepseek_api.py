"""Test DeepSeek API connection.

Usage:
    python scripts/test_deepseek_api.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(project_root, ".env"))

api_key = os.environ.get("DEEPSEEK_API_KEY")
if not api_key:
    print("ERROR: DEEPSEEK_API_KEY not found in .env")
    print("Please create .env in project root with: DEEPSEEK_API_KEY=sk-xxxxx")
    sys.exit(1)

print(f"API Key found: {api_key[:8]}...{api_key[-4:]}")
print("Testing DeepSeek API...\n")

from agentos.utils import call_model

try:
    response = call_model(
        messages=[{"role": "user", "content": "你好，请用一句话回复。"}],
        api_key=api_key
    )
    print(f"SUCCESS")
    print(f"Response: {response}")
except Exception as e:
    print(f"FAILED")
    print(f"Error: {e}")
