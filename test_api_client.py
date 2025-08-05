#!/usr/bin/env python3
"""
API client test script to demonstrate calling the FastAPI /hackrx/run endpoint
with the provided Indian Constitution document and legal questions.
"""

import asyncio
import json
import logging
from typing import List, Dict, Any
import httpx
import sys
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# API Configuration
API_BASE_URL = "http://localhost:8000"  # Adjust if running on different host/port
BEARER_TOKEN = "your-bearer-token-here"  # Replace with actual token if authentication is enabled

# Document and questions from the user's request
DOCUMENT_URL = "https://hackrx.blob.core.windows.net/assets/indian_constitution.pdf?sv=2023-01-03&st=2025-07-28T06%3A42%3A00Z&se=2026-11-29T06%3A42%3A00Z&sr=b&sp=r&sig=5Gs%2FOXqP3zY00lgciu4BZjDV5QjTDIx7fgnfdz6Pu24%3D"

QUESTIONS = [
    "If my car is stolen, what case will it be in law?",
    "If I am arrested without a warrant, is that legal?",
    "If someone denies me a job because of my caste, is that allowed?",
    "If the government takes my land for a project, can I stop it?",
    "If my child is forced to work in a factory, is that legal?",
    "If I am stopped from speaking at a protest, is that against my rights?",
    "If a religious place stops me from entering because I'm a woman, is that constitutional?",
    "If I change my religion, can the government stop me?",
    "If the police torture someone in custody, what right is being violated?",
    "If I'm denied admission to a public university because I'm from a backward community, can I do something?"
]

class APIClient:
    """Client for testing the FastAPI application."""
    
    def __init__(self, base_url: str, bearer_token: str = None):
        self.base_url = base_url
        self.bearer_token = bearer_token
        self.client = httpx.AsyncClient(timeout=300.0)  # 5 minute timeout for document processing
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.client.aclose()
    
    async def health_check(self) -> Dict[str, Any]:
        """Check if the API is healthy."""
        try:
            response = await self.client.get(f"{self.base_url}/health")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            raise
    
    async def process_document_questions(self, document_url: str, questions: List[str]) -> Dict[str, Any]:
        """Call the /hackrx/run endpoint to process document and questions."""
        payload = {
            "documents": document_url,
            "questions": questions
        }
        
        headers = {}
        if self.bearer_token:
            headers["Authorization"] = f"Bearer {self.bearer_token}"
        
        try:
            logger.info(f"Sending request to {self.base_url}/hackrx/run")
            logger.info(f"Document: {document_url}")
            logger.info(f"Questions: {len(questions)}")
            
            response = await self.client.post(
                f"{self.base_url}/hackrx/run",
                json=payload,
                headers=headers
            )
            
            response.raise_for_status()
            return response.json()
            
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error {e.response.status_code}: {e.response.text}")
            raise
        except Exception as e:
            logger.error(f"Request failed: {e}")
            raise

async def main():
    """Main function to test the API."""
    logger.info("Starting FastAPI Client Test")
    logger.info("=" * 50)
    
    async with APIClient(API_BASE_URL, BEARER_TOKEN) as client:
        try:
            # First, check if the API is healthy
            logger.info("Checking API health...")
            health = await client.health_check()
            logger.info(f"API Health: {health}")
            
            # Process the document and questions
            logger.info("Processing document and questions...")
            result = await client.process_document_questions(DOCUMENT_URL, QUESTIONS)
            
            # Display results
            logger.info("Processing completed successfully!")
            logger.info("=" * 50)
            logger.info("RESULTS:")
            logger.info("=" * 50)
            
            print(f"\nDocument: {DOCUMENT_URL}")
            print(f"Status: Success")
            print(f"\nQuestions and Answers:")
            print("=" * 50)
            
            for i, (question, answer) in enumerate(zip(QUESTIONS, result["answers"]), 1):
                print(f"\nQ{i}: {question}")
                print(f"A{i}: {answer}")
                print("-" * 50)
            
            # Save results to JSON file
            output_file = "api_test_results.json"
            result_data = {
                "document_url": DOCUMENT_URL,
                "questions": QUESTIONS,
                "answers": result["answers"],
                "processing_status": "completed"
            }
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(result_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Results saved to {output_file}")
            
        except httpx.ConnectError:
            logger.error("Failed to connect to the API. Make sure the FastAPI server is running.")
            logger.error("You can start it with: python app.py")
            return 1
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 401:
                logger.error("Authentication failed. Check your bearer token.")
            else:
                logger.error(f"API request failed with status {e.response.status_code}")
            return 1
        except Exception as e:
            logger.error(f"Error during API test: {e}")
            return 1
    
    return 0

def print_usage():
    """Print usage instructions."""
    print("\nFastAPI Client Test Usage:")
    print("=" * 50)
    print("1. Start the FastAPI server:")
    print("   python app.py")
    print("\n2. In another terminal, run this client:")
    print("   python test_api_client.py")
    print("\n3. Make sure you have the required API keys configured in .env")
    print("   or the server will fail to start.")
    print("\nNote: This script expects the server to be running on http://localhost:8000")
    print("Modify API_BASE_URL in this script if using a different address.")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] in ["-h", "--help"]:
        print_usage()
        sys.exit(0)
    
    exit_code = asyncio.run(main())
    
    if exit_code != 0:
        print_usage()
    
    sys.exit(exit_code)
