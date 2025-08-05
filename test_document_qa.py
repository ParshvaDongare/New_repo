#!/usr/bin/env python3
"""
Test script to simulate document processing and question answering
for the provided Indian Constitution document and legal questions.
"""

import asyncio
import json
import logging
from typing import List, Dict, Any
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Mock document URL and questions from the user's request
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

class MockDocumentProcessor:
    """Mock document processor that simulates the FastAPI application logic."""
    
    def __init__(self):
        self.processed_docs = set()
    
    async def process_document_and_questions(self, doc_url: str, questions: List[str]) -> Dict[str, Any]:
        """Simulate the main /hackrx/run endpoint functionality."""
        logger.info(f"Processing document: {doc_url}")
        logger.info(f"Questions to answer: {len(questions)}")
        
        # Simulate document ID generation
        import hashlib
        doc_id = hashlib.md5(doc_url.encode()).hexdigest()
        
        # Simulate document processing
        if doc_id not in self.processed_docs:
            logger.info("Simulating document ingestion...")
            await self._simulate_document_ingestion(doc_url, doc_id)
            self.processed_docs.add(doc_id)
        else:
            logger.info("Document already processed, using cached version")
        
        # Process questions
        logger.info("Processing questions...")
        answers = []
        for i, question in enumerate(questions, 1):
            logger.info(f"Processing question {i}/{len(questions)}: {question}")
            answer = await self._process_question(question, doc_id, doc_url)
            answers.append(answer)
            
            # Simulate delay between questions
            await asyncio.sleep(0.5)
        
        return {
            "document_id": doc_id,
            "document_url": doc_url,
            "questions": questions,
            "answers": answers,
            "processing_status": "completed"
        }
    
    async def _simulate_document_ingestion(self, doc_url: str, doc_id: str):
        """Simulate document download, parsing, chunking, and indexing."""
        logger.info("Step 1: Downloading and extracting document pages...")
        await asyncio.sleep(2)  # Simulate download time
        
        logger.info("Step 2: Chunking document content...")
        await asyncio.sleep(1)  # Simulate chunking time
        
        logger.info("Step 3: Generating embeddings...")
        await asyncio.sleep(3)  # Simulate embedding generation
        
        logger.info("Step 4: Indexing to vector database...")
        await asyncio.sleep(1)  # Simulate indexing time
        
        logger.info("Document ingestion completed successfully")
    
    async def _process_question(self, question: str, doc_id: str, doc_url: str) -> str:
        """Process a single question using pattern matching similar to the real app."""
        q_lower = question.lower()
        
        # These responses are based on the actual logic in app.py
        if any(term in q_lower for term in ["stolen", "theft", "car", "vehicle"]):
            return "This would be treated as theft under criminal law (Indian Penal Code), not constitutional law. You should file a police report (FIR) for investigation and recovery."
        
        elif any(term in q_lower for term in ["arrest", "warrant", "police"]) and "without" in q_lower:
            return "Arrest without warrant is legal for cognizable offences, but Article 22 guarantees that arrested persons must be informed of grounds and produced before magistrate within 24 hours."
        
        elif any(term in q_lower for term in ["job", "employment", "caste", "discrimination"]):
            return "Article 15 prohibits discrimination based on caste, and Article 16 ensures equal opportunity in public employment. Caste-based job denial is unconstitutional."
        
        elif any(term in q_lower for term in ["religion", "change", "convert"]) and "government" in q_lower:
            return "Article 25 guarantees freedom of conscience and right to freely profess, practice, and propagate religion. Government cannot stop religious conversion."
        
        elif any(term in q_lower for term in ["land", "acquisition"]) and "government" in q_lower:
            return "Government can acquire land for public use under eminent domain (Article 300A), but must follow legal procedures and provide compensation. You can challenge the process in court."
        
        elif any(term in q_lower for term in ["child", "work", "factory", "labor"]):
            return "Article 24 explicitly prohibits employment of children below 14 years in factories, mines, or hazardous employment. Child labor is illegal."
        
        elif any(term in q_lower for term in ["protest", "speech", "speak", "assembly"]):
            return "Article 19(1)(a) and 19(1)(b) guarantee freedom of speech and peaceful assembly. Stopping you without valid legal reason violates these fundamental rights."
        
        elif any(term in q_lower for term in ["religious", "temple", "woman", "enter", "deny"]):
            return "Article 15 prohibits discrimination on grounds of sex. Denying women entry to religious places is unconstitutional, as upheld by Supreme Court judgments."
        
        elif any(term in q_lower for term in ["torture", "police", "custody", "beat"]):
            return "Police torture violates Article 21 (right to life and personal liberty), including freedom from cruel, inhuman, or degrading treatment. Custodial torture is unconstitutional."
        
        elif any(term in q_lower for term in ["university", "admission", "backward", "community", "deny"]):
            return "Article 15(4) allows special provisions for backward classes, but Article 29(2) prohibits discrimination in state-funded institutions. You can challenge discriminatory admission practices."
        
        else:
            # Fallback for unrecognized questions
            return "Based on the Indian Constitution, this question requires specific legal analysis. Please consult the relevant constitutional provisions or seek legal advice for a detailed answer."

async def main():
    """Main function to run the test."""
    logger.info("Starting Document Q&A Test")
    logger.info("=" * 50)
    
    processor = MockDocumentProcessor()
    
    try:
        # Process the document and questions
        result = await processor.process_document_and_questions(DOCUMENT_URL, QUESTIONS)
        
        # Display results
        logger.info("Processing completed successfully!")
        logger.info("=" * 50)
        logger.info("RESULTS:")
        logger.info("=" * 50)
        
        print(f"\nDocument: {result['document_url']}")
        print(f"Document ID: {result['document_id']}")
        print(f"Status: {result['processing_status']}")
        print(f"\nQuestions and Answers:")
        print("=" * 50)
        
        for i, (question, answer) in enumerate(zip(result['questions'], result['answers']), 1):
            print(f"\nQ{i}: {question}")
            print(f"A{i}: {answer}")
            print("-" * 50)
        
        # Also save results to JSON file
        output_file = "test_results.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Results saved to {output_file}")
        
    except Exception as e:
        logger.error(f"Error during processing: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
