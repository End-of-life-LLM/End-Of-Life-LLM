#!/usr/bin/env python3
"""
Main entry point for the Cloud LLM Model system.
Demonstrates how to use the RAG system within the larger application.
"""

import os
import argparse
import logging
from dotenv import load_dotenv
from Core.controller import Controller

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def setup_environment():
    """Load environment variables and API keys."""
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.warning("OPENAI_API_KEY not found in environment variables.")
        api_key = input("Please enter your OpenAI API key: ")
    
    return api_key


def demo_rag_system(controller, args):
    """Demonstrate RAG system functionality."""
    logger.info("Running RAG system demo...")
    
    if args.index_files:
        for file_path in args.index_files:
            logger.info(f"Indexing file: {file_path}")
            chunks = controller.rag_controller.index_text_file(file_path)
            logger.info(f"Indexed {chunks} chunks from {file_path}")
        
        
        controller.rag_controller.save_index("vector_index")
        logger.info("Index saved to 'vector_index' directory")
    
    
    if os.path.exists("vector_index") and not controller.rag_controller.is_loaded():
        logger.info("Loading existing index from 'vector_index' directory")
        controller.rag_controller.load_index("vector_index")
    
    
    while True:
        question = input("\nEnter your question (or 'exit' to quit): ")
        if question.lower() in ["exit", "quit", "q"]:
            break
        
        # Process the query
        result = controller.rag_controller.query(question, k=5)
        
        print("\n" + "="*50)
        print("Answer:")
        print(result["answer"])
        print("="*50)
        
        # Show sources if requested
        if args.show_sources:
            print("\nSources:")
            for i, source in enumerate(result["source_documents"]):
                print(f"\nSource {i+1} (score: {source['score']:.4f}):")
                print(f"From: {source['metadata'].get('source', 'Unknown')}")
                print(f"Chunk ID: {source['metadata'].get('chunk_id', 'Unknown')}")
                print(f"Text snippet: {source['text'][:150]}...")


def demo_conversation(controller):
    """Demonstrate simple conversation with the model."""
    logger.info("Starting conversation with model...")
    print("\nStarting conversation. Type 'exit' to quit.")
    
    while controller.model.message_manager():
        pass
    
    print("Conversation ended.")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Cloud LLM Model System")
    parser.add_argument("--rag", action="store_true", help="Run RAG system demo")
    parser.add_argument("--converse", action="store_true", help="Start conversation with model")
    parser.add_argument("--index-files", nargs="+", help="Files to index for RAG system")
    parser.add_argument("--show-sources", action="store_true", help="Show sources in RAG results")
    
    args = parser.parse_args()
    
    # Setup environment
    api_key = setup_environment()
    
    # Create controller
    controller = Controller(api_key=api_key)
    
    if args.rag:
        demo_rag_system(controller, args)
    elif args.converse:
        demo_conversation(controller)
    else:
        print("Please specify --rag or --converse to run a demo.")
        print("Example usage:")
        print("  python main.py --rag --index-files document1.txt document2.txt")
        print("  python main.py --rag --show-sources")
        print("  python main.py --converse")


if __name__ == "__main__":
    main()