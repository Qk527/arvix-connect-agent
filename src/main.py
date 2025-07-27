#!/usr/bin/env python3
"""
Simple CLI interface for the Arxiv Learning Agent.

This script provides an interactive command-line interface to:
1. Learn about topics by downloading Arxiv papers
2. Ask questions about stored knowledge  
3. Explore what topics the system knows about
"""

import os
import sys
from dotenv import load_dotenv

# Add src to path so we can import the agent
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from agent import ArxivLearningAgent


def print_banner():
    """Print a welcome banner."""
    print("=" * 60)
    print("🤖 Arxiv Learning Agent")
    print("=" * 60)
    print("A simple agent to learn from scientific papers and answer questions!")
    print("\nAvailable commands:")
    print("  learn <topic>     - Learn about a topic from Arxiv papers")
    print("  ask <question>    - Ask a question about stored knowledge")
    print("  explore           - See what topics the system knows about")
    print("  help              - Show this help message")
    print("  quit/exit         - Exit the program")
    print("=" * 60)


def print_learning_result(result):
    """Print the result of a learning operation."""
    if result["success"]:
        print(f"✅ {result['message']}")
        print(f"📄 Processed {result['papers_processed']} papers")
        print(f"📝 Created {result['chunks_created']} text chunks")
        
        if result.get("papers"):
            print("\n📚 Papers learned:")
            for i, paper in enumerate(result["papers"], 1):
                print(f"  {i}. {paper['title']}")
                print(f"     Authors: {paper['authors']}")
                print(f"     ArXiv ID: {paper['arxiv_id']}")
                print(f"     Published: {paper['published']}")
                print()
    else:
        print(f"❌ {result['message']}")


def print_answer_result(result):
    """Print the result of a question answering operation."""
    print(f"🤔 Answer: {result['answer']}")
    
    if result.get("sources"):
        print("\n📖 Sources:")
        for i, source in enumerate(result["sources"], 1):
            print(f"  {i}. {source['title']}")
            print(f"     Authors: {source['authors']}")
            print(f"     ArXiv ID: {source['arxiv_id']}")
            print(f"     Relevant text: {source['relevance_chunk']}")
            print()


def print_exploration_result(result):
    """Print the result of knowledge exploration."""
    print(f"🔍 {result['message']}")
    
    if result['total_documents'] > 0:
        print(f"\n📊 Statistics:")
        print(f"  • Total documents: {result['total_documents']}")
        print(f"  • Total papers: {result['total_papers']}")
        print(f"  • Topics learned: {len(result['topics'])}")
        
        if result.get("topics"):
            print(f"\n🎯 Topics:")
            for topic in result["topics"]:
                print(f"  • {topic}")
        
        if result.get("papers"):
            print(f"\n📚 Papers in knowledge base:")
            for i, paper in enumerate(result["papers"], 1):
                print(f"  {i}. {paper['title']}")
                print(f"     Topic: {paper['search_topic']}")
                print(f"     ArXiv ID: https://arxiv.org/abs/{paper['title'].split('/')[-1] if '/' in str(paper['title']) else 'unknown'}")
                print()


def main():
    """Main CLI loop."""
    # Load environment variables
    load_dotenv()
    
    # Check for OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Error: OPENAI_API_KEY not found in environment variables.")
        print("Please copy example.env to .env and add your OpenAI API key.")
        return
    
    # Initialize the agent
    print("🚀 Initializing Arxiv Learning Agent...")
    try:
        agent = ArxivLearningAgent()
        print("✅ Agent initialized successfully!")
    except Exception as e:
        print(f"❌ Error initializing agent: {e}")
        return
    
    print_banner()
    
    # Main interaction loop
    while True:
        try:
            user_input = input("\n💬 Enter a command: ").strip()
            
            if not user_input:
                continue
                
            # Parse command
            parts = user_input.split(maxsplit=1)
            command = parts[0].lower()
            
            if command in ["quit", "exit"]:
                print("👋 Goodbye!")
                break
                
            elif command == "help":
                print_banner()
                
            elif command == "learn":
                if len(parts) < 2:
                    print("❌ Please specify a topic to learn about.")
                    print("Example: learn machine learning")
                    continue
                
                topic = parts[1]
                print(f"🔍 Learning about '{topic}'...")
                print("⏳ This may take a moment to download and process papers...")
                
                result = agent.learn_topic(topic)
                print_learning_result(result)
                
            elif command == "ask":
                if len(parts) < 2:
                    print("❌ Please specify a question to ask.")
                    print("Example: ask What is transformer architecture?")
                    continue
                
                question = parts[1]
                print(f"🤔 Thinking about: '{question}'...")
                
                result = agent.ask_question(question)
                print_answer_result(result)
                
            elif command == "explore":
                print("🔍 Exploring knowledge base...")
                
                result = agent.explore_knowledge()
                print_exploration_result(result)
                
            else:
                print(f"❌ Unknown command: {command}")
                print("Type 'help' to see available commands.")
                
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ An error occurred: {e}")
            print("Please try again or type 'help' for available commands.")


if __name__ == "__main__":
    main() 