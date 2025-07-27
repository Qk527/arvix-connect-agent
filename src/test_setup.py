#!/usr/bin/env python3
"""
Test script to verify that the Arxiv Learning Agent setup is working correctly.

Run this script after installation to check:
1. All dependencies are installed
2. OpenAI API key is configured
3. Basic agent functionality works
"""

import os
import sys
from dotenv import load_dotenv

def test_imports():
    """Test that all required packages can be imported."""
    print("🔍 Testing imports...")
    
    try:
        import langchain
        print("✅ langchain imported successfully")
    except ImportError as e:
        print(f"❌ langchain import failed: {e}")
        return False
    
    try:
        from langchain_community.retrievers import ArxivRetriever
        print("✅ ArxivRetriever imported successfully")
    except ImportError as e:
        print(f"❌ ArxivRetriever import failed: {e}")
        return False
    
    try:
        from langchain_openai import OpenAIEmbeddings, ChatOpenAI
        print("✅ OpenAI components imported successfully")
    except ImportError as e:
        print(f"❌ OpenAI components import failed: {e}")
        return False
    
    try:
        import chromadb
        print("✅ chromadb imported successfully")
    except ImportError as e:
        print(f"❌ chromadb import failed: {e}")
        return False
    
    try:
        import arxiv
        print("✅ arxiv package imported successfully")
    except ImportError as e:
        print(f"❌ arxiv package import failed: {e}")
        return False
    
    return True

def test_environment():
    """Test environment variables."""
    print("\n🔍 Testing environment variables...")
    
    load_dotenv()
    
    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        print("❌ OPENAI_API_KEY not found in environment")
        print("   Please copy example.env to .env and add your OpenAI API key")
        return False
    
    if openai_key == "your_openai_api_key_here":
        print("❌ OPENAI_API_KEY still has placeholder value")
        print("   Please update .env with your actual OpenAI API key")
        return False
    
    if len(openai_key) < 20:
        print("❌ OPENAI_API_KEY seems too short")
        print("   Please check that you've entered the correct API key")
        return False
    
    print("✅ OPENAI_API_KEY found and appears valid")
    return True

def test_agent_initialization():
    """Test basic agent initialization."""
    print("\n🔍 Testing agent initialization...")
    
    try:
        # Add src to path
        sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
        from agent import ArxivLearningAgent
        
        # Try to initialize the agent
        agent = ArxivLearningAgent()
        print("✅ Agent initialized successfully")
        
        # Test basic functionality
        result = agent.explore_knowledge()
        if isinstance(result, dict) and "message" in result:
            print("✅ Agent explore_knowledge() works")
        else:
            print("❌ Agent explore_knowledge() returned unexpected result")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Agent initialization failed: {e}")
        return False

def test_arxiv_connection():
    """Test connection to Arxiv."""
    print("\n🔍 Testing Arxiv connection...")
    
    try:
        from langchain_community.retrievers import ArxivRetriever
        
        # Try a very limited search to test connectivity
        retriever = ArxivRetriever(load_max_docs=1)
        papers = retriever.invoke("machine learning")
        
        if papers and len(papers) > 0:
            print("✅ Arxiv connection works")
            print(f"   Sample paper: {papers[0].metadata.get('Title', 'Unknown')}")
            return True
        else:
            print("❌ No papers retrieved from Arxiv")
            return False
            
    except Exception as e:
        print(f"❌ Arxiv connection failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🚀 Testing Arxiv Learning Agent Setup")
    print("=" * 50)
    
    all_passed = True
    
    # Test imports
    if not test_imports():
        all_passed = False
        print("\n💡 Fix: Run 'pip install -r requirements.txt'")
    
    # Test environment
    if not test_environment():
        all_passed = False
        print("\n💡 Fix: Copy example.env to .env and add your OpenAI API key")
    
    # Test agent initialization (only if previous tests passed)
    if all_passed and not test_agent_initialization():
        all_passed = False
        print("\n💡 Fix: Check the error message above for details")
    
    # Test Arxiv connection (only if previous tests passed)
    if all_passed and not test_arxiv_connection():
        all_passed = False
        print("\n💡 Fix: Check your internet connection")
    
    print("\n" + "=" * 50)
    if all_passed:
        print("🎉 All tests passed! Your setup is ready.")
        print("   Run 'python main.py' to start the agent.")
    else:
        print("❌ Some tests failed. Please fix the issues above.")
        print("   Then run this test script again.")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 