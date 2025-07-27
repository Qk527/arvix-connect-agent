# 🤖 Arxiv Learning Agent

A simple LangChain-based agent that learns from scientific papers on Arxiv and answers questions using RAG (Retrieval-Augmented Generation).

## 🎯 Features

The agent has three main functions:

1. **📚 Learn**: Download and process papers from Arxiv on any topic
2. **🤔 Ask**: Answer questions using stored knowledge or indicate when learning is needed
3. **🔍 Explore**: Browse what topics and papers the system already knows about

## 🏗️ Architecture

### Core Components

- **ArxivRetriever**: Downloads papers from Arxiv using LangChain's integration
- **Vector Store**: Uses Chroma DB to store document embeddings locally
- **Text Splitter**: Breaks papers into manageable chunks for processing
- **RAG Chain**: Combines retrieval with GPT-3.5-turbo for question answering
- **CLI Interface**: Simple command-line interface for interaction

### How It Works

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Arxiv Papers  │ -> │  Text Splitting  │ -> │   Embeddings    │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                         │
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  User Question  │ -> │   RAG Retrieval  │ <- │  Vector Store   │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │
                                ▼
                        ┌──────────────────┐
                        │   LLM Response   │
                        └──────────────────┘
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- OpenAI API key

### Installation

1. **Clone or download this project**
   ```bash
   cd arxiv-learning-agent
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**
   ```bash
   # Copy the example environment file
   cp example.env .env
   
   # Edit .env and add your OpenAI API key
   # OPENAI_API_KEY=your_openai_api_key_here
   ```

4. **Run the agent**
   ```bash
   python src/main.py
   ```

## 📖 Usage Examples

### Learning About a Topic
```
💬 Enter a command: learn transformer architecture

🔍 Learning about 'transformer architecture'...
⏳ This may take a moment to download and process papers...
✅ Successfully learned about 'transformer architecture'
📄 Processed 5 papers
📝 Created 45 text chunks

📚 Papers learned:
  1. Attention Is All You Need
     Authors: Ashish Vaswani, Noam Shazeer, Niki Parmar...
     ArXiv ID: 1706.03762
     Published: 2017-06-12
```

### Asking Questions
```
💬 Enter a command: ask What is the transformer architecture?

🤔 Thinking about: 'What is the transformer architecture?'...
🤔 Answer: The transformer architecture is a neural network model that relies entirely on attention mechanisms, dispensing with recurrence and convolutions entirely...

📖 Sources:
  1. Attention Is All You Need
     Authors: Ashish Vaswani, Noam Shazeer...
     ArXiv ID: 1706.03762
     Relevant text: The Transformer is a model architecture eschewing recurrence...
```

### Exploring Knowledge
```
💬 Enter a command: explore

🔍 Exploring knowledge base...
🔍 Knowledge base contains 3 papers across 2 topics

📊 Statistics:
  • Total documents: 45
  • Total papers: 3
  • Topics learned: 2

🎯 Topics:
  • transformer architecture
  • machine learning
```

## 🛠️ Configuration

### Environment Variables

- `OPENAI_API_KEY`: Required. Your OpenAI API key for embeddings and LLM
- `LANGSMITH_API_KEY`: Optional. For LangSmith tracing
- `LANGSMITH_TRACING`: Optional. Set to "true" to enable tracing

### Agent Parameters

You can modify these in `src/agent.py`:

- `chunk_size`: Size of text chunks (default: 1000)
- `chunk_overlap`: Overlap between chunks (default: 200)
- `max_papers`: Maximum papers to download per topic (default: 5)
- `temperature`: LLM creativity (default: 0.1 for factual responses)

## 📁 Project Structure

```
arxiv-learning-agent/
├── src/
│   └── agent.py              # Main agent class
│   └── main.py              # CLI interface
├── requirements.txt          # Python dependencies
├── example.env              # Environment variables template
├── README.md               # This file
└── knowledge_base/         # Created automatically - vector store data
```

## 🔧 Technical Details

### Dependencies

- **langchain**: Core framework for building the agent
- **langchain-community**: Provides ArxivRetriever integration
- **langchain-openai**: OpenAI integration for LLM and embeddings
- **chromadb**: Vector database for storing embeddings
- **arxiv**: Python library for Arxiv API access
- **python-dotenv**: Environment variable management

### Key Classes

#### ArxivLearningAgent
The main agent class with three core methods:
- `learn_topic(topic, max_papers)`: Downloads and processes papers
- `ask_question(question)`: Answers questions using RAG
- `explore_knowledge()`: Shows available knowledge

#### Data Flow
1. **Learning**: Arxiv → Documents → Chunks → Embeddings → Vector Store
2. **Querying**: Question → Retrieval → Context + Question → LLM → Answer

## 🎓 Educational Value

This project demonstrates:

- **Agent Architecture**: Simple but effective agent design patterns
- **RAG Implementation**: Practical retrieval-augmented generation
- **Vector Databases**: Using embeddings for semantic search
- **LangChain Integration**: Real-world use of LangChain components
- **API Integration**: Working with external APIs (Arxiv, OpenAI)

## 🚀 Extensions & Improvements

Consider these enhancements for learning:

1. **Multiple Data Sources**: Add support for other academic databases
2. **Better Paper Analysis**: Classify papers as theoretical vs. practical
3. **Citation Tracking**: Track relationships between papers
4. **Web Interface**: Build a simple web UI using Streamlit
5. **Better Chunking**: Implement paper-section-aware chunking
6. **Query Expansion**: Improve search with query expansion techniques

## 🐛 Troubleshooting

### Common Issues

**"No module named 'agent'"**
- Make sure you're running from the project root directory
- The `main.py` file adds `src/` to the Python path

**"OPENAI_API_KEY not found"**
- Copy `example.env` to `.env`
- Add your OpenAI API key to the `.env` file

**"Rate limit exceeded"**
- You may be hitting OpenAI API limits
- Wait a moment and try again
- Consider reducing `max_papers` parameter

**"No papers found for topic"**
- Try a more general topic (e.g., "machine learning" instead of very specific terms)
- Check that the topic has papers available on Arxiv

### Memory Usage
- The vector database is stored locally in `knowledge_base/`
- Each paper adds ~1-5MB depending on length
- You can delete `knowledge_base/` to start fresh

## 📄 License

This project is for educational purposes. Feel free to modify and extend it for learning!

## 🤝 Contributing

This is an educational project! Feel free to:
- Add new features
- Improve the code structure
- Fix bugs
- Add more data sources
- Enhance the user interface

---

**Happy Learning! 🎓** 