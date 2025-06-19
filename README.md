Cloud LLM Model with RAG and Web Intelligence
A comprehensive AI chat assistant that combines OpenAI's language models with Retrieval Augmented Generation (RAG) capabilities, intelligent web scraping, and vector-based document search using FAISS. This system enables contextually-aware conversations by leveraging both user-uploaded documents and dynamically scraped web content.
# Major Components
1. OpenAI Models Integration
The foundation of the system, providing both language generation and embedding capabilities.
Components:

GPT Models: Primary conversational AI (GPT-4-turbo)
Embedding Models: Text-to-vector conversion (text-embedding-3-large, text-embedding-3-small)
Vision Models: Image analysis for PDF processing (GPT-4 Vision)

Key Features:

Multi-model support with automatic fallbacks
Rate limiting and error handling
Token counting and optimization
Temperature and parameter control
Conversation context management

2. RAG (Retrieval Augmented Generation) System
The intelligent document processing and retrieval system that enables AI to answer questions using your specific content.
Components:

Document Processor: Smart text chunking for different content types
Query Engine: Semantic search and context retrieval
Answer Generator: Combines retrieved context with LLM responses
Source Attribution: Tracks and cites information sources

Key Features:

Semantic chunking (handles code, tables, prose differently)
Hybrid search (vector + keyword)
Context window optimization
Relevance scoring and reranking
Multi-document synthesis
Source citation in responses

3. FAISS Vector Database
High-performance vector storage and similarity search engine for semantic document retrieval.
Components:

Vector Index: Stores document embeddings for fast similarity search
Similarity Search: Cosine similarity-based retrieval
Metadata Storage: Document source and chunk information
Caching Layer: Optimized query performance

Key Features:

Fast approximate nearest neighbor search
Scalable to millions of documents
Memory-efficient storage
Batch processing capabilities
Index persistence and loading
Custom distance metrics

4. Web Scraping Intelligence
Automated web content discovery and processing system with reliability assessment.
Components:

Multi-Engine Search: DuckDuckGo, Bing, Google Scholar integration
Content Extractor: Intelligent article text extraction
Reliability Scorer: Domain and content quality assessment
Article Manager: Automated fetching and storage coordination

Key Features:

Multi-search engine fallbacks
Paywall detection and handling
Content cleaning and normalization
Domain reliability scoring (.edu, .gov prioritized)
Parallel processing for efficiency
Automatic article indexing for RAG integration

# System Architecture Overview
The system integrates four major components to provide intelligent, context-aware AI responses:

User Interaction flows through the main controller
Web Scraping continuously discovers and processes content
FAISS Vector Database stores and searches document embeddings
RAG System retrieves relevant context for enhanced responses
OpenAI Models provide embeddings, chat completion, and image analysis


# System Flow Visualization 
Check the diagram below 
```
https://ibb.co/BHCn0yrk 
```
Quick Start Installation
1. Clone and Setup
```
git clone https://github.com/End-of-life-LLM/End-Of-Life-LLM
cd End-Of-Life-LLM
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```
2. Install All Dependencies
```
pip install -r requirements.txt
```
4. Download NLTK Data
```
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"
```
6. Initialize Directory Structure
```
python setup_directories.py
```
7. Create a .env file in the main directory that incudes the following
```
OPENAI_API_KEY=<Your openai api key>
```
(Check this video on how to get the key https://www.youtube.com/watch?v=SzPE_AE0eEo) 
8. Launch Application
```
python main.py
```
Navigate to http://localhost:5000 and follow the setup wizard.


# Component Interaction Details
RAG ↔ FAISS Integration
python# Document Processing Flow
Document → Text Chunks → OpenAI Embeddings → FAISS Vector Storage
Query → OpenAI Embedding → FAISS Search → Relevant Chunks → RAG Response
Web Scraping → RAG Pipeline
python# Automatic Content Enhancement
Search Query → Multi-Engine Search → Content Extraction → 
Quality Assessment → Article Storage → Embedding Generation → 
FAISS Indexing → Available for RAG Queries
OpenAI Models Coordination
python# Multi-Model Usage
Text Content → text-embedding-3-large → Vector Embeddings
User Query + Context → GPT-4 → Contextual Response
PDF Images → GPT-4 Vision → Image Descriptions → Text Processing


# Key Benefits of This Architecture

Scalable Knowledge Base: FAISS enables fast search across millions of documents
Real-time Content Discovery: Web scraping continuously expands knowledge
Multi-modal Processing: Handles text, images, and structured data
Quality Assurance: Reliability scoring ensures trustworthy sources
Efficient Resource Usage: Caching and optimization reduce API costs



# Advanced Usage Examples
Building Domain-Specific Knowledge
# Upload domain documents + setup web scraping for continuous updates
1. Upload research papers → RAG indexing
2. Setup search queries: "machine learning 2024", "AI research papers"
3. Enable auto-fetching → Continuous knowledge expansion
4. Query: "What are the latest developments in transformer architectures?"
Multi-source Intelligence
bash# Combine personal documents with web intelligence
1. Upload company documents → Internal knowledge base
2. Scrape industry news → External intelligence
3. Query: "How do recent industry trends affect our product strategy?"
4. Get responses citing both internal docs and recent web articles
This architecture ensures that your AI assistant becomes increasingly intelligent over time, combining the power of large language models with your specific knowledge domain and real-time web intelligence.
