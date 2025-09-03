# AI-Powered Internal Knowledge Assistant

An intelligent enterprise knowledge assistant that combines **RAG**, **vector embeddings**, and **natural language processing** to provide unified access to both structured database data and unstructured documents.

## Key Technologies & Features

### **Core AI/ML Technologies**
• **RAG** - Combines document retrieval with LLM generation for accurate, contextual responses
• **FAISS Vector Database** - High-performance similarity search for semantic document retrieval
• **OpenAI Embeddings** - Text-embedding-3-small model for document vectorization
• **LangChain Framework** - Orchestrates LLM interactions and document processing
• **Natural Language to SQL** - Automatic SQL generation from conversational queries

### **Backend Architecture**
• **FastAPI** - Modern, high-performance web framework with automatic API documentation
• **SQLite Database** - Structured data storage with foreign key relationships and indexes
• **Hybrid Query Processing** - Intelligent routing between database and document sources
• **Vector Search** - Semantic similarity search across company documents
• **Real-time Chat Interface** - HTTP-based conversational AI interface

### **Data Processing & Storage**
• **Document Chunking** - RecursiveCharacterTextSplitter for optimal text segmentation
• **Vector Store Management** - Persistent FAISS index with metadata tracking
• **Database Schema Design** - Normalized relational model with performance indexes
• **Query Analysis Engine** - Confidence-based routing between data sources

## Project Highlights

### **Intelligent Query Routing**
• **Automatic Source Detection** - Analyzes queries to determine database vs document search
• **Confidence Scoring** - Uses ML-based confidence metrics for optimal routing
• **Hybrid Responses** - Combines structured data with contextual document information

### **Advanced Document Processing**
• **Semantic Search** - Vector-based similarity search across HR policies, meeting notes, and technical docs
• **Context-Aware Retrieval** - RAG system provides precise answers with source attribution
• **Document Summarization** - LLM-powered content summarization and precise answer extraction

### **Database Intelligence**
• **Natural Language SQL** - Converts conversational queries to optimized SQL statements
• **Schema-Aware Generation** - LLM generates SQL with full database schema context
• **Query Validation** - Safety checks and syntax validation for generated SQL

## Technical Implementation

### **Performance Optimizations**
• **Vector Index Caching** - Persistent FAISS index for fast similarity search
• **Database Indexing** - Strategic indexes on frequently queried columns
• **Query Optimization** - Rule-based SQL generation with LLM fallback

### **Security & Safety**
• **Read-Only Database Access** - SELECT-only operations with SQL injection protection
• **Input Validation** - Comprehensive input sanitization and validation
• **API Key Management** - Secure environment variable handling

## Data Sources

### **Structured Data (SQLite)**
• **Employee Database** - 100 employees across 10 departments with hierarchical relationships
• **Sales Records** - 300 sales transactions with customer and product data
• **Project Management** - 50 projects with budgets, timelines, and manager assignments

### **Unstructured Documents**
• **HR Policies** - Employee benefits, performance management, remote work policies
• **Meeting Notes** - Engineering standups, executive reviews, product development meetings
• **Technical Documentation** - API guides, system architecture, integration specifications

## Quick Start

### Prerequisites
• Python 3.10+
• OpenAI API key

### Installation
```bash
# Clone repository
git clone <repository-url>
cd AI-Powered-Internal-Knowledge-Assistant

# Install dependencies
pip install -r requirements.txt

# Set environment variables
export OPENAI_API_KEY=your_api_key_here

# Initialize database
python database.py

# Start application
python api.py
```

### Access Points
• **Web Interface**: http://localhost:8000
• **API Documentation**: http://localhost:8000/docs
• **Health Check**: http://localhost:8000/health

## Usage Examples

### Database Queries
• "Show me the top 5 employees by sales revenue"
• "How many employees are in the Engineering department?"
• "List all active projects with their budgets"

### Document Queries
• "What are the employee benefits?"
• "What was discussed in the latest engineering standup?"
• "What is the system architecture?"

### Hybrid Queries
• "Show me sales data and related policies"
• "List project managers and their meeting notes"
• "What are the benefits for high-performing employees?"