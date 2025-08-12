# AI-Powered Internal Knowledge Assistant

An intelligent assistant that automatically decides when to use database vs documents and generates SQL from natural language queries.

## Project Goal

Create a unified AI system that can:
- **Analyze queries** to determine data source (database vs documents)
- **Generate SQL** from natural language for database queries
- **Search documents** semantically for policy/meeting information
- **Provide hybrid responses** combining structured data with contextual information
- **Visualize results** with charts for numeric data


## Files Overview

| File | Description |
|------|-------------|
| `api.py` | FastAPI backend with chat endpoint |
| `streamlit_app.py` | Advanced Streamlit UI |
| `start_app.py` | Easy startup script for web interface |
| `start_streamlit.py` | Easy startup script for Streamlit UI |
| `intelligent_agent.py` | Core AI agent with query analysis |
| `database.py` | Database setup and management |
| `setup_database.py` | Database initialization script |
| `database_schema.md` | Detailed database schema documentation |
| `chart_generator.py` | Matplotlib charts for data visualization |
| `tools/database_tool.py` | SQL execution and schema tools |
| `tools/document_search_tool.py` | Vector search for documents |
| `static/index.html` | Web chat interface |
| `requirements.txt` | Python dependencies |
| `.env` | Environment variables (create from template) |
| `.gitignore` | Git ignore patterns |

## Quick Setup

### 1. Environment Setup
```bash
git clone <repository>
cd AI-Powered-Internal-Knowledge-Assistant
python -m venv venv
venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

### 2. Configuration
```bash
# Create .env file with your OpenAI API key
echo "OPENAI_API_KEY=your_key_here" > .env
echo "OPENAI_MODEL=gpt-4o-mini" >> .env
```

### 3. Initialize Database
```bash
python setup_database.py
```

### 4. Run Application
```bash
python start_app.py
# Open http://localhost:8000
```

## Example Queries

### Database Queries
- "Show me the top 5 employees by sales revenue"
- "How many employees are in Engineering department?"
- "List all active projects with budgets"
- "Find employees who joined last year"

### Document Queries
- "What are the remote work policies?"
- "What was discussed in the latest engineering standup?"
- "What are the employee benefits?"
- "What is the system architecture?"

### Hybrid Queries
- "Show me sales data and related policies"
- "List project managers and their meeting notes"

## Features

- **Smart Query Analysis**: Automatically detects query type (DB vs docs)
- **SQL Generation**: Converts natural language to SQLite queries
- **Document Search**: Semantic search using vector embeddings
- **Data Visualization**: Automatic charts for numeric results
- **Pretty Output**: Formatted tables and bullet-point summaries
- **Dual UI**: Web interface + Streamlit app
- **Query Logging**: Track and debug all interactions

## API Endpoints

- `GET /` - Web chat interface
- `POST /chat` - Main chat endpoint
- `GET /health` - System status
- `GET /capabilities` - Available features
- `GET /logs` - Query history

## Data Sources

**Database Tables:**
- `employees` (id, name, department, role, salary, etc.)
- `sales` (id, date, product, revenue, employee_id, etc.)
- `projects` (id, name, status, budget, manager_id, etc.)

**Document Categories:**
- HR Policies (benefits, performance, remote work)
- Meeting Notes (standups, reviews, development)
- Technical Docs (architecture, API guides)

## Use Cases

- **HR Analytics**: Employee performance, department insights
- **Sales Analysis**: Revenue trends, top performers
- **Project Management**: Status tracking, budget analysis
- **Policy Research**: Quick access to company guidelines
- **Meeting Insights**: Historical discussion summaries