# AI-Powered Internal Knowledge Assistant

An intelligent assistant that automatically decides when to use database vs documents and generates SQL from natural language queries.

<img width="1200" height="800" alt="Screenshot (481)" src="https://github.com/user-attachments/assets/53065f1d-7c38-4653-b3df-fff483c28dde" />


## Project Goal

Create a unified AI system that can:
- **Analyze queries** to determine data source (database vs documents)
- **Generate SQL** from natural language for database queries
- **Search documents** semantically for policy/meeting information
- **Provide hybrid responses** combining structured data with contextual information
- **Visualize results** with charts for numeric data


## Quick Start

### Prerequisites
- Python 3.10+ 
- OpenAI API key

### Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set environment variables:**
   ```bash
   # Windows
   set OPENAI_API_KEY=your_api_key_here
   
   # Linux/Mac
   export OPENAI_API_KEY=your_api_key_here
   ```

3. **Initialize database:**
   ```bash
   python setup_database.py
   ```

4. **Start the application:**
   ```bash
   python start_app.py
   ```

5. **Open your browser:**
   Navigate to `http://localhost:8000`

## Features

###  **Intelligent Query Routing**
- Automatically determines whether to query database or search documents
- Uses confidence scoring to choose the best data source
- Provides hybrid responses combining multiple sources

###  **Database Operations**
- Natural language to SQL conversion
- Employee data, sales records, inventory management
- Automatic chart generation for numeric data
- Sample data included for testing

###  **Document Search**
- Semantic search across company documents
- HR policies, meeting notes, technical documentation
- Context-aware responses with source attribution

###  **Modern Web Interface**
- Clean, responsive design
- Real-time chat interface
- Sample queries for easy testing
- System information display

## Usage Examples

### Database Queries
- "Show me all employees in the Engineering department"
- "What's the total sales for Q1 2024?"
- "Create a chart of employee salaries by department"

### Document Queries
- "What's our remote work policy?"
- "What was discussed in the last executive meeting?"
- "How do I integrate with the API?"

### Hybrid Queries
- "Show me engineering employees and their benefits"
- "What's our sales performance and what policies affect it?"

## Project Structure

```
AI-Powered-Internal-Knowledge-Assistant/
├── api.py                 # FastAPI backend
├── start_app.py          # Web interface startup
├── intelligent_agent.py  # Core AI logic
├── database.py           # Database operations
├── chart_generator.py    # Visualization
├── setup_database.py     # Database setup
├── requirements.txt      # Dependencies
├── database_schema.md    # Schema documentation
├── company.db           # SQLite database
├── .gitignore           # Git ignore patterns
├── static/
│   └── index.html       # Web frontend
├── tools/
│   ├── database_tool.py
│   └── document_search_tool.py
└── documents/
    ├── hr_policies/
    ├── meeting_notes/
    └── technical_docs/
```

## Configuration

### Environment Variables
- `OPENAI_API_KEY`: Your OpenAI API key (required)

### Customization
- **Add Documents**: Place files in the `documents/` subdirectories
- **Modify Database**: Edit `setup_database.py` for different sample data
- **Change Port**: Modify the port in `start_app.py`

## Troubleshooting

### Common Issues
- **API Key Error**: Ensure `OPENAI_API_KEY` is set correctly
- **Database Error**: Run `python setup_database.py` to recreate database
- **Port Already in Use**: Change the port in `start_app.py`

### Debugging
- **Enable Debug Mode**: Uncomment the Response Details section in `static/index.html`
- **View Logs**: Check console output for detailed logging
- **Database Schema**: See `database_schema.md` for table structures

## Document Categories

### HR Policies (`documents/hr_policies/`)
- Employee benefits and policies
- Performance management guidelines
- Remote work policies

### Meeting Notes (`documents/meeting_notes/`)
- Engineering team standups
- Executive quarterly reviews
- Product development meetings

### Technical Docs (`documents/technical_docs/`)
- API integration guides
- System architecture documentation
- Technical specifications
