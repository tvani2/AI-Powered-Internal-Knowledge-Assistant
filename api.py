#!/usr/bin/env python3
"""
FastAPI backend for AI-Powered Internal Knowledge Assistant
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional, Dict, Any
import os
import logging
import uuid
import socket
import psutil
from datetime import datetime

# Import our agent
from intelligent_agent import IntelligentAgent, QueryType
from database import DatabaseManager
from chart_generator import chart_generator

# Configure detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('app.log')
    ]
)
logger = logging.getLogger(__name__)

# Debug: Check if we can import all required modules
logger.info("=== Starting API initialization ===")
logger.info(f"Current working directory: {os.getcwd()}")
logger.info(f"Python executable: {os.sys.executable}")

try:
    import fastapi
    logger.info(f"FastAPI version: {fastapi.__version__}")
except ImportError as e:
    logger.error(f"Failed to import FastAPI: {e}")

try:
    import uvicorn
    logger.info(f"Uvicorn version: {uvicorn.__version__}")
except ImportError as e:
    logger.error(f"Failed to import Uvicorn: {e}")

# Initialize FastAPI app
app = FastAPI(
    title="AI-Powered Internal Knowledge Assistant",
    description="An intelligent enterprise knowledge assistant with database and document search capabilities",
    version="1.0.0"
)

logger.info("FastAPI app created successfully")

# Add CORS middleware for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logger.info("CORS middleware added")

# Mount static files
if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")
    logger.info("Static files mounted at /static")
else:
    logger.warning("Static directory not found")

# Initialize components
logger.info("Initializing components...")
try:
    logger.info("Creating IntelligentAgent...")
    agent = IntelligentAgent()
    logger.info("IntelligentAgent created successfully")
    
    logger.info("Creating DatabaseManager...")
    db_manager = DatabaseManager()
    logger.info("DatabaseManager created successfully")
    
    logger.info("Intelligent agent and database manager initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize components: {e}")
    logger.error(f"Exception type: {type(e)}")
    import traceback
    logger.error(f"Traceback: {traceback.format_exc()}")
    agent = None
    db_manager = None

# No authentication required - open access

# Pydantic models for request/response
class ChatRequest(BaseModel):
    query: str
    user_id: Optional[str] = None
    session_id: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    query_type: Optional[str] = None
    confidence: Optional[float] = None
    reasoning: Optional[str] = None
    sql_generated: Optional[str] = None
    documents_searched: Optional[list] = None
    processing_time: Optional[float] = None
    sources: Optional[Dict[str, Any]] = None
    chart: Optional[str] = None

class HealthResponse(BaseModel):
    status: str
    agent_ready: bool
    database_connected: bool
    documents_available: bool

class QueryLogResponse(BaseModel):
    logs: list
    total_count: int

@app.get("/")
async def root():
    """Serve the chat interface"""
    logger.info("Root endpoint accessed")
    if os.path.exists("static/index.html"):
        logger.info("Serving static index.html")
        return FileResponse("static/index.html")
    else:
        logger.info("Static index.html not found, returning API info")
        return {
            "message": "AI-Powered Internal Knowledge Assistant API",
            "version": "1.0.0",
            "endpoints": {
                "chat": "/chat",
                "health": "/health",
                "capabilities": "/capabilities",
                "logs": "/logs"
            }
        }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    logger.info("Health check endpoint accessed")
    try:
        # Check if agent is ready
        agent_ready = agent is not None
        logger.info(f"Agent ready: {agent_ready}")
        
        # Check database connection
        try:
            from tools.database_tool import DatabaseTool
            db_tool = DatabaseTool()
            schema = db_tool.get_schema()
            database_connected = len(schema) > 0
            logger.info(f"Database connected: {database_connected}, schema tables: {list(schema.keys())}")
        except Exception as e:
            logger.error(f"Database connection failed: {e}")
            database_connected = False
        
        # Check documents availability
        import os
        documents_available = os.path.exists("documents") and len(os.listdir("documents")) > 0
        logger.info(f"Documents available: {documents_available}")
        
        status = "healthy" if all([agent_ready, database_connected, documents_available]) else "degraded"
        logger.info(f"Overall health status: {status}")
        
        return HealthResponse(
            status=status,
            agent_ready=agent_ready,
            database_connected=database_connected,
            documents_available=documents_available
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return HealthResponse(
            status="unhealthy",
            agent_ready=False,
            database_connected=False,
            documents_available=False
        )



@app.get("/capabilities")
async def get_capabilities():
    """Get system capabilities"""
    try:
        from tools.database_tool import DatabaseTool
        db_tool = DatabaseTool()
        schema = db_tool.get_schema()
        
        return {
            "database_tables": list(schema.keys()),
            "document_categories": [
                "HR policies",
                "Meeting notes", 
                "Technical documentation"
            ],
            "features": [
                "Natural language to SQL conversion",
                "Document semantic search",
                "Intelligent query routing",
                "Hybrid database/document responses",
                "Query logging and debugging"
            ],
            "sample_queries": [
                "Show me the top 5 employees by sales revenue",
                "What are the employee benefits?",
                "List all active projects with their budgets",
                "What was discussed in the latest engineering standup?"
            ]
        }
    except Exception as e:
        logger.error(f"Failed to get capabilities: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve capabilities")

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Main chat endpoint that processes user queries
    """
    import time
    
    if not agent:
        raise HTTPException(status_code=503, detail="Intelligent agent not available")
    
    start_time = time.time()
    session_id = request.session_id or str(uuid.uuid4())
    user_id = request.user_id or "anonymous"
    
    try:
        logger.info(f"Processing query: {request.query}")
        
        # Analyze the query first
        analysis = agent.analyze_query(request.query)
        
        # Process the query
        response = agent.process_query(request.query)
        
        processing_time = time.time() - start_time
        
        # Prepare response data
        response_data = {
            "response": response,
            "query_type": analysis.query_type.value,
            "confidence": analysis.confidence,
            "reasoning": analysis.reasoning,
            "processing_time": round(processing_time, 2)
        }
        
        # Add SQL info if generated
        if analysis.suggested_sql:
            response_data["sql_generated"] = analysis.suggested_sql
        
        # Add document search info if used
        if analysis.suggested_document_queries:
            response_data["documents_searched"] = analysis.suggested_document_queries
        
        # Prepare sources information
        sources = {}
        if analysis.suggested_sql:
            sources["database"] = {
                "sql": analysis.suggested_sql,
                "tables_accessed": _extract_tables_from_sql(analysis.suggested_sql)
            }
        if analysis.suggested_document_queries:
            sources["documents"] = {
                "queries": analysis.suggested_document_queries,
                "categories": _categorize_document_queries(analysis.suggested_document_queries)
            }
        
        response_data["sources"] = sources
        
        # Format response for better readability
        response_data["response"] = _format_response(response_data["response"], analysis.query_type.value)
        
        # Generate chart for database queries with numeric data
        if analysis.query_type.value == "database" and "|" in response_data["response"]:
            chart_data = chart_generator.generate_chart(response_data["response"], request.query)
            if chart_data:
                response_data["chart"] = chart_data
        
        # Log the query
        if db_manager:
            db_manager.log_query(
                user_query=request.query,
                query_type=analysis.query_type.value,
                confidence=analysis.confidence,
                sql_generated=analysis.suggested_sql,
                documents_searched=", ".join(analysis.suggested_document_queries) if analysis.suggested_document_queries else None,
                response=response,
                processing_time=processing_time,
                user_id=user_id,
                session_id=session_id
            )
        

        
        logger.info(f"Query processed successfully in {processing_time:.2f}s")
        
        return ChatResponse(**response_data)
        
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        processing_time = time.time() - start_time
        
        # Log the error
        if db_manager:
            db_manager.log_query(
                user_query=request.query,
                query_type="error",
                confidence=0.0,
                response=f"Error: {str(e)}",
                processing_time=processing_time,
                error_message=str(e),
                user_id=user_id,
                session_id=session_id
            )
        

        
        return ChatResponse(
            response=f"I apologize, but I encountered an error while processing your query: {str(e)}",
            query_type="error",
            confidence=0.0,
            reasoning="Error occurred during processing",
            processing_time=round(processing_time, 2)
        )

def _extract_tables_from_sql(sql):
        """Extract table names from SQL query"""
        if not sql:
            return []
        
        # Simple table extraction - in production, use a proper SQL parser
        tables = []
        sql_upper = sql.upper()
        
        if "FROM" in sql_upper:
            # Extract table after FROM
            from_parts = sql_upper.split("FROM")
            for part in from_parts[1:]:
                if "JOIN" in part:
                    # Handle JOINs
                    join_parts = part.split("JOIN")
                    for join_part in join_parts:
                        table = join_part.strip().split()[0]
                        if table not in tables:
                            tables.append(table)
                else:
                    # Simple FROM
                    table = part.strip().split()[0]
                    if table not in tables:
                        tables.append(table)
        
        return tables

def _categorize_document_queries(queries):
    """Categorize document queries by type"""
    categories = {
        "hr_policies": [],
        "meeting_notes": [],
        "technical_docs": []
    }
    
    for query in queries:
        query_lower = query.lower()
        if any(word in query_lower for word in ["policy", "benefit", "hr", "employee"]):
            categories["hr_policies"].append(query)
        elif any(word in query_lower for word in ["meeting", "standup", "review", "discussion"]):
            categories["meeting_notes"].append(query)
        elif any(word in query_lower for word in ["technical", "api", "architecture", "system"]):
            categories["technical_docs"].append(query)
    
    return categories

def _format_response(response: str, query_type: str) -> str:
    """Format response for better readability based on query type"""
    if query_type == "database":
        # Format database results as tables
        if "|" in response or "Employee" in response or "Sales" in response:
            lines = response.split('\n')
            formatted_lines = []
            for line in lines:
                if '|' in line:
                    # Format table rows
                    formatted_lines.append(line.strip())
                elif any(keyword in line for keyword in ["Employee", "Sales", "Project", "Department"]):
                    # Format headers
                    formatted_lines.append(f"\n**{line.strip()}**")
                else:
                    formatted_lines.append(line.strip())
            return '\n'.join(formatted_lines)
    
    elif query_type == "document":
        # Format document results with bullet points
        if "•" not in response and "-" not in response:
            # Add bullet points for policy summaries
            lines = response.split('\n')
            formatted_lines = []
            for line in lines:
                line = line.strip()
                if line and not line.startswith('•') and not line.startswith('-'):
                    if any(keyword in line.lower() for keyword in ["policy", "benefit", "guideline", "rule"]):
                        formatted_lines.append(f"• {line}")
                    else:
                        formatted_lines.append(line)
                else:
                    formatted_lines.append(line)
            return '\n'.join(formatted_lines)
    
    return response

@app.get("/sample-queries")
async def get_sample_queries():
    """Get sample queries for testing"""
    return {
        "database_queries": [
            "Show me the top 5 employees by sales revenue",
            "How many employees are in the Engineering department?",
            "List all active projects with their budgets",
            "Find employees who joined in the last year",
            "Show sales performance by product category"
        ],
        "document_queries": [
            "What are the remote work policies?",
            "What was discussed in the latest engineering standup?",
            "What are the employee benefits?",
            "What is the system architecture?",
            "What are the performance management guidelines?"
        ],
        "hybrid_queries": [
            "Show me sales data and related policies",
            "List project managers and their meeting notes",
            "What are the benefits for high-performing employees?"
        ]
    }



@app.get("/logs", response_model=QueryLogResponse)
async def get_query_logs(limit: int = 50, user_id: Optional[str] = None):
    """Get query logs for debugging"""
    try:
        if not db_manager:
            raise HTTPException(status_code=503, detail="Database manager not available")
        
        logs_df = db_manager.get_query_logs(limit=limit, user_id=user_id)
        
        # Convert DataFrame to list of dictionaries
        logs = []
        for _, row in logs_df.iterrows():
            log_entry = {
                "id": row["id"],
                "timestamp": row["timestamp"],
                "user_query": row["user_query"],
                "query_type": row["query_type"],
                "confidence": row["confidence"],
                "sql_generated": row["sql_generated"],
                "documents_searched": row["documents_searched"],
                "response": row["response"][:200] + "..." if row["response"] and len(row["response"]) > 200 else row["response"],
                "processing_time": row["processing_time"],
                "error_message": row["error_message"],
                "user_id": row["user_id"],
                "session_id": row["session_id"]
            }
            logs.append(log_entry)
        
        return QueryLogResponse(logs=logs, total_count=len(logs))
        
    except Exception as e:
        logger.error(f"Failed to retrieve query logs: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve query logs")

if __name__ == "__main__":
    import uvicorn
    
    logger.info("=== Starting server ===")
    
    # Check if OpenAI API key is set
    if not os.getenv("OPENAI_API_KEY"):
        logger.warning("OPENAI_API_KEY not set. LLM features will not work.")
        logger.info("Set OPENAI_API_KEY environment variable to enable full functionality.")
    
    # Check if port 8000 is available
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        result = sock.connect_ex(('localhost', 8000))
        sock.close()
        if result == 0:
            logger.error("Port 8000 is already in use!")
            logger.info("Please stop any existing server or use a different port")
            logger.info("You can check what's using port 8000 with: netstat -ano | findstr :8000")
        else:
            logger.info("Port 8000 is available")
    except Exception as e:
        logger.error(f"Error checking port availability: {e}")
    
    # Check network interfaces
    try:
        import socket
        hostname = socket.gethostname()
        local_ip = socket.gethostbyname(hostname)
        logger.info(f"Hostname: {hostname}")
        logger.info(f"Local IP: {local_ip}")
    except Exception as e:
        logger.error(f"Error getting network info: {e}")
    
    logger.info("Starting uvicorn server...")
    logger.info("Server will be available at:")
    logger.info("  - http://localhost:8000")
    logger.info("  - http://127.0.0.1:8000")
    logger.info("  - http://0.0.0.0:8000")
    
    # Run the server
    try:
        uvicorn.run(
            "api:app",
            host="0.0.0.0",
            port=8000,
            reload=True,
            log_level="debug"
        )
    except Exception as e:
        logger.error(f"Failed to start server: {e}")
        logger.error(f"Exception type: {type(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
