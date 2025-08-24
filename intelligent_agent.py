import os
import json
import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import pandas as pd
import numpy as np
from pathlib import Path

from dotenv import load_dotenv
from langchain.agents import AgentExecutor, initialize_agent
from langchain.agents import AgentType
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts.chat import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.schema import BaseMessage, HumanMessage, SystemMessage
from langchain.text_splitter import RecursiveCharacterTextSplitter
import faiss
import pickle

from tools.database_tool import DatabaseTool
from tools.document_search_tool import DocumentSearchTool


class QueryType(Enum):
    DATABASE = "database"
    DOCUMENTS = "documents"
    HYBRID = "hybrid"
    CONVERSATIONAL = "conversational"
    UNKNOWN = "unknown"


@dataclass
class QueryAnalysis:
    query_type: QueryType
    confidence: float
    reasoning: str
    suggested_sql: Optional[str] = None
    suggested_document_queries: Optional[List[str]] = None


class DocumentPreprocessor:
    """
    Handles document preprocessing, chunking, and vector store operations
    """
    
    def __init__(self, chunk_size: int = 300, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.embeddings = None
        self.vector_store = None
        self.chunks = []
        
        # Initialize embeddings if API key is available
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key and api_key != "your-api-key-here":
            try:
                self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
                print("Embeddings initialized successfully")
            except Exception as e:
                print(f"Warning: Could not initialize embeddings: {e}")
        else:
            print("Warning: OPENAI_API_KEY not set. Vector search will be disabled.")
    
    def chunk_document(self, content: str, source: str) -> List[Dict[str, Any]]:
        """Chunk a single document into smaller sections"""
        if not content.strip():
            return []
        
        # Use RecursiveCharacterTextSplitter for better chunking
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        chunks = splitter.split_text(content)
        
        return [
            {
                "content": chunk.strip(),
                "source": source,
                "chunk_id": f"{source}_{i}",
                "length": len(chunk)
            }
            for i, chunk in enumerate(chunks) if chunk.strip()
        ]
    
    def chunk_all_documents(self, documents_dir: str) -> List[Dict[str, Any]]:
        """Chunk all documents in the specified directory"""
        all_chunks = []
        documents_path = Path(documents_dir)
        
        if not documents_path.exists():
            print(f"Documents directory {documents_dir} not found")
            return all_chunks
        
        # Find all text files recursively
        text_files = list(documents_path.rglob("*.txt")) + list(documents_path.rglob("*.md"))
        
        for file_path in text_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                source = str(file_path.relative_to(documents_path))
                chunks = self.chunk_document(content, source)
                all_chunks.extend(chunks)
                
                print(f"Chunked {source}: {len(chunks)} chunks")
                
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
        
        print(f"Total chunks created: {len(all_chunks)}")
        return all_chunks
    
    def create_vector_store(self, chunks: List[Dict[str, Any]], store_path: str) -> bool:
        """Create vector store from chunks"""
        if not self.embeddings:
            print("Embeddings not available. Cannot create vector store.")
            return False
        
        if not chunks:
            print("No chunks provided for vector store creation.")
            return False
        
        try:
            # Extract content and metadata
            texts = [chunk["content"] for chunk in chunks]
            metadatas = [
                {
                    "source": chunk["source"],
                    "chunk_id": chunk["chunk_id"],
                    "length": chunk["length"]
                }
                for chunk in chunks
            ]
            
            # Create embeddings
            print("Creating embeddings...")
            embeddings_list = self.embeddings.embed_documents(texts)
            
            # Convert to numpy array
            embeddings_array = np.array(embeddings_list).astype('float32')
            
            # Create FAISS index
            dimension = embeddings_array.shape[1]
            index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
            index.add(embeddings_array)
            
            # Save index and metadata
            store_path = Path(store_path)
            store_path.mkdir(exist_ok=True)
            
            faiss.write_index(index, str(store_path / "index.faiss"))
            
            # Save chunks and metadata
            with open(store_path / "index.pkl", "wb") as f:
                pickle.dump({
                    "chunks": chunks,
                    "metadatas": metadatas,
                    "embeddings": embeddings_list
                }, f)
            
            self.vector_store = {
                "index": index,
                "chunks": chunks,
                "metadatas": metadatas,
                "embeddings": embeddings_list
            }
            
            print(f"Vector store created successfully at {store_path}")
            return True
            
        except Exception as e:
            print(f"Error creating vector store: {e}")
            return False
    
    def load_vector_store(self, store_path: str = "vector_store") -> bool:
        """Load existing vector store"""
        try:
            store_path = Path(store_path)
            index_path = store_path / "index.faiss"
            metadata_path = store_path / "index.pkl"
            
            if not index_path.exists() or not metadata_path.exists():
                print("Vector store files not found")
                return False
            
            # Load FAISS index
            index = faiss.read_index(str(index_path))
            
            # Load metadata
            with open(metadata_path, "rb") as f:
                data = pickle.load(f)
            
            self.vector_store = {
                "index": index,
                "chunks": data["chunks"],
                "metadatas": data["metadatas"],
                "embeddings": data["embeddings"]
            }
            
            print("Vector store loaded successfully")
            return True
            
        except Exception as e:
            print(f"Error loading vector store: {e}")
            return False
    
    def search_documents(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Search documents using vector similarity"""
        if not self.vector_store or not self.embeddings:
            print("Vector store or embeddings not available")
            return []
        
        try:
            # Create query embedding
            query_embedding = self.embeddings.embed_query(query)
            query_vector = np.array([query_embedding]).astype('float32')
            
            # Search in FAISS index
            scores, indices = self.vector_store["index"].search(query_vector, k)
            
            results = []
            for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
                if idx < len(self.vector_store["chunks"]):
                    chunk = self.vector_store["chunks"][idx]
                    metadata = self.vector_store["metadatas"][idx]
                    
                    results.append({
                        "content": chunk["content"],
                        "source": chunk["source"],
                        "score": float(score),
                        "method": "vector_search",
                        "chunk_id": chunk["chunk_id"]
                    })
            
            return results
            
        except Exception as e:
            print(f"Error searching documents: {e}")
            return []
    
    def re_rank_results(self, results: List[Dict[str, Any]], query: str) -> List[Dict[str, Any]]:
        """Re-rank results based on relevance to the query"""
        if not results:
            return results
        
        # Simple re-ranking based on content relevance
        query_lower = query.lower()
        query_words = set(word for word in query_lower.split() if len(word) > 2)
        
        for result in results:
            content_lower = result["content"].lower()
            
            # Calculate word overlap score
            content_words = set(word for word in content_lower.split() if len(word) > 2)
            word_overlap = len(query_words.intersection(content_words))
            overlap_score = word_overlap / len(query_words) if query_words else 0
            
            # Combine vector similarity with word overlap
            vector_score = result["score"]
            final_score = (vector_score * 0.7) + (overlap_score * 0.3)
            
            result["final_score"] = final_score
            result["word_overlap"] = overlap_score
        
        # Sort by final score
        results.sort(key=lambda x: x["final_score"], reverse=True)
        
        return results


class IntelligentAgent:
    """
    An intelligent agent that can:
    1. Analyze user queries to determine whether to use database or documents
    2. Generate SQL automatically from natural language queries
    3. Provide hybrid responses combining both data sources
    4. Use vector search for semantic document retrieval
    5. Re-rank results for better relevance
    """
    
    def __init__(self):
        try:
            load_dotenv()
            print("Successfully loaded .env file")
        except Exception as e:
            print(f"Warning: Could not load .env file: {e}")
            # Set environment variables manually (use placeholder)
            os.environ["OPENAI_MODEL"] = "gpt-4o-mini"
            os.environ["OPENAI_API_KEY"] = "your-api-key-here"
        
        # Initialize LLM only if API key is available
        self.llm = None
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key and api_key != "your-api-key-here":
            try:
                self.llm = ChatOpenAI(
                    model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"), 
                    temperature=0
                )
                print("OpenAI LLM initialized successfully")
            except Exception as e:
                print(f"Warning: Could not initialize OpenAI LLM: {e}")
                print("LLM features will be disabled. Set OPENAI_API_KEY environment variable to enable.")
        else:
            print("Warning: OPENAI_API_KEY not set. LLM features will be disabled.")
            print("Set OPENAI_API_KEY environment variable to enable full functionality.")
        
        self.db_tool = DatabaseTool()
        
        # Initialize document preprocessor and vector store
        self.doc_preprocessor = DocumentPreprocessor(chunk_size=300, chunk_overlap=50)
        self.vector_store_loaded = self.doc_preprocessor.load_vector_store()
        
        if self.vector_store_loaded:
            print("Vector store loaded successfully - semantic search enabled")
        else:
            print("Vector store not available - falling back to simple search")
        
        # Initialize document summaries
        self.document_summaries = self._load_document_summaries()
    
    def analyze_query(self, query: str) -> QueryAnalysis:
        """
        Analyze a user query to determine the best approach (database, documents, or hybrid)
        """
        
        # Simple keyword-based analysis for faster processing
        query_lower = query.lower()
        
        # Document-related keywords - expanded and more specific
        doc_keywords = [
            'policy', 'policies', 'procedure', 'guideline', 'rule', 'remote work', 
            'remote', 'work', 'benefit', 'benefits', 'meeting', 'standup', 'review', 'documentation',
            'technical', 'architecture', 'api', 'system', 'hr', 'human resources',
            'performance management', 'employee handbook', 'work from home', 'what are',
            'health', 'insurance', 'plan', 'coverage', 'enrollment', 'qualifying', 'event',
            'mid-year', 'mid year', 'midyear', 'change', 'modify', 'update', 'switch', 'enroll',
            'dental', 'vision', 'medical', '401k', 'retirement', 'pto', 'vacation', 'sick',
            'leave', 'time off', 'holiday', 'pay', 'salary', 'compensation', 'perk',
            'assistance', 'counseling', 'transportation', 'technology', 'equipment',
            'faq', 'question', 'answer', 'how to', 'what if', 'when can', 'where to',
            'discussed', 'discussion', 'latest', 'engineering', 'team', 'updates', 'blockers', 'progress'
        ]
        
        # Database-related keywords
        db_keywords = [
            'employees', 'sales', 'revenue', 'project', 'budget',
            'department', 'salary', 'hire', 'count', 'how many', 'list all',
            'top', 'bottom', 'average', 'sum', 'total', 'manager', 'customer',
            'data', 'table', 'query', 'select', 'database'
        ]
        
        # Count matches with better logic
        doc_matches = 0
        db_matches = 0
        
        # Check for document keywords with phrase matching
        for keyword in doc_keywords:
            if keyword in query_lower:
                doc_matches += 1
                # Give extra weight for specific health/policy-related terms
                if keyword in ['health', 'plan', 'mid-year', 'mid year', 'midyear', 'change', 'insurance', 'benefit', 'benefits']:
                    doc_matches += 2
                # Give extra weight for meeting/standup terms
                if keyword in ['meeting', 'standup', 'discussed', 'discussion', 'engineering', 'team', 'updates', 'blockers', 'progress']:
                    doc_matches += 3
        
        # Check for database keywords
        for keyword in db_keywords:
            if keyword in query_lower and keyword != 'employee':  # 'employee' can be in both contexts
                db_matches += 1
                # Give extra weight for sales-related terms
                if keyword in ['sales', 'revenue', 'data']:
                    db_matches += 2
        
        # Initialize variables
        suggested_sql = None
        suggested_document_queries = None
        
        # Special case for health plan queries - these should always go to documents
        health_plan_indicators = ['health plan', 'health insurance', 'medical plan', 'mid year', 'mid-year', 'midyear']
        if any(indicator in query_lower for indicator in health_plan_indicators):
            query_type = QueryType.DOCUMENTS
            confidence = 0.95
            reasoning = "Health plan or mid-year change query - checking policy documents"
            suggested_document_queries = [query]
            return QueryAnalysis(
                query_type=query_type,
                confidence=confidence,
                reasoning=reasoning,
                suggested_sql=suggested_sql,
                suggested_document_queries=suggested_document_queries
            )
        
        # Special case for hybrid queries that combine sales data and policies
        hybrid_indicators = ['sales data and', 'sales and policies', 'data and policies', 'sales with policies']
        if any(indicator in query_lower for indicator in hybrid_indicators):
            query_type = QueryType.HYBRID
            confidence = 0.9
            reasoning = "Hybrid query requesting both sales data and policy information"
            # Use more specific document queries for sales-related policies
            suggested_document_queries = ["sales policies", "sales procedures", "sales guidelines"]
            suggested_sql = "SELECT date, product, revenue, customer_name FROM sales ORDER BY date DESC LIMIT 10"
            return QueryAnalysis(
                query_type=query_type,
                confidence=confidence,
                reasoning=reasoning,
                suggested_sql=suggested_sql,
                suggested_document_queries=suggested_document_queries
            )
        
        # Special case for project managers and meeting notes
        project_meeting_indicators = ['project manager', 'project managers', 'managers and meeting', 'managers and notes']
        if any(indicator in query_lower for indicator in project_meeting_indicators):
            query_type = QueryType.HYBRID
            confidence = 0.9
            reasoning = "Hybrid query requesting both project manager data and meeting notes"
            suggested_document_queries = ["meeting notes", "project meeting", "manager meeting"]
            suggested_sql = "SELECT p.name as project_name, e.name as manager_name, e.department FROM projects p JOIN employees e ON p.manager_id = e.id ORDER BY p.name"
            return QueryAnalysis(
                query_type=query_type,
                confidence=confidence,
                reasoning=reasoning,
                suggested_sql=suggested_sql,
                suggested_document_queries=suggested_document_queries
            )
        
        # Determine query type
        if doc_matches > 0 and db_matches == 0:
            query_type = QueryType.DOCUMENTS
            confidence = min(0.9, 0.5 + (doc_matches * 0.1))
            reasoning = f"Query contains {doc_matches} document-related keywords"
            suggested_document_queries = [query]
            return QueryAnalysis(
                query_type=query_type,
                confidence=confidence,
                reasoning=reasoning,
                suggested_sql=None,
                suggested_document_queries=suggested_document_queries
            )
        elif db_matches > 0 and doc_matches == 0:
            query_type = QueryType.DATABASE
            confidence = min(0.9, 0.5 + (db_matches * 0.1))
            reasoning = f"Query contains {db_matches} database-related keywords"
            suggested_sql = None  # Will be generated later
            return QueryAnalysis(
                query_type=query_type,
                confidence=confidence,
                reasoning=reasoning,
                suggested_sql=suggested_sql,
                suggested_document_queries=None
            )
        elif doc_matches > 0 and db_matches > 0:
            # If both are present, prioritize documents for policy and health-related queries
            if any(word in query_lower for word in ['policy', 'policies', 'remote work', 'remote', 'work', 'health', 'insurance', 'plan', 'benefit', 'benefits']):
                query_type = QueryType.DOCUMENTS
                confidence = 0.9
                reasoning = f"Policy/benefits-related query with {doc_matches} document keywords"
                suggested_document_queries = [query]
                return QueryAnalysis(
                    query_type=query_type,
                    confidence=confidence,
                    reasoning=reasoning,
                    suggested_sql=None,
                    suggested_document_queries=suggested_document_queries
                )
            else:
                query_type = QueryType.HYBRID
                confidence = 0.8
                reasoning = f"Query contains both document ({doc_matches}) and database ({db_matches}) keywords"
                suggested_document_queries = [query]
                suggested_sql = None
                return QueryAnalysis(
                    query_type=query_type,
                    confidence=confidence,
                    reasoning=reasoning,
                    suggested_sql=suggested_sql,
                    suggested_document_queries=suggested_document_queries
                )
        else:
            # Check for conversational queries
            conversational_indicators = ['how are you', 'hello', 'hi', 'good morning', 'good afternoon', 'good evening', 'thanks', 'thank you']
            if any(indicator in query_lower for indicator in conversational_indicators):
                query_type = QueryType.CONVERSATIONAL
                confidence = 0.95
                reasoning = "Conversational query - providing friendly response"
                return QueryAnalysis(
                    query_type=query_type,
                    confidence=confidence,
                    reasoning=reasoning,
                    suggested_sql=None,
                    suggested_document_queries=None
                )
        
        # Fallback: try to determine based on question words and content
        if any(word in query_lower for word in ['what', 'how', 'when', 'where', 'why', 'can']):
            if any(word in query_lower for word in ['policy', 'procedure', 'guideline', 'remote work', 'remote', 'health', 'insurance', 'plan', 'benefit', 'benefits', 'mid-year', 'mid year']):
                query_type = QueryType.DOCUMENTS
                confidence = 0.7
                reasoning = "Question about policies/benefits/procedures"
                suggested_document_queries = [query]
            else:
                query_type = QueryType.DATABASE
                confidence = 0.6
                reasoning = "General question, defaulting to database"
                suggested_sql = None
        else:
            query_type = QueryType.DOCUMENTS  # Changed from UNKNOWN to DOCUMENTS as fallback
            confidence = 0.5
            reasoning = "Unclear query type, defaulting to document search"
            suggested_document_queries = [query]
        
        return QueryAnalysis(
            query_type=query_type,
            confidence=confidence,
            reasoning=reasoning,
            suggested_sql=suggested_sql,
            suggested_document_queries=suggested_document_queries
        )
    
    def _load_document_summaries(self) -> Dict[str, Dict[str, Any]]:
        """Load document summaries with keywords and descriptions"""
        return {
            "employee_benefits.txt": {
                "keywords": ["benefits", "health insurance", "dental", "vision", "401k", "retirement", "pto", "vacation", "sick leave", "life insurance", "disability", "enrollment", "mid year", "mid-year", "qualifying events", "cobra", "vesting"],
                "description": "Comprehensive employee benefits policy covering health, dental, vision, retirement, PTO, and enrollment procedures",
                "questions": ["What are the employee benefits?", "How do I enroll in benefits?", "Can I change my health plan mid-year?", "What is the 401k match?", "How much PTO do I get?", "What happens to my benefits if I leave?"]
            },
            "remote_work_policy.txt": {
                "keywords": ["remote work", "work from home", "hybrid", "telecommute", "equipment", "ergonomic", "internet", "security", "communication", "performance", "expectations", "eligibility"],
                "description": "Remote work policy covering eligibility, expectations, equipment, security, and performance standards",
                "questions": ["What are the remote work policies?", "Am I eligible for remote work?", "What equipment is provided?", "What are the work hours for remote work?", "How is remote work performance measured?"]
            },
            "performance_management.txt": {
                "keywords": ["performance", "review", "evaluation", "goals", "feedback", "appraisal", "rating", "improvement", "discipline", "appeal", "process"],
                "description": "Performance management guidelines including reviews, feedback, goals, and improvement processes",
                "questions": ["How does performance management work?", "What is the review process?", "How do I appeal a performance rating?", "What are the performance goals?", "How often are reviews conducted?"]
            },
            "engineering_team_standup.txt": {
                "keywords": ["standup", "meeting", "engineering", "team", "updates", "blockers", "progress", "daily", "agenda"],
                "description": "Engineering team daily standup meeting notes and updates",
                "questions": ["What was discussed in the engineering standup?", "What are the current blockers?", "What progress was made?", "What are the team updates?"]
            },
            "executive_quarterly_review.txt": {
                "keywords": ["executive", "quarterly", "review", "business", "strategy", "goals", "performance", "financial", "leadership"],
                "description": "Executive quarterly business review covering strategy, goals, and performance",
                "questions": ["What was discussed in the executive review?", "What are the business goals?", "How is the company performing?", "What is the strategic direction?"]
            },
            "product_development_meeting.txt": {
                "keywords": ["product", "development", "meeting", "features", "roadmap", "priorities", "timeline", "requirements"],
                "description": "Product development meeting notes covering features, roadmap, and priorities",
                "questions": ["What was discussed in the product meeting?", "What features are being developed?", "What is the product roadmap?", "What are the development priorities?"]
            },
            "api_integration_guide.txt": {
                "keywords": ["api", "integration", "guide", "documentation", "endpoints", "authentication", "examples", "code"],
                "description": "API integration guide with endpoints, authentication, and code examples",
                "questions": ["How do I integrate with the API?", "What are the API endpoints?", "How do I authenticate?", "Are there code examples?"]
            },
            "system_architecture.txt": {
                "keywords": ["architecture", "system", "design", "components", "infrastructure", "technology", "stack", "diagram"],
                "description": "System architecture documentation covering components, infrastructure, and technology stack",
                "questions": ["What is the system architecture?", "What technologies are used?", "How is the system designed?", "What are the main components?"]
            }
        }
    
    def _summarize_schema(self, schema: Dict[str, List[Dict[str, Any]]]) -> str:
        """Create a human-readable summary of the database schema"""
        summary = []
        for table, columns in schema.items():
            col_names = [col["name"] for col in columns]
            summary.append(f"- {table}: {', '.join(col_names)}")
        return "\n".join(summary)
    
    def generate_sql(self, query: str) -> Optional[str]:
        """
        Generate SQL from natural language query
        """
        # Simple keyword-based SQL generation for faster processing
        query_lower = query.lower()
        
        # Common SQL patterns
        if "employee" in query_lower and "sales" in query_lower:
            if "top" in query_lower:
                # Extract the number from the query (top 3, top 5, top 10, etc.)
                import re
                number_match = re.search(r'top\s+(\d+)', query_lower)
                limit = int(number_match.group(1)) if number_match else 5
                return f"SELECT e.name, e.department, SUM(s.revenue) as total_revenue FROM employees e JOIN sales s ON e.id = s.employee_id GROUP BY e.id ORDER BY total_revenue DESC LIMIT {limit}"
            else:
                return "SELECT e.name, e.department, s.revenue, s.product FROM employees e JOIN sales s ON e.id = s.employee_id ORDER BY s.revenue DESC LIMIT 10"
        
        elif "employee" in query_lower and "department" in query_lower:
            if "count" in query_lower or "how many" in query_lower:
                dept = "Engineering" if "engineering" in query_lower else "Sales" if "sales" in query_lower else "Marketing"
                return f"SELECT COUNT(*) as employee_count FROM employees WHERE department = '{dept}'"
            else:
                return "SELECT name, department, role FROM employees ORDER BY department, name LIMIT 10"
        
        elif "project" in query_lower:
            if "manager" in query_lower and ("meeting" in query_lower or "notes" in query_lower):
                return "SELECT p.name as project_name, e.name as manager_name, e.department FROM projects p JOIN employees e ON p.manager_id = e.id ORDER BY p.name"
            elif "active" in query_lower:
                return "SELECT p.name, p.description, p.budget, e.name as manager FROM projects p JOIN employees e ON p.manager_id = e.id WHERE p.status = 'Active' ORDER BY p.priority"
            else:
                return "SELECT name, status, budget, priority FROM projects ORDER BY priority DESC LIMIT 10"
        
        elif "sales" in query_lower:
            if "performance" in query_lower and "category" in query_lower:
                return "SELECT category, SUM(revenue) as total_revenue, COUNT(*) as sales_count FROM sales GROUP BY category ORDER BY total_revenue DESC"
            else:
                return "SELECT date, product, revenue, customer_name FROM sales ORDER BY date DESC LIMIT 10"
        
        elif "hire" in query_lower or "joined" in query_lower:
            return "SELECT name, department, hire_date FROM employees WHERE hire_date >= date('now', '-1 year') ORDER BY hire_date DESC"
        
        # Fallback to LLM-based generation for complex queries (only if LLM is available)
        if self.llm:
            try:
                db_schema = self.db_tool.get_schema()
                schema_summary = self._summarize_schema(db_schema)
                
                sql_prompt = f"""
You are an expert SQL generator. Convert the natural language query into a valid SQLite SELECT statement.

Available database schema:
{schema_summary}

Important guidelines:
- Only generate SELECT statements (no INSERT, UPDATE, DELETE)
- Use proper SQLite syntax
- Include appropriate JOINs when needed
- Use aggregation functions (COUNT, SUM, AVG, etc.) when appropriate
- Limit results when appropriate (LIMIT clause)
- Use proper date functions for date comparisons
- Return only the SQL query, no explanations

Query: "{query}"

SQL:"""

                response = self.llm.invoke(sql_prompt)
                sql = response.content.strip()
                
                # Basic validation
                if not sql.upper().startswith("SELECT"):
                    return None
                
                # Test if SQL is valid
                try:
                    self.db_tool.execute_sql(sql)
                    return sql
                except Exception:
                    return None
                    
            except Exception as e:
                print(f"Error generating SQL with LLM: {e}")
                return None
        else:
            print("LLM not available for complex SQL generation. Using keyword-based approach only.")
            return None
    
    def execute_database_query(self, sql: str) -> Dict[str, Any]:
        """Execute a database query and return results"""
        try:
            return self.db_tool.execute_sql(sql)
        except Exception as e:
            return {"error": str(e), "rows": [], "count": 0}
    
    def execute_document_search(self, queries: List[str]) -> List[Dict[str, Any]]:
        """Execute document searches and return results"""
        # Always try vector search first for semantic search
        if self.vector_store_loaded:
            print("Using vector search for semantic document retrieval...")
            vector_results = self._vector_document_search(queries)
            if vector_results:
                return vector_results
        
        # Fall back to simple search only if vector search fails
        print("Falling back to simple document search...")
        return self._simple_document_search(queries)
    
    def _vector_document_search(self, queries: List[str]) -> List[Dict[str, Any]]:
        """Search documents using vector similarity"""
        all_results = []
        
        for query in queries:
            print(f"Searching for: '{query}' using vector search...")
            
            # Get vector search results with more candidates
            vector_results = self.doc_preprocessor.search_documents(query, k=10)
            
            if vector_results:
                print(f"Found {len(vector_results)} vector search results")
                
                # Re-rank results for better relevance
                re_ranked_results = self.doc_preprocessor.re_rank_results(vector_results, query)
                
                # Extract precise answers from the best results
                for result in re_ranked_results[:3]:  # Top 3 results
                    print(f"Processing result from: {result['source']} (score: {result['score']:.3f})")
                    precise_answer = self._extract_precise_answer(result["content"], query)
                    result["content"] = precise_answer
                    all_results.append(result)
                
                # If we found good vector results, return them
                if all_results:
                    print(f"Returning {len(all_results)} vector search results")
                    return all_results
            else:
                print("No vector search results found")
        
        print("No good vector results, falling back to simple search")
        return self._simple_document_search(queries)
    
    def _extract_precise_answer(self, content: str, query: str) -> str:
        """Extract a precise answer from content based on the query using RAG discipline"""
        if not self.llm:
            # Fallback to simple extraction if LLM not available
            return self._simple_extract_answer(content, query)
        
        # Use RAG system prompt for intelligent answer generation
        rag_prompt = f"""You are an AI assistant that answers user questions based only on the retrieved context provided. 
Follow these steps strictly:

1. Read the user's question carefully. Identify the specific information being asked.
2. Examine the retrieved context (chunks of documents). 
   - Select only the parts that directly answer the question. 
   - Ignore irrelevant or generic sentences.
3. Generate a clear, concise answer in your own words, grounded in the retrieved context. 
   - Do not just repeat entire chunks of text. 
   - Summarize or rephrase if needed.
4. If the retrieved context does not contain the answer, say: 
   "I could not find that information in the available documents."
5. Never invent or guess information. Do not use outside knowledge.
6. If the user's query is ambiguous, ask a clarifying question before answering.

Output format:
- Direct answer to the question in natural language.
- If useful, you may cite or briefly quote the relevant part of the context.

User Question: "{query}"

Retrieved Context:
{content}

Answer:"""

        try:
            response = self.llm.invoke(rag_prompt)
            answer = response.content.strip()
            
            # Ensure the answer is not too long and is actually answering the question
            if len(answer) > 500:
                # If too long, ask for a more concise version
                concise_prompt = f"""The previous answer was too long. Please provide a more concise answer to this question: "{query}"

Previous answer: {answer}

Please provide a shorter, more focused answer (under 200 words):"""
                
                response = self.llm.invoke(concise_prompt)
                answer = response.content.strip()
            
            return answer
            
        except Exception as e:
            print(f"Error generating RAG answer: {e}")
            # Fallback to simple extraction
            return self._simple_extract_answer(content, query)
    
    def _simple_extract_answer(self, content: str, query: str) -> str:
        """Simple fallback answer extraction when LLM is not available"""
        query_lower = query.lower()
        
        # For health plan mid-year changes, look for the specific FAQ
        if 'mid' in query_lower and 'year' in query_lower and 'health' in query_lower:
            # Look for the specific Q&A pattern
            lines = content.split('\n')
            for i, line in enumerate(lines):
                if 'Q:' in line and 'mid' in line.lower() and 'year' in line.lower():
                    # Found the question, get the answer from next line
                    if i + 1 < len(lines) and 'A:' in lines[i + 1]:
                        return f"{line.strip()}\n{lines[i + 1].strip()}"
                    # Or look for answer in the same line
                    elif 'A:' in line:
                        return line.strip()
        
        # For employee benefits queries, look for the overview section
        if 'benefit' in query_lower and 'employee' in query_lower:
            lines = content.split('\n')
            for i, line in enumerate(lines):
                if 'OVERVIEW' in line and i + 1 < len(lines):
                    # Get the overview paragraph
                    overview = lines[i + 1].strip()
                    if overview:
                        return overview
                    # If no overview text, get the first meaningful paragraph
                    for j in range(i + 1, min(i + 10, len(lines))):
                        if lines[j].strip() and not lines[j].strip().startswith(('1.', '2.', '3.')):
                            return lines[j].strip()
        
        # For remote work queries, look for specific policy details
        if 'remote' in query_lower and 'work' in query_lower:
            lines = content.split('\n')
            for i, line in enumerate(lines):
                if 'PURPOSE' in line and i + 1 < len(lines):
                    purpose = lines[i + 1].strip()
                    if purpose:
                        return purpose
        
        # For meeting/standup queries, extract key highlights
        if any(word in query_lower for word in ['standup', 'meeting', 'discussed', 'discussion']):
            # Return the full content so it can be properly formatted
            return content
        
        # For other queries, look for Q&A patterns
        if 'Q:' in content and 'A:' in content:
            # Simple Q&A extraction
            qa_start = content.find('Q:')
            if qa_start != -1:
                qa_end = content.find('Q:', qa_start + 2)
                if qa_end == -1:
                    qa_end = len(content)
                qa_section = content[qa_start:qa_end].strip()
                return qa_section
        
        # Fallback: return first meaningful paragraph
        lines = content.split('\n')
        for line in lines:
            line = line.strip()
            if line and len(line) > 20 and not line.startswith(('1.', '2.', '3.', '4.', '5.')):
                return line
        
        # Final fallback: return first 200 characters
        return content[:200] + "..." if len(content) > 200 else content
    
    def _summarize_text(self, text: str, max_length: int = 200) -> str:
        """Summarize text content if it's too long using LLM"""
        if len(text) <= max_length:
            return text
        
        # Use LLM for summarization if available
        if self.llm:
            try:
                summary_prompt = f"""Summarize the following text in a concise, informative way. Keep it under {max_length} characters and preserve the most important information:

{text}

Summary:"""
                
                response = self.llm.invoke(summary_prompt)
                summary = response.content.strip()
                
                # Ensure summary is not longer than original and fits within limit
                if len(summary) < len(text) and len(summary) <= max_length:
                    return summary
            except Exception as e:
                print(f"Error summarizing with LLM: {e}")
        
        # Fallback: simple truncation if LLM is not available or fails
        if len(text) > max_length:
            return text[:max_length-3] + "..."
        
        return text

    def _simple_document_search(self, queries: List[str]) -> List[Dict[str, Any]]:
        """Enhanced document search using document summaries and keywords"""
        import os
        import glob
        
        results = []
        
        # Search in documents directory
        doc_dir = "documents"
        if not os.path.exists(doc_dir):
            return results
        
        # Get all text files recursively
        text_files = []
        for pattern in ["**/*.txt", "**/*.md"]:
            files = glob.glob(os.path.join(doc_dir, pattern), recursive=True)
            text_files.extend(files)
        
        for query in queries:
            query_lower = query.lower().strip()
            query_words = [word for word in query_lower.split() if len(word) > 2]
            
            # Step 1: Search using document summaries (fastest and most accurate)
            summary_matches = []
            for filename, summary in self.document_summaries.items():
                score = 0
                
                # Check keyword matches
                for keyword in summary["keywords"]:
                    if keyword in query_lower:
                        score += 2.0  # High weight for keyword matches
                
                # Check question similarity
                for question in summary["questions"]:
                    question_lower = question.lower()
                    # Check if query words appear in the question
                    word_matches = sum(1 for word in query_words if word in question_lower)
                    if word_matches > 0:
                        score += (word_matches / len(query_words)) * 1.5
                
                # Check description similarity
                desc_lower = summary["description"].lower()
                desc_word_matches = sum(1 for word in query_words if word in desc_lower)
                if desc_word_matches > 0:
                    score += (desc_word_matches / len(query_words)) * 1.0
                
                if score > 0.5:  # Threshold for summary-based matches
                    summary_matches.append((filename, score))
            
            # Sort summary matches by score
            summary_matches.sort(key=lambda x: x[1], reverse=True)
            
            # Step 2: If we found good summary matches, use them
            if summary_matches:
                best_match = summary_matches[0]
                filename, score = best_match
                
                # Find the actual file
                for file_path in text_files:
                    if os.path.basename(file_path) == filename:
                        try:
                            with open(file_path, 'r', encoding='utf-8') as f:
                                content = f.read()
                            
                            if content.strip():
                                matched_content = self._extract_precise_answer(content, query)
                                results.append({
                                    "content": matched_content.strip(),
                                    "source": filename,
                                    "score": score,
                                    "method": "summary_match"
                                })
                                return results  # Return the best match
                        except Exception:
                            continue
            
            # Step 3: Fallback to content-based search if no good summary matches
            content_matches = []
            for file_path in text_files:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    if not content.strip():
                        continue
                    
                    content_lower = content.lower()
                    filename = os.path.basename(file_path)
                    
                    # Content-based scoring
                    score = 0
                    
                    # Exact query match
                    if query_lower in content_lower:
                        score = 1.0
                    
                    # Word matching
                    word_matches = sum(1 for word in query_words if word in content_lower)
                    if word_matches > 0:
                        score = (word_matches / len(query_words)) * 0.8
                    
                    # Phrase matching
                    phrases = ['mid year', 'mid-year', 'health plan', 'health insurance', 
                              'remote work', 'benefits', 'performance', 'policy', 'policies']
                    for phrase in phrases:
                        if phrase in query_lower and phrase in content_lower:
                            score += 0.3
                    
                    if score > 0.3:
                        matched_content = self._extract_precise_answer(content, query)
                        content_matches.append({
                            "content": matched_content.strip(),
                            "source": filename,
                            "score": score,
                            "method": "content_search"
                        })
                        
                except Exception:
                    continue
            
            # Step 4: Return the best content match if no summary matches
            if content_matches:
                content_matches.sort(key=lambda x: x["score"], reverse=True)
                results.append(content_matches[0])
                return results
        
        # Step 5: No matches found
        return results
    
    def format_database_response(self, results: dict, limit: int = None) -> str:
        if "error" in results:
            return f"<p style='color:red;'>Database error: {results['error']}</p>"
        
        rows = results.get("rows", [])
        if not rows:
            return "<p>No data found matching your query.</p>"
        
        df = pd.DataFrame(rows)

        # Apply client-requested limit if given
        if limit is not None and limit < len(df):
            df = df.head(limit)
        
        # Clean up column names for better display
        column_rename = {}
        for col in df.columns:
            if col == 'total_revenue':
                column_rename[col] = 'Total Revenue'
            elif col == 'employee_count':
                column_rename[col] = 'Employee Count'
            elif col == 'sales_count':
                column_rename[col] = 'Sales Count'
            elif col == 'hire_date':
                column_rename[col] = 'Hire Date'
            elif col == 'customer_name':
                column_rename[col] = 'Customer Name'
            elif col == 'budget':
                column_rename[col] = 'Budget'
            elif col == 'project_name':
                column_rename[col] = 'Project Name'
            elif col == 'manager_name':
                column_rename[col] = 'Manager Name'
            else:
                # Capitalize first letter of each word
                column_rename[col] = col.replace('_', ' ').title()
        
        df = df.rename(columns=column_rename)
        
        # Format numeric values nicely
        for col in df.columns:
            if ('Revenue' in col or 'Budget' in col) and df[col].dtype in ['float64', 'int64']:
                df[col] = df[col].apply(lambda x: f"${x:,.2f}" if pd.notnull(x) else "")
            elif df[col].dtype in ['float64', 'int64'] and col not in ['Revenue', 'Budget']:
                df[col] = df[col].apply(lambda x: f"{x:,.0f}" if pd.notnull(x) else "")
        
        styled = (
            df.style
            .set_table_styles(
                [
                    {"selector": "table",
                    "props": "border-collapse: separate; border-spacing: 0; border: none; "
                            "border-radius: 12px; overflow: hidden; "
                            "font-family: 'Segoe UI', sans-serif; font-size: 14px; margin: 15px 0; "
                            "box-shadow: 0 8px 25px rgba(0,0,0,0.1), 0 4px 10px rgba(0,0,0,0.05);"},
                    {"selector": "thead th",
                    "props": "background: #007AFF; color: white; "
                            "padding: 14px 12px; text-align: left; font-weight: 600; border: none; "
                            "text-shadow: 0 1px 3px rgba(0,0,0,0.3); font-size: 13px; letter-spacing: 0.5px;"},
                    {"selector": "thead th:first-child",
                    "props": "border-top-left-radius: 15px;"},
                    {"selector": "thead th:last-child", 
                    "props": "border-top-right-radius: 15px;"},
                    {"selector": "tbody td",
                    "props": "padding: 12px; border: none; border-bottom: 1px solid #e9ecef; text-align: left; background-color: white; transition: all 0.2s ease;"},
                    {"selector": "tbody tr",
                    "props": "background-color: white; transition: all 0.2s ease;"},
                    {"selector": "tbody tr:hover",
                    "props": "background-color: #f8f9fa; transform: translateY(-1px); box-shadow: 0 2px 8px rgba(0,0,0,0.1);"},
                    {"selector": "tbody tr:last-child td:first-child",
                    "props": "border-bottom-left-radius: 15px;"},
                    {"selector": "tbody tr:last-child td:last-child",
                    "props": "border-bottom-right-radius: 15px;"},
                ]
            )
            .hide(axis="index")
        )
        
        return styled.to_html()
        
    
    def format_document_response(self, results: List[Dict[str, Any]]) -> str:
        """Format document search results into a readable response"""
        if not results:
            # Provide helpful feedback when no documents are found
            available_topics = []
            for filename, summary in self.document_summaries.items():
                available_topics.extend(summary["keywords"][:3])  # Show first 3 keywords per document
            
            unique_topics = list(set(available_topics))[:10]  # Limit to 10 unique topics
            
            return f"""I couldn't find a specific answer to your question in our documents. 

Available topics in our knowledge base include: {', '.join(unique_topics)}

Please try rephrasing your question or ask about one of these topics. For example:
- "What are the employee benefits?"
- "How does remote work work?"
- "What is the performance review process?"
- "What was discussed in recent meetings?" """
        
        # Show the best result - content is already processed by RAG prompt
        best_result = results[0]
        content = best_result["content"]
        source = best_result["source"]
        
        # The content is already processed by the RAG prompt, so return it directly
        return content.strip()
    
    def _format_document_content(self, content: str) -> str:
        """Format document content for better readability with proper bullet points and structure"""
        if not content:
            return content
        
        lines = content.split('\n')
        formatted_lines = []
        
        for line in lines:
            line = line.strip()
            if not line:
                formatted_lines.append('')
                continue
            
            # Handle bullet points (•, -, *)
            if line.startswith(('•', '-', '*')):
                # Ensure bullet points start at the beginning of the line
                bullet_char = line[0]
                rest_of_line = line[1:].strip()
                formatted_lines.append(f"{bullet_char} {rest_of_line}")
            
            # Handle section headers (ALL CAPS with optional spaces/dashes)
            elif line.isupper() and len(line) > 5:
                formatted_lines.append('')  # Add space before headers
                formatted_lines.append(f"**{line}**")
                formatted_lines.append('')  # Add space after headers
            
            # Handle numbered lists (1., 2., etc.)
            elif line and line[0].isdigit() and ('.' in line[:5] or ')' in line[:5]):
                formatted_lines.append(line)
            
            # Handle lines with dates/times (contains Date:, Time:, Location:)
            elif any(keyword in line for keyword in ['Date:', 'Time:', 'Location:', 'STATUS', 'GOAL', 'TARGET']):
                formatted_lines.append(f"**{line}**")
            
            # Regular lines
            else:
                formatted_lines.append(line)
        
        # Join lines and clean up multiple empty lines
        formatted_text = '\n'.join(formatted_lines)
        
        # Clean up excessive whitespace
        while '\n\n\n' in formatted_text:
            formatted_text = formatted_text.replace('\n\n\n', '\n\n')
        
        return formatted_text.strip()
    
    def process_query(self, query: str) -> str:
        """
        Main method to process a user query intelligently
        """
        # Step 1: Analyze the query
        analysis = self.analyze_query(query)
        
        response_parts = []
        
        # Step 2: Handle database queries
        if analysis.query_type in [QueryType.DATABASE, QueryType.HYBRID]:
            sql = analysis.suggested_sql or self.generate_sql(query)
            
            if sql:
                db_results = self.execute_database_query(sql)
                db_response = self.format_database_response(db_results)
                response_parts.append(db_response)
            else:
                response_parts.append("Could not generate a valid SQL query for your request.")
        
        # Step 3: Handle document queries
        if analysis.query_type in [QueryType.DOCUMENTS, QueryType.HYBRID]:
            doc_queries = analysis.suggested_document_queries or [query]
            doc_results = self.execute_document_search(doc_queries)
            
            if doc_results:
                doc_response = self.format_document_response(doc_results)
                response_parts.append(doc_response)
            else:
                response_parts.append("No relevant documents found for your query.")
        
        # Step 4: Handle conversational queries
        if analysis.query_type == QueryType.CONVERSATIONAL:
            response_parts.append(
                "Hello! I'm doing well, thank you for asking! 😊 I'm here to help you with questions about:\n"
                "• Employee benefits and policies\n"
                "• Sales data and performance\n"
                "• Project information\n"
                "• Meeting notes and updates\n"
                "• Technical documentation\n\n"
                "What would you like to know about?"
            )
        
        # Step 5: Handle unknown queries (this should rarely happen now)
        if analysis.query_type == QueryType.UNKNOWN:
            # Try document search as fallback
            doc_results = self.execute_document_search([query])
            if doc_results:
                doc_response = self.format_document_response(doc_results)
                response_parts.append(doc_response)
            else:
                response_parts.append(
                    "I'm not sure how to best answer your query. "
                    "Could you please rephrase it or be more specific about what information you're looking for?"
                )
        
        # Combine responses
        if len(response_parts) == 1:
            return response_parts[0]
        else:
            return "\n\n".join(response_parts)


def main():
    """Test the intelligent agent with sample queries"""
    agent = IntelligentAgent()
    
    test_queries = [
        "Show me the top 5 employees by sales revenue",
        "What are the remote work policies?",
        "List all active projects and their managers",
        "What was discussed in the latest engineering standup?",
        "How many employees are in the Engineering department?",
        "What are the employee benefits?",
        "Show sales performance by product category",
        "What is the system architecture?",
        "Find employees who joined in the last year",
        "What are the performance management guidelines?",
        "mid year health plan change",  # Added the specific query
        "health plan mid year change",
        "can I change my health insurance mid year"
    ]
    
    print("=== Intelligent Agent Test ===\n")
    
    for i, query in enumerate(test_queries, 1):
        print(f"Query {i}: {query}")
        print("-" * 50)
        response = agent.process_query(query)
        print(response)
        print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    main()