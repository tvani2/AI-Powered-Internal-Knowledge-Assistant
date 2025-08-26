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
from conversation_handler import ConversationHandler
from document_handler import DocumentHandler
from database_handler import DatabaseHandler


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

from hybrid_handler import HybridHandler


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
        
        # Initialize database handler
        self.database_handler = DatabaseHandler()
        
        # Initialize document handler (encapsulates document search utilities)
        self.document_handler = DocumentHandler(llm=self.llm)
        
        # Initialize hybrid handler (controller)
        self.hybrid_handler = HybridHandler(self.database_handler, self.document_handler)
        
        # Initialize document summaries (via handler)
        self.document_summaries = self.document_handler._load_document_summaries()
        
        # Initialize conversation handler
        self.conversation = ConversationHandler()
    
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
                db_schema = self.database_handler.get_schema()
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
                if self.database_handler.try_execute_for_validation(sql):
                    return sql
                    return None
                    
            except Exception as e:
                print(f"Error generating SQL with LLM: {e}")
                return None
        else:
            print("LLM not available for complex SQL generation. Using keyword-based approach only.")
            return None
    
    def execute_database_query(self, sql: str) -> Dict[str, Any]:
        """Execute a database query and return results"""
        return self.database_handler.execute_database_query(sql)
    
    def format_database_response(self, results: dict, limit: int = None) -> str:
        return self.database_handler.format_database_response(results, limit)
    
    def execute_document_search(self, queries: List[str]) -> List[Dict[str, Any]]:
        """Execute document searches and return results"""
        return self.document_handler.execute_document_search(queries)
    
    def _vector_document_search(self, queries: List[str]) -> List[Dict[str, Any]]:
        """Search documents using vector similarity"""
        return self.document_handler._vector_document_search(queries)
    
    def _extract_precise_answer(self, content: str, query: str) -> str:
        """Extract a precise answer from content based on the query using RAG discipline"""
        return self.document_handler._extract_precise_answer(content, query)
    
    def _simple_extract_answer(self, content: str, query: str) -> str:
        """Simple fallback answer extraction when LLM is not available"""
        return self.document_handler._simple_extract_answer(content, query)
    
    def _summarize_text(self, text: str, max_length: int = 200) -> str:
        """Summarize text content if it's too long using LLM"""
        return self.document_handler._summarize_text(text, max_length)

    def _simple_document_search(self, queries: List[str]) -> List[Dict[str, Any]]:
        """Enhanced document search using document summaries and keywords"""
        return self.document_handler._simple_document_search(queries)
    
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
        """Format document content for better readability with proper HTML structure"""
        return self.document_handler._format_document_content(content)
    
    def process_query(self, query: str) -> str:
        """
        Main method to process a user query intelligently
        """
        # Step 1: Check for personal/out-of-scope questions first
        personal_questions = self._is_personal_question(query)
        if personal_questions:
            return self._handle_personal_question(query)
        
        # Step 2: Handle conversational queries
        if self._is_conversational_query(query):
            return self._handle_conversational_query(query)
        
        # Step 3: Controller logic - try SQL first, then fallback to documents
        return self._answer_query_controller(query)
    
    def _answer_query_controller(self, query: str) -> str:
        """
        Controller layer that implements the decision tree:
        1. Try SQL query first
        2. If SQL fails, try document retrieval
        3. If both fail, return error message
        """
        analysis = self.analyze_query(query)
        return self.hybrid_handler.answer_query(
            query=query,
            analysis=analysis,
            generate_sql_fn=self.generate_sql,
            format_db_fn=self.format_database_response,
            analyze_query_fn=self.analyze_query,
        )
    
    def _run_sql_query(self, query: str) -> Optional[Dict[str, Any]]:
        """Run SQL query and return results if successful"""
        analysis = self.analyze_query(query)
        return self.hybrid_handler._run_sql_query(query, analysis, self.generate_sql)
    
    def _format_sql_result(self, sql_result: Dict[str, Any]) -> str:
        """Format SQL results into clean natural language"""
        return self.hybrid_handler._format_sql_result(sql_result, self.format_database_response)
    
    def _run_doc_retrieval(self, query: str) -> Optional[List[Dict[str, Any]]]:
        """Run document retrieval and return results if successful"""
        return self.hybrid_handler._run_doc_retrieval(query, self.analyze_query)
    
    def _format_doc_result(self, doc_results: List[Dict[str, Any]]) -> str:
        """Format document results into clean natural language"""
        return self.hybrid_handler._format_doc_result(doc_results)
    
    def _is_conversational_query(self, query: str) -> bool:
        """Check if the query is conversational or a closing."""
        return self.conversation.is_conversational_query(query) or self.conversation.is_closing(query)
    
    def _handle_conversational_query(self, query: str) -> str:
        """Handle conversational queries or closings via ConversationHandler."""
        if self.conversation.is_closing(query):
            return self.conversation.handle_closing(query)
        return self.conversation.handle_conversational_query(query)
    
    def _is_personal_question(self, query: str) -> bool:
        """Delegate personal question detection to ConversationHandler."""
        return self.conversation.is_personal_question(query)
    
    def _handle_personal_question(self, query: str) -> str:
        """Delegate personal question response to ConversationHandler."""
        return self.conversation.handle_personal_question(query)


def main():
    """Test the intelligent agent with sample queries"""
    agent = IntelligentAgent()
    
if __name__ == "__main__":
    main()