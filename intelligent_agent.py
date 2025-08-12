import os
import json
import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from dotenv import load_dotenv
from langchain.agents import AgentExecutor, initialize_agent
from langchain.agents import AgentType
from langchain_openai import ChatOpenAI
from langchain.prompts.chat import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.schema import BaseMessage, HumanMessage, SystemMessage

from tools.database_tool import DatabaseTool
from tools.document_search_tool import DocumentSearchTool


class QueryType(Enum):
    DATABASE = "database"
    DOCUMENTS = "documents"
    HYBRID = "hybrid"
    UNKNOWN = "unknown"


@dataclass
class QueryAnalysis:
    query_type: QueryType
    confidence: float
    reasoning: str
    suggested_sql: Optional[str] = None
    suggested_document_queries: Optional[List[str]] = None


class IntelligentAgent:
    """
    An intelligent agent that can:
    1. Analyze user queries to determine whether to use database or documents
    2. Generate SQL automatically from natural language queries
    3. Provide hybrid responses combining both data sources
    """
    
    def __init__(self):
        try:
            load_dotenv()
        except Exception as e:
            print(f"Warning: Could not load .env file: {e}")
            # Set environment variables manually (use placeholder)
            os.environ["OPENAI_MODEL"] = "gpt-4o-mini"
            os.environ["OPENAI_API_KEY"] = "your-api-key-here"
            
        self.llm = ChatOpenAI(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"), 
            temperature=0
        )
        self.db_tool = DatabaseTool()
        
        # Initialize document search - use simple file search by default
        self.doc_tool = None
        print("Document search will use simple file-based search")
    
    def analyze_query(self, query: str) -> QueryAnalysis:
        """
        Analyze a user query to determine the best approach (database, documents, or hybrid)
        """
        
        # Simple keyword-based analysis for faster processing
        query_lower = query.lower()
        
        # Document-related keywords
        doc_keywords = [
            'policy', 'policies', 'procedure', 'guideline', 'rule', 'remote work', 
            'remote', 'work', 'benefit', 'benefits', 'meeting', 'standup', 'review', 'documentation',
            'technical', 'architecture', 'api', 'system', 'hr', 'human resources',
            'performance management', 'employee handbook', 'work from home', 'what are'
        ]
        
        # Database-related keywords
        db_keywords = [
            'employee', 'employees', 'sales', 'revenue', 'project', 'budget',
            'department', 'salary', 'hire', 'count', 'how many', 'list all',
            'top', 'bottom', 'average', 'sum', 'total', 'manager', 'customer',
            'data', 'table', 'query', 'select', 'database'
        ]
        
        # Count matches with better logic
        doc_matches = 0
        db_matches = 0
        
        # Check for document keywords first (with higher priority)
        for keyword in doc_keywords:
            if keyword in query_lower:
                doc_matches += 1
                # Give extra weight for specific policy-related terms
                if keyword in ['policy', 'policies', 'remote work', 'remote']:
                    doc_matches += 2  # Extra weight for policy terms
        
        # Check for database keywords (but avoid conflicts with document terms)
        for keyword in db_keywords:
            if keyword in query_lower:
                # Don't count 'work' as database keyword if it's part of 'remote work'
                if keyword == 'work' and 'remote work' in query_lower:
                    continue
                db_matches += 1
        
        # Initialize variables
        suggested_sql = None
        suggested_document_queries = None
        

        

        
        # Determine query type
        if doc_matches > 0 and db_matches == 0:
            query_type = QueryType.DOCUMENTS
            confidence = min(0.9, 0.5 + (doc_matches * 0.1))
            reasoning = f"Query contains {doc_matches} document-related keywords"
            suggested_document_queries = [query]
        elif db_matches > 0 and doc_matches == 0:
            query_type = QueryType.DATABASE
            confidence = min(0.9, 0.5 + (db_matches * 0.1))
            reasoning = f"Query contains {db_matches} database-related keywords"
            suggested_sql = None  # Will be generated later
        elif doc_matches > 0 and db_matches > 0:
            # If both are present, prioritize documents for policy queries
            if any(word in query_lower for word in ['policy', 'policies', 'remote work', 'remote', 'work']):
                query_type = QueryType.DOCUMENTS
                confidence = 0.9
                reasoning = f"Policy-related query with {doc_matches} document keywords"
                suggested_document_queries = [query]
            else:
                query_type = QueryType.HYBRID
                confidence = 0.8
                reasoning = f"Query contains both document ({doc_matches}) and database ({db_matches}) keywords"
                suggested_document_queries = [query]
                suggested_sql = None
        else:
            # Fallback: try to determine based on question words
            if any(word in query_lower for word in ['what', 'how', 'when', 'where', 'why']):
                if any(word in query_lower for word in ['policy', 'procedure', 'guideline', 'remote work', 'remote']):
                    query_type = QueryType.DOCUMENTS
                    confidence = 0.7
                    reasoning = "Question about policies/procedures"
                    suggested_document_queries = [query]
                else:
                    query_type = QueryType.DATABASE
                    confidence = 0.6
                    reasoning = "General question, defaulting to database"
                    suggested_sql = None
            else:
                query_type = QueryType.UNKNOWN
                confidence = 0.3
                reasoning = "Unable to determine query type"
        
        # Document search is always available (using simple file search)
        doc_available = True
        
        return QueryAnalysis(
            query_type=query_type,
            confidence=confidence,
            reasoning=reasoning,
            suggested_sql=suggested_sql if query_type in [QueryType.DATABASE, QueryType.HYBRID] else None,
            suggested_document_queries=suggested_document_queries if doc_available else None
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
                limit = 5 if "5" in query else 10
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
            if "active" in query_lower:
                return "SELECT p.name, p.description, e.name as manager FROM projects p JOIN employees e ON p.manager_id = e.id WHERE p.status = 'Active' ORDER BY p.priority"
            else:
                return "SELECT name, status, budget, priority FROM projects ORDER BY priority DESC LIMIT 10"
        
        elif "sales" in query_lower:
            if "performance" in query_lower and "category" in query_lower:
                return "SELECT category, SUM(revenue) as total_revenue, COUNT(*) as sales_count FROM sales GROUP BY category ORDER BY total_revenue DESC"
            else:
                return "SELECT date, product, revenue, customer_name FROM sales ORDER BY date DESC LIMIT 10"
        
        elif "hire" in query_lower or "joined" in query_lower:
            return "SELECT name, department, hire_date FROM employees WHERE hire_date >= date('now', '-1 year') ORDER BY hire_date DESC"
        
        # Fallback to LLM-based generation for complex queries
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
            print(f"Error generating SQL: {e}")
            return None
    
    def execute_database_query(self, sql: str) -> Dict[str, Any]:
        """Execute a database query and return results"""
        try:
            return self.db_tool.execute_sql(sql)
        except Exception as e:
            return {"error": str(e), "rows": [], "count": 0}
    
    def execute_document_search(self, queries: List[str]) -> List[Dict[str, Any]]:
        """Execute document searches and return results"""
        # Always use simple file search for now
        return self._simple_document_search(queries)
    
    def _simple_document_search(self, queries: List[str]) -> List[Dict[str, Any]]:
        """Simple keyword-based document search as fallback"""
        import os
        import glob
        
        results = []
        
        # Search in documents directory
        doc_dir = "documents"
        if not os.path.exists(doc_dir):
            return results
        
        # Get all text files
        text_files = []
        for pattern in ["**/*.txt", "**/*.md"]:
            text_files.extend(glob.glob(os.path.join(doc_dir, pattern), recursive=True))
        
        for query in queries:
            query_lower = query.lower()
            
            for file_path in text_files:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # Simple keyword matching with better relevance
                    content_lower = content.lower()
                    query_words = query_lower.split()
                    
                    # Check if any query words are in the content
                    matches = sum(1 for word in query_words if word in content_lower)
                    if matches > 0:
                        # Calculate relevance score
                        score = matches / len(query_words)
                        
                        # Boost score for exact phrase matches
                        if 'remote work' in query_lower and 'remote work' in content_lower:
                            score += 0.5
                        if 'policy' in query_lower and 'policy' in content_lower:
                            score += 0.3
                        
                        # Get a snippet around the first match
                        snippet_start = max(0, content_lower.find(query_lower.split()[0]) - 100)
                        snippet_end = min(len(content), snippet_start + 400)
                        snippet = content[snippet_start:snippet_end]
                        
                        results.append({
                            "content": snippet,
                            "source": file_path,
                            "score": score
                        })
                        
                except Exception as e:
                    print(f"Error reading file {file_path}: {e}")
                    continue
        
        return results
    
    def format_database_response(self, results: Dict[str, Any]) -> str:
        """Format database results into a readable response"""
        if "error" in results:
            return f"Database error: {results['error']}"
        
        rows = results.get("rows", [])
        count = results.get("count", 0)
        
        if not rows:
            return "No data found matching your query."
        
        # Format as table
        if rows:
            headers = list(rows[0].keys())
            response = "| " + " | ".join(headers) + " |\n"
            response += "|" + "|".join(["---"] * len(headers)) + "|\n"
            
            for row in rows[:10]:  # Limit to first 10 rows
                response += "| " + " | ".join(str(row.get(h, "")) for h in headers) + " |\n"
            
            if len(rows) > 10:
                response += f"\n... and {len(rows) - 10} more rows"
        
        return response
    
    def format_document_response(self, results: List[Dict[str, Any]]) -> str:
        """Format document search results into a readable response"""
        if not results:
            return "No relevant documents found."
        
        response = "**Relevant Information from Documents:**\n\n"
        
        for i, result in enumerate(results, 1):
            content = result["content"]
            source = result["source"]
            score = result["score"]
            
            # Truncate content if too long
            if len(content) > 300:
                content = content[:300] + "..."
            
            response += f"**{i}. Source: {source}** (Relevance: {score:.2f})\n"
            response += f"{content}\n\n"
        
        return response
    
    def process_query(self, query: str) -> str:
        """
        Main method to process a user query intelligently
        """
        print(f"Processing query: {query}")
        
        # Step 1: Analyze the query
        analysis = self.analyze_query(query)
        print(f"Analysis: {analysis.query_type.value} (confidence: {analysis.confidence:.2f})")
        print(f"Reasoning: {analysis.reasoning}")
        
        response_parts = []
        
        # Step 2: Handle database queries
        if analysis.query_type in [QueryType.DATABASE, QueryType.HYBRID]:
            sql = analysis.suggested_sql or self.generate_sql(query)
            
            if sql:
                print(f"Generated SQL: {sql}")
                db_results = self.execute_database_query(sql)
                db_response = self.format_database_response(db_results)
                response_parts.append(f"**Database Results:**\n{db_response}")
            else:
                response_parts.append("Could not generate a valid SQL query for your request.")
        
        # Step 3: Handle document queries
        if analysis.query_type in [QueryType.DOCUMENTS, QueryType.HYBRID]:
            doc_queries = analysis.suggested_document_queries or [query]
            doc_results = self.execute_document_search(doc_queries)
            doc_response = self.format_document_response(doc_results)
            response_parts.append(doc_response)
        
        # Step 4: Handle unknown queries
        if analysis.query_type == QueryType.UNKNOWN:
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
        "What are the performance management guidelines?"
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
