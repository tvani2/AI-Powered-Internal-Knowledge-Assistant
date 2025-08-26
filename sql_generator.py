from typing import Optional

from database_handler import DatabaseHandler
from intelligent_agent import QueryAnalysis  # type reference only; not required at runtime


class SQLGenerator:
    """
    Generates SQL from natural language queries. Uses simple rules first, falls back to LLM if available.
    """

    def __init__(self, database_handler: DatabaseHandler, llm=None) -> None:
        self.database_handler = database_handler
        self.llm = llm

    def generate(self, query: str) -> Optional[str]:
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
                
            except Exception:
                return None
        else:
            # LLM not available
            return None

    def _summarize_schema(self, schema: dict) -> str:
        """Create a human-readable summary of the database schema"""
        summary = []
        for table, columns in schema.items():
            col_names = [col["name"] for col in columns]
            summary.append(f"- {table}: {', '.join(col_names)}")
        return "\n".join(summary)
