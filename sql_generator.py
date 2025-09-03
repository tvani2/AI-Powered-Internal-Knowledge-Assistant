from typing import Optional, Dict, Any, List
import re

from database_handler import DatabaseHandler


class SQLGenerator:
    """
    Hybrid SQL generator with rule-based patterns for common queries and LLM fallback for complex cases.
    
    Architecture:
    1. Rule-based layer: Fast, deterministic handling of common pattern
    2. LLM fallback layer: Schema-aware generation for complex/unseen queries
    3. Validation layer: Safety checks and schema compliance
    """

    def __init__(self, database_handler: DatabaseHandler, llm=None) -> None:
        self.database_handler = database_handler
        self.llm = llm
        
        # Core query patterns - these should cover 80% of common queries
        self.query_patterns = self._initialize_query_patterns()

    def _initialize_query_patterns(self) -> List[Dict[str, Any]]:
        """Initialize the rule-based query patterns"""
        return [
            # Employee performance queries
            {
                'keywords': ['high performing', 'top performers', 'best employees', 'performance'],
                'sql_template': "SELECT name, department, role, salary FROM employees ORDER BY salary DESC LIMIT 10",
                'description': "High performance employees by salary",
                'priority': 1
            },
            
            # Department-specific queries (highest priority)
            {
                'keywords': ['top', 'revenue', 'engineering', 'sales', 'marketing', 'legal', 'research', 'operations', 'finance', 'hr', 'support', 'product'],
                'sql_template': "SELECT e.name, e.department, CASE WHEN e.department = 'Sales' THEN COALESCE(SUM(s.revenue), 0) ELSE e.salary END as total_revenue FROM employees e LEFT JOIN sales s ON e.id = s.employee_id WHERE e.department = '{dept}' GROUP BY e.id, e.name, e.department, e.salary ORDER BY total_revenue DESC LIMIT {limit}",
                'description': "Top employees by revenue (Sales) or salary (other departments)",
                'priority': 3,
                'dynamic_params': {'limit': 5, 'dept': None}
            },
            
            # Sales performance queries (fallback)
            {
                'keywords': ['top', 'sales', 'revenue'],
                'sql_template': "SELECT e.name, e.department, SUM(s.revenue) as total_revenue FROM employees e JOIN sales s ON e.id = s.employee_id WHERE e.department = '{dept}' GROUP BY e.id ORDER BY total_revenue DESC LIMIT {limit}",
                'description': "Top employees by sales revenue",
                'priority': 0.5,

                'dynamic_params': {'limit': 5, 'dept': 'Sales'}
            },
            
            # Department queries
            {
                'keywords': ['department', 'count', 'how many'],
                'sql_template': "SELECT COUNT(*) as employee_count FROM employees WHERE department = '{dept}'",
                'description': "Employee count by department",
                'priority': 2,
                'dynamic_params': {'dept': 'Engineering'}
            },
            
            # Project queries
            {
                'keywords': ['project', 'manager'],
                'sql_template': "SELECT p.name as project_name, e.name as manager_name, e.department FROM projects p JOIN employees e ON p.manager_id = e.id ORDER BY p.name",
                'description': "Projects with managers",
                'priority': 2
            },
            
            # Hiring queries
            {
                'keywords': ['hire', 'joined', 'recent'],
                'sql_template': "SELECT name, department, hire_date FROM employees WHERE hire_date >= date('now', '-1 year') ORDER BY hire_date DESC",
                'description': "Recently hired employees",
                'priority': 3
            }
        ]

    def generate(self, query: str) -> Optional[str]:
        """
        Generate SQL using hybrid approach: rule-based first, LLM fallback second
        """
        query_lower = query.lower()
        
        # Step 1: Try rule-based generation (fast, deterministic)
        rule_based_sql = self._generate_rule_based(query_lower)
        if rule_based_sql:
            # Validate the generated SQL
            if self._validate_sql(rule_based_sql):
                return rule_based_sql
            else:
                print(f"Warning: Rule-based SQL failed validation: {rule_based_sql}")
        
        # Step 2: Fallback to LLM generation (covers complex/unseen queries)
        if self.llm:
            llm_sql = self._generate_llm_based(query)
            if llm_sql and self._validate_sql(llm_sql):
                return llm_sql
        
        return None

    def _generate_rule_based(self, query_lower: str) -> Optional[str]:
        """Generate SQL using rule-based patterns"""
        best_match = None
        best_score = 0
        
        for pattern in self.query_patterns:
            score = self._calculate_pattern_score(query_lower, pattern)
            if score > best_score:
                best_score = score
                best_match = pattern
        
        if best_match and best_score > 0.5:  # Threshold for pattern matching
            return self._apply_pattern_template(best_match, query_lower)
        
        return None

    def _calculate_pattern_score(self, query: str, pattern: Dict[str, Any]) -> float:
        """Calculate how well a query matches a pattern"""
        score = 0
        keywords = pattern['keywords']
        
        for keyword in keywords:
            if keyword in query:
                score += 1
                # Bonus for exact phrase matches
                if f" {keyword} " in f" {query} ":
                    score += 0.5
        
        # Normalize by pattern priority
        score = score / len(keywords) * pattern['priority']
        return score

    def _apply_pattern_template(self, pattern: Dict[str, Any], query: str) -> str:
        """Apply a pattern template with dynamic parameter substitution"""
        sql = pattern['sql_template']
        
        # Handle dynamic parameters
        if 'dynamic_params' in pattern:
            params = pattern['dynamic_params']
            
            # Extract limit from query if present
            if 'limit' in params:
                limit_match = re.search(r'top\s+(\d+)', query)
                if limit_match:
                    sql = sql.replace('{limit}', limit_match.group(1))
                else:
                    sql = sql.replace('{limit}', str(params['limit']))
            
            # Extract department from query if present
            if 'dept' in params:
                dept = self._extract_department(query)
                if dept:
                    sql = sql.replace('{dept}', dept)
                elif params['dept'] is not None:
                    # Use default department if specified
                    sql = sql.replace('{dept}', params['dept'])
        
        return sql

    def _extract_department(self, query: str) -> Optional[str]:
        """Extract department name from query"""
        dept_keywords = {
            'engineering': 'Engineering',
            'sales': 'Sales', 
            'marketing': 'Marketing',
            'legal': 'Legal',
            'research': 'Research',
            'operations': 'Operations',
            'finance': 'Finance',
            'hr': 'HR',
            'support': 'Support',
            'product': 'Product'
        }
        
        query_lower = query.lower()
        
        for keyword, dept in dept_keywords.items():
            if keyword in query_lower:
                return dept
        
        return None

    def _generate_llm_based(self, query: str) -> Optional[str]:
        """Generate SQL using LLM with schema-aware prompting"""
        try:
            db_schema = self.database_handler.get_schema()
            schema_summary = self._summarize_schema(db_schema)
            
            sql_prompt = f"""
You are an expert SQL generator for a business intelligence system. Convert the natural language query into a valid SQLite SELECT statement.

Available database schema:
{schema_summary}

Critical safety rules:
- ONLY generate SELECT statements (NO INSERT, UPDATE, DELETE, DROP, ALTER, etc.)
- Use proper SQLite syntax
- Include appropriate JOINs when needed
- Use aggregation functions (COUNT, SUM, AVG, etc.) when appropriate
- Limit results when appropriate (LIMIT clause)
- Use proper date functions for date comparisons
- Ensure all table/column references exist in the schema

Query: "{query}"

Return ONLY the SQL query, no explanations or markdown formatting:"""

            response = self.llm.invoke(sql_prompt)
            sql = response.content.strip()
            
            # Clean up common LLM artifacts
            sql = self._clean_llm_response(sql)
            
            return sql
            
        except Exception as e:
            print(f"LLM SQL generation failed: {e}")
            return None

    def _clean_llm_response(self, sql: str) -> str:
        """Clean up common LLM response artifacts"""
        # Remove markdown code blocks
        sql = re.sub(r'```sql\s*', '', sql)
        sql = re.sub(r'```\s*', '', sql)
        
        # Remove leading/trailing whitespace
        sql = sql.strip()
        
        # Remove explanatory text after the query
        if ';' in sql:
            sql = sql.split(';')[0] + ';'
        
        return sql

    def _validate_sql(self, sql: str) -> bool:
        """Validate generated SQL for safety and schema compliance"""
        if not sql:
            return False
        
        # Safety checks
        dangerous_keywords = ['DROP', 'DELETE', 'INSERT', 'UPDATE', 'ALTER', 'CREATE', 'TRUNCATE']
        sql_upper = sql.upper()
        
        for keyword in dangerous_keywords:
            if keyword in sql_upper:
                print(f"Safety check failed: Dangerous keyword '{keyword}' found")
                return False
        
        # Basic syntax check
        if not sql_upper.startswith('SELECT'):
            print("Safety check failed: Query must start with SELECT")
            return False
        
        # Schema compliance check (if database handler supports it)
        if hasattr(self.database_handler, 'try_execute_for_validation'):
            try:
                return self.database_handler.try_execute_for_validation(sql)
            except Exception as e:
                print(f"Schema validation failed: {e}")
                return False
        
        return True

    def _summarize_schema(self, schema: dict) -> str:
        """Create a human-readable summary of the database schema"""
        summary = []
        for table, columns in schema.items():
            col_names = [col["name"] for col in columns]
            summary.append(f"- {table}: {', '.join(col_names)}")
        return "\n".join(summary)

    def add_pattern(self, keywords: List[str], sql_template: str, description: str, priority: int = 1):
        """Add a new query pattern for future use"""
        self.query_patterns.append({
            'keywords': keywords,
            'sql_template': sql_template,
            'description': description,
            'priority': priority
        })
        
        # Sort by priority for better matching
        self.query_patterns.sort(key=lambda x: x['priority'], reverse=True)

    def get_pattern_coverage(self) -> Dict[str, Any]:
        """Get statistics about pattern coverage"""
        total_patterns = len(self.query_patterns)
        high_priority = len([p for p in self.query_patterns if p['priority'] >= 2])
        
        return {
            'total_patterns': total_patterns,
            'high_priority_patterns': high_priority,
            'patterns': [{'keywords': p['keywords'], 'description': p['description']} for p in self.query_patterns]
        }
