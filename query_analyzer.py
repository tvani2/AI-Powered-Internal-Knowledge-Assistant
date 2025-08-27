from typing import List, Optional

from intelligent_agent import QueryAnalysis, QueryType


class QueryAnalyzer:
    """
    Determines whether a query targets documents, database, or hybrid.
    """

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
            'discussed', 'discussion', 'latest', 'team', 'updates', 'blockers', 'progress'
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
        suggested_sql: Optional[str] = None
        suggested_document_queries: Optional[List[str]] = None
        
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
            # More specific document queries to find relevant meeting content
            suggested_document_queries = [
                "project development meeting", 
                "executive quarterly review", 
                "engineering team standup",
                "project managers meeting",
                "product development meeting"
            ]
            # Better SQL to get project managers with more context
            suggested_sql = """
                SELECT 
                    p.name as project_name, 
                    e.name as manager_name, 
                    e.department,
                    p.status,
                    p.budget,
                    p.deadline
                FROM projects p 
                JOIN employees e ON p.manager_id = e.id 
                WHERE p.status IN ('Active', 'Planning', 'On Hold')
                ORDER BY p.name
            """
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
