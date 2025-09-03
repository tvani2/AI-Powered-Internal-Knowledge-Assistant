from typing import List, Dict, Any, Optional

from database_handler import DatabaseHandler
from document_handler import DocumentHandler
from intelligent_agent import QueryAnalysis, QueryType 


class HybridHandler:
    """
    Hybrid orchestration extracted from IntelligentAgent.
    Implements the SQL-then-documents controller and formatting delegates.
    """

    def __init__(self, database_handler: DatabaseHandler, document_handler: DocumentHandler) -> None:
        self.database_handler = database_handler
        self.document_handler = document_handler

    def answer_query(self, query: str, analysis: QueryAnalysis, generate_sql_fn, format_db_fn, analyze_query_fn) -> str:
        sql_result, doc_result = None, None

        if analysis.query_type in [QueryType.DATABASE, QueryType.HYBRID]:
            sql_result = self._run_sql_query(query, analysis, generate_sql_fn)

        if analysis.query_type in [QueryType.DOCUMENTS, QueryType.HYBRID]:
            doc_result = self._run_doc_retrieval(query, analyze_query_fn)

        # Hybrid logic
        if analysis.query_type == QueryType.HYBRID:
            combined = ""
            if sql_result:
                combined += self._format_sql_result(sql_result, format_db_fn)
            if doc_result:
                if combined:
                    combined += "\n\n"  # separate sections
                combined += self._format_doc_result(doc_result)
            return combined or "I could not find that information in the available data."

        # Database-only
        if sql_result:
            return self._format_sql_result(sql_result, format_db_fn)

        # Documents-only
        if doc_result:
            return self._format_doc_result(doc_result)

        return "I could not find that information in the available data."


    def _run_sql_query(self, query: str, analysis: QueryAnalysis, generate_sql_fn) -> Optional[Dict[str, Any]]:
        """Run SQL query and return results if successful"""
        sql = analysis.suggested_sql or generate_sql_fn(query)
        
        if sql:
            db_results = self.database_handler.execute_database_query(sql)
            if "error" not in db_results and db_results.get("rows"):
                return db_results
        
        return None

    def _format_sql_result(self, sql_result: Dict[str, Any], format_db_fn) -> str:
        """Format SQL results into clean natural language"""
        return format_db_fn(sql_result)

    def _run_doc_retrieval(self, query: str, analyze_query_fn) -> Optional[List[Dict[str, Any]]]:
        """Run document retrieval and return results if successful"""
        analysis = analyze_query_fn(query)
        doc_queries = analysis.suggested_document_queries or [query]
        
        # For hybrid queries about project managers and meetings, use more specific search
        if any(word in query.lower() for word in ['project manager', 'meeting', 'standup', 'review']):
            # Expand search to ensure we get complete meeting content
            expanded_queries = []
            for base_query in doc_queries:
                expanded_queries.extend([
                    base_query,
                    f"{base_query} content",
                    f"{base_query} details",
                    f"{base_query} notes"
                ])
            doc_queries = expanded_queries
        
        # For hybrid queries about sales and policies, expand search terms
        elif any(word in query.lower() for word in ['sales', 'revenue', 'business', 'policies', 'market']):
            expanded_queries = []
            for base_query in doc_queries:
                expanded_queries.extend([
                    base_query,
                    f"{base_query} strategy",
                    f"{base_query} policies",
                    f"{base_query} performance",
                    f"{base_query} data"
                ])
            doc_queries = expanded_queries
        
        doc_results = self.document_handler.execute_document_search(doc_queries)
        
        if doc_results:
            return doc_results
        
        return None

    def _format_doc_result(self, doc_results: List[Dict[str, Any]]) -> str:
        """Format document results into clean natural language"""
        return self.document_handler.format_document_response(doc_results)
