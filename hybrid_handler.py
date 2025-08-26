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
        """
        Controller layer that implements the decision tree:
        1. Try SQL query first
        2. If SQL fails, try document retrieval
        3. If both fail, return error message
        """
        # Step 1: Check if it's a structured (SQL) type question
        if analysis.query_type in [QueryType.DATABASE, QueryType.HYBRID]:
            sql_result = self._run_sql_query(query, analysis, generate_sql_fn)
            if sql_result:  # non-empty result
                return self._format_sql_result(sql_result, format_db_fn)
            # else fallback to docs
        
        # Step 2: Try document retrieval (RAG)
        doc_result = self._run_doc_retrieval(query, analyze_query_fn)
        if doc_result:
            return self._format_doc_result(doc_result)
        
        # Step 3: Both failed
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
        doc_results = self.document_handler.execute_document_search(doc_queries)
        
        if doc_results:
            return doc_results
        
        return None

    def _format_doc_result(self, doc_results: List[Dict[str, Any]]) -> str:
        """Format document results into clean natural language"""
        return self.document_handler.format_document_response(doc_results)
