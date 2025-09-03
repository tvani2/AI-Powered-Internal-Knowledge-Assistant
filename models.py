#!/usr/bin/env python3
"""
Shared data models for the intelligent agent system
"""

from enum import Enum
from typing import List, Optional


class QueryType(Enum):
    """Types of queries the system can handle"""
    DATABASE = "database"
    DOCUMENTS = "documents"
    HYBRID = "hybrid"
    CONVERSATIONAL = "conversational"
    UNKNOWN = "unknown"


class QueryAnalysis:
    """Result of query analysis"""
    
    def __init__(
        self,
        query_type: QueryType,
        confidence: float,
        reasoning: str,
        suggested_sql: Optional[str] = None,
        suggested_document_queries: Optional[List[str]] = None
    ):
        self.query_type = query_type
        self.confidence = confidence
        self.reasoning = reasoning
        self.suggested_sql = suggested_sql
        self.suggested_document_queries = suggested_document_queries
