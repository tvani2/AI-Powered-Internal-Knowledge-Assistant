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
from conversation_handler import ConversationHandler
from document_handler import DocumentHandler
from database_handler import DatabaseHandler
from bootstrap import EnvironmentManager, LLMInitializer


from models import QueryType, QueryAnalysis

from hybrid_handler import HybridHandler
from query_analyzer import QueryAnalyzer
from sql_generator import SQLGenerator


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
        EnvironmentManager.load_environment()
        self.llm = LLMInitializer.initialize_llm()
        
        self.database_handler = DatabaseHandler()
        
        self.document_handler = DocumentHandler(llm=self.llm)
        
        self.hybrid_handler = HybridHandler(self.database_handler, self.document_handler)
        
        self.query_analyzer = QueryAnalyzer()
        
        self.sql_generator = SQLGenerator(self.database_handler, llm=self.llm)
        
        self.document_summaries = self.document_handler._load_document_summaries()
        
        self.conversation = ConversationHandler()
    
    def analyze_query(self, query: str) -> QueryAnalysis:
        return self.query_analyzer.analyze_query(query)
    
    def generate_sql(self, query: str) -> Optional[str]:
        return self.sql_generator.generate(query)

    def format_database_response(self, results: dict, limit: int = None) -> str:
        return self.database_handler.format_database_response(results, limit)
    
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