#!/usr/bin/env python3
"""
Streamlit UI for AI-Powered Internal Knowledge Assistant
"""

import streamlit as st
import requests
import json
import time
import uuid
from datetime import datetime
import pandas as pd

# Configuration
API_BASE = "http://localhost:8000"

def init_session_state():
    """Initialize session state variables"""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "user_id" not in st.session_state:
        st.session_state.user_id = f"user_{uuid.uuid4().hex[:8]}"
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())



def check_api_health():
    """Check if the API is healthy"""
    try:
        response = requests.get(f"{API_BASE}/health", timeout=5)
        return response.status_code == 200, response.json()
    except:
        return False, None



def send_chat_message(query):
    """Send a chat message to the API"""
    headers = {
        "Content-Type": "application/json"
    }
    
    data = {
        "query": query,
        "user_id": "anonymous",
        "session_id": st.session_state.session_id
    }
    
    try:
        response = requests.post(
            f"{API_BASE}/chat",
            headers=headers,
            json=data,
            timeout=30
        )
        
        if response.status_code == 200:
            return response.json()

        else:
            st.error(f"API Error: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        st.error(f"Connection error: {str(e)}")
        return None

def get_query_logs():
    """Get query logs for debugging"""
    try:
        response = requests.get(
            f"{API_BASE}/logs",
            params={"limit": 100}
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            return None
    except:
        return None

def display_sources(sources):
    """Display sources used in the response"""
    if not sources:
        return
    
    st.subheader("Sources Used")
    
    # Database sources
    if "database" in sources:
        with st.expander("Database Sources", expanded=True):
            db_sources = sources["database"]
            st.write("**SQL Generated:**")
            st.code(db_sources.get("sql", "No SQL generated"), language="sql")
            
            if "tables_accessed" in db_sources and db_sources["tables_accessed"]:
                st.write("**Tables Accessed:**")
                for table in db_sources["tables_accessed"]:
                    st.write(f"• {table}")
    
    # Document sources
    if "documents" in sources:
        with st.expander("Document Sources", expanded=True):
            doc_sources = sources["documents"]
            
            if "queries" in doc_sources:
                st.write("**Document Queries:**")
                for query in doc_sources["queries"]:
                    st.write(f"• {query}")
            
            if "categories" in doc_sources:
                categories = doc_sources["categories"]
                if any(categories.values()):
                    st.write("**Document Categories:**")
                    for category, queries in categories.items():
                        if queries:
                            category_name = category.replace("_", " ").title()
                            st.write(f"**{category_name}:**")
                            for query in queries:
                                st.write(f"  • {query}")

def display_response_details(response_data):
    """Display response details and metadata"""
    with st.expander("Response Details", expanded=False):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Query Type", response_data.get("query_type", "Unknown"))
        
        with col2:
            confidence = response_data.get("confidence", 0)
            st.metric("Confidence", f"{confidence:.1%}" if confidence else "N/A")
        
        with col3:
            processing_time = response_data.get("processing_time", 0)
            st.metric("Processing Time", f"{processing_time}s" if processing_time else "N/A")
        
        if response_data.get("reasoning"):
            st.write("**Reasoning:**")
            st.write(response_data["reasoning"])

def main():
    """Main Streamlit application"""
    st.set_page_config(
        page_title="AI Knowledge Assistant",
        page_icon="",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize session state
    init_session_state()
    
    # Sidebar for system info
    with st.sidebar:
        st.title("System Info")
        st.write("AI-Powered Knowledge Assistant")
        st.write("No authentication required")
        st.write("Ready to help with your queries!")
        
        st.divider()
        
        # System status
        st.subheader("System Status")
        is_healthy, health_data = check_api_health()
        
        if is_healthy:
            st.success("API Connected")
            if health_data:
                st.write(f"**Status:** {health_data['status']}")
                st.write(f"**Agent:** {'Ready' if health_data['agent_ready'] else 'Not Ready'}")
                st.write(f"**Database:** {'Connected' if health_data['database_connected'] else 'Not Connected'}")
                st.write(f"**Documents:** {'Available' if health_data['documents_available'] else 'Not Available'}")
        else:
            st.error("API Not Connected")
            st.info("Make sure the FastAPI server is running on localhost:8000")
        
        st.divider()
        
        # Sample queries
        st.subheader("Sample Queries")
        
        st.write("**Database Queries:**")
        if st.button("Top 5 employees by sales"):
            st.session_state.sample_query = "Show me the top 5 employees by sales revenue"
            st.rerun()
        
        if st.button("Engineering department count"):
            st.session_state.sample_query = "How many employees are in the Engineering department?"
            st.rerun()
        
        st.write("**Document Queries:**")
        if st.button("Employee benefits"):
            st.session_state.sample_query = "What are the employee benefits?"
            st.rerun()
        
        if st.button("Remote work policies"):
            st.session_state.sample_query = "What are the remote work policies?"
            st.rerun()
        
        st.write("**Hybrid Queries:**")
        if st.button("Sales data and policies"):
            st.session_state.sample_query = "Show me sales data and related policies"
            st.rerun()
        
        st.divider()
        
        # Query logs
        if st.button("View Query Logs"):
            st.session_state.show_logs = True
            st.rerun()
        
        # Clear chat
        if st.button("Clear Chat"):
            st.session_state.messages = []
            st.rerun()
    
    # Main chat interface
    st.title("AI-Powered Internal Knowledge Assistant")
    st.markdown("Ask questions about employees, sales, projects, policies and more!")
    
    # Check authentication
    if not st.session_state.authenticated:
        st.warning("Please authenticate using the sidebar to start chatting.")
        return
    
    # Handle sample query
    if hasattr(st.session_state, 'sample_query'):
        query = st.session_state.sample_query
        del st.session_state.sample_query
    else:
        # Chat input
        query = st.chat_input("Ask me anything...")
    
    # Process query
    if query:
        if not st.session_state.authenticated:
            st.error("Please login first to send messages")
            return
            
        # Add user message
        st.session_state.messages.append({"role": "user", "content": query})
        
        # Show user message
        with st.chat_message("user"):
            st.write(query)
        
        # Get AI response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response_data = send_chat_message(query)
            
            if response_data:
                # Display response
                st.write(response_data["response"])
                
                # Display sources
                if response_data.get("sources"):
                    display_sources(response_data["sources"])
                
                # Display response details
                display_response_details(response_data)
                
                # Add assistant message to session state
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": response_data["response"],
                    "metadata": response_data
                })
            else:
                st.error("Failed to get response from the API.")
    
    # Display chat history
    for message in st.session_state.messages:
        if message["role"] == "user":
            with st.chat_message("user"):
                st.write(message["content"])
        else:
            with st.chat_message("assistant"):
                st.write(message["content"])
                
                # Show sources for assistant messages with metadata
                if "metadata" in message and message["metadata"].get("sources"):
                    display_sources(message["metadata"]["sources"])
    
    # Query logs view
    if hasattr(st.session_state, 'show_logs') and st.session_state.show_logs:
        st.divider()
        st.subheader("Query Logs")
        
        logs_data = get_query_logs()
        if logs_data and logs_data["logs"]:
            logs_df = pd.DataFrame(logs_data["logs"])
            
            # Format timestamp
            if "timestamp" in logs_df.columns:
                logs_df["timestamp"] = pd.to_datetime(logs_df["timestamp"])
            
            # Display logs
            st.dataframe(
                logs_df[["timestamp", "user_query", "query_type", "confidence", "processing_time"]],
                use_container_width=True
            )
            
            # Detailed view
            if st.button("Show Detailed Logs"):
                st.json(logs_data["logs"])
        else:
            st.info("No query logs found.")
        
        if st.button("Close Logs"):
            del st.session_state.show_logs
            st.rerun()

if __name__ == "__main__":
    main()

