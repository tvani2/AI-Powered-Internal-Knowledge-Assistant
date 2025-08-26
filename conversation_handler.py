"""
Personal and conversational query handling
"""

from typing import Dict, Any


class ConversationHandler:
    """
    Handles conversational queries, personal questions and closing statements
    """
    
    # Centralized response dictionary for personal questions
    PERSONAL_RESPONSES = {
        "age": {
            "keywords": ["how old are you", "what is your age", "when were you born"],
            "response": "I'm an AI assistant, so I don't have an age in the traditional sense. I was created to help with company knowledge and data queries. How can I assist you with your work-related questions?"
        },
        "human": {
            "keywords": ["are you real", "are you human", "do you have feelings"],
            "response": "I'm an AI assistant designed to help with company information and data analysis. I'm not human, but I'm here to help you with your work-related questions!"
        },
        "opinions": {
            "keywords": ["what do you think", "do you like", "do you enjoy", "what is your opinion"],
            "response": "I'm an AI assistant focused on helping with company data and policies. I don't have personal opinions or preferences, but I'm happy to help you find the information you need!"
        },
        "identity": {
            "keywords": ["what is your name", "who are you", "what are you"],
            "response": "I'm an AI-powered internal knowledge assistant. I help employees find information about company policies, data anddocumentation. How can I help you today?"
        },
        "capabilities": {
            "keywords": ["what can you do", "how do you work", "what are your capabilities"],
            "response": "I can help you with company data queries, policy information, meeting notes andtechnical documentation. I use semantic search and database integration to provide accurate answers."
        },
        "personal_life": {
            "keywords": ["do you sleep", "do you eat", "do you dream", "are you married", "do you have family", "do you have friends"],
            "response": "I'm an AI assistant designed to help with company information and data. I don't have personal experiences or feelings, but I'm here to help you with your work-related questions!"
        }
    }
    
    def is_conversational_query(self, query: str) -> bool:
        """Check if query is conversational/greeting"""
        query_lower = query.lower().strip()
        conversational_phrases = [
            "hello", "hi", "hey", "good morning", "good afternoon", "good evening",
            "how are you", "how's it going", "what's up", "nice to meet you",
            "thanks", "thank you", "appreciate it", "goodbye", "bye", "see you",
            "have a good day", "have a nice day", "take care"
        ]
        return any(phrase in query_lower for phrase in conversational_phrases)
    
    def is_closing(self, query: str) -> bool:
        """Check if query is a closing/goodbye"""
        query_lower = query.lower().strip()
        closing_phrases = [
            "goodbye", "bye", "see you", "see ya", "take care", "have a good day",
            "have a nice day", "talk to you later", "until next time", "farewell",
            "good night", "goodnight", "see you tomorrow", "see you later"
        ]
        return any(phrase in query_lower for phrase in closing_phrases)
    
    def is_personal_question(self, query: str) -> bool:
        """Check if query is a personal question about the AI"""
        query_lower = query.lower().strip()
        
        # Check against all personal response categories
        for category, data in self.PERSONAL_RESPONSES.items():
            for keyword in data["keywords"]:
                if keyword in query_lower:
                    return True
        return False
    
    def handle_closing(self, query: str) -> str:
        """Handle closing/goodbye queries"""
        return "Goodbye! Feel free to ask me any questions about company data, policies, or documentation when you need help."
    
    def handle_personal_question(self, query: str) -> str:
        """Handle personal questions about the AI"""
        query_lower = query.lower().strip()
        
        # Find matching response category
        for category, data in self.PERSONAL_RESPONSES.items():
            for keyword in data["keywords"]:
                if keyword in query_lower:
                    return data["response"]
        
        # Default response for unmatched personal questions
        return "I'm an AI assistant designed to help with company information and data. I'm here to help you with work-related questions!"
    
    def handle_conversational_query(self, query: str) -> str:
        """Handle conversational/greeting queries"""
        query_lower = query.lower().strip()
        
        if any(greeting in query_lower for greeting in ["hello", "hi", "hey"]):
            return "Hello! I'm your AI assistant. I can help you with company data, policies anddocumentation. What would you like to know?"
        elif any(thanks in query_lower for thanks in ["thanks", "thank you", "appreciate"]):
            return "You're welcome! I'm happy to help. Is there anything else you'd like to know about the company?"
        elif "how are you" in query_lower:
            return "I'm functioning well and ready to help you with any questions about company data, policies, or documentation!"
        else:
            return "Hello! I'm here to help with company information. What can I assist you with today?"
