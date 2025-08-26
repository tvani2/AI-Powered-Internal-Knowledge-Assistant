import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

class EnvironmentManager:
    """Handles loading environment variables safely."""

    @staticmethod
    def load_environment():
        try:
            load_dotenv()
            print("Successfully loaded .env file")
        except Exception as e:
            print(f"Warning: Could not load .env file: {e}")
            # Set placeholder values
            os.environ["OPENAI_MODEL"] = "gpt-4o-mini"
            os.environ["OPENAI_API_KEY"] = "your-api-key-here"


class LLMInitializer:
    """Handles initialization of the LLM (OpenAI)."""

    @staticmethod
    def initialize_llm():
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key and api_key != "your-api-key-here":
            try:
                llm = ChatOpenAI(
                    model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                    temperature=0
                )
                print("OpenAI LLM initialized successfully")
                return llm
            except Exception as e:
                print(f"Warning: Could not initialize OpenAI LLM: {e}")
                print("LLM features will be disabled. Set OPENAI_API_KEY environment variable to enable.")
                return None
        else:
            print("Warning: OPENAI_API_KEY not set. LLM features will be disabled.")
            print("Set OPENAI_API_KEY environment variable to enable full functionality.")
            return None