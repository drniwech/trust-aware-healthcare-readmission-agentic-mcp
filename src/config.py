"""
Central configuration for the Trust-Aware Healthcare Readmission Platform
"""
import os
from dotenv import load_dotenv

load_dotenv()

# ============== MODEL CONFIGURATION ==============
# Change this in .env to switch providers
# Options: "openai:gpt-4o-mini" or "ollama:llama3.2" (or any model you pulled)
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "ollama:llama3.2")

# Ollama settings
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

print(f"🔧 Using model: {DEFAULT_MODEL}")

# You can add more config here later
# e.g.
# MAX_TOKENS = 12000
# TEMPERATURE = 0.2