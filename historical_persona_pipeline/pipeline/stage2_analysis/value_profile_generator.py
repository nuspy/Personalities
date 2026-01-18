import json
import logging
from typing import Dict, Any, Optional
from ...pipeline.utils.llm_provider import LLMProvider

logger = logging.getLogger(__name__)

class ValueProfileGenerator:
    """Generates dynamic value profiles using an LLM."""
    
    SYSTEM_PROMPT = """You are an expert historian and cultural anthropologist. 
Your task is to generate a JSON value dictionary for a specific historical or cultural context.
The output must strictly follow this JSON structure:
{
  "value_name_1": {
    "keywords": {
      "en": ["keyword1", "keyword2"],
      "it": ["parola_chiave1", "parola_chiave2"],
      "original_language_code": ["native_term1", "native_term2"] 
    },
    "weight": 0.0 to 1.0,
    "description": "Brief description of the value"
  },
  ...
}

GUIDELINES:
1. Identify 5-8 core values relevant to the requested context.
2. For each value, provide keywords in English, Italian, and the relevant historical language (e.g., Latin for Rome, Japanese for Samurai).
3. Assign a weight based on importance (1.0 = critical, 0.5 = secondary).
4. Do NOT include Markdown formatting (like ```json). Return ONLY raw JSON.
"""

    def __init__(self, llm_provider: LLMProvider):
        self.llm = llm_provider

    def generate_profile(self, context_description: str, languages: list = ["en", "it"]) -> Dict[str, Any]:
        """
        Generates a value profile for the given context.
        
        Args:
            context_description: e.g., "Samurai Bushido Code 16th Century" or "Victorian Era Morality"
            languages: List of languages for keywords (English and Italian are default in system prompt, but can be emphasized)
        """
        prompt = f"""Generate a cultural value dictionary for: '{context_description}'.
        
        Ensure keywords are provided in: {', '.join(languages)} and the original historical language if applicable.
        """
        
        try:
            logger.info(f"Generating value profile for: {context_description}")
            response_text = self.llm.generate(
                prompt=prompt,
                system_prompt=self.SYSTEM_PROMPT,
                json_mode=True
            )
            
            # Clean response if necessary (sometimes LLMs add markdown even if asked not to)
            cleaned_text = response_text.strip()
            if cleaned_text.startswith("```json"):
                cleaned_text = cleaned_text[7:]
            if cleaned_text.endswith("```"):
                cleaned_text = cleaned_text[:-3]
                
            profile = json.loads(cleaned_text)
            return profile
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response as JSON: {e}")
            logger.debug(f"Raw response: {response_text}")
            raise ValueError("The LLM produced invalid JSON. Please try again.")
        except Exception as e:
            logger.error(f"Error generating profile: {e}")
            raise
