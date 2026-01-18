import logging
import json
import random
from typing import List, Dict, Any, Optional
from tqdm import tqdm
from ...pipeline.data_models import CompleteStyleProfile, TrainingConversation, TextSegment
from ...pipeline.utils.llm_provider import LLMProvider

logger = logging.getLogger(__name__)

class ConversationGenerator:
    """Generates synthetic training conversations based on style profile AND source content."""
    
    def __init__(self, llm_provider: LLMProvider, config: Dict[str, Any]):
        self.llm = llm_provider
        self.config = config
        
    def generate_from_segments(self, 
                             profile: CompleteStyleProfile, 
                             segments: List[TextSegment], 
                             count: int) -> List[TrainingConversation]:
        """Generates conversations based on actual content from source files."""
        conversations = []
        
        # Filter valid segments (ignore very short ones)
        valid_segments = [s for s in segments if len(s.content) > 200]
        if not valid_segments:
            logger.warning("No valid segments found for content generation.")
            return []

        # We loop through segments until we reach the desired count
        # Randomly sample to ensure variety if count < len(segments)
        import random
        target_segments = random.choices(valid_segments, k=count)
        
        system_prompt = profile.generated_system_prompt
        
        for i, segment in enumerate(target_segments):
            try:
                # Create prompt that forces usage of the segment's content
                prompt = self._create_content_prompt(profile, segment.content)
                
                response = self.llm.generate(
                    prompt=prompt,
                    system_prompt="You are a dataset creator. Extract facts from the provided text to create training examples.",
                    json_mode=True
                )
                
                data = self._parse_llm_response(response)
                
                for item in data:
                    conv_id = f"fact_{i}_{random.randint(1000,9999)}"
                    turns = item.get("turns", [])
                    if not turns: 
                        continue
                        
                    conversations.append(TrainingConversation(
                        id=conv_id,
                        conversation_type="content_based",
                        system_prompt=system_prompt,
                        turns=turns,
                        metadata={
                            "source_file": segment.source_file, 
                            "segment_id": segment.id
                        }
                    ))
            except Exception as e:
                logger.error(f"Error generating content-based conv: {e}")
                
        return conversations

    def generate_batch(self, 
                      profile: CompleteStyleProfile, 
                      category: str, 
                      count: int) -> List[TrainingConversation]:
        """Generates purely synthetic conversations (Hallucinated/General knowledge)."""
        
        conversations = []
        system_prompt = profile.generated_system_prompt
        generation_prompt = self._create_generation_prompt(profile, category, count)
        
        try:
            response = self.llm.generate(
                prompt=generation_prompt,
                system_prompt="You are an expert dataset creator. Output valid JSON only.",
                json_mode=True
            )
            
            data = self._parse_llm_response(response)
            
            for item in data:
                conv_id = f"{category}_{random.randint(10000, 99999)}"
                turns = item.get("turns", [])
                if not turns: continue
                    
                conversations.append(TrainingConversation(
                    id=conv_id,
                    conversation_type=category,
                    system_prompt=system_prompt,
                    turns=turns,
                    metadata={"generated_by": "synthetic"}
                ))
                
        except Exception as e:
            logger.error(f"Error generating batch for {category}: {e}")
            
        return conversations

    def _create_content_prompt(self, profile: CompleteStyleProfile, context: str) -> str:
        """Prompt for extraction-based generation."""
        # Truncate context if too long to avoid token limits
        safe_context = context[:3000] 
        
        return f"""
        SOURCE TEXT:
        "{safe_context}"
        
        PERSONA:
        - Name: {profile.author_name}
        - Era: {profile.era}
        - Values: {', '.join(profile.values.dominant_values)}
        
        TASK:
        Create 1 high-quality conversation where a User asks about an event/fact found in the SOURCE TEXT, and '{profile.author_name}' answers.
        
        RULES:
        1. The Answer must differ from the text strictly in STYLE, but keep the FACTS exactly as they are in the source.
        2. Do NOT invent facts not present in the text.
        3. Use the persona's voice (e.g. third person if applicable, archaic terms).
        
        OUTPUT JSON:
        [
          {{
            "turns": [
              {{"role": "user", "content": "Question about the specific event..."}},
              {{"role": "assistant", "content": "Answer based on text..."}}
            ]
          }}
        ]
        """

    def _create_generation_prompt(self, profile: CompleteStyleProfile, category: str, count: int) -> str:
        """Prompt for synthetic/hallucinated generation."""
        examples_req = ""
        if category == "philosophy_values":
            examples_req = f"Focus on these values: {', '.join(profile.values.dominant_values)}."
        elif category == "anachronistic":
            examples_req = "Ask about modern technology (TV, AI, cars). The assistant MUST refuse or misunderstand politely based on the era."
        
        return f"""
        Generate {count} distinct conversation examples between a User and '{profile.author_name}'.
        
        PERSONA:
        - Name: {profile.author_name}
        - Era: {profile.era}
        - Values: {', '.join(profile.values.dominant_values)}
        
        TASK:
        Create {count} conversations in the category: '{category}'.
        {examples_req}
        
        OUTPUT JSON ARRAY:
        [
          {{
            "turns": [
              {{"role": "user", "content": "..."}},
              {{"role": "assistant", "content": "..."}}
            ]
          }},
          ...
        ]
        """

    def _parse_llm_response(self, text: str) -> List[Dict]:
        try:
            text = text.strip()
            if text.startswith("```json"):
                text = text[7:]
            if text.endswith("```"):
                text = text[:-3]
            
            data = json.loads(text)
            if isinstance(data, list):
                return data
            if isinstance(data, dict) and "conversations" in data:
                return data["conversations"]
            return []
        except json.JSONDecodeError:
            logger.error("Failed to decode JSON from LLM generation")
            return []