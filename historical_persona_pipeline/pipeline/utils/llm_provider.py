import os
import json
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
import requests

logger = logging.getLogger(__name__)

class LLMProvider(ABC):
    @abstractmethod
    def generate(self, prompt: str, system_prompt: Optional[str] = None, json_mode: bool = False) -> str:
        pass

class LMStudioProvider(LLMProvider):
    def __init__(self, base_url: str = "http://localhost:1234/v1", api_key: str = "lm-studio", model: str = "local-model"):
        self.base_url = base_url
        self.api_key = api_key
        self.model = model

    def generate(self, prompt: str, system_prompt: Optional[str] = None, json_mode: bool = False) -> str:
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"}
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.7,
            "stream": False
        }
        
        if json_mode:
            payload["response_format"] = {"type": "json_object"}

        try:
            response = requests.post(f"{self.base_url}/chat/completions", headers=headers, json=payload, timeout=120)
            response.raise_for_status()
            return response.json()['choices'][0]['message']['content']
        except Exception as e:
            logger.error(f"LM Studio API Error: {e}")
            raise

class OpenAICompatibleProvider(LLMProvider):
    """Generic provider for Nebius, Groq, DeepSeek, etc."""
    def __init__(self, base_url: str, api_key: str, model: str):
        self.base_url = base_url
        self.api_key = api_key
        self.model = model

    def generate(self, prompt: str, system_prompt: Optional[str] = None, json_mode: bool = False) -> str:
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"}
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": 4096
        }
        
        if json_mode:
            payload["response_format"] = {"type": "json_object"}

        try:
            response = requests.post(f"{self.base_url}/chat/completions", headers=headers, json=payload, timeout=60)
            response.raise_for_status()
            return response.json()['choices'][0]['message']['content']
        except Exception as e:
            logger.error(f"OpenAI Compatible API Error: {e}")
            raise

class AnthropicProvider(LLMProvider):
    def __init__(self, api_key: str, model: str = "claude-3-sonnet-20240229"):
        import anthropic
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model

    def generate(self, prompt: str, system_prompt: Optional[str] = None, json_mode: bool = False) -> str:
        try:
            kwargs = {
                "model": self.model,
                "max_tokens": 4096,
                "messages": [{"role": "user", "content": prompt}]
            }
            if system_prompt:
                kwargs["system"] = system_prompt
                
            response = self.client.messages.create(**kwargs)
            return response.content[0].text
        except Exception as e:
            logger.error(f"Anthropic API Error: {e}")
            raise

class LLMFactory:
    @staticmethod
    def create(config: Dict[str, Any]) -> LLMProvider:
        provider_type = config.get("llm_provider", "lm_studio")
        
        if provider_type == "lm_studio":
            return LMStudioProvider(
                base_url=config.get("llm_base_url", "http://localhost:1234/v1"),
                api_key=config.get("llm_api_key", "lm-studio"),
                model=config.get("llm_model_name", "local-model")
            )
        elif provider_type == "nebius" or provider_type == "openai_compatible":
            return OpenAICompatibleProvider(
                base_url=config.get("llm_base_url"),
                api_key=config.get("llm_api_key"),
                model=config.get("llm_model_name")
            )
        elif provider_type == "anthropic":
            return AnthropicProvider(
                api_key=config.get("llm_api_key"),
                model=config.get("llm_model_name", "claude-3-sonnet-20240229")
            )
        else:
            raise ValueError(f"Unknown LLM provider: {provider_type}")
