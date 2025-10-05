# models/hf_inference.py
from typing import Any, Dict, Optional
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from .base import ModelInterface
import requests
import json
import time

# Use same schema for consistency
class AIResponse(BaseModel):
    summary: str = Field(description="Summary of the user's message")
    sentiment: int = Field(description="Sentiment score 0-100")
    response: str = Field(description="Suggested response to the user")
    action: str = Field(description="Recommended action for the support rep")


class HuggingFaceInferenceAdapter(ModelInterface):
    """
    Simple remote adapter using Hugging Face Inference API.
    Note: requires HF_API_TOKEN. This adapter does not use LangChain LLM objects
    but follows the same ModelInterface.
    """
    def __init__(self, hf_token: str, model_id: str = "google/flan-t5-large"):
        self.hf_token = hf_token
        self.model_id = model_id
        self.json_parser = JsonOutputParser(pydantic_object=AIResponse)
        self.api_url = f"https://api-inference.huggingface.co/models/{self.model_id}"

    def initialize(self) -> None:
        # nothing heavy to initialize, just verify token maybe
        headers = {"Authorization": f"Bearer {self.hf_token}"}
        resp = requests.request("GET", self.api_url, headers=headers, timeout=10)
        # If 200 or 404 metadata response, we assume token is valid for inference
        # We don't raise here to keep it permissive; user will see errors at invoke time.

    def get_prompt_template(self) -> PromptTemplate:
        return PromptTemplate(
            template="""
You are a helpful and creative AI assistant.

{system_prompt}

Please answer first naturally, then at the end emit JSON using this format:
{format_prompt}

User Message:
{user_prompt}
""",
            input_variables=["system_prompt", "format_prompt", "user_prompt"]
        )

    def _call_hf(self, prompt: str, retries: int = 3) -> str:
        headers = {"Authorization": f"Bearer {self.hf_token}", "Content-Type": "application/json"}
        payload = {"inputs": prompt, "options": {"wait_for_model": True}}
        for attempt in range(1, retries + 1):
            r = requests.post(self.api_url, headers=headers, json=payload, timeout=60)
            if r.status_code == 200:
                data = r.json()
                # Inference API may return a string or a list depending on model
                if isinstance(data, dict) and "error" in data:
                    if attempt < retries:
                        time.sleep(1)
                        continue
                    else:
                        raise RuntimeError(f"HF inference error: {data['error']}")
                if isinstance(data, list):
                    # many models return [{"generated_text": "..."}]
                    text = data[0].get("generated_text") or data[0].get("text") or str(data)
                elif isinstance(data, str):
                    text = data
                else:
                    text = json.dumps(data)
                return text
            else:
                # non-200
                if attempt < retries:
                    time.sleep(1)
                    continue
                else:
                    raise RuntimeError(f"HuggingFace API error {r.status_code}: {r.text}")
        raise RuntimeError("HF call exhausted retries")

    def respond(self, system_prompt: str, user_prompt: str, format_instructions: str, multimodal_inputs: Optional[Dict[str, Any]] = None):
        prompt_tpl = self.get_prompt_template()
        prompt = prompt_tpl.format(system_prompt=system_prompt, format_prompt=format_instructions, user_prompt=user_prompt)
        raw_text = self._call_hf(prompt)
        # Try to parse JSON block from raw_text using JSON parser via manual extraction
        match = None
        import re
        m = re.search(r"\{.*\}", raw_text, re.DOTALL)
        if m:
            try:
                parsed = self.json_parser.parse(m.group(0))
                return parsed
            except Exception:
                # parser failed, return raw text
                return raw_text
        else:
            return raw_text
