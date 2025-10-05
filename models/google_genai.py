# models/google_genai.py
from typing import Any, Dict, Optional
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field
from .base import ModelInterface

# # Structured schema (same as you used)
class AIResponse(BaseModel):
    summary: str = Field(description="Summary of the user's message")
    sentiment: int = Field(description="Sentiment score 0-100")
    response: str = Field(description="Suggested response to the user")
    action: str = Field(description="Recommended action for the support rep")


class GoogleGenAIAdapter(ModelInterface):
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.model = None
        self.json_parser = JsonOutputParser(pydantic_object=AIResponse)

    def initialize(self):
        # instantiate the LangChain wrapper for Gemini
        self.model = ChatGoogleGenerativeAI(
            model="gemini-2.5-pro",
            google_api_key=self.api_key,
            temperature=0.4,
            max_output_tokens=1024,
            top_p=0.9,
        )

    def get_prompt_template(self) -> PromptTemplate:
        return PromptTemplate(
            template="""
                        You are a helpful and creative AI assistant.

                        {system_prompt}

                        Finally, make sure the structured output follows exactly:
                        {format_prompt}

                        User Message:
                        {user_prompt}
                    """,
            input_variables=["system_prompt", "format_prompt", "user_prompt"]
        )

    def respond(self, system_prompt: str, user_prompt: str, format_instructions: str, multimodal_inputs: Optional[Dict[str, Any]] = None):
        if self.model is None:
            self.initialize()

        template = self.get_prompt_template()
        chain = template | self.model | self.json_parser
        # Run the chain as you instructed earlier
        return chain.invoke({
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "format_prompt": format_instructions
        })
    
