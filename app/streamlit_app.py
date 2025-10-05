# app/streamlit_app.py
import streamlit as st
import pandas as pd
from models.google_genai import GoogleGenAIAdapter, AIResponse
from models.hf_inference import HuggingFaceInferenceAdapter
from models.base import ModelInterface
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
import json
import io
import pandas as pd
import io
import json
import re

# ----------------- CONFIG (replace keys here if you want hard-coded) -----------------
GOOGLE_API_KEY = "xxx"    
HF_API_TOKEN = "YOUR_HF_API_TOKEN_HERE"
# -------------------------------------------------------------------------------------



def main():
    st.set_page_config(page_title="Multimodal LLM UI", layout="wide")

    st.title("Multimodel LLM Demo — Select model & output format")

    # Available models mapping
    models_map = {
        "Google Gemini (Gemini 2.5 - remote)": lambda: GoogleGenAIAdapter(api_key=GOOGLE_API_KEY),
        "HuggingFace Inference (remote)": lambda: HuggingFaceInferenceAdapter(hf_token=HF_API_TOKEN, model_id="google/flan-t5-large"),
    }

    # Input fields

    user_input = st.text_area("User prompt:", height=150, value="What is the captial of India?")
    col0, col1, col2, col3 = st.columns([2,1,1,1])
    with col1:
        model_choice = st.selectbox("Choose model:", list(models_map.keys()))
    with col2:
        output_format = st.selectbox("Choose output format:", ["JSON (parsed)", "CSV", "RAW (text)"])
    with col3: 
        st.write("  ")
        # st.write("  ")
        submit = st.button("Submit", use_container_width=True)

    # Instantiate model lazily
    if submit:
        st.info(f"Running model: {model_choice} — format: {output_format}")
        # create adapter
        adapter_factory = models_map[model_choice]
        adapter = adapter_factory()
        # try:
        #     adapter.initialize()
        # except Exception as e:
        #     st.warning(f"Model initialization warning: {e}")

        system_prompt = "You are an empathetic assistant that analyzes and summarises user messages."

        # Call model
        try:
            # Dynamically select format instructions based on output_format
            if output_format == "JSON (parsed)":
                json_parser = JsonOutputParser(pydantic_object=AIResponse)
                result = adapter.respond(
                    system_prompt=system_prompt,
                    user_prompt=user_input,
                    format_instructions=json_parser.get_format_instructions()
                )
                try:
                    # Display parsed JSON
                    if hasattr(result, "model_dump_json"):
                        st.json(json.loads(result.model_dump_json()))
                    elif isinstance(result, dict):
                        st.json(result)
                    else:
                        # Fallback: try to extract JSON block
                        m = re.search(r"\{.*\}", str(result), re.DOTALL)
                        if m:
                            try:
                                parsed = json.loads(m.group(0))
                                st.json(parsed)
                            except Exception:
                                st.write("Could not parse JSON; showing raw text:")
                                st.write(result)
                        else:
                            st.write("No JSON found; showing raw text:")
                            st.write(result)
                except Exception as e:
                    st.write(result)


            elif output_format == "CSV":
                parser = JsonOutputParser(pydantic_object=AIResponse)
                result = adapter.respond(
                    system_prompt=system_prompt,
                    user_prompt=user_input,
                    format_instructions=parser.get_format_instructions()
                )
                # Convert parsed model/dict to CSV
                if hasattr(result, "model_dump"):
                    d = result.model_dump()
                    df = pd.DataFrame([d])
                elif isinstance(result, dict):
                    df = pd.DataFrame([result])
                else:
                    st.write("CSV not available for raw text; showing raw text:")
                    st.write(result)
                    df = None

                if df is not None:
                    st.dataframe(df)
                    csv_buf = io.StringIO()
                    df.to_csv(csv_buf, index=False)
                    st.download_button(
                        "Download CSV", csv_buf.getvalue(), file_name="ai_output.csv", mime="text/csv"
                    )

            elif output_format == "RAW (text)":
                # RAW output: call without format_instructions
                result = adapter.respond(
                    system_prompt=system_prompt,
                    user_prompt=user_input
                )
                st.subheader("Raw model output")
                if hasattr(result, "response"):
                    st.write(result.response)
                else:
                    st.write(result)

        except Exception as e:
            st.error(f"Error generating AI response: {e}")
