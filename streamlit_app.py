"""
This module creates a Streamlit web application for optimizing prompts using the Groq API.
It allows users to enter prompts in basic or advanced modes, configure model parameters,
and view the optimized results.
"""
import streamlit as st
import os
import sys
import importlib.util
from dotenv import load_dotenv
from dataclasses import dataclass
from typing import List, Optional

load_dotenv()

@dataclass
class PromptCanvas:
    """
    A data class to hold prompt details for advanced prompt engineering.
    """
    persona: str = ""
    audience: str = ""
    task: str = ""
    steps: Optional[List[str]] = None
    context: str = ""
    references: Optional[List[str]] = None
    output_format: str = ""
    tonality: str = ""

    def __post_init__(self):
        """
        Initializes the steps and references lists if they are None.
        """
        if self.steps is None:
            self.steps = []
        if self.references is None:
            self.references = []

def import_module_from_path(module_name, file_path):
    """
    Imports a module from a given file path.

    Args:
        module_name (str): The name to give the module.
        file_path (str): The path to the module file.

    Returns:
        module: The imported module.
    """
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

# Load default model, temperature, and max tokens from environment variables or set defaults
DEFAULT_MODEL = os.getenv('GROMPT_DEFAULT_MODEL', 'llama-3.3-70b-versatile')
DEFAULT_TEMPERATURE = float(os.getenv('GROMPT_DEFAULT_TEMPERATURE', '0.5'))
DEFAULT_MAX_TOKENS = int(os.getenv('GROMPT_DEFAULT_MAX_TOKENS', '1024'))

# Streamlit app configuration
st.sidebar.title("Configuration")
GROQ_API_KEY = st.sidebar.text_input("Enter your GROQ API Key:", type="password")

if not GROQ_API_KEY:
    st.sidebar.warning("Please enter your GROQ API Key to use the app.")

st.title("Grompt - Prompt Optimizer")
st.write("Grompt uses Groq's LLM services to instantly optimize prompts.")

# Add tabs for Basic and Advanced modes
tab1, tab2 = st.tabs(["Basic", "Advanced (Prompt Canvas)"])

# Basic mode tab
with tab1:
    user_prompt = st.text_area("Enter your prompt:", height=100)

# Advanced mode tab
with tab2:
    with st.expander("Persona & Audience", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            persona = st.text_input("Persona/Role", placeholder="e.g., expert technical writer")
        with col2:
            audience = st.text_input("Target Audience", placeholder="e.g., software developers")
    
    task = st.text_area("Task/Intent", placeholder="Describe the specific task...")
    steps = st.text_area("Steps", placeholder="Enter steps, one per line...")
    context = st.text_area("Context", placeholder="Provide relevant background...")
    references = st.text_area("References", placeholder="Enter references, one per line...")
    
    with st.expander("Output Format & Tone", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            output_format = st.selectbox("Output Format", 
                ["Natural Text", "Technical Documentation", "Code", "Markdown"])
        with col2:
            tonality = st.text_input("Tone", placeholder="e.g., professional, technical")
    
    canvas_prompt = st.text_area("Your Prompt:", height=100)

# Shared model settings
col1, col2, col3 = st.columns(3)
with col1:
    model = st.selectbox("Select Model", [
        "llama-3.3-70b-versatile",
        "llama3-groq-8b-8192-tool-use-preview",
        "llama3-70b-8192",
        "llama3-8b-8192"
    ], index=0)
with col2:
    temperature = st.slider("Temperature", 0.0, 1.0, DEFAULT_TEMPERATURE, 0.1)
with col3:
    max_tokens = st.number_input("Max Tokens", 1, 32768, DEFAULT_MAX_TOKENS)

optimized_prompt = None  # Initialize optimized_prompt

# Button to trigger prompt optimization
if st.button("Optimize Prompt"):
    if not GROQ_API_KEY:
        st.error("Please enter your GROQ API Key in the sidebar.")
    elif user_prompt or canvas_prompt:
        os.environ['GROQ_API_KEY'] = GROQ_API_KEY # Set the API key in environment
        
        try:
            # Import the Grompt module
            Grompt = import_module_from_path("Grompt", "Grompt.py")
        except Exception as e:
            st.error(f"Unable to import 'Grompt': {str(e)}")
            st.stop()
        
        with st.spinner("Optimizing prompt..."):
            if canvas_prompt:  # Advanced mode
                # Create a PromptCanvas object from user inputs
                canvas = PromptCanvas(
                    persona=persona,
                    audience=audience,
                    task=task,
                    steps=[s.strip() for s in steps.split('\n') if steps and s.strip()],
                    context=context,
                    references=[r.strip() for r in references.split('\n') if references and r.strip()],
                    output_format=output_format,
                    tonality=tonality
                )
                # Rephrase the prompt using the PromptCanvas
                optimized_prompt = Grompt.rephrase_prompt(
                    canvas_prompt, model, temperature, max_tokens, canvas=canvas
                )
            else:  # Basic mode
                # Rephrase the prompt without a PromptCanvas
                optimized_prompt = Grompt.rephrase_prompt(
                    user_prompt, model, temperature, max_tokens
                )
                
            if optimized_prompt:
                st.subheader("Optimized Prompt:")
                st.write(optimized_prompt)
    else:
        st.warning("Please enter a prompt to optimize.")

st.markdown("---")
st.write("Powered by Groq LLM services.")

st.sidebar.markdown("---")
st.sidebar.info(
    "Note: Your API key is used only for this session and is not stored. "
    "Always keep your API keys confidential."
)

st.sidebar.markdown("---")
st.sidebar.markdown("[View on GitHub]https://github.com/Jagarut/Gprompt")
st.sidebar.markdown("Created by J. Gravelle<br>Debugged by ChusDeBoss", unsafe_allow_html=True)

