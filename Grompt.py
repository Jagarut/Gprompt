"""
This module provides functionality to rephrase prompts using the Groq API.
It includes functions to craft system messages, rephrase prompts, and a command-line interface.
"""
import argparse
import os
from dotenv import load_dotenv
from groq import Groq
from typing import Optional, List
from prompt_canvas import PromptCanvas

load_dotenv()

# Load default model, temperature, and max tokens from environment variables or set defaults
DEFAULT_MODEL = os.getenv('GROMPT_DEFAULT_MODEL', 'llama-3.3-70b-versatile')
DEFAULT_TEMPERATURE = float(os.getenv('GROMPT_DEFAULT_TEMPERATURE', '0.5'))
DEFAULT_MAX_TOKENS = int(os.getenv('GROMPT_DEFAULT_MAX_TOKENS', '1024'))

def craft_system_message(canvas: Optional[PromptCanvas] = None, prompt: str = "") -> str:
    """
    Crafts a system message for the Groq API based on the provided PromptCanvas or a user prompt.

    If a PromptCanvas is provided, it constructs a detailed system message using the canvas attributes.
    Otherwise, it uses the user prompt to create a rephrasing system message.

    Args:
        canvas (Optional[PromptCanvas]): An optional PromptCanvas object containing prompt details.
        prompt (str): A user-provided prompt string.

    Returns:
        str: A formatted system message string.
    """
    if canvas:
        # Construct a detailed system message from the PromptCanvas
        steps_str = '\n'.join(f'- {step}' for step in canvas.steps) if canvas.steps else "None"
        references_str = ', '.join(canvas.references) if canvas.references else "None"
        return f"""You are a {canvas.persona} focused on delivering results for {canvas.audience}.

Task: {canvas.task}

Step-by-Step Approach:
{steps_str}

Context: {canvas.context}

References: {references_str}

Output Requirements:
- Format: {canvas.output_format}
- Tone: {canvas.tonality}"""
    else:
        # If no canvas, use the user prompt to create a rephrasing system message
        return get_rephrased_user_prompt(prompt)

def get_rephrased_user_prompt(prompt: str) -> str:
    """
    Creates a system message for rephrasing a user prompt.

    This function generates a system message that instructs the LLM to act as a prompt engineer
    and optimize the given user prompt.

    Args:
        prompt (str): The user-provided prompt string.

    Returns:
        str: A formatted system message string for rephrasing.
    """
    return f"""You are a professional prompt engineer. Optimize this prompt by making it clearer, more concise, and more effective.
    User request: "{prompt}"
    Rephrased:"""

def rephrase_prompt(prompt: str, 
                   model: str = DEFAULT_MODEL,
                   temperature: float = DEFAULT_TEMPERATURE, 
                   max_tokens: int = DEFAULT_MAX_TOKENS,
                   canvas: Optional[PromptCanvas] = None) -> str:
    """
    Rephrases a given prompt using the Groq API.

    This function takes a user prompt and optionally a PromptCanvas, and uses the Groq API
    to rephrase the prompt based on the provided context.

    Args:
        prompt (str): The prompt to be rephrased.
        model (str): The Groq model to use.
        temperature (float): The temperature setting for the model.
        max_tokens (int): The maximum number of tokens for the response.
        canvas (Optional[PromptCanvas]): An optional PromptCanvas object containing prompt details.

    Returns:
        str: The rephrased prompt.

    Raises:
        Exception: If there is an error during the prompt engineering process.
    """
    try:
        groq = Groq() # Initialize the Groq client
        system_message = craft_system_message(canvas, prompt) # Craft the system message
        
        # Call the Groq API to get a rephrased prompt
        response = groq.chat.completions.create(
            messages=[{"role": "user", "content": system_message}],
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        
        return response.choices[0].message.content.strip() # Return the rephrased prompt
    except Exception as e:
        raise Exception(f"Prompt engineering error: {str(e)}")

def main():
    """
    Main function to parse command line arguments and rephrase a prompt.
    """
    parser = argparse.ArgumentParser(description="Rephrase prompts using Groq LLM.")
    parser.add_argument("prompt", help="The prompt to rephrase")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--max_tokens", type=int, default=DEFAULT_MAX_TOKENS)
    
    args = parser.parse_args()
    
    try:
        rephrased = rephrase_prompt(args.prompt, args.model, args.temperature, args.max_tokens)
        print("Rephrased prompt:")
        print(rephrased)
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
