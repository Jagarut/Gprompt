from dataclasses import dataclass
from typing import List, Optional

@dataclass
class PromptCanvas:
    """
    A data class to hold prompt details for advanced prompt engineering.
    """
    persona: str
    audience: str
    task: str
    steps: Optional[List[str]]
    context: str
    references: Optional[List[str]]
    output_format: str
    tonality: str
