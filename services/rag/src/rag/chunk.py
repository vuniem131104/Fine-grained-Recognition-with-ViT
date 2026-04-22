from pydantic import BaseModel
from litellm import token_counter

def tokens_calculator(text: str, model: str = "gpt-4o") -> int:
    """Calculate the number of tokens for a given model and text.

    Args:
        text (str): The text to analyze.
        model (str): The model to use for token calculation.

    Returns:
        int: The number of tokens in the text.
    """
    return token_counter(model=model, text=text)

class ChunkerInput(BaseModel):
    contents: str

class ChunkerOutput(BaseModel):
    chunks: list[str]


class ChunkerService:
    
    def __init__(self, max_tokens: int = 1000, min_tokens: int = 500):
        self.max_tokens = max_tokens
        self.min_tokens = min_tokens
        
    def process(self, inputs: ChunkerInput) -> ChunkerOutput:
        chunks = self._split_by_headers(inputs.contents)
        return ChunkerOutput(chunks=chunks)

    def _split_by_headers(self, content: str) -> list[str]:
        lines = content.split('\n')
        chunks = []
        current_chunk = []
        current_tokens = 0
        header_stack = []  # Stack to track hierarchy of headers
        first_line_processed = False
        
        for line_idx, line in enumerate(lines):
            # Check if line is a header (only first line with # or lines with ##, ###, etc.)
            is_header = False
            if line.startswith('#'):
                # First line: only treat as header if it's exactly # (level 1)
                if line_idx == 0 and not first_line_processed:
                    first_line_processed = True
                    if line.startswith('# ') and not line.startswith('##'):
                        is_header = True
                        header_level = 1
                # All other lines: treat as header if it's ## or higher (not single #)
                elif line.startswith('## '):
                    is_header = True
                    header_level = len(line) - len(line.lstrip('#'))
            
            if is_header:
                # Update header stack based on level
                # Keep headers up to the parent level
                header_stack = [h for h in header_stack if h[0] < header_level]
                header_stack.append((header_level, line))
                
                # Check if we should start a new chunk for major sections (## level)
                if header_level == 2:
                    # If current chunk has content and meets minimum token requirement
                    if current_chunk and current_tokens >= self.min_tokens:
                        chunk_content = '\n'.join(current_chunk)
                        chunks.append(chunk_content)
                        current_chunk = []
                        current_tokens = 0
                    
                    # Start new chunk with full header hierarchy (excluding the current header)
                    # We'll add the current header after this
                    if current_chunk == []:  # Only add context if starting fresh
                        current_chunk = self._build_header_context(header_stack[:-1])
                        current_tokens = tokens_calculator('\n'.join(current_chunk))
                
                current_chunk.append(line)
                current_tokens = tokens_calculator('\n'.join(current_chunk))
                
            else:
                # Regular content line (including comment lines like #...)
                line_tokens = tokens_calculator(line)
                potential_tokens = current_tokens + line_tokens
                
                # Check if adding this line would exceed max tokens
                if potential_tokens > self.max_tokens and current_chunk:
                    # Save current chunk if it meets minimum requirement
                    if current_tokens >= self.min_tokens:
                        chunk_content = '\n'.join(current_chunk)
                        chunks.append(chunk_content)
                        
                        # Start new chunk with header context
                        current_chunk = self._build_header_context(header_stack)
                        current_tokens = tokens_calculator('\n'.join(current_chunk))
                    
                current_chunk.append(line)
                current_tokens = tokens_calculator('\n'.join(current_chunk))
        
        # Add the last chunk if it has content
        if current_chunk:
            chunk_content = '\n'.join(current_chunk)
            chunks.append(chunk_content)
        
        return chunks
    
    def _build_header_context(self, header_stack: list[tuple[int, str]]) -> list[str]:
        context_lines = []
        for level, header in header_stack:
            context_lines.append(header)
        return context_lines
