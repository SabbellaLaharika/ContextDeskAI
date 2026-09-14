from typing import List, Tuple
import tiktoken
from src.context_engine import Message

try:
    TOKENIZER = tiktoken.get_encoding("cl100k_base")
except Exception:
    TOKENIZER = tiktoken.encoding_for_model("gpt-4o-mini")

def count_tokens(text: str) -> int:
    """Calculates tokens using cl100k_base encoding."""
    if not text:
        return 0
    return len(TOKENIZER.encode(text))

def count_message_tokens(msg: Message) -> int:
    """Calculates total tokens for a single message object (role + content + message overhead)."""
    return count_tokens(msg.role) + count_tokens(msg.content) + 4

def calculate_recent_tokens(recent: List[Message]) -> int:
    """Sum total tokens across all messages in recent history."""
    return sum(count_message_tokens(msg) for msg in recent)

def apply_token_budget(recent: List[Message], max_tokens: int) -> Tuple[List[Message], int]:
    """
    Evicts oldest messages (FIFO) from recent history until the total tokens <= max_tokens.
    System prompt and pinned facts do NOT count towards max_tokens limit.
    Returns (pruned_recent_messages, total_recent_tokens).
    """
    pruned = list(recent)
    total = calculate_recent_tokens(pruned)
    
    while total > max_tokens and pruned:
        pruned.pop(0)
        total = calculate_recent_tokens(pruned)
        
    return pruned, total
