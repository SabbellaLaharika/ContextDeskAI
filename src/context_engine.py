import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from pydantic import BaseModel, Field
import tiktoken

LOGS_DIR = Path("./logs")

class Message(BaseModel):
    role: str  # "user" or "assistant"
    content: str

class PinnedFacts(BaseModel):
    ticket_id: Optional[str] = None

class AgentContext(BaseModel):
    session_id: str
    active_topic: str = "general"
    pinned: PinnedFacts = Field(default_factory=PinnedFacts)
    recent: List[Message] = Field(default_factory=list)
    total_recent_tokens: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "active_topic": self.active_topic,
            "pinned": {
                "ticket_id": self.pinned.ticket_id
            },
            "recent": [msg.model_dump() for msg in self.recent],
            "total_recent_tokens": self.total_recent_tokens
        }

# In-memory store for fast session lookup
SESSION_STORE: Dict[str, AgentContext] = {}

# Initialize tiktoken encoder
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
    """Calculates total tokens for a single message object."""
    # Standard format: role + content + message overhead tokens
    return count_tokens(msg.role) + count_tokens(msg.content) + 4

def calculate_recent_tokens(recent: List[Message]) -> int:
    """Sum total tokens across all messages in recent history."""
    return sum(count_message_tokens(msg) for msg in recent)

def apply_token_budget(recent: List[Message], max_tokens: int) -> Tuple[List[Message], int]:
    """
    Evicts oldest messages from recent until total_recent_tokens <= max_tokens.
    Returns (pruned_recent, total_tokens).
    """
    pruned = list(recent)
    total = calculate_recent_tokens(pruned)
    
    while total > max_tokens and pruned:
        pruned.pop(0)
        total = calculate_recent_tokens(pruned)
        
    return pruned, total

def extract_ticket_id(text: str) -> Optional[str]:
    """Dynamically identifies IT Ticket IDs matching format IT-XXXX (X is digit)."""
    match = re.search(r"\bIT-\d{4}\b", text, re.IGNORECASE)
    if match:
        return match.group(0).upper()
    return None

def infer_active_topic(text: str, current_topic: str = "general") -> str:
    """
    Infers the active topic based on user message content.
    Categorizes into distinct strings (e.g. 'general', 'wifi_support', 'ticket_inquiry').
    """
    lower_text = text.lower()
    
    # Check for ticket inquiry
    if extract_ticket_id(text) or "ticket" in lower_text:
        return "ticket_inquiry"
        
    # Check for wifi / network support
    if any(k in lower_text for k in ["wifi", "wi-fi", "internet", "network", "connect", "mac"]):
        return "wifi_support"
        
    # Check for generic greeting
    if any(k in lower_text for k in ["hi", "hello", "hey", "help"]):
        if current_topic == "general" or not current_topic:
            return "general"
            
    return current_topic or "general"

def get_log_filepath(session_id: str) -> Path:
    """Returns path for context log file: ./logs/context_{session_id}.json"""
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    return LOGS_DIR / f"context_{session_id}.json"

def load_context(session_id: str, create_if_missing: bool = False) -> Optional[AgentContext]:
    """Loads session context from in-memory store or disk file."""
    # Check in-memory store
    if session_id in SESSION_STORE:
        return SESSION_STORE[session_id]
        
    # Check disk log file
    file_path = get_log_filepath(session_id)
    if file_path.exists():
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                pinned_data = data.get("pinned", {})
                pinned = PinnedFacts(ticket_id=pinned_data.get("ticket_id"))
                recent_data = data.get("recent", [])
                recent = [Message(**msg) for msg in recent_data]
                
                context = AgentContext(
                    session_id=data.get("session_id", session_id),
                    active_topic=data.get("active_topic", "general"),
                    pinned=pinned,
                    recent=recent,
                    total_recent_tokens=data.get("total_recent_tokens", calculate_recent_tokens(recent))
                )
                SESSION_STORE[session_id] = context
                return context
        except Exception:
            pass
            
    if create_if_missing:
        return AgentContext(session_id=session_id)
        
    return None

def save_context(context: AgentContext) -> None:
    """Persists context object to ./logs/context_{session_id}.json and updates SESSION_STORE."""
    # Update in-memory store
    SESSION_STORE[context.session_id] = context
    
    # Write to disk
    file_path = get_log_filepath(context.session_id)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(context.to_dict(), f, indent=2)
