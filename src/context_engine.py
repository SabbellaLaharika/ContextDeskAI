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

from src.token_budget import (
    count_tokens,
    count_message_tokens,
    calculate_recent_tokens,
    apply_token_budget
)

from src.extractor import extract_ticket_id, infer_active_topic

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
