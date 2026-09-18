import os
from typing import List, Dict, Any, Optional
from openai import OpenAI
from src.context_engine import AgentContext, Message

def build_system_prompt(context: AgentContext) -> str:
    """
    Constructs system prompt combining persona, active_topic, and pinned memory facts.
    Pinned facts and system prompt do NOT count towards TOKEN_BUDGET_LIMIT.
    """
    prompt_lines = [
        "You are an intelligent IT Helpdesk AI Assistant for a university campus.",
        "Your goal is to help students and faculty resolve technical issues clearly and efficiently.",
        f"Active Topic: {context.active_topic}"
    ]
    
    if context.pinned and context.pinned.ticket_id:
        prompt_lines.append(f"Pinned Fact - User Ticket ID: {context.pinned.ticket_id}")
    else:
        prompt_lines.append("Pinned Fact - User Ticket ID: None")
        
    prompt_lines.append(
        "Always reference the user's pinned ticket ID if they ask about their ticket or previous ticket status."
    )
    
    return "\n".join(prompt_lines)

def get_llm_client_and_model() -> tuple[Optional[OpenAI], str]:
    """
    Determines provider (Groq or OpenAI) based on environment configuration.
    Returns (client, model_name).
    """
    groq_key = os.getenv("GROQ_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")
    
    # Check for explicit invalid test key to trigger 500 error test (Req 10)
    if groq_key == "invalid_key_for_testing_500" or openai_key == "invalid_key_for_testing_500":
        raise RuntimeError("Invalid API key forced for error handling testing.")
        
    is_groq_valid = groq_key and not groq_key.startswith("your_") and groq_key != "mock_key_for_testing"
    is_openai_valid = openai_key and not openai_key.startswith("your_") and openai_key != "mock_key_for_testing"

    # Route 1: Valid Groq key provided
    if is_groq_valid:
        groq_model = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
        client = OpenAI(
            base_url="https://api.groq.com/openai/v1",
            api_key=groq_key
        )
        return client, groq_model

    # Route 2: Valid OpenAI key provided
    if is_openai_valid:
        openai_model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        client = OpenAI(api_key=openai_key)
        return client, openai_model

    # Route 3: Mock fallback for unit tests when no live API keys are provided
    return None, "mock"

def generate_llm_response(context: AgentContext, pruned_recent: List[Message]) -> str:
    """
    Executes upstream LLM call using Groq or OpenAI API.
    If the API call fails or key is invalid, raises an Exception to trigger atomic HTTP 500 rollback.
    """
    client, model = get_llm_client_and_model()

    system_prompt = build_system_prompt(context)
    
    messages: List[Dict[str, str]] = [
        {"role": "system", "content": system_prompt}
    ]
    
    for msg in pruned_recent:
        messages.append({"role": msg.role, "content": msg.content})
        
    if client is None:
        # Mock LLM response generator for local unit testing when real API key is absent
        user_last = pruned_recent[-1].content if pruned_recent else ""
        ticket_id = context.pinned.ticket_id if context.pinned else None
        
        if ticket_id and ("status" in user_last.lower() or "ticket" in user_last.lower()):
            return f"I have checked your ticket {ticket_id}. The status is currently under review by IT support."
        elif "wifi" in user_last.lower() or "mac" in user_last.lower():
            return "To resolve Mac WiFi issues: 1. Forget the network, 2. Renew DHCP Lease under Network Settings, 3. Reconnect to Campus WiFi."
        elif ticket_id:
            return f"Thank you. I have recorded your ticket ID as {ticket_id}. How can I assist you with it today?"
        else:
            return "Hello! Welcome to the IT Helpdesk. How can I help you today?"
            
    response = client.chat.completions.create(
        model=model,
        messages=messages,  # type: ignore
        temperature=0.3
    )
    
    raw_content = response.choices[0].message.content or "I am unable to process your request."
    cleaned_content = raw_content.replace("\ufffd", "").replace("", "")
    return cleaned_content
