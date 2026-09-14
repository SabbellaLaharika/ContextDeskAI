import pytest
from src.context_engine import Message
from src.token_budget import count_tokens, count_message_tokens, calculate_recent_tokens, apply_token_budget

def test_count_tokens():
    tokens = count_tokens("Hello world")
    assert tokens > 0
    assert isinstance(tokens, int)

def test_count_message_tokens():
    msg = Message(role="user", content="Hello world")
    tokens = count_message_tokens(msg)
    assert tokens == count_tokens("user") + count_tokens("Hello world") + 4

def test_apply_token_budget_within_limit():
    messages = [
        Message(role="user", content="Hi"),
        Message(role="assistant", content="Hello")
    ]
    pruned, total = apply_token_budget(messages, max_tokens=100)
    assert len(pruned) == 2
    assert total <= 100

def test_apply_token_budget_fifo_eviction():
    msg1 = Message(role="user", content="Message A: " + "word " * 30)
    msg2 = Message(role="assistant", content="Response A: " + "word " * 30)
    msg3 = Message(role="user", content="Message B: Short message")
    
    messages = [msg1, msg2, msg3]
    # Total tokens for msg1+msg2+msg3 is ~80-100 tokens. Set budget limit = 25 tokens.
    pruned, total = apply_token_budget(messages, max_tokens=25)
    
    # Oldest messages (msg1 and msg2) should be evicted FIFO, keeping msg3
    assert total <= 25
    assert len(pruned) == 1
    assert pruned[0].content == msg3.content
