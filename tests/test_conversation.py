from jarvis.core.conversation import Conversation, ConversationManager
from jarvis.llm.base import Message


def test_conversation_creation():
    """Test creating a new conversation and verifying its initial state."""
    manager = ConversationManager(system_prompt="Test Prompt")
    conv = manager.new_conversation()

    assert isinstance(conv, Conversation)
    assert manager.get_active() == conv
    assert len(conv.messages) == 1
    assert conv.messages[0].role == "system"
    assert conv.messages[0].content == "Test Prompt"


def test_add_messages():
    """Test adding user and assistant messages."""
    manager = ConversationManager()
    manager.add_user_message("Hello")
    conv = manager.get_active()

    assert len(conv.user_messages) == 1
    assert conv.user_messages[0].content == "Hello"

    manager.add_assistant_message("Hi")
    assert len(conv.assistant_messages) == 1
    assert conv.assistant_messages[0].content == "Hi"


def test_token_estimate():
    """Test the rudimentary token estimation logic."""
    conv = Conversation()
    conv.add_message(Message.user("Hello World!"))
    assert conv.token_estimate == 3


def test_history_trimming():
    """Test that large histories are trimmed appropriately based on max_tokens."""
    conv = Conversation()
    conv.add_message(Message.system("System context"))
    for i in range(10):
        conv.add_message(Message.user("Short msg"))

    trimmed = conv.to_llm_messages(max_tokens=10)
    assert len(trimmed) < 11
    assert trimmed[0].role == "system"


def test_long_latest_message_is_retained_within_budget():
    conv = Conversation()
    conv.add_message(Message.system("System context"))
    conv.add_message(Message.user("START " + "x" * 10000 + " END"))
    trimmed = conv.to_llm_messages(max_tokens=100)
    assert trimmed[-1].role == "user"
    assert trimmed[-1].content.startswith("START")
    assert trimmed[-1].content.endswith("END")
    assert sum(len(m.content) for m in trimmed) <= 400
