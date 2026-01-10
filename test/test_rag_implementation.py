import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from task._constants import API_KEY
from task.embeddings.embeddings_client import DialEmbeddingsClient
from task.embeddings.text_processor import TextProcessor, SearchMode
from task.chat.chat_completion_client import DialChatCompletionClient
from task.app import SYSTEM_PROMPT, USER_PROMPT, main
from task.models.conversation import Conversation
from task.models.message import Message
from task.models.role import Role
from task.utils.text import chunk_text

def test_constants_import():
    assert API_KEY is not None

def test_embeddings_client_import():
    assert hasattr(DialEmbeddingsClient, '__init__')
    assert hasattr(DialEmbeddingsClient, 'get_embeddings')

def test_text_processor_import():
    assert hasattr(TextProcessor, '__init__')
    assert hasattr(TextProcessor, 'process_text_file')
    assert hasattr(TextProcessor, 'search')

def test_chat_completion_client_import():
    assert hasattr(DialChatCompletionClient, '__init__')
    assert hasattr(DialChatCompletionClient, 'get_completion')

def test_app_import():
    assert isinstance(SYSTEM_PROMPT, str)
    assert isinstance(USER_PROMPT, str)
    assert callable(main)

def test_system_prompt():
    assert isinstance(SYSTEM_PROMPT, str)
    assert len(SYSTEM_PROMPT) > 0
    assert "RAG" in SYSTEM_PROMPT
    assert "microwave" in SYSTEM_PROMPT.lower()

def test_user_prompt():
    assert isinstance(USER_PROMPT, str)
    assert len(USER_PROMPT) > 0
    assert "{context}" in USER_PROMPT
    assert "{question}" in USER_PROMPT
    formatted = USER_PROMPT.format(context="test context", question="test question")
    assert "test context" in formatted
    assert "test question" in formatted

def test_role_enum():
    from task.models.role import Role
    assert Role.USER == "user"
    assert Role.AI == "assistant"
    assert Role.SYSTEM == "system"

def test_message_creation():
    message = Message(Role.USER, "Test content")
    assert message.role == Role.USER
    assert message.content == "Test content"
    message_dict = message.to_dict()
    assert message_dict == {"role": "user", "content": "Test content"}

def test_conversation():
    conversation = Conversation()
    assert len(conversation.messages) == 0
    message = Message(Role.USER, "Test message")
    conversation.add_message(message)
    assert len(conversation.messages) == 1
    assert conversation.messages[0] == message

def test_chunk_text_basic():
    text = "Hello World Programming"
    chunks = chunk_text(text, chunk_size=8, overlap=3)
    assert isinstance(chunks, list)
    assert all(isinstance(chunk, str) for chunk in chunks)
    if chunks:
        assert len(chunks) >= 1
        reconstructed = "".join(chunks)
        assert "Hello" in (reconstructed if len(reconstructed) >= 5 else text)

def test_chunk_text_edge_cases():
    result = chunk_text("", 10, 2)
    assert result == []
    result = chunk_text("Short", 10, 2)
    assert result == ["Short"]
    result = chunk_text("A", 5, 1)
    assert result == ["A"]

def test_search_mode_values():
    assert SearchMode.EUCLIDIAN_DISTANCE == "euclidean"
    assert SearchMode.COSINE_DISTANCE == "cosine"

def test_dial_embeddings_client_init():
    client = DialEmbeddingsClient('test-model', 'test-key')
    assert client.deployment_name == 'test-model'
    assert client.api_key == 'test-key'
    assert '/openai/deployments/test-model/embeddings' in client.endpoint

def test_dial_chat_completion_client_init():
    client = DialChatCompletionClient('test-model', 'test-key')
    expected_endpoint = 'https://ai-proxy.lab.epam.com/openai/deployments/test-model/chat/completions'
    assert client._endpoint == expected_endpoint
    assert client._api_key == 'test-key'

def test_dial_chat_completion_client_empty_key():
    try:
        DialChatCompletionClient('test-model', '')
        assert False, "Should raise ValueError"
    except ValueError:
        pass
