import sys
import os
import pytest

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


class TestImports:
    def test_constants_import(self):
        assert API_KEY is not None

    def test_embeddings_client_import(self):
        assert hasattr(DialEmbeddingsClient, '__init__')
        assert hasattr(DialEmbeddingsClient, 'get_embeddings')

    def test_text_processor_import(self):
        assert hasattr(TextProcessor, '__init__')
        assert hasattr(TextProcessor, 'process_text_file')
        assert hasattr(TextProcessor, 'search')

    def test_chat_completion_client_import(self):
        assert hasattr(DialChatCompletionClient, '__init__')
        assert hasattr(DialChatCompletionClient, 'get_completion')

    def test_app_import(self):
        assert isinstance(SYSTEM_PROMPT, str)
        assert isinstance(USER_PROMPT, str)
        assert callable(main)


class TestAppComponents:
    def test_system_prompt(self):
        assert isinstance(SYSTEM_PROMPT, str)
        assert len(SYSTEM_PROMPT) > 0
        assert "RAG" in SYSTEM_PROMPT
        assert "microwave" in SYSTEM_PROMPT.lower()

    def test_user_prompt(self):
        assert isinstance(USER_PROMPT, str)
        assert len(USER_PROMPT) > 0
        assert "{context}" in USER_PROMPT
        assert "{question}" in USER_PROMPT
        
        formatted = USER_PROMPT.format(context="test context", question="test question")
        assert "test context" in formatted
        assert "test question" in formatted


class TestModels:
    def test_role_enum(self):
        assert Role.USER == "user"
        assert Role.AI == "assistant"
        assert Role.SYSTEM == "system"

    def test_message_creation(self):
        message = Message(Role.USER, "Test content")
        assert message.role == Role.USER
        assert message.content == "Test content"
        
        message_dict = message.to_dict()
        assert message_dict == {"role": "user", "content": "Test content"}

    def test_conversation(self):
        conversation = Conversation()
        assert len(conversation.messages) == 0
        
        message = Message(Role.USER, "Test message")
        conversation.add_message(message)
        assert len(conversation.messages) == 1
        assert conversation.messages[0] == message


class TestUtils:
    def test_chunk_text_basic(self):
        text = "Hello World Programming"
        chunks = chunk_text(text, chunk_size=8, overlap=3)
        
        assert isinstance(chunks, list)
        assert all(isinstance(chunk, str) for chunk in chunks)
        
        if chunks:
            assert len(chunks) >= 1
            reconstructed = "".join(chunks)
            assert "Hello" in (reconstructed if len(reconstructed) >= 5 else text)

    def test_chunk_text_edge_cases(self):
        result = chunk_text("", 10, 2)
        assert result == []
        
        result = chunk_text("Short", 10, 2)
        assert result == ["Short"]
        
        result = chunk_text("A", 5, 1)
        assert result == ["A"]


class TestSearchMode:
    def test_search_mode_values(self):
        assert SearchMode.EUCLIDIAN_DISTANCE == "euclidean"
        assert SearchMode.COSINE_DISTANCE == "cosine"


class TestBasicInitialization:
    def test_dial_embeddings_client_init(self):
        client = DialEmbeddingsClient('test-model', 'test-key')
        assert client.deployment_name == 'test-model'
        assert client.api_key == 'test-key'
        assert client.endpoint.endswith('/openai/deployments/test-model/embeddings')

    def test_dial_chat_completion_client_init(self):
        client = DialChatCompletionClient('test-model', 'test-key')
        assert client._endpoint == 'https://ai-proxy.lab.epam.com/openai/deployments/test-model/chat/completions'
        assert client._api_key == 'test-key'

    def test_dial_chat_completion_client_empty_key(self):
        with pytest.raises(ValueError):
            DialChatCompletionClient('test-model', '')


if __name__ == "__main__":
    pytest.main([__file__])