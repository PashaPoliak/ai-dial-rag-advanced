import logging
from unittest.mock import Mock, patch, MagicMock, mock_open
import sys
import os
import pytest

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from task.embeddings.embeddings_client import DialEmbeddingsClient
from task.embeddings.text_processor import TextProcessor, SearchMode
from task.chat.chat_completion_client import DialChatCompletionClient
from task.app import SYSTEM_PROMPT, USER_PROMPT, main
from task.models.conversation import Conversation
from task.models.message import Message
from task.models.role import Role
from task.utils.text import chunk_text


class TestEmbeddingsClient:
    @pytest.fixture
    def client(self):
        return DialEmbeddingsClient('test-model', 'test-key')

    def test_initialization(self):
        client = DialEmbeddingsClient('model-name', 'api-key')
        assert client.deployment_name == 'model-name'
        assert client.api_key == 'api-key'
        assert client.endpoint.endswith('/openai/deployments/model-name/embeddings')

    @patch('requests.post')
    def test_get_embeddings_success(self, mock_post, client):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "data": [
                {"index": 0, "embedding": [0.1, 0.2, 0.3]},
                {"index": 1, "embedding": [0.4, 0.5, 0.6]}
            ]
        }
        mock_post.return_value = mock_response

        inputs = ["hello world", "test input"]
        result = client.get_embeddings(inputs, dimensions=3)

        assert len(result) == 2
        assert result[0] == [0.1, 0.2, 0.3]
        assert result[1] == [0.4, 0.5, 0.6]
        
        mock_post.assert_called_once()
        args, kwargs = mock_post.call_args
        assert kwargs['json']['input'] == inputs
        assert kwargs['json']['dimensions'] == 3

    @patch('requests.post')
    def test_get_embeddings_failure(self, mock_post, client):
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"
        mock_post.return_value = mock_response

        with pytest.raises(Exception) as context:
            client.get_embeddings(["test"], dimensions=3)
        
        assert "HTTP 500" in str(context.value)


class TestTextProcessor:
    @pytest.fixture
    def setup_processor(self):
        embeddings_client = Mock()
        db_config = {
            'host': 'localhost',
            'port': 5433,
            'database': 'vectordb',
            'user': 'postgres',
            'password': 'postgres'
        }
        processor = TextProcessor(embeddings_client, db_config)
        return embeddings_client, processor, db_config

    def test_initialization(self, setup_processor):
        embeddings_client, processor, db_config = setup_processor
        assert processor.embeddings_client == embeddings_client
        assert processor.db_config == db_config

    def test_search_mode_enum(self):
        assert SearchMode.EUCLIDIAN_DISTANCE == "euclidean"
        assert SearchMode.COSINE_DISTANCE == "cosine"

    @patch('task.embeddings.text_processor.psycopg2.connect')
    def test_truncate_table(self, mock_connect, setup_processor):
        embeddings_client, processor, db_config = setup_processor
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_connect.return_value.__enter__.return_value = mock_conn
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
        
        processor._truncate_table()
        
        mock_cursor.execute.assert_called_once()
        mock_conn.commit.assert_called_once()

    @patch('task.embeddings.text_processor.chunk_text', return_value=["chunk1", "chunk2", "chunk3"])
    @patch('builtins.open', new_callable=mock_open, read_data="test content for chunking")
    @patch.object(TextProcessor, '_save_chunk')
    def test_process_text_file(self, mock_save_chunk, mock_file, mock_chunk_text, setup_processor):
        embeddings_client, _, db_config = setup_processor
        embeddings_client.get_embeddings.return_value = {0: [0.1, 0.2, 0.3], 1: [0.4, 0.5, 0.6], 2: [0.7, 0.8, 0.9]}
        
        processor = TextProcessor(embeddings_client, db_config)
        
        processor.process_text_file("test.txt", chunk_size=10, overlap=2, dimensions=3, truncate_table=False)
        
        mock_file.assert_called_once_with("test.txt", 'r', encoding='utf-8')
        
        mock_chunk_text.assert_called()
        
        embeddings_client.get_embeddings.assert_called()
        
        assert mock_save_chunk.call_count == 3

    @patch('task.embeddings.text_processor.psycopg2.connect')
    def test_save_chunk(self, mock_connect, setup_processor):
        embeddings_client, processor, db_config = setup_processor
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_connect.return_value.__enter__.return_value = mock_conn
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
        
        processor._save_chunk("test text", [0.1, 0.2, 0.3], "test_doc.txt")
        
        mock_cursor.execute.assert_called_once()
        mock_conn.commit.assert_called_once()


class TestAppComponents:
    def test_system_prompt_exists(self):
        assert isinstance(SYSTEM_PROMPT, str)
        assert len(SYSTEM_PROMPT) > 0
        assert "RAG" in SYSTEM_PROMPT
        assert "microwave" in SYSTEM_PROMPT.lower()

    def test_user_prompt_exists(self):
        assert isinstance(USER_PROMPT, str)
        assert len(USER_PROMPT) > 0
        assert "{context}" in USER_PROMPT
        assert "{question}" in USER_PROMPT


class TestChatCompletionClient:
    def test_initialization(self):
        client = DialChatCompletionClient('model-name', 'api-key')
        assert client._endpoint == 'https://ai-proxy.lab.epam.com/openai/deployments/model-name/chat/completions'
        assert client._api_key == 'api-key'

    def test_initialization_empty_api_key(self):
        with pytest.raises(ValueError) as context:
            DialChatCompletionClient('model-name', '')
        
        assert "API key cannot be null or empty" in str(context.value)

        with pytest.raises(ValueError) as context:
            DialChatCompletionClient('model-name', ' ')
        
        assert "API key cannot be null or empty" in str(context.value)

    @patch('requests.post')
    def test_get_completion_success(self, mock_post):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Test response"}}]
        }
        mock_post.return_value = mock_response

        client = DialChatCompletionClient('model-name', 'test-key')
        messages = [Message(Role.USER, "Test message")]
        result = client.get_completion(messages)

        assert result.role == Role.AI
        assert result.content == "Test response"
        
        mock_post.assert_called_once()
        args, kwargs = mock_post.call_args
        assert kwargs['headers']["api-key"] == 'test-key'

    @patch('requests.post')
    def test_get_completion_failure(self, mock_post):
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"
        mock_post.return_value = mock_response

        client = DialChatCompletionClient('model-name', 'test-key')
        messages = [Message(Role.USER, "Test message")]

        with pytest.raises(Exception) as context:
            client.get_completion(messages)
        
        assert "HTTP 500" in str(context.value)


class TestConversationModels:
    def test_role_enum_values(self):
        assert Role.USER == "user"
        assert Role.AI == "assistant"
        assert Role.SYSTEM == "system"

    def test_message_creation(self):
        message = Message(Role.USER, "Test content")
        assert message.role == Role.USER
        assert message.content == "Test content"

    def test_message_to_dict(self):
        message = Message(Role.USER, "Test content")
        message_dict = message.to_dict()
        assert message_dict == {"role": "user", "content": "Test content"}

    def test_conversation_creation(self):
        conversation = Conversation()
        assert len(conversation.messages) == 0

    def test_conversation_add_message(self):
        conversation = Conversation()
        message = Message(Role.USER, "Test message")
        conversation.add_message(message)
        
        assert len(conversation.messages) == 1
        assert conversation.messages[0] == message


class TestUtils:
    def test_chunk_text_basic(self):
        text = "Hello World Programming"
        chunks = chunk_text(text, chunk_size=8, overlap=3)
        
        assert len(chunks) >= 1
        
        for i, chunk in enumerate(chunks[:-1]):
            assert len(chunk) <= 8
        
        assert isinstance(chunks, list)
        assert all(isinstance(chunk, str) for chunk in chunks)

    def test_chunk_text_edge_cases(self):
        assert chunk_text("", 10, 2) == []
        
        result = chunk_text("Short", 10, 2)
        assert result == ["Short"]
        
        result = chunk_text("Exactly10", 9, 2)
        assert result == ["Exactly10"]


if __name__ == '__main__':
    pytest.main([__file__])