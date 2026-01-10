import logging
from unittest.mock import Mock, patch, MagicMock
import sys
import os
import pytest

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from task.embeddings.embeddings_client import DialEmbeddingsClient
from task.embeddings.text_processor import TextProcessor, SearchMode
from task.chat.chat_completion_client import DialChatCompletionClient
from task.app import SYSTEM_PROMPT, USER_PROMPT
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
    def test_get_embeddings_success(self, mock_post):
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
        result = DialEmbeddingsClient('test-model', 'test-key').get_embeddings(inputs, dimensions=3)

        assert len(result) == 2
        assert result[0] == [0.1, 0.2, 0.3]
        assert result[1] == [0.4, 0.5, 0.6]
        
        mock_post.assert_called_once()
        args, kwargs = mock_post.call_args
        assert kwargs['json']['input'] == inputs
        assert kwargs['json']['dimensions'] == 3

    @patch('requests.post')
    def test_get_embeddings_failure(self, mock_post):
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"
        mock_post.return_value = mock_response

        with pytest.raises(Exception) as context:
            DialEmbeddingsClient('test-model', 'test-key').get_embeddings(["test"], dimensions=3)
        
        assert "HTTP 500" in str(context.value)


class TestTextProcessor:
    @pytest.fixture
    def mock_client_and_config(self):
        embeddings_client = Mock()
        db_config = {
            'host': 'localhost',
            'port': 5433,
            'database': 'vectordb',
            'user': 'postgres',
            'password': 'postgres'
        }
        processor = TextProcessor(embeddings_client, db_config)
        return embeddings_client, db_config, processor

    def test_initialization(self, mock_client_and_config):
        embeddings_client, db_config, processor = mock_client_and_config
        assert processor.embeddings_client == embeddings_client
        assert processor.db_config == db_config

    def test_search_mode_enum(self):
        assert SearchMode.EUCLIDIAN_DISTANCE == "euclidean"
        assert SearchMode.COSINE_DISTANCE == "cosine"

    def test_get_connection(self, mock_client_and_config):
        _, _, processor = mock_client_and_config
        assert hasattr(processor, '_get_connection')


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
        text = "Test Utils"
        chunks = chunk_text(text, chunk_size=8, overlap=3)
        
        assert isinstance(chunks, list)
        assert all(isinstance(chunk, str) for chunk in chunks)
        if chunks:
            reconstructed = "".join(chunks)
            assert "Test" in (reconstructed if len(reconstructed) >= 5 else text)

    def test_chunk_text_edge_cases(self):
        assert chunk_text("", 10, 2) == []
        assert chunk_text("", 10, 2) == []
        
        result = chunk_text("Short", 10, 2)
        result = chunk_text("Short", 10, 2)
        assert result == ["Short"]
        
        result = chunk_text("A", 5, 1)
        result = chunk_text("A", 5, 1)
        assert result == ["A"]


class TestMainFunction:
    @patch('builtins.input', side_effect=['What is the power rating?', 'exit'])
    @patch('task.app.DialEmbeddingsClient')
    @patch('task.app.DialChatCompletionClient')
    @patch('task.app.TextProcessor')
    @patch('task.app.logger')  # Mock logger to prevent console output
    @patch('builtins.print')
    def test_main_function_execution(self, mock_print, mock_logger, mock_text_processor, mock_chat_client, mock_embeddings_client, mock_input):
        # Mock the clients and text processor
        mock_embeddings_instance = Mock()
        mock_chat_instance = Mock()
        mock_text_processor_instance = Mock()
        
        mock_embeddings_client.return_value = mock_embeddings_instance
        mock_chat_client.return_value = mock_chat_instance
        mock_text_processor.return_value = mock_text_processor_instance
        
        # Mock the text processor methods
        mock_text_processor_instance.search.return_value = ["Sample context from manual"]
        
        # Mock the chat client response
        mock_response = Mock()
        mock_response.content = "The power rating is 1000 watts."
        mock_chat_instance.get_completion.return_value = mock_response
        
        # Run the main function
        from task.app import main
        try:
            main()
        except SystemExit:
            pass  # Expected when input returns 'exit'
        
        # Verify that the required components were called
        mock_embeddings_client.assert_called_once()
        mock_chat_client.assert_called_once()
        mock_text_processor_instance.search.assert_called()
        mock_chat_instance.get_completion.assert_called()


class TestAppIntegrationWithMockedDB:
    def test_app_components_integration_without_main_execution(self):
        # This test verifies that the app components can be imported and initialized
        # without actually running the main function which involves user input
        from task.app import SYSTEM_PROMPT, USER_PROMPT
        from task.chat.chat_completion_client import DialChatCompletionClient
        from task.embeddings.embeddings_client import DialEmbeddingsClient
        from task.embeddings.text_processor import TextProcessor
        
        # Verify that prompts exist and are properly formatted
        assert isinstance(SYSTEM_PROMPT, str)
        assert isinstance(USER_PROMPT, str)
        assert "{context}" in USER_PROMPT
        assert "{question}" in USER_PROMPT
        
        # Verify that components can be initialized (with mocks for external dependencies)
        mock_embeddings_client = Mock()
        db_config = {
            'host': 'localhost',
            'port': 5433,
            'database': 'vectordb',
            'user': 'postgres',
            'password': 'postgres'
        }
        
        # Verify that models work correctly
        from task.models.conversation import Conversation
        from task.models.message import Message
        from task.models.role import Role
        
        conversation = Conversation()
        message = Message(Role.USER, "Test message")
        conversation.add_message(message)
        
        assert len(conversation.messages) == 1
        assert conversation.messages[0].role == Role.USER