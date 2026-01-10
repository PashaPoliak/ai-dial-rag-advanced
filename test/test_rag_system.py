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
from task.app import SYSTEM_PROMPT, USER_PROMPT, main


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


class TestAppComponents:
    def test_system_prompt_exists(self):
        assert isinstance(SYSTEM_PROMPT, str)
        assert len(SYSTEM_PROMPT) > 0
        assert "RAG" in SYSTEM_PROMPT

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


if __name__ == '__main__':
    pytest.main([__file__])