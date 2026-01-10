import sys
import os
import pytest
from unittest.mock import Mock, patch, MagicMock

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from task.app import main, SYSTEM_PROMPT, USER_PROMPT
from task.chat.chat_completion_client import DialChatCompletionClient
from task.models.message import Message
from task.models.role import Role
from task.embeddings.text_processor import TextProcessor


class TestAppMainFunction:
    def test_prompts_exist_and_contain_expected_content(self):
        assert isinstance(SYSTEM_PROMPT, str)
        assert len(SYSTEM_PROMPT) > 0
        assert "RAG" in SYSTEM_PROMPT
        assert "microwave" in SYSTEM_PROMPT.lower()
        
        assert isinstance(USER_PROMPT, str)
        assert len(USER_PROMPT) > 0
        assert "{context}" in USER_PROMPT
        assert "{question}" in USER_PROMPT


class TestChatCompletionClientAdditional:
    def test_get_completion_with_print_request(self):
        client = DialChatCompletionClient('model-name', 'test-key')
        
        messages = [Message(Role.USER, "Test message")]
        
        with patch('builtins.print') as mock_print:
            with patch('requests.post') as mock_post:
                mock_response = MagicMock()
                mock_response.status_code = 200
                mock_response.json.return_value = {
                    "choices": [{"message": {"content": "Test response"}}]
                }
                mock_post.return_value = mock_response
                
                result = client.get_completion(messages, print_request=True)
                
                mock_print.assert_called()
                
                assert result.role == Role.AI
                assert result.content == "Test response"

    def test_get_completion_no_choices(self):
        client = DialChatCompletionClient('model-name', 'test-key')
        
        messages = [Message(Role.USER, "Test message")]
        
        with patch('requests.post') as mock_post:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "choices": []
            }
            mock_post.return_value = mock_response
            
            with pytest.raises(ValueError) as context:
                client.get_completion(messages)
            
            assert "No Choice has been present in the response" in str(context.value)

    @patch('task.chat.chat_completion_client.requests.post')
    def test_get_messages_str(self, mock_post):
        client = DialChatCompletionClient('model-name', 'test-key')
        
        messages = [
            Message(Role.USER, "First message"),
            Message(Role.AI, "Second message")
        ]
        
        result = client._get_messages_str(messages)
        
        assert "Role: USER" in result
        assert "First message" in result
        assert "Role: ASSISTANT" in result
        assert "Second message" in result


class TestTextProcessorAdditional:
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
        return embeddings_client, processor

    @patch('task.embeddings.text_processor.psycopg2.connect')
    def test_search_method(self, mock_connect, setup_processor):
        embeddings_client, processor = setup_processor
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_connect.return_value.__enter__.return_value = mock_conn
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
        
        embeddings_client.get_embeddings.return_value = {0: [0.1, 0.2, 0.3]}
        
        mock_cursor.fetchall.return_value = [{'text': 'Test result 1'}, {'text': 'Test result 2'}]
        
        results = processor.search("test query", top_k=2)
        
        embeddings_client.get_embeddings.assert_called_once()
        
        mock_cursor.execute.assert_called_once()
        
        assert results == ['Test result 1', 'Test result 2']


if __name__ == '__main__':
    pytest.main([__file__])