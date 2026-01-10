import sys
import os
import pytest
from unittest.mock import Mock, patch, MagicMock

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from task.app import main, SYSTEM_PROMPT, USER_PROMPT
from task.chat.chat_completion_client import DialChatCompletionClient
from task.embeddings.text_processor import TextProcessor, SearchMode


class TestAppMainBlock:
    def test_main_block_execution(self):
        import task.app
        assert hasattr(task.app, 'main')
        assert callable(getattr(task.app, 'main'))


class TestTextProcessorAdditionalMethods:
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
    def test_truncate_table_method(self, mock_connect, setup_processor):
        embeddings_client, processor = setup_processor
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_connect.return_value.__enter__.return_value = mock_conn
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
        
        processor._truncate_table()
        
        mock_cursor.execute.assert_called_once_with("TRUNCATE TABLE vectors RESTART IDENTITY;")
        mock_conn.commit.assert_called_once()

    def test_get_connection_method(self, setup_processor):
        embeddings_client, processor = setup_processor
        assert hasattr(processor, '_get_connection')

    @patch('task.embeddings.text_processor.psycopg2.connect')
    def test_search_with_different_modes(self, mock_connect, setup_processor):
        embeddings_client, processor = setup_processor
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_connect.return_value.__enter__.return_value = mock_conn
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
        
        embeddings_client.get_embeddings.return_value = {0: [0.1, 0.2, 0.3]}
        
        mock_cursor.fetchall.return_value = [{'text': 'Test result'}]
        
        results = processor.search("test query", search_mode=SearchMode.COSINE_DISTANCE, top_k=1)
        assert results == ['Test result']
        
        results = processor.search("test query", search_mode=SearchMode.EUCLIDIAN_DISTANCE, top_k=1)
        assert results == ['Test result']
        
        assert mock_cursor.execute.call_count == 2

    def test_search_mode_enum_values(self):
        assert SearchMode.EUCLIDIAN_DISTANCE == "euclidean"
        assert SearchMode.COSINE_DISTANCE == "cosine"


class TestChatCompletionClientEdgeCases:
    def test_initialization_empty_api_key_variations(self):
        with pytest.raises(ValueError) as context:
            DialChatCompletionClient('model-name', ' ')
        
        assert "API key cannot be null or empty" in str(context.value)
        
        with pytest.raises(ValueError) as context:
            DialChatCompletionClient('model-name', '\t\n')
        
        assert "API key cannot be null or empty" in str(context.value)


if __name__ == '__main__':
    pytest.main([__file__])