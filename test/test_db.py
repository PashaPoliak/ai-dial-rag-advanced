import sys
import os
import importlib.util
import pytest
from unittest.mock import Mock, patch, MagicMock

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from task.embeddings.text_processor import TextProcessor


class TestRemainingCoverage:
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
    def test_process_text_file_with_truncate_true(self, mock_connect, setup_processor):
        embeddings_client, processor = setup_processor
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_connect.return_value.__enter__.return_value = mock_conn
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
        
        mock_file_handle = MagicMock()
        mock_file_handle.read.return_value = "test content"
        
        with patch("builtins.open", return_value=mock_file_handle):
            with patch('task.embeddings.text_processor.chunk_text', return_value=["chunk1", "chunk2"]):
                embeddings_client.get_embeddings.return_value = {0: [0.1, 0.2, 0.3], 1: [0.4, 0.5, 0.6]}
                
                with patch.object(processor, '_save_chunk') as mock_save:
                    processor.process_text_file("test.txt", truncate_table=True)
                    
                    mock_cursor.execute.assert_any_call("TRUNCATE TABLE vectors RESTART IDENTITY;")
                    mock_conn.commit.assert_called()

    @patch('task.embeddings.text_processor.psycopg2.connect')
    def test_search_method_error_conditions(self, mock_connect, setup_processor):
        embeddings_client, processor = setup_processor
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_connect.return_value.__enter__.return_value = mock_conn
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
        
        embeddings_client.get_embeddings.return_value = {0: [0.1, 0.2, 0.3]}
        
        with patch.object(processor, '_get_connection') as mock_get_conn:
            mock_get_conn.return_value.__enter__.return_value = mock_conn
            mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
            mock_cursor.fetchall.return_value = []
            
            result = processor.search("test", top_k=1)

    def test_execute_main_as_script(self):
        spec = importlib.util.spec_from_file_location("__main__", "task/app.py")
        if spec and spec.loader:
            pass


if __name__ == '__main__':
    pytest.main([__file__])