from enum import StrEnum

import psycopg2
from psycopg2.extras import RealDictCursor

from task.embeddings.embeddings_client import DialEmbeddingsClient
from task.utils.text import chunk_text


class SearchMode(StrEnum):
    EUCLIDIAN_DISTANCE = "euclidean"  # Euclidean distance (<->)
    COSINE_DISTANCE = "cosine"  # Cosine distance (<=>)


class TextProcessor:
    """Processor for text documents that handles chunking, embedding, storing, and retrieval"""

    def __init__(self, embeddings_client: DialEmbeddingsClient, db_config: dict):
        self.embeddings_client = embeddings_client
        self.db_config = db_config

    def _get_connection(self):
        """Get database connection"""
        return psycopg2.connect(
            host=self.db_config['host'],
            port=self.db_config['port'],
            database=self.db_config['database'],
            user=self.db_config['user'],
            password=self.db_config['password']
        )

    def _truncate_table(self):
        """Truncate the vectors table to remove all existing entries"""
        with self._get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute("TRUNCATE TABLE vectors RESTART IDENTITY;")
                conn.commit()

    def process_text_file(self, file_path: str, chunk_size: int = 150, overlap: int = 40, dimensions: int = 1536, truncate_table: bool = True):
        """Process a text file by chunking, embedding, and storing in the database"""
        if truncate_table:
            self._truncate_table()
        
        # Load content from file
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        
        # Generate chunks using the utility function
        chunks = chunk_text(content, chunk_size, overlap)
        
        # Generate embeddings for all chunks at once
        chunk_embeddings = self.embeddings_client.get_embeddings(chunks, dimensions)
        
        # Save each chunk with its embedding to the database
        for i, chunk in enumerate(chunks):
            embedding = chunk_embeddings[i]
            self._save_chunk(chunk, embedding, file_path)
    
    def _save_chunk(self, text: str, embedding: list[float], document_name: str):
        """Save a text chunk with its embedding to the database"""
        with self._get_connection() as conn:
            with conn.cursor() as cursor:
                # Convert embedding list to string representation for PostgreSQL
                embedding_str = "{" + ",".join(map(str, embedding)) + "}"
                
                cursor.execute("""
                    INSERT INTO vectors (document_name, text, embedding)
                    VALUES (%s, %s, %s::vector)
                """, (document_name, text, embedding_str))
                conn.commit()




    def search(self, user_request: str, search_mode: SearchMode = SearchMode.COSINE_DISTANCE, top_k: int = 5, min_score: float = 0.5, dimensions: int = 1536) -> list[str]:
        """Search for relevant text chunks based on user request using vector similarity"""
        # Generate embedding for user request
        request_embedding = self.embeddings_client.get_embeddings([user_request], dimensions)
        embedding_vector = request_embedding[0]  # Get the first (and only) embedding
        
        # Convert embedding list to string representation for PostgreSQL
        embedding_str = "{" + ",".join(map(str, embedding_vector)) + "}"
        
        # Determine the distance operator based on search mode
        if search_mode == SearchMode.EUCLIDIAN_DISTANCE:
            operator = "<->"
        elif search_mode == SearchMode.COSINE_DISTANCE:
            operator = "<=>"
        else:
            raise ValueError(f"Unsupported search mode: {search_mode}")
        
        with self._get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                # Query to find similar vectors based on the selected distance metric
                query = f"""
                    SELECT text, embedding {operator} %s::vector AS distance
                    FROM vectors
                    WHERE embedding {operator} %s::vector <= %s
                    ORDER BY distance
                    LIMIT %s;
                """
                cursor.execute(query, (embedding_str, embedding_str, min_score, top_k))
                results = cursor.fetchall()
                
        # Extract text from results
        return [row['text'] for row in results]

