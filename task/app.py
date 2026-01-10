import logging

from task._constants import API_KEY
from task.chat.chat_completion_client import DialChatCompletionClient
from task.embeddings.embeddings_client import DialEmbeddingsClient
from task.embeddings.text_processor import TextProcessor, SearchMode
from task.models.conversation import Conversation
from task.models.message import Message
from task.models.role import Role

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Create system prompt with info that it is RAG powered assistant.
# Explain user message structure (firstly will be provided RAG context and the user question).
# Provide instructions that LLM should use RAG Context when answer on User Question, will restrict LLM to answer
# questions that are not related microwave usage, not related to context or out of history scope
SYSTEM_PROMPT = """
You are a RAG-powered assistant specialized in providing information about microwave ovens based on the provided manual context. Your responses must be grounded in the provided context and you should only answer questions related to microwave usage, safety, maintenance, and specifications as detailed in the manual. Do not provide information about topics unrelated to microwaves or outside the scope of the provided context. If a question cannot be answered based on the context, politely state that you don't have that information in the manual.
"""

# Provide structured system prompt, with RAG Context and User Question sections.
USER_PROMPT = """
### RAG CONTEXT:
{context}

### USER QUESTION:
{question}

Please provide an accurate answer based on the provided context.
"""


def main():
    # Create embeddings client with 'text-embedding-3-small-1' model
    embeddings_client = DialEmbeddingsClient('text-embedding-3-small-1', API_KEY)
    
    # Create chat completion client
    chat_client = DialChatCompletionClient('gpt-4o-1', API_KEY)
    
    # Create text processor, DB config: {'host': 'localhost','port': 5433,'database': 'vectordb','user': 'postgres','password': 'postgres'}
    db_config = {
        'host': 'localhost',
        'port': 5433,
        'database': 'vectordb',
        'user': 'postgres',
        'password': 'postgres'
    }
    text_processor = TextProcessor(embeddings_client, db_config)
    
    # Process the microwave manual document
    logger.info("Processing microwave manual...")
    text_processor.process_text_file('./task/embeddings/microwave_manual.txt', chunk_size=300, overlap=40, dimensions=1536, truncate_table=True)
    logger.info("Document processed and stored in vector database.")
    
    # Initialize conversation
    conversation = Conversation()
    conversation.add_message(Message(Role.SYSTEM, SYSTEM_PROMPT))
    
    logger.info("\nRAG-powered Microwave Assistant is ready!")
    logger.info("Ask questions about the microwave manual (type 'exit' to quit):\n")
    
    # Console chat loop
    while True:
        # Get user input from console
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            break

        # Retrieve context using the text processor
        retrieved_contexts = text_processor.search(user_input, top_k=5, min_score=0.5)
        context_str = "\n".join(retrieved_contexts)

        # Perform augmentation - format the user prompt with context and question
        augmented_prompt = USER_PROMPT.format(context=context_str, question=user_input)

        # Add user message to conversation
        conversation.add_message(Message(Role.USER, augmented_prompt))

        # Perform generation - get response from the chat client
        response = chat_client.get_completion(conversation.messages)

        # Add AI response to conversation
        conversation.add_message(response)

        # Print the AI response (keeping print for interactive console output)
        print(f"\nAssistant: {response.content}\n")


if __name__ == "__main__":
    main()



# TODO:
#  PAY ATTENTION THAT YOU NEED TO RUN Postgres DB ON THE 5433 WITH PGVECTOR EXTENSION!
#  RUN docker-compose.yml