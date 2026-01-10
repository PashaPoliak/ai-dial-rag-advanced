import logging
from unittest.mock import Mock, patch
import sys
import os
import pytest

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from task.app import SYSTEM_PROMPT, USER_PROMPT, main
from task.models.conversation import Conversation
from task.models.message import Message
from task.models.role import Role


class TestAppFunctionality:
    def test_system_prompt_exists_and_formatted_correctly(self):
        assert isinstance(SYSTEM_PROMPT, str)
        assert len(SYSTEM_PROMPT) > 0
        assert "RAG" in SYSTEM_PROMPT
        assert "microwave" in SYSTEM_PROMPT.lower()
        assert "manual" in SYSTEM_PROMPT.lower()

    def test_user_prompt_exists_and_formatted_correctly(self):
        assert isinstance(USER_PROMPT, str)
        assert len(USER_PROMPT) > 0
        assert "{context}" in USER_PROMPT
        assert "{question}" in USER_PROMPT
        formatted = USER_PROMPT.format(context="test context", question="test question")
        assert "test context" in formatted
        assert "test question" in formatted

class TestAppIntegration:
    def test_conversation_flow_simulation(self):
        conversation = Conversation()
        
        conversation.add_message(Message(Role.SYSTEM, SYSTEM_PROMPT))
        assert len(conversation.messages) == 1
        assert conversation.messages[0].role == Role.SYSTEM
        
        context = "Sample context from manual"
        question = "What is the power rating?"
        user_message_content = USER_PROMPT.format(context=context, question=question)
        conversation.add_message(Message(Role.USER, user_message_content))
        
        assert len(conversation.messages) == 2
        assert conversation.messages[1].role == Role.USER
        assert context in conversation.messages[1].content
        assert question in conversation.messages[1].content
        
        ai_response = Message(Role.AI, "The power rating is 1000 watts.")
        conversation.add_message(ai_response)
        
        assert len(conversation.messages) == 3
        assert conversation.messages[2].role == Role.AI


if __name__ == '__main__':
    pytest.main([__file__])