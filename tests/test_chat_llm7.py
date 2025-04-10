import pytest
from unittest.mock import Mock, patch
from unittest.mock import MagicMock
from langchain_llm7 import ChatLLM7
import json
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)


class TestChatLLM7NonStreaming:
    @pytest.fixture
    def mock_response(self):
        mock_resp = Mock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "choices": [{
                "message": {
                    "content": "Test response",
                    "role": "assistant"
                }
            }],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 20,
                "total_tokens": 30
            }
        }
        return mock_resp

    def test_non_streaming_invoke(self, mock_response):
        with patch('requests.post', return_value=mock_response) as mock_post:
            model = ChatLLM7(model="gpt-4o-mini-2024-07-18")
            message = model.invoke([HumanMessage(content="Hello!")])

            assert message.content == "Test response"
            assert message.usage_metadata["input_tokens"] == 2
            assert message.usage_metadata["output_tokens"] == 2
            assert message.usage_metadata["total_tokens"] == 4


    def test_custom_parameters(self, mock_response):
        with patch('requests.post', return_value=mock_response):
            model = ChatLLM7(
                model="llama-3.3-70b-instruct-fp8-fast",
                temperature=0.7,
                max_tokens=100,
                stop=["\n"],
                timeout=30
            )

            messages = [
                SystemMessage(content="You are a helpful assistant"),
                HumanMessage(content="Hi there!")
            ]

            result = model.invoke(messages)

            assert model._identifying_params == {
                "model_name": "llama-3.3-70b-instruct-fp8-fast",
                "temperature": 0.7,
                "max_tokens": 100,
                "base_url": "https://api.llm7.io/v1"
            }


class TestChatLLM7Streaming:
    @pytest.fixture
    def mock_streaming_response(self):
        mock_resp = Mock()
        mock_resp.status_code = 200
        mock_resp.iter_lines.return_value = [
            b'data: {"choices": [{"delta": {"content": "Hello"}}]}',
            b'data: {"choices": [{"delta": {"content": " world"}}]}',
            b'data: [DONE]'
        ]
        return mock_resp

    def test_streaming_generation(self, mock_streaming_response):
        mock_post = MagicMock()
        mock_post.return_value.__enter__.return_value = mock_streaming_response

        with patch('requests.post', new=mock_post):
            model = ChatLLM7(streaming=True)
            messages = [HumanMessage(content="Say hello")]

            chunks = list(model.stream(messages))
            assert len(chunks) == 2
            assert chunks[0].content == "Hello"
            assert chunks[1].content == " world"

            # Verify streaming request format
            args, kwargs = mock_post.call_args
            assert kwargs['json']['stream'] is True


class TestErrorHandling:
    def test_api_error_handling(self):
        mock_resp = Mock()
        mock_resp.status_code = 500
        mock_resp.text = "Server error"

        with patch('requests.post', return_value=mock_resp), pytest.raises(ValueError) as excinfo:
            model = ChatLLM7()
            model.invoke([HumanMessage(content="Test")])

        assert "API request failed with status 500" in str(excinfo.value)

    def test_invalid_message_type(self):
        model = ChatLLM7()
        with pytest.raises(ValueError):
            model._format_messages([Mock(spec=BaseMessage)])


class TestMessageFormatting:
    def test_message_conversion(self):
        model = ChatLLM7()
        messages = [
            SystemMessage(content="System prompt"),
            HumanMessage(content="User message"),
            AIMessage(content="AI response")
        ]

        formatted = model._format_messages(messages)

        assert formatted == [
            {"role": "system", "content": "System prompt"},
            {"role": "user", "content": "User message"},
            {"role": "assistant", "content": "AI response"}
        ]


class TestModelProperties:
    def test_llm_type(self):
        model = ChatLLM7()
        assert model._llm_type == "llm7-chat"

    def test_identifying_params(self):
        model = ChatLLM7(
            model="llama-3.3-70b-instruct-fp8-fast",
            temperature=0.5,
            max_tokens=150
        )
        params = model._identifying_params
        assert params == {
            "model_name": "llama-3.3-70b-instruct-fp8-fast",
            "temperature": 0.5,
            "max_tokens": 150,
            "base_url": "https://api.llm7.io/v1"
        }