[![PyPI version](https://badge.fury.io/py/langchain-llm7.svg)](https://badge.fury.io/py/langchain-llm7)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Downloads](https://static.pepy.tech/badge/langchain-llm7)](https://pepy.tech/project/langchain-llm7)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-blue)](https://www.linkedin.com/in/eugene-evstafev-716669181/)

# LangChain LLM7 Integration

Official LangChain compatibility layer for LLM7 API services.

## Installation

```bash
pip install langchain-llm7
```

## Features

- 🚀 Native integration with LangChain's BaseChatModel interface
- ⚡ Support for both streaming and non-streaming responses
- 🔧 Customizable model parameters (temperature, max_tokens, stop sequences)
- 📊 Token usage metadata tracking
- 🛠 Robust error handling and retry mechanisms

## Usage

### Basic Implementation

```python
from langchain_llm7 import ChatLLM7
from langchain_core.messages import HumanMessage

# Initialize with default parameters
llm = ChatLLM7()

# Basic invocation
response = llm.invoke([HumanMessage(content="Hello!")])
print(response.content)
```

### Streaming Responses

```python
# Enable streaming
llm = ChatLLM7(streaming=True)

for chunk in llm.stream([HumanMessage(content="Tell me about quantum computing")]):
    print(chunk.content, end="", flush=True)
```

### Advanced Configuration

```python
# Custom model configuration
llm = ChatLLM7(
    model="llama-3.3-70b-instruct-fp8-fast",
    temperature=0.7,
    max_tokens=500,
    stop=["\n", "Observation:"],
    timeout=45
)
```

## Parameters

| Parameter     | Description                                  | Default                      |
|---------------|----------------------------------------------|------------------------------|
| `model`       | Model version to use                         | "gpt-4o-mini-2024-07-18"     |
| `base_url`    | API endpoint URL                             | "https://api.llm7.io/v1"      |
| `temperature` | Sampling temperature (0.0-2.0)              | 1.0                          |
| `max_tokens`  | Maximum number of tokens to generate         | None                         |
| `timeout`     | Request timeout in seconds                  | 120                          |
| `stop`        | Stop sequences for response generation      | None                         |
| `streaming`   | Enable streaming response mode              | False                        |

## Error Handling

The library provides detailed error messages for:
- API communication failures
- Invalid message formats
- Unsupported message types
- Response parsing errors

```python
try:
    llm.invoke([{"invalid": "message"}])
except ValueError as e:
    print(f"Error: {e}")
```

## Testing

To run the test suite:

```bash
pip install pytest
pytest tests/
```

## Documentation

For complete documentation see:
- [LangChain Core Documentation](https://python.langchain.com)
- [LLM7 API Reference](https://api.llm7.io/)

## Contributing

Contributions are welcome! Please open an issue or submit a PR:
- [GitHub Repository](https://github.com/chigwell/langchain_llm7)

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Contact

Eugene Evstafev  

[LinkedIn Profile](https://www.linkedin.com/in/eugene-evstafev-716669181/)
