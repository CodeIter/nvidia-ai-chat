# NVIDIA AI Chat

A command-line interface for interacting with NVIDIA's AI chat models.

## Description

This tool provides a convenient way to chat with NVIDIA's AI models from your terminal. It supports:
- Interactive chat sessions
- Non-interactive mode for single prompts
- Conversation history management
- Customizable model parameters (temperature, top_p, etc.)
- Streaming responses

## Installation

To build the application, you need to have Go installed.

```bash
make build
```

This will create an executable named `nvidia-ai-chat` (or `nvidia-ai-chat.exe` on Windows) in the project directory.

## Usage

### Authentication

The tool requires an NVIDIA AI access token. You can provide it in one of two ways:
1. Set the `NVIDIA_BUILD_AI_ACCESS_TOKEN` environment variable:
   ```bash
   export NVIDIA_BUILD_AI_ACCESS_TOKEN="your_token_here"
   ```
2. Use the `-k` or `--access_token` flag:
   ```bash
   ./nvidia-ai-chat -k "your_token_here"
   ```

### Interactive Mode

To start an interactive chat session, simply run the application:

```bash
./nvidia-ai-chat
```

This will create a new conversation file in `~/.cache/nvidia-chat/`. You can also specify a conversation file to resume a previous chat:

```bash
./nvidia-ai-chat /path/to/conversation.json
```

In interactive mode, you can use the following commands:
- `/exit`, `/quit`: Exit the program.
- `/history`: Print the full conversation JSON.
- `/clear`: Clear the conversation messages.
- `/save <file>`: Save the conversation to a new file.
- `/persist-system <file>`: Persist a system prompt from a file.
- `/model <model_name>`: Switch the model for the current session.
- `/temperature <0..1>`: Set the temperature for the current session.
- `/help`: Show the help message.

### Non-Interactive Mode

To get a response for a single prompt, use the `--prompt` flag:

```bash
./nvidia-ai-chat --prompt="Hello, world!"
```

You can also pipe the prompt from stdin:

```bash
echo "Hello, world!" | ./nvidia-ai-chat --prompt=-
```

### Options

For a full list of options, run:

```bash
./nvidia-ai-chat --help
```

## License

This project is licensed under the MIT License — see the [LICENSE](./LICENSE) file for the full text and copyright information.
