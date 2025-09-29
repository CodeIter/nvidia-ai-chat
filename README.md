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

1.  **Environment Variable**: The tool checks for the following environment variables in order: `NVIDIA_BUILD_AI_ACCESS_TOKEN`, `NVIDIA_ACCESS_TOKEN`, `ACCESS_TOKEN`, `NVIDIA_API_KEY`, `API_KEY`.
    ```bash
    export NVIDIA_BUILD_AI_ACCESS_TOKEN="your_token_here"
    ```
2.  **Command-Line Flag**: Use the `-k` or `--access-token` flag to provide the token directly. This overrides any environment variables.
    ```bash
    ./nvidia-ai-chat -k "your_token_here"
    ```

### Conversation Management

By default, `nvidia-ai-chat` stores your conversations in `~/.cache/nvidia-chat/`.

-   **Starting a New Chat**: If you run the tool without specifying a file, it creates a new timestamped conversation file (e.g., `conversation-20231027-123456.json`) and prints its path.
-   **Resuming a Chat**: To continue a previous conversation, pass the path to the conversation file as an argument:
    ```bash
    ./nvidia-ai-chat /path/to/your/conversation.json
    ```

### Interactive Mode

To start an interactive chat session, run the application, optionally specifying a conversation file:

```bash
./nvidia-ai-chat
```

This will create a new conversation file in `~/.cache/nvidia-chat/`. You can also specify a conversation file to resume a previous chat:

```bash
./nvidia-ai-chat /path/to/conversation.json
```

In interactive mode, you can use the following commands:
- `/help`: Show the help message.
- `/exit`, `/quit`: Exit the program.
- `/history`: Print the full conversation JSON.
- `/clear`: Clear the conversation messages.
- `/save <file>`: Save the conversation to a new file.
- `/list`: List supported models.
- `/model <model_name>`: Switch model for the session.
- `/modelinfo [name]`: List settings for a model (defaults to current).
- `/askfor_model_setting`: Interactively set model parameters.
- `/persist-settings`: Save the current session's settings to the conversation file.
- `/persist-system <file>`: Persist a system prompt from a file.
- `/exportlast [-t] <file>`: Export last AI response to a markdown file (-t filters thinking).
- `/exportlastn [-t] <n> <file>`: Export last n AI responses.
- `/exportn [-t] <n> <file>`: Export the Nth-to-last AI response.
- `/randomodel`: Switch to a random supported model.

For any model setting, you can use `/<setting_name> <value>` or `/<setting_name> unset`.
For example: `/temperature 0.8`, `/stop unset`

### Non-Interactive Mode

To get a response for a single prompt without entering an interactive session, use the `--prompt` flag. The tool will print the AI's response to standard output and exit.

The `--prompt` flag can accept:
- A string of text directly:
  ```bash
  ./nvidia-ai-chat --prompt="Translate 'hello' to French"
  ```
- A path to a file containing the prompt:
  ```bash
  ./nvidia-ai-chat --prompt=./my_prompt.txt
  ```
- A hyphen (`-`) to read the prompt from standard input (stdin):
  ```bash
  echo "Summarize this article" | ./nvidia-ai-chat --prompt=-
  ```

You can combine this with other flags, such as specifying a model:
```bash
./nvidia-ai-chat --model="google/codegemma-7b" --prompt="Write a python function to check for prime numbers"
```

You can also use non-interactive mode with an existing conversation file to provide context to the model:
```bash
./nvidia-ai-chat --prompt="What was the last thing we talked about?" /path/to/conversation.json
```

### Options

For a full list of options, run `./nvidia-ai-chat --help`.

#### General Options

-   `-h, --help`: Show the help message and exit.
-   `-l, --list`: List supported models and exit.
-   `-m, --model NAME`: Specify the model ID to use (e.g., `mistralai/mistral-small-24b-instruct`).
-   `-k, --access-token KEY`: Provide your API key directly.
-   `--prompt TEXT|FILE|-`: Enable non-interactive mode and provide the prompt.
-   `-s, --sys-prompt-file PATH`: Path to a file containing a system prompt to use for the session.
-   `-S`: Persist the system prompt provided via `-s` to the conversation file.
-   `--save-settings`: Persist the current session's model settings to the conversation file.
-   `--modelinfo NAME`: Show detailed settings and capabilities for a specific model and exit.

#### Model Setting Options

These flags override the default settings for the current session. For model-specific details, ranges, and defaults, use the `/modelinfo <model_name>` command in interactive mode.

-   `--temperature <0..1>`: Set the sampling temperature.
-   `--top-p <0.01..1>`: Set the top-p sampling mass.
-   `--max-tokens <number>`: Set the maximum number of tokens to generate.
-   `--frequency-penalty <-2..2>`: Set the frequency penalty.
-   `--presence-penalty <-2..2>`: Set the presence penalty.
-   `--stop <string>`: Set a custom stop sequence.
-   `--stream <true|false>`: Enable or disable streaming responses.
-   `--history-limit <number>`: Set the maximum number of messages to keep in the conversation history.
-   `--reasoning-effort <low|medium|high>`: Control the reasoning effort for capable models.
-   ... and many more model-specific parameters. Use `/modelinfo` to discover them.

## License

This project is licensed under the MIT License — see the [LICENSE](./LICENSE) file for the full text and copyright information.
