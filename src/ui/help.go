// Path: src/ui/help.go
// Rationale: This file contains the printHelp function, which displays the command-line help message.

package main

import "fmt"

func printHelp(cfg map[string]string) {
	fmt.Printf(`%snvidia-chat (go)%s
Usage: nvidia-chat [OPTIONS] CONVERSATION_FILE

If CONVERSATION_FILE is omitted, one will be created at:
  %s/conversation-<timestamp>.json
and its path will be printed.

Options:
  -m MODEL                model id (default: %s)
  -T TEMPERATURE          sampling temperature 0..1 (default: %s)
  -P TOP_P                top_p 0.01..1 (default: %s)
  -f FREQUENCY_PENALTY    -2..2 (default: %s)
  -r PRESENCE_PENALTY     -2..2 (default: %s)
  -M MAX_TOKENS           1..4096 (default: %s)
  -L LIMIT                max messages allowed in conversation file (default: %d)
  -s SYS_PROMPT_FILE      path to system prompt text file (content used for this run)
  -S                      persist -s into the conversation file's top-level "system" (replaces previous)
  --save-settings         persist current model settings into the conversation file's top-level "settings"
  --no-stream             disable streaming (equivalent to --stream=false)
  --stream=true|false     explicitly enable/disable streaming
  --reasoning EFFORT      reasoning effort: low | medium | high (default: %s)
  --stop STRING           stop string (empty = omitted)
  --prompt=text|file|-    non-interactive mode: print AI response for given prompt and exit
  -k ACCESS_TOKEN         provide API key (overrides env)
  -l, --list              list supported models (built-in subset) and exit
  -h, --help              show this help

Interactive commands (enter on a new line):
  /exit, /quit            exit the program
  /history                print full conversation JSON
  /clear                  clear conversation messages
  /save <file>            save conversation to a new file
  /persist-system         prompts for a file path to persist as the system prompt
  /model <model_name>     switch model for this session
  /temperature <0..1>     set temperature for this session
  /top_p <0.01..1>        set top_p for this session
  /max_tokens <1..4096>   set max_tokens for this session
  /stop <string>          set stop string for this session
  /persist-settings       save the current session's settings to the conversation file

Parameter explanations:
  Reasoning Effort : Controls the effort level for reasoning in reasoning-capable models.
    'low' provides basic reasoning, 'medium' provides balanced reasoning, and
    'high' provides detailed step-by-step reasoning.

  Stop string: A string (or list of strings) where the API will stop generating further tokens.
    The returned text will not contain the stop sequence. If empty, the 'stop' field is not sent.

  Max Tokens: The maximum number of tokens to generate in any given call. Generation will stop
    once this token count is reached.

  Presence Penalty: Positive values penalize new tokens based on whether they appear in the text so far,
    increasing model likelihood to talk about new topics.

  Frequency Penalty: How much to penalize new tokens based on their existing frequency in the text so far,
    decreasing model likelihood to repeat the same line verbatim.

  Top P: The top-p sampling mass used for text generation. Not recommended to modify both temperature and top_p.

  Temperature: Sampling temperature for generation. Higher -> less deterministic output.

Important safety message shown at conversation start:
  AI models generate responses and outputs based on complex algorithms and machine learning techniques,
  and those responses or outputs may be inaccurate, harmful, biased or indecent. By testing this model,
  you assume the risk of any harm caused by any response or output of the model. Please do not upload
  any confidential information or personal data unless expressly permitted. Your use is logged for
  security purposes.

For full models list and details: https://build.nvidia.com/
`, bold, normal, cfg["HISTORY_DIR"], cfg["MODEL"], cfg["TEMPERATURE"], cfg["TOP_P"], cfg["FREQUENCY_PENALTY"], cfg["PRESENCE_PENALTY"], cfg["MAX_TOKENS"], defaultHistoryLimit, cfg["REASONING_EFFORT"])
}
