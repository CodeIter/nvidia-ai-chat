package main

import (
	"bufio"
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"io/ioutil"
	"math/rand"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
	"time"
)

var (
	// defaults (same as your zsh script)
	defaultBaseURL       = "https://integrate.api.nvidia.com/v1"
	defaultModel         = "openai/gpt-oss-120b"
	defaultTemperature   = "1"
	defaultTopP          = "1"
	defaultFrequency     = "0"
	defaultPresence      = "0"
	defaultMaxTokens     = "4096"
	defaultStream        = "true"
	defaultReasoning     = "medium"
	defaultStop          = ""
	defaultHistorySubdir = ".cache/nvidia-chat"
	defaultHistoryLimit  = 40
	modelsList           = []string{
		"openai/gpt-oss-120b",
		"gpt-oss-120b",
		"seed-oss-36b-instruct",
		"qwen3-coder-480b-a35b-instruct",
		"nvidia-nemotron-nano-9b-v2",
		"llama-3.3-nemotron-super-49b-v1.5",
		"mistral-nemotron",
		"mistral-small-24b-instruct",
		"deepseek-v3.1",
		"deepseek-r1-distill-qwen-32b",
		"deepseek-r1-distill-llama-8b",
		"deepseek-r1-0528",
		"qwen3-next-80b-a3b-instruct",
		"qwen3-next-80b-a3b-thinking",
		"kimi-k2-instruct-0905",
		"codegemma-7b",
		"gemma-7b",
		"mixtral-8x22b-instruct-v0.1",
	}
	apiEnvNames = []string{"NVIDIA_BUILD_AI_ACCESS_TOKEN", "NVIDIA_ACCESS_TOKEN", "ACCESS_TOKEN", "NVIDIA_API_KEY", "API_KEY"}
)

type Settings struct {
	Model            string  `json:"model"`
	Temperature      float64 `json:"temperature"`
	TopP             float64 `json:"top_p"`
	FrequencyPenalty float64 `json:"frequency_penalty"`
	PresencePenalty  float64 `json:"presence_penalty"`
	MaxTokens        int     `json:"max_tokens"`
	Stream           bool    `json:"stream"`
	ReasoningEffort  string  `json:"reasoning_effort"`
	Stop             string  `json:"stop,omitempty"`
	HistoryLimit     int     `json:"history_limit,omitempty"`
}

type Message struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type ConversationFile struct {
	System   string    `json:"system"`
	Settings Settings  `json:"settings"`
	Messages []Message `json:"messages"`
}

func tput(name string) string {
	out, err := exec.Command("tput", name).Output()
	if err != nil {
		return ""
	}
	return string(out)
}

var (
	bold   = tput("bold")
	normal = tput("sgr0")
	blue   = tput("setaf 4")
	green  = tput("setaf 2")
	red    = tput("setaf 1")
)

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
  /exportlast [-t] <file> export last AI response to a markdown file. (-t: filter thinking)
  /exportlastn [-t] <n> <file> export last n AI responses to a markdown file. (-t: filter thinking)
  /exportn [-t] <n> <file> export the Nth-to-last AI response to a markdown file. (-t: filter thinking)

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

// helpers
func mustAtoi(s string, def int) int {
	if v, err := strconv.Atoi(s); err == nil {
		return v
	}
	return def
}
func mustParseFloat(s string, def float64) float64 {
	if v, err := strconv.ParseFloat(s, 64); err == nil {
		return v
	}
	return def
}

func ensureHistoryFileStructure(path string, cfg map[string]string) error {
	// if file doesn't exist, create it with defaults
	if _, err := os.Stat(path); os.IsNotExist(err) {
		dir := filepath.Dir(path)
		if err := os.MkdirAll(dir, 0o755); err != nil {
			return err
		}
		// build default file
		stream := cfg["STREAM"] == "true"
		s := Settings{
			Model:            cfg["MODEL"],
			Temperature:      mustParseFloat(cfg["TEMPERATURE"], 1.0),
			TopP:             mustParseFloat(cfg["TOP_P"], 1.0),
			FrequencyPenalty: mustParseFloat(cfg["FREQUENCY_PENALTY"], 0),
			PresencePenalty:  mustParseFloat(cfg["PRESENCE_PENALTY"], 0),
			MaxTokens:        mustAtoi(cfg["MAX_TOKENS"], 4096),
			Stream:           stream,
			ReasoningEffort:  cfg["REASONING_EFFORT"],
		}
		cf := ConversationFile{
			System:   "",
			Settings: s,
			Messages: []Message{},
		}
		b, _ := json.MarshalIndent(cf, "", "  ")
		return ioutil.WriteFile(path, b, 0o644)
	}
	// file exists: verify shape; if not, back up and recreate
	data, err := ioutil.ReadFile(path)
	if err != nil {
		return err
	}
	var tmp interface{}
	if err := json.Unmarshal(data, &tmp); err != nil {
		// back up and recreate
		backup := path + ".bak." + strconv.FormatInt(time.Now().Unix(), 10)
		_ = os.Rename(path, backup)
		return ensureHistoryFileStructure(path, cfg)
	}
	// ensure object with messages array
	obj, ok := tmp.(map[string]interface{})
	if !ok {
		backup := path + ".bak." + strconv.FormatInt(time.Now().Unix(), 10)
		_ = os.Rename(path, backup)
		return ensureHistoryFileStructure(path, cfg)
	}
	if msgs, ok := obj["messages"]; !ok {
		backup := path + ".bak." + strconv.FormatInt(time.Now().Unix(), 10)
		_ = os.Rename(path, backup)
		return ensureHistoryFileStructure(path, cfg)
	} else {
		if _, ok := msgs.([]interface{}); !ok {
			backup := path + ".bak." + strconv.FormatInt(time.Now().Unix(), 10)
			_ = os.Rename(path, backup)
			return ensureHistoryFileStructure(path, cfg)
		}
	}
	return nil
}

func readConversation(path string) (*ConversationFile, error) {
	data, err := ioutil.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var cf ConversationFile
	if err := json.Unmarshal(data, &cf); err != nil {
		return nil, err
	}
	return &cf, nil
}

func writeConversation(path string, cf *ConversationFile) error {
	b, err := json.MarshalIndent(cf, "", "  ")
	if err != nil {
		return err
	}
	tmp := path + ".tmp"
	if err := ioutil.WriteFile(tmp, b, 0o644); err != nil {
		return err
	}
	return os.Rename(tmp, path)
}

func appendMessage(path, role, content string) error {
	cf, err := readConversation(path)
	if err != nil {
		return err
	}
	cf.Messages = append(cf.Messages, Message{Role: role, Content: content})
	return writeConversation(path, cf)
}

func messageCount(path string) (int, error) {
	cf, err := readConversation(path)
	if err != nil {
		return 0, err
	}
	return len(cf.Messages), nil
}

func persistSystemToFile(path, content string) error {
	cf, err := readConversation(path)
	if err != nil {
		return err
	}
	cf.System = content
	return writeConversation(path, cf)
}

func persistSettingsToFile(path string, cfg map[string]string) error {
	cf, err := readConversation(path)
	if err != nil {
		return err
	}
	stream := cfg["STREAM"] == "true"
	s := Settings{
		Model:            cfg["MODEL"],
		Temperature:      mustParseFloat(cfg["TEMPERATURE"], 1.0),
		TopP:             mustParseFloat(cfg["TOP_P"], 1.0),
		FrequencyPenalty: mustParseFloat(cfg["FREQUENCY_PENALTY"], 0),
		PresencePenalty:  mustParseFloat(cfg["PRESENCE_PENALTY"], 0),
		MaxTokens:        mustAtoi(cfg["MAX_TOKENS"], 4096),
		Stream:           stream,
		ReasoningEffort:  cfg["REASONING_EFFORT"],
		HistoryLimit:     mustAtoi(cfg["HISTORY_LIMIT"], defaultHistoryLimit),
	}
	if cfg["STOP"] != "" {
		s.Stop = cfg["STOP"]
	}
	cf.Settings = s
	return writeConversation(path, cf)
}

func applyFileSettingsAsDefaults(path string, cfg map[string]string, provided map[string]bool) error {
	cf, err := readConversation(path)
	if err != nil {
		return err
	}
	// apply if not provided
	if !provided["MODEL"] && cf.Settings.Model != "" {
		cfg["MODEL"] = cf.Settings.Model
	}
	if !provided["TEMPERATURE"] && cf.Settings.Temperature != 0 {
		cfg["TEMPERATURE"] = fmt.Sprintf("%g", cf.Settings.Temperature)
	}
	if !provided["TOP_P"] && cf.Settings.TopP != 0 {
		cfg["TOP_P"] = fmt.Sprintf("%g", cf.Settings.TopP)
	}
	if !provided["FREQUENCY_PENALTY"] && cf.Settings.FrequencyPenalty != 0 {
		cfg["FREQUENCY_PENALTY"] = fmt.Sprintf("%g", cf.Settings.FrequencyPenalty)
	}
	if !provided["PRESENCE_PENALTY"] && cf.Settings.PresencePenalty != 0 {
		cfg["PRESENCE_PENALTY"] = fmt.Sprintf("%g", cf.Settings.PresencePenalty)
	}
	if !provided["MAX_TOKENS"] && cf.Settings.MaxTokens != 0 {
		cfg["MAX_TOKENS"] = fmt.Sprintf("%d", cf.Settings.MaxTokens)
	}
	if !provided["STREAM"] {
		if cf.Settings.Stream {
			cfg["STREAM"] = "true"
		} else {
			cfg["STREAM"] = "false"
		}
	}
	if !provided["REASONING_EFFORT"] && cf.Settings.ReasoningEffort != "" {
		cfg["REASONING_EFFORT"] = cf.Settings.ReasoningEffort
	}
	if !provided["STOP"] && cf.Settings.Stop != "" {
		cfg["STOP"] = cf.Settings.Stop
	}
	if !provided["HISTORY_LIMIT"] && cf.Settings.HistoryLimit != 0 {
		cfg["HISTORY_LIMIT"] = fmt.Sprintf("%d", cf.Settings.HistoryLimit)
	}
	return nil
}

func validateNumericRanges(cfg map[string]string) error {
	// temperature 0..1
	t, err := strconv.ParseFloat(cfg["TEMPERATURE"], 64)
	if err != nil || t < 0 || t > 1 {
		return fmt.Errorf("Invalid temperature (0..1): %s", cfg["TEMPERATURE"])
	}
	tp, err := strconv.ParseFloat(cfg["TOP_P"], 64)
	if err != nil || tp < 0.01 || tp > 1 {
		return fmt.Errorf("Invalid top_p (0.01..1): %s", cfg["TOP_P"])
	}
	freq, err := strconv.ParseFloat(cfg["FREQUENCY_PENALTY"], 64)
	if err != nil || freq < -2 || freq > 2 {
		return fmt.Errorf("Invalid frequency_penalty (-2..2): %s", cfg["FREQUENCY_PENALTY"])
	}
	pres, err := strconv.ParseFloat(cfg["PRESENCE_PENALTY"], 64)
	if err != nil || pres < -2 || pres > 2 {
		return fmt.Errorf("Invalid presence_penalty (-2..2): %s", cfg["PRESENCE_PENALTY"])
	}
	mt, err := strconv.Atoi(cfg["MAX_TOKENS"])
	if err != nil || mt < 1 || mt > 4096 {
		return fmt.Errorf("Invalid max_tokens (1..4096): %s", cfg["MAX_TOKENS"])
	}
	if cfg["REASONING_EFFORT"] != "low" && cfg["REASONING_EFFORT"] != "medium" && cfg["REASONING_EFFORT"] != "high" {
		return fmt.Errorf("Invalid reasoning effort (low|medium|high): %s", cfg["REASONING_EFFORT"])
	}
	if cfg["STREAM"] != "true" && cfg["STREAM"] != "false" {
		return fmt.Errorf("Invalid stream flag (true|false): %s", cfg["STREAM"])
	}
	return nil
}

// Build payload JSON (messages must already include system if desired)
func buildPayload(cfg map[string]string, messages []Message) ([]byte, error) {
	stream := cfg["STREAM"] == "true"
	temp, _ := strconv.ParseFloat(cfg["TEMPERATURE"], 64)
	topP, _ := strconv.ParseFloat(cfg["TOP_P"], 64)
	freq, _ := strconv.ParseFloat(cfg["FREQUENCY_PENALTY"], 64)
	pres, _ := strconv.ParseFloat(cfg["PRESENCE_PENALTY"], 64)
	maxTokens, _ := strconv.Atoi(cfg["MAX_TOKENS"])

	payload := map[string]interface{}{
		"model":             cfg["MODEL"],
		"messages":          messages,
		"temperature":       temp,
		"top_p":             topP,
		"frequency_penalty": freq,
		"presence_penalty":  pres,
		"max_tokens":        maxTokens,
		"stream":            stream,
		"reasoning_effort":  cfg["REASONING_EFFORT"],
	}
	if cfg["STOP"] != "" {
		payload["stop"] = cfg["STOP"]
	}
	return json.Marshal(payload)
}

// streaming JSON chunk structures (we only extract needed bits)
type ChoiceDelta struct {
	Content          *string `json:"content,omitempty"`
	ReasoningContent *string `json:"reasoning_content,omitempty"`
}
type ChoiceStream struct {
	Delta   *ChoiceDelta           `json:"delta,omitempty"`
	Message map[string]interface{} `json:"message,omitempty"` // fallback
}
type StreamChunk struct {
	Choices []ChoiceStream `json:"choices"`
}

func handleStream(respBody io.Reader, convFile string) (string, error) {
	scanner := bufio.NewScanner(respBody)
	assistantTextBuf := &bytes.Buffer{}
	inReasoning := false

	// Ensure scanner can read very long lines if needed
	const maxCapacity = 1024 * 1024
	buf := make([]byte, 0, 64*1024)
	scanner.Buffer(buf, maxCapacity)

	for scanner.Scan() {
		line := scanner.Text()
		// SSE style: lines may start with "data: "
		if strings.HasPrefix(line, "data: ") {
			line = strings.TrimPrefix(line, "data: ")
		}
		line = strings.TrimSpace(line)
		if line == "" {
			// skip event separators
			continue
		}
		if line == "[DONE]" {
			continue
		}

		// Try to parse JSON chunk
		var chunk StreamChunk
		if err := json.Unmarshal([]byte(line), &chunk); err != nil {
			// Not parsable -> skip
			continue
		}
		if len(chunk.Choices) == 0 {
			continue
		}
		choice := chunk.Choices[0]
		// Try delta.reasoning_content and delta.content
		var reasoning, content string
		if choice.Delta != nil {
			if choice.Delta.ReasoningContent != nil {
				reasoning = *choice.Delta.ReasoningContent
			}
			if choice.Delta.Content != nil {
				content = *choice.Delta.Content
			}
		} else {
			// fallback: some servers may put content under message
			if msg := choice.Message; msg != nil {
				if v, ok := msg["reasoning_content"].(string); ok {
					reasoning = v
				}
				if v, ok := msg["content"].(string); ok {
					content = v
				}
			}
		}

		if reasoning != "" {
			if !inReasoning {
				fmt.Printf("\n%s\n", green+"[Begin of Assistant Reasoning]"+normal)
				assistantTextBuf.WriteString("[Begin of Assistant Reasoning]\n")
				inReasoning = true
			}
			// JSON unmarshal already unescaped sequences; print directly
			fmt.Print(reasoning)
			assistantTextBuf.WriteString(reasoning)
		}
		if content != "" {
			if inReasoning {
				fmt.Printf("\n%s\n\n", green+"[/End of Assistant Reasoning]"+normal)
				assistantTextBuf.WriteString("\n[/End of Assistant Reasoning]\n\n")
				inReasoning = false
			}
			fmt.Print(content)
			assistantTextBuf.WriteString(content)
		}
	}

	if inReasoning {
		fmt.Printf("\n%s\n\n", green+"[/End of Assistant Reasoning]"+normal)
		assistantTextBuf.WriteString("\n[/End of Assistant Reasoning]\n\n")
		inReasoning = false
	}

	if err := scanner.Err(); err != nil {
		// Non-fatal; return what we have
		return assistantTextBuf.String(), err
	}

	fmt.Println()
	return assistantTextBuf.String(), nil
}

func handleNonStream(body []byte) (string, error) {
	// try to extract .choices[0].delta.reasoning_content or .choices[0].message.reasoning_content and content fields
	var j map[string]interface{}
	if err := json.Unmarshal(body, &j); err != nil {
		return "", err
	}
	var reasoning string
	var content string

	if choices, ok := j["choices"].([]interface{}); ok && len(choices) > 0 {
		if first, ok := choices[0].(map[string]interface{}); ok {
			// delta.reasoning_content
			if delta, ok := first["delta"].(map[string]interface{}); ok {
				if rc, ok := delta["reasoning_content"].(string); ok {
					reasoning = rc
				}
				if c, ok := delta["content"].(string); ok {
					content = c
				}
			}
			// fallback: message.reasoning_content
			if msg, ok := first["message"].(map[string]interface{}); ok {
				if rc, ok := msg["reasoning_content"].(string); ok && reasoning == "" {
					reasoning = rc
				}
				if c, ok := msg["content"].(string); ok && content == "" {
					content = c
				}
			}
		}
	}

	outBuf := &bytes.Buffer{}
	if reasoning != "" {
		fmt.Printf("\n%s\n", green+"[Begin of Assistant Reasoning]"+normal)
		fmt.Print(reasoning)
		fmt.Printf("\n%s\n\n", green+"[/End of Assistant Reasoning]"+normal)
		outBuf.WriteString("[Begin of Assistant Reasoning]\n")
		outBuf.WriteString(reasoning)
		outBuf.WriteString("\n[End of Assistant Reasoning]\n\n")
	}
	if content != "" {
		fmt.Print(content)
		outBuf.WriteString(content)
	}
	if outBuf.Len() == 0 {
		// no assistant content parsed; print raw
		fmt.Printf("%s\n", string(body))
		return "", errors.New("no assistant content parsed from response")
	}
	return outBuf.String(), nil
}

// processMessage sends the given userInput as a user message, calls the API (stream or non-stream),
// prints the assistant output and persists the assistant message to convFile.
func processMessage(userInput, convFile string, cfg map[string]string, sysPromptContent, accessToken string) error {
	// append user message
	if err := appendMessage(convFile, "user", userInput); err != nil {
		return fmt.Errorf("append user message: %w", err)
	}

	// re-check limit
	count, err := messageCount(convFile)
	if err != nil {
		return fmt.Errorf("message count: %w", err)
	}
	limit, _ := strconv.Atoi(cfg["HISTORY_LIMIT"])
	if count > limit {
		return fmt.Errorf("after adding your message, the conversation file exceeded the limit (%d)", limit)
	}

	// Determine effective system prompt: precedence -s content > persisted .system in file > none
	effectiveSystem := sysPromptContent
	if effectiveSystem == "" {
		cf, err := readConversation(convFile)
		if err == nil {
			effectiveSystem = cf.System
		}
	}

	// Build messages: prepend system prompt if non-empty, then .messages
	cf2, err := readConversation(convFile)
	if err != nil {
		return fmt.Errorf("read conversation: %w", err)
	}
	var messages []Message
	if effectiveSystem != "" {
		messages = append(messages, Message{Role: "system", Content: effectiveSystem})
	}
	messages = append(messages, cf2.Messages...)

	// Build payload
	payloadBytes, err := buildPayload(cfg, messages)
	if err != nil {
		return fmt.Errorf("build payload: %w", err)
	}

	// Prepare HTTP request
	url := cfg["BASE_URL"] + "/chat/completions"
	req, _ := http.NewRequest("POST", url, bytes.NewReader(payloadBytes))
	req.Header.Set("Authorization", "Bearer "+accessToken)
	req.Header.Set("Content-Type", "application/json")

	client := &http.Client{Timeout: 0}
	if cfg["STREAM"] == "true" {
		// streaming mode
		resp, err := client.Do(req)
		if err != nil {
			return fmt.Errorf("request failed: %w", err)
		}
		if resp.StatusCode >= 400 {
			body, _ := ioutil.ReadAll(resp.Body)
			resp.Body.Close()
			return fmt.Errorf("api error: %s\n%s", resp.Status, string(body))
		}
		assistantText, err := handleStream(resp.Body, convFile)
		resp.Body.Close()
		if assistantText != "" {
			if err2 := appendMessage(convFile, "assistant", assistantText); err2 != nil {
				// non-fatal append error, but surface it
				return fmt.Errorf("append assistant message: %w", err2)
			}
		}
		return err
	} else {
		// non-streaming mode
		resp, err := client.Do(req)
		if err != nil {
			return fmt.Errorf("request failed: %w", err)
		}
		body, _ := ioutil.ReadAll(resp.Body)
		resp.Body.Close()
		if resp.StatusCode >= 400 {
			return fmt.Errorf("api error: %s\n%s", resp.Status, string(body))
		}
		assistantText, _ := handleNonStream(body)
		if assistantText != "" {
			if err := appendMessage(convFile, "assistant", assistantText); err != nil {
				return fmt.Errorf("append assistant message: %w", err)
			}
		}
		return nil
	}
}

func getAPIKeyFromEnv() string {
	for _, n := range apiEnvNames {
		if v := os.Getenv(n); v != "" {
			return v
		}
	}
	return ""
}

func readSingleLine(reader io.Reader, delimiters []string, trimDelimiter bool) (string, error) {
	if reader == nil {
		reader = os.Stdin
	}
	if len(delimiters) == 0 {
		delimiters = []string{"\r\n", "\r", "\n"}
	}
	br := bufio.NewReader(reader)
	var line bytes.Buffer
	for {
		b, err := br.ReadByte()
		if err != nil {
			if err == io.EOF {
				// If no data was read, propagate EOF
				if line.Len() == 0 {
					return "", io.EOF
				}
				// Return last partial line along with EOF
				return line.String(), io.EOF
			}
			return "", err
		}
		line.WriteByte(b)
		for _, delim := range delimiters {
			delimBytes := []byte(delim)
			if bytes.HasSuffix(line.Bytes(), delimBytes) {
				resultBytes := line.Bytes()
				if trimDelimiter {
					resultBytes = bytes.TrimSuffix(resultBytes, delimBytes)
				}
				return string(resultBytes), nil
			}
		}
	}
}

func readLines(reader io.Reader, delimiters []string, trimDelimiter bool) ([]string, error) {
	if reader == nil {
		reader = os.Stdin
	}
	if len(delimiters) == 0 {
		delimiters = []string{"\r\n", "\r", "\n"}
	}
	lines := make([]string, 0)
	var lastErr error
	for {
		line, err := readSingleLine(reader, delimiters, trimDelimiter)
		if err != nil {
			lastErr = err
			if err == io.EOF {
				if line != "" {
					lines = append(lines, line)
				}
				break
			}
			return nil, err
		}
		if line != "" || lastErr != io.EOF {
			lines = append(lines, line)
		}
	}
	if lastErr != nil && lastErr != io.EOF {
		return nil, lastErr
	}

	return lines, nil
}

func main() {
	rand.Seed(time.Now().UnixNano())
	// Default cfg map
	cfg := map[string]string{
		"BASE_URL":          defaultBaseURL,
		"MODEL":             defaultModel,
		"TEMPERATURE":       defaultTemperature,
		"TOP_P":             defaultTopP,
		"FREQUENCY_PENALTY": defaultFrequency,
		"PRESENCE_PENALTY":  defaultPresence,
		"MAX_TOKENS":        defaultMaxTokens,
		"STREAM":            defaultStream,
		"REASONING_EFFORT":  defaultReasoning,
		"STOP":              defaultStop,
		"HISTORY_DIR":       filepath.Join(os.Getenv("HOME"), defaultHistorySubdir),
		"HISTORY_LIMIT":     fmt.Sprintf("%d", defaultHistoryLimit),
	}

	// -----------------------
	// Parse options (robust)
	// -----------------------
	provided := map[string]bool{}
	rawArgs := os.Args[1:]
	var positionalArgs []string

	ACCESS_TOKEN := ""
	SYS_PROMPT_FILE := ""
	PERSIST_SYSTEM := false
	SAVE_SETTINGS := false
	LIST_ONLY := false
	PROMPT_MODE := "" // for --prompt

	// helper to get next argument (used when flag and its value are separate tokens)
	nextArg := func(i *int) (string, error) {
		*i++
		if *i >= len(rawArgs) {
			return "", fmt.Errorf("missing value for %s", rawArgs[*i-1])
		}
		return rawArgs[*i], nil
	}

	i := 0
	for i < len(rawArgs) {
		a := rawArgs[i]

		if a == "--" {
			// stop parsing flags; remaining args are positional
			positionalArgs = append(positionalArgs, rawArgs[i+1:]...)
			break
		}

		if !strings.HasPrefix(a, "-") {
			positionalArgs = append(positionalArgs, a)
			i++
			continue
		}

		// at this point, 'a' is a flag
		key := a
		val := ""
		// handle --flag=value and -f=value
		if strings.Contains(a, "=") {
			parts := strings.SplitN(a, "=", 2)
			key = parts[0]
			val = parts[1]
		}

		switch key {
		// flags that take a value
		case "-m", "--model":
			if val == "" {
				v, err := nextArg(&i)
				if err != nil {
					fmt.Fprintf(os.Stderr, "%s%s%s\n", red, err.Error(), normal)
					os.Exit(1)
				}
				val = v
			}
			cfg["MODEL"] = val
			provided["MODEL"] = true
		case "-T", "--temperature":
			if val == "" {
				v, err := nextArg(&i)
				if err != nil {
					fmt.Fprintf(os.Stderr, "%s%s%s\n", red, err.Error(), normal)
					os.Exit(1)
				}
				val = v
			}
			cfg["TEMPERATURE"] = val
			provided["TEMPERATURE"] = true
		case "-P", "--top-p":
			if val == "" {
				v, err := nextArg(&i)
				if err != nil {
					fmt.Fprintf(os.Stderr, "%s%s%s\n", red, err.Error(), normal)
					os.Exit(1)
				}
				val = v
			}
			cfg["TOP_P"] = val
			provided["TOP_P"] = true
		case "-f", "--frequency-penalty":
			if val == "" {
				v, err := nextArg(&i)
				if err != nil {
					fmt.Fprintf(os.Stderr, "%s%s%s\n", red, err.Error(), normal)
					os.Exit(1)
				}
				val = v
			}
			cfg["FREQUENCY_PENALTY"] = val
			provided["FREQUENCY_PENALTY"] = true
		case "-r", "--presence-penalty":
			if val == "" {
				v, err := nextArg(&i)
				if err != nil {
					fmt.Fprintf(os.Stderr, "%s%s%s\n", red, err.Error(), normal)
					os.Exit(1)
				}
				val = v
			}
			cfg["PRESENCE_PENALTY"] = val
			provided["PRESENCE_PENALTY"] = true
		case "-M", "--max-tokens":
			if val == "" {
				v, err := nextArg(&i)
				if err != nil {
					fmt.Fprintf(os.Stderr, "%s%s%s\n", red, err.Error(), normal)
					os.Exit(1)
				}
				val = v
			}
			cfg["MAX_TOKENS"] = val
			provided["MAX_TOKENS"] = true
		case "-L", "--limit":
			if val == "" {
				v, err := nextArg(&i)
				if err != nil {
					fmt.Fprintf(os.Stderr, "%s%s%s\n", red, err.Error(), normal)
					os.Exit(1)
				}
				val = v
			}
			cfg["HISTORY_LIMIT"] = val
			provided["HISTORY_LIMIT"] = true
		case "-s", "--sys-prompt-file":
			if val == "" {
				v, err := nextArg(&i)
				if err != nil {
					fmt.Fprintf(os.Stderr, "%s%s%s\n", red, err.Error(), normal)
					os.Exit(1)
				}
				val = v
			}
			SYS_PROMPT_FILE = val
			provided["SYS_PROMPT_FILE"] = true
		case "-k", "--access-token":
			if val == "" {
				v, err := nextArg(&i)
				if err != nil {
					fmt.Fprintf(os.Stderr, "%s%s%s\n", red, err.Error(), normal)
					os.Exit(1)
				}
				val = v
			}
			ACCESS_TOKEN = val
		case "--reasoning":
			if val == "" {
				v, err := nextArg(&i)
				if err != nil {
					fmt.Fprintf(os.Stderr, "%s%s%s\n", red, err.Error(), normal)
					os.Exit(1)
				}
				val = v
			}
			cfg["REASONING_EFFORT"] = val
			provided["REASONING_EFFORT"] = true
		case "--stop":
			if val == "" {
				v, err := nextArg(&i)
				if err != nil {
					fmt.Fprintf(os.Stderr, "%s%s%s\n", red, err.Error(), normal)
					os.Exit(1)
				}
				val = v
			}
			cfg["STOP"] = val
			provided["STOP"] = true
		case "--prompt":
			if val == "" {
				v, err := nextArg(&i)
				if err != nil {
					fmt.Fprintf(os.Stderr, "%s%s%s\n", red, err.Error(), normal)
					os.Exit(1)
				}
				val = v
			}
			PROMPT_MODE = val
		case "--stream":
			if val == "true" {
				cfg["STREAM"] = "true"
			} else if val == "false" {
				cfg["STREAM"] = "false"
			} else {
				fmt.Fprintf(os.Stderr, "%sInvalid value for --stream: %s. Use true or false.%s\n", red, val, normal)
				os.Exit(1)
			}
			provided["STREAM"] = true

		// boolean flags
		case "-S":
			PERSIST_SYSTEM = true
		case "--no-stream":
			cfg["STREAM"] = "false"
			provided["STREAM"] = true
		case "--save-settings":
			SAVE_SETTINGS = true
		case "-l", "--list":
			LIST_ONLY = true
		case "-h", "--help":
			printHelp(cfg)
			return
		default:
			fmt.Fprintf(os.Stderr, "Unknown option: %s\n", a)
			printHelp(cfg)
			os.Exit(1)
		}
		i++
	}
	args := positionalArgs

	// If list requested
	if LIST_ONLY {
		fmt.Printf("%sSupported models (built-in subset):%s\n", bold, normal)
		for _, m := range modelsList {
			fmt.Printf("  %s\n", m)
		}
		fmt.Println()
		fmt.Println("View the full models list and details at: https://build.nvidia.com/")
		return
	}

	// API key selection from env if not provided
	if ACCESS_TOKEN == "" {
		ACCESS_TOKEN = getAPIKeyFromEnv()
	}
	if ACCESS_TOKEN == "" {
		fmt.Fprintf(os.Stderr, "%sNo API key provided.%s Set NVIDIA_BUILD_AI_ACCESS_TOKEN or pass -k ACCESS_TOKEN\n", red, normal)
		os.Exit(1)
	}

	// conversation file
	convFile := ""
	if len(args) > 0 {
		convFile = args[0]
		// expand ~
		if strings.HasPrefix(convFile, "~") {
			home := os.Getenv("HOME")
			convFile = home + convFile[1:]
		}
	}

	// read system prompt file
	sysPromptContent := ""
	if SYS_PROMPT_FILE != "" {
		if _, err := os.Stat(SYS_PROMPT_FILE); os.IsNotExist(err) {
			fmt.Fprintf(os.Stderr, "%sSystem prompt file not found: %s%s\n", red, SYS_PROMPT_FILE, normal)
			os.Exit(1)
		}
		b, _ := ioutil.ReadFile(SYS_PROMPT_FILE)
		sysPromptContent = string(b)
	}

	// Non-interactive prompt mode
	if PROMPT_MODE != "" {
		var promptText string
		var err error
		if PROMPT_MODE == "-" {
			// from stdin
			b, e := ioutil.ReadAll(os.Stdin)
			if e != nil {
				fmt.Fprintf(os.Stderr, "%sFailed to read from stdin: %v%s\n", red, e, normal)
				os.Exit(1)
			}
			promptText = string(b)
		} else if fileExists(PROMPT_MODE) {
			// from file
			b, e := ioutil.ReadFile(PROMPT_MODE)
			if e != nil {
				fmt.Fprintf(os.Stderr, "%sFailed to read prompt file: %v%s\n", red, e, normal)
				os.Exit(1)
			}
			promptText = string(b)
		} else {
			// as-is
			promptText = PROMPT_MODE
		}

		if convFile != "" {
			// Non-interactive with a conversation file
			if err := ensureHistoryFileStructure(convFile, cfg); err != nil {
				fmt.Fprintf(os.Stderr, "%sFailed to setup conversation file: %v%s\n", red, err, normal)
				os.Exit(1)
			}
			if err := applyFileSettingsAsDefaults(convFile, cfg, provided); err != nil {
				fmt.Fprintf(os.Stderr, "%sWarning applying file settings: %v%s\n", red, err, normal)
			}
			if err := validateNumericRanges(cfg); err != nil {
				fmt.Fprintf(os.Stderr, "%s%s%s\n", red, err.Error(), normal)
				os.Exit(1)
			}
			if SAVE_SETTINGS {
				if err := persistSettingsToFile(convFile, cfg); err != nil {
					fmt.Fprintf(os.Stderr, "%sFailed to persist settings: %v%s\n", red, err, normal)
					os.Exit(1)
				}
				fmt.Fprintf(os.Stderr, "%sPersisted current settings into %s%s\n", green, convFile, normal)
			}
			err = processMessage(promptText, convFile, cfg, sysPromptContent, ACCESS_TOKEN)
			if err != nil {
				fmt.Fprintf(os.Stderr, "%sError: %v%s\n", red, err, normal)
				os.Exit(1)
			}
		} else {
			// Non-interactive, no conversation file
			err = processSinglePrompt(promptText, cfg, sysPromptContent, ACCESS_TOKEN)
			if err != nil {
				fmt.Fprintf(os.Stderr, "%sError: %v%s\n", red, err, normal)
				os.Exit(1)
			}
		}
		return
	}

	// Interactive mode
	if convFile == "" {
		// create new default path
		hdir := os.Getenv("XDG_CACHE_HOME")
		if hdir == "" {
			hdir = filepath.Join(os.Getenv("HOME"), ".cache")
		}
		cfg["HISTORY_DIR"] = filepath.Join(hdir, "nvidia-chat")
		ts := time.Now().Format("20060102-150405")
		convFile = filepath.Join(cfg["HISTORY_DIR"], "conversation-"+ts+".json")
		fmt.Fprintf(os.Stderr, "Creating conversation file: %s\n", convFile)
	}

	// ensure conversation file exists and has structure
	if err := ensureHistoryFileStructure(convFile, cfg); err != nil {
		fmt.Fprintf(os.Stderr, "%sFailed to setup conversation file: %v%s\n", red, err, normal)
		os.Exit(1)
	}
	fmt.Fprintf(os.Stderr, "%sConversation file:%s %s\n", green, normal, convFile)

	// Apply persisted settings as defaults if user did not provide those options explicitly
	if err := applyFileSettingsAsDefaults(convFile, cfg, provided); err != nil {
		// non-fatal: warn
		fmt.Fprintf(os.Stderr, "%sWarning applying file settings: %v%s\n", red, err, normal)
	}

	// Validate numeric ranges
	if err := validateNumericRanges(cfg); err != nil {
		fmt.Fprintf(os.Stderr, "%s%s%s\n", red, err.Error(), normal)
		os.Exit(1)
	}

	// If persist system requested but no -s provided -> exit
	if PERSIST_SYSTEM && sysPromptContent == "" {
		fmt.Fprintf(os.Stderr, "%sPersist system requested (-S) but no -s SYS_PROMPT_FILE provided.%s Provide -s path and -S together to persist system prompt into the conversation file.\n", red, normal)
		os.Exit(1)
	}

	// Check message count vs limit
	count, err := messageCount(convFile)
	if err != nil {
		fmt.Fprintf(os.Stderr, "%sFailed reading conversation file: %v%s\n", red, err, normal)
		os.Exit(1)
	}
	limit, _ := strconv.Atoi(cfg["HISTORY_LIMIT"])
	if limit <= 0 {
		fmt.Fprintf(os.Stderr, "%sInvalid limit (-L): %s%s\n", red, cfg["HISTORY_LIMIT"], normal)
		os.Exit(1)
	}
	if count >= limit {
		fmt.Fprintf(os.Stderr, "%sConversation message limit reached.%s\nFile: %s\nMessages in file: %d\nConfigured limit: %d\n\nThis program will NOT remove or rotate messages automatically.\nOptions:\n  - Increase limit via -L option and re-run\n  - Use a different conversation file (pass new filename)\n  - Manually edit the file to remove old messages\n\nExiting.\n", red, normal, convFile, count, limit)
		os.Exit(1)
	}

	// Persist settings or system if requested before interactive loop
	if SAVE_SETTINGS {
		if err := persistSettingsToFile(convFile, cfg); err != nil {
			fmt.Fprintf(os.Stderr, "%sFailed to persist settings: %v%s\n", red, err, normal)
			os.Exit(1)
		}
		fmt.Fprintf(os.Stderr, "%sPersisted current settings into conversation file's .settings%s\n", green, normal)
	}
	if PERSIST_SYSTEM {
		if err := persistSystemToFile(convFile, sysPromptContent); err != nil {
			fmt.Fprintf(os.Stderr, "%sFailed to persist system prompt: %v%s\n", red, err, normal)
			os.Exit(1)
		}
		fmt.Fprintf(os.Stderr, "%sPersisted system prompt into conversation file's .system%s\n", green, normal)
	}

	// Interactive banner
	fmt.Fprintln(os.Stderr)
	fmt.Fprintln(os.Stderr, `AI models generate responses and outputs based on complex algorithms and
machine learning techniques, and those responses or outputs may be
inaccurate, harmful, biased or indecent. By testing this model, you assume
the risk of any harm caused by any response or output of the model. Please
do not upload any confidential information or personal data unless
expressly permitted. Your use is logged for security purposes.
`)
	fmt.Fprintf(os.Stderr, "%sNVIDIA chat (go)%s model=%s temperature=%s top_p=%s max_tokens=%s stream=%s freq_penalty=%s pres_penalty=%s reasoning=%s stop=%q\n", bold, normal, cfg["MODEL"], cfg["TEMPERATURE"], cfg["TOP_P"], cfg["MAX_TOKENS"], cfg["STREAM"], cfg["FREQUENCY_PENALTY"], cfg["PRESENCE_PENALTY"], cfg["REASONING_EFFORT"], cfg["STOP"])
	fmt.Fprintf(os.Stderr, "Conversation file: %s\n", convFile)
	fmt.Fprintln(os.Stderr, "Type your message and end it by Ctrl+D. Commands: /exit /quit /history /clear /save <file> /persist-system (see help)")

	// trap SIGINT handled by default (Ctrl+C ends program)

	lines := make([]string, 0)

	// interactive loop
	for {
		fmt.Fprintf(os.Stderr, "\n%s: ", blue+"You"+normal)

		// read first line
		firstLine, err := readSingleLine(nil, []string{"\r\n", "\r", "\n"}, true)
		if err != nil && err != io.EOF {
			fmt.Fprintf(os.Stderr, "%sFailed reading input: %v%s\n", red, err, normal)
			return
		}
		if firstLine == "" {
			// EOF with no input -> restart loop
			continue
		}

		firstLineTrimmed := strings.TrimSpace(firstLine)
		if strings.HasPrefix(firstLineTrimmed, "/") {
			// Check if it's a command
			if handled := handleInteractiveInput(firstLineTrimmed, convFile, cfg); handled {
				continue
			}
		}

		// If it wasn't a command, read the rest of the multi-line input until EOF
		if err == nil { // only if we didn't get an EOF on the first read
			remainingLines, err := readLines(nil, []string{"\r\n", "\r", "\n"}, true)
			if err != nil && err != io.EOF {
				fmt.Fprintf(os.Stderr, "%sFailed reading multi-line input: %v%s\n", red, err, normal)
				continue
			}
			lines = append([]string{firstLine}, remainingLines...)
		}

		userInput := strings.Join(lines, "\n")
		userInput = strings.TrimSpace(userInput)

		if userInput == "" {
			continue
		}

		// append user message
		if err := appendMessage(convFile, "user", userInput); err != nil {
			fmt.Fprintf(os.Stderr, "%sFailed appending message: %v%s\n", red, err, normal)
			continue
		}
		// re-check limit
		count, _ := messageCount(convFile)
		limit, _ := strconv.Atoi(cfg["HISTORY_LIMIT"])
		if count > limit {
			fmt.Fprintf(os.Stderr, "%sAfter adding your message, the conversation file exceeded the limit (%d).%s\nI did not remove messages. Increase limit with -L or use another file.\n", red, limit, normal)
			os.Exit(1)
		}

		// Determine effective system prompt: precedence -s content > persisted .system in file > none
		effectiveSystem := ""
		if sysPromptContent != "" {
			effectiveSystem = sysPromptContent
		} else {
			cf, _ := readConversation(convFile)
			effectiveSystem = cf.System
		}

		// Build messages: prepend system prompt if non-empty, then .messages
		var messages []Message
		cf2, err := readConversation(convFile)
		if err != nil {
			fmt.Fprintf(os.Stderr, "%sFailed reading conversation to build payload: %v%s\n", red, err, normal)
			continue
		}
		if effectiveSystem != "" {
			messages = append(messages, Message{Role: "system", Content: effectiveSystem})
		}
		messages = append(messages, cf2.Messages...)

		// Build payload
		payloadBytes, err := buildPayload(cfg, messages)
		if err != nil {
			fmt.Fprintf(os.Stderr, "%sFailed building payload: %v%s\n", red, err, normal)
			continue
		}

		// Prepare HTTP request
		url := cfg["BASE_URL"] + "/chat/completions"
		req, _ := http.NewRequest("POST", url, bytes.NewReader(payloadBytes))
		req.Header.Set("Authorization", "Bearer "+ACCESS_TOKEN)
		req.Header.Set("Content-Type", "application/json")

		client := &http.Client{}
		if cfg["STREAM"] == "true" {
			// streaming mode
			resp, err := client.Do(req)
			if err != nil {
				fmt.Fprintf(os.Stderr, "%sRequest failed: %v%s\n", red, err, normal)
				continue
			}
			if resp.StatusCode >= 400 {
				body, _ := ioutil.ReadAll(resp.Body)
				fmt.Fprintf(os.Stderr, "%sAPI error: %s%s\n%s\n", red, resp.Status, normal, string(body))
				resp.Body.Close()
				continue
			}
			fmt.Fprintf(os.Stderr, "\n%s\n", blue+"Assistant:"+normal)
			assistantText, err := handleStream(resp.Body, convFile)
			resp.Body.Close()
			if err != nil {
				// print error but continue
			}
			if strings.TrimSpace(assistantText) != "" {
				if err := appendMessage(convFile, "assistant", assistantText); err != nil {
					fmt.Fprintf(os.Stderr, "%sFailed appending assistant message: %v%s\n", red, err, normal)
				}
			}
		} else {
			// non-streaming mode
			resp, err := client.Do(req)
			if err != nil {
				fmt.Fprintf(os.Stderr, "%sRequest failed: %v%s\n", red, err, normal)
				continue
			}
			body, _ := ioutil.ReadAll(resp.Body)
			resp.Body.Close()
			if resp.StatusCode >= 400 {
				fmt.Fprintf(os.Stderr, "%sAPI error: %s%s\n%s\n", red, resp.Status, normal, string(body))
				continue
			}
			fmt.Fprintf(os.Stderr, "\n%s\n", blue+"Assistant:"+normal)
			assistantText, err := handleNonStream(body)
			if err != nil {
				// we printed raw body already; don't treat as fatal
			}
			if strings.TrimSpace(assistantText) != "" {
				if err := appendMessage(convFile, "assistant", assistantText); err != nil {
					fmt.Fprintf(os.Stderr, "%sFailed appending assistant message: %v%s\n", red, err, normal)
				}
			}
		}
	}
}

// Returns true if the command was a special/handled interactive command
// handleInteractiveInput returns true if the input was a special command that was handled here.
// Otherwise returns false so the caller will continue normal message processing.
func filterThinkingBlock(content string) string {
	re := regexp.MustCompile(`(?s)\[Begin of Assistant Reasoning\].*?\[/End of Assistant Reasoning\]\s*\n?`)
	return re.ReplaceAllString(content, "")
}

func exportLastN(n int, convFile, targetFile string, filterThinking bool) error {
	cf, err := readConversation(convFile)
	if err != nil {
		return fmt.Errorf("reading conversation file: %w", err)
	}

	var aiResponses []string
	for i := len(cf.Messages) - 1; i >= 0; i-- {
		if cf.Messages[i].Role == "assistant" {
			aiResponses = append(aiResponses, cf.Messages[i].Content)
			if len(aiResponses) == n {
				break
			}
		}
	}

	if len(aiResponses) == 0 {
		return fmt.Errorf("no assistant responses found")
	}

	// Reverse the slice to get the correct order
	for i, j := 0, len(aiResponses)-1; i < j; i, j = i+1, j-1 {
		aiResponses[i], aiResponses[j] = aiResponses[j], aiResponses[i]
	}

	if filterThinking {
		for i, resp := range aiResponses {
			aiResponses[i] = filterThinkingBlock(resp)
		}
	}

	content := strings.Join(aiResponses, "\n\n---\n\n")
	return ioutil.WriteFile(targetFile, []byte(content), 0o644)
}

func exportNth(n int, convFile, targetFile string, filterThinking bool) error {
	cf, err := readConversation(convFile)
	if err != nil {
		return fmt.Errorf("reading conversation file: %w", err)
	}

	var aiResponses []string
	for _, msg := range cf.Messages {
		if msg.Role == "assistant" {
			aiResponses = append(aiResponses, msg.Content)
		}
	}

	if len(aiResponses) == 0 {
		return fmt.Errorf("no assistant responses found")
	}

	// n is 1-based from the end
	index := len(aiResponses) - n
	if index < 0 || index >= len(aiResponses) {
		return fmt.Errorf("index out of bounds: specified %d, but there are only %d assistant responses", n, len(aiResponses))
	}

	content := aiResponses[index]
	if filterThinking {
		content = filterThinkingBlock(content)
	}
	return ioutil.WriteFile(targetFile, []byte(content), 0o644)
}

func parseTFlag(parts []string) (bool, []string) {
	filterThinking := false
	var newParts []string
	newParts = append(newParts, parts[0])
	for _, p := range parts[1:] {
		if p == "-t" {
			filterThinking = true
		} else {
			newParts = append(newParts, p)
		}
	}
	return filterThinking, newParts
}

func handleInteractiveInput(userInput, convFile string, cfg map[string]string) bool {
	trimmed := strings.TrimSpace(userInput)
	parts := strings.Fields(trimmed)
	if len(parts) == 0 {
		return false
	}
	command := parts[0]

	switch command {
	case "/exit", "/quit":
		fmt.Fprintln(os.Stderr, "Bye.")
		os.Exit(0)
		return true

	case "/history":
		fmt.Fprintf(os.Stderr, "%s:\n", convFile)
		b, err := ioutil.ReadFile(convFile)
		if err != nil {
			fmt.Fprintf(os.Stderr, "%sFailed reading conversation: %v%s\n", red, err, normal)
			return true
		}
		fmt.Println(string(b))
		return true

	case "/clear":
		cf, err := readConversation(convFile)
		if err != nil {
			fmt.Fprintf(os.Stderr, "%sFailed reading conversation: %v%s\n", red, err, normal)
			return true
		}
		cf.Messages = []Message{}
		if err := writeConversation(convFile, cf); err != nil {
			fmt.Fprintf(os.Stderr, "%sFailed clearing messages: %v%s\n", red, err, normal)
		} else {
			fmt.Fprintf(os.Stderr, "%sMessages cleared%s\n", green, normal)
		}
		return true

	case "/save":
		if len(parts) < 2 {
			fmt.Fprintln(os.Stderr, "Usage: /save path")
			return true
		}
		target := parts[1]
		if err := copyFile(convFile, target); err != nil {
			fmt.Fprintf(os.Stderr, "%sFailed to save: %v%s\n", red, err, normal)
		} else {
			fmt.Fprintf(os.Stderr, "Saved to %s\n", target)
		}
		return true

	case "/persist-system":
		if len(parts) < 2 {
			fmt.Fprintln(os.Stderr, "Usage: /persist-system <file>")
			return true
		}
		path := parts[1]
		if !fileExists(path) {
			fmt.Fprintf(os.Stderr, "%sFile not found: %s%s\n", red, path, normal)
			return true
		}
		content, err := ioutil.ReadFile(path)
		if err != nil {
			fmt.Fprintf(os.Stderr, "%sFailed to read file: %v%s\n", red, err, normal)
			return true
		}
		if err := persistSystemToFile(convFile, string(content)); err != nil {
			fmt.Fprintf(os.Stderr, "%sFailed to persist system prompt: %v%s\n", red, err, normal)
		} else {
			fmt.Fprintf(os.Stderr, "%sPersisted system prompt from %s into conversation file's .system%s\n", green, path, normal)
		}
		return true

	case "/model", "/temperature", "/top_p", "/frequency_penalty", "/presence_penalty", "/max_tokens", "/reasoning", "/stop", "/limit", "/stream":
		if len(parts) < 2 {
			fmt.Fprintf(os.Stderr, "Usage: %s <value>\n", command)
			return true
		}
		val := parts[1]
		key := strings.TrimPrefix(command, "/")
		key = strings.ToUpper(key)
		if key == "REASONING" {
			key = "REASONING_EFFORT"
		}
		if key == "LIMIT" {
			key = "HISTORY_LIMIT"
		}
		// quick validation
		tempCfg := make(map[string]string)
		for k, v := range cfg {
			tempCfg[k] = v
		}
		tempCfg[key] = val
		if err := validateNumericRanges(tempCfg); err != nil {
			fmt.Fprintf(os.Stderr, "%sInvalid value: %v%s\n", red, err, normal)
			return true
		}
		cfg[key] = val
		fmt.Fprintf(os.Stderr, "%s%s set to %s%s\n", green, key, val, normal)
		return true

	case "/persist-settings":
		if err := persistSettingsToFile(convFile, cfg); err != nil {
			fmt.Fprintf(os.Stderr, "%sFailed to persist settings: %v%s\n", red, err, normal)
		} else {
			fmt.Fprintf(os.Stderr, "%sPersisted current settings to %s%s\n", green, convFile, normal)
		}
		return true
	case "/exportlast":
		filterThinking, newParts := parseTFlag(parts)
		parts = newParts
		if len(parts) < 2 {
			fmt.Fprintln(os.Stderr, "Usage: /exportlast [-t] <file>")
			return true
		}
		targetFile := parts[1]
		if err := exportLastN(1, convFile, targetFile, filterThinking); err != nil {
			fmt.Fprintf(os.Stderr, "%sFailed to export: %v%s\n", red, err, normal)
		} else {
			fmt.Fprintf(os.Stderr, "%sExported last response to %s%s\n", green, targetFile, normal)
		}
		return true
	case "/exportn":
		filterThinking, newParts := parseTFlag(parts)
		parts = newParts
		if len(parts) < 3 {
			fmt.Fprintln(os.Stderr, "Usage: /exportn [-t] <n> <file>")
			return true
		}
		n, err := strconv.Atoi(parts[1])
		if err != nil || n <= 0 {
			fmt.Fprintf(os.Stderr, "%sInvalid number: %s%s\n", red, parts[1], normal)
			return true
		}
		targetFile := parts[2]
		if err := exportNth(n, convFile, targetFile, filterThinking); err != nil {
			fmt.Fprintf(os.Stderr, "%sFailed to export: %v%s\n", red, err, normal)
		} else {
			fmt.Fprintf(os.Stderr, "%sExported %d(th) last response to %s%s\n", green, n, targetFile, normal)
		}
		return true
	case "/exportlastn":
		filterThinking, newParts := parseTFlag(parts)
		parts = newParts
		if len(parts) < 3 {
			fmt.Fprintln(os.Stderr, "Usage: /exportlastn [-t] <n> <file>")
			return true
		}
		n, err := strconv.Atoi(parts[1])
		if err != nil || n <= 0 {
			fmt.Fprintf(os.Stderr, "%sInvalid number: %s%s\n", red, parts[1], normal)
			return true
		}
		targetFile := parts[2]
		if err := exportLastN(n, convFile, targetFile, filterThinking); err != nil {
			fmt.Fprintf(os.Stderr, "%sFailed to export: %v%s\n", red, err, normal)
		} else {
			fmt.Fprintf(os.Stderr, "%sExported last %d responses to %s%s\n", green, n, targetFile, normal)
		}
		return true
	case "/randomodel":
		newModel := modelsList[rand.Intn(len(modelsList))]
		cfg["MODEL"] = newModel
		fmt.Fprintf(os.Stderr, "%sSwitched model to %s%s\n", green, newModel, normal)
		return true
	case "/help":
		fmt.Fprintln(os.Stderr, "Available commands:")
		fmt.Fprintln(os.Stderr, "  /exit, /quit: Exit the program")
		fmt.Fprintln(os.Stderr, "  /history: Print full conversation JSON")
		fmt.Fprintln(os.Stderr, "  /clear: Clear conversation messages")
		fmt.Fprintln(os.Stderr, "  /save <file>: Save conversation to a new file")
		fmt.Fprintln(os.Stderr, "  /persist-system <file>: Persist a system prompt from a file")
		fmt.Fprintln(os.Stderr, "  /model <model_name>: Switch model for this session")
		fmt.Fprintln(os.Stderr, "  /temperature <0..1>: Set temperature for this session")
		fmt.Fprintln(os.Stderr, "  /top_p <0.01..1>: Set top_p for this session")
		fmt.Fprintln(os.Stderr, "  /max_tokens <1..4096>: Set max_tokens for this session")
		fmt.Fprintln(os.Stderr, "  /stop <string>: Set stop string for this session")
		fmt.Fprintln(os.Stderr, "  /limit <number>: Set history limit for this session")
		fmt.Fprintln(os.Stderr, "  /stream <true|false>: Enable/disable streaming for this session")
		fmt.Fprintln(os.Stderr, "  /persist-settings: Save the current session's settings to the conversation file")
		fmt.Fprintln(os.Stderr, "  /exportlast [-t] <file>: Export last AI response to a markdown file. (-t: filter thinking)")
		fmt.Fprintln(os.Stderr, "  /exportlastn [-t] <n> <file>: Export last n AI responses to a markdown file. (-t: filter thinking)")
		fmt.Fprintln(os.Stderr, "  /exportn [-t] <n> <file>: Export the Nth-to-last AI response to a markdown file. (-t: filter thinking)")
		fmt.Fprintln(os.Stderr, "  /randomodel: Switch to a random model")
		fmt.Fprintln(os.Stderr, "  /help: Show this help message")
		return true

	default:
		return false
	}
}

func copyFile(src, dst string) error {
	in, err := os.Open(src)
	if err != nil {
		return err
	}
	defer in.Close()
	out, err := os.Create(dst)
	if err != nil {
		return err
	}
	defer out.Close()
	_, err = io.Copy(out, in)
	if err != nil {
		return err
	}
	return out.Sync()
}

func fileExists(path string) bool {
	_, err := os.Stat(path)
	return err == nil
}

// Quieter stream handler for --prompt mode
func handleStreamQuiet(respBody io.Reader) error {
	scanner := bufio.NewScanner(respBody)
	const maxCapacity = 1024 * 1024
	buf := make([]byte, 0, 64*1024)
	scanner.Buffer(buf, maxCapacity)

	for scanner.Scan() {
		line := scanner.Text()
		if strings.HasPrefix(line, "data: ") {
			line = strings.TrimPrefix(line, "data: ")
		}
		line = strings.TrimSpace(line)
		if line == "" || line == "[DONE]" {
			continue
		}
		var chunk StreamChunk
		if err := json.Unmarshal([]byte(line), &chunk); err != nil {
			continue
		}
		if len(chunk.Choices) > 0 {
			choice := chunk.Choices[0]
			var content string
			if choice.Delta != nil && choice.Delta.Content != nil {
				content = *choice.Delta.Content
			} else if msg := choice.Message; msg != nil {
				if v, ok := msg["content"].(string); ok {
					content = v
				}
			}
			if content != "" {
				fmt.Print(content)
			}
		}
	}
	return scanner.Err()
}

// Quieter non-stream handler for --prompt mode
func handleNonStreamQuiet(body []byte) error {
	var j map[string]interface{}
	if err := json.Unmarshal(body, &j); err != nil {
		fmt.Print(string(body)) // fallback to printing raw body
		return err
	}
	var content string
	if choices, ok := j["choices"].([]interface{}); ok && len(choices) > 0 {
		if first, ok := choices[0].(map[string]interface{}); ok {
			if msg, ok := first["message"].(map[string]interface{}); ok {
				if c, ok := msg["content"].(string); ok {
					content = c
				}
			}
		}
	}

	if content != "" {
		fmt.Print(content)
	} else {
		fmt.Print(string(body)) // fallback
	}
	return nil
}

// processSinglePrompt is for non-interactive mode. It sends a single prompt and prints the response.
func processSinglePrompt(userInput string, cfg map[string]string, sysPromptContent, accessToken string) error {
	var messages []Message
	if sysPromptContent != "" {
		messages = append(messages, Message{Role: "system", Content: sysPromptContent})
	}
	messages = append(messages, Message{Role: "user", Content: userInput})

	payloadBytes, err := buildPayload(cfg, messages)
	if err != nil {
		return fmt.Errorf("build payload: %w", err)
	}

	url := cfg["BASE_URL"] + "/chat/completions"
	req, _ := http.NewRequest("POST", url, bytes.NewReader(payloadBytes))
	req.Header.Set("Authorization", "Bearer "+accessToken)
	req.Header.Set("Content-Type", "application/json")

	client := &http.Client{Timeout: 0}
	resp, err := client.Do(req)
	if err != nil {
		return fmt.Errorf("request failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode >= 400 {
		body, _ := ioutil.ReadAll(resp.Body)
		return fmt.Errorf("api error: %s\n%s", resp.Status, string(body))
	}

	if cfg["STREAM"] == "true" {
		return handleStreamQuiet(resp.Body)
	} else {
		body, _ := ioutil.ReadAll(resp.Body)
		return handleNonStreamQuiet(body)
	}
}
