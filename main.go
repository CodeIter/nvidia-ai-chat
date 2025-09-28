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
	"sort"
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
	defaultReasoning     = "low"
	defaultStop          = ""
	defaultHistorySubdir = ".cache/nvidia-chat"
	defaultHistoryLimit  = 40
	modelsList           = []string{
		"openai/gpt-oss-120b",
		"bytedance/seed-oss-36b-instruct",
		"qwen/qwen3-coder-480b-a35b-instruct",
		"nvidia/nvidia-nemotron-nano-9b-v2",
		"nvidia/llama-3.3-nemotron-super-49b-v1.5",
		"mistralai/mistral-nemotron",
		"mistralai/mistral-small-24b-instruct",
		"deepseek-ai/deepseek-v3.1",
		"deepseek-ai/deepseek-r1-distill-qwen-32b",
		"deepseek-ai/deepseek-r1-distill-llama-8b",
		"deepseek-ai/deepseek-r1-0528",
		"qwen/qwen3-next-80b-a3b-instruct",
		"qwen/qwen3-next-80b-a3b-thinking",
		"moonshotai/kimi-k2-instruct-0905",
		"google/codegemma-7b",
		"google/gemma-7b",
		"mistralai/mixtral-8x22b-instruct-v0.1",
	}
	apiEnvNames = []string{"NVIDIA_BUILD_AI_ACCESS_TOKEN", "NVIDIA_ACCESS_TOKEN", "ACCESS_TOKEN", "NVIDIA_API_KEY", "API_KEY"}
)

// ModelSettings represents the settings for a single model or the default settings.
// It's a map to flexibly accommodate various parameters across different models.
type ModelSettings map[string]interface{}

// TopLevelSettings holds the overall settings in the conversation file.
type TopLevelSettings struct {
	Stream       bool                   `json:"stream"`
	HistoryLimit int                    `json:"history_limit"`
	Default      ModelSettings          `json:"default"`
	Models       map[string]ModelSettings `json:"models"`
}

type Message struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// ConversationFile is the top-level structure for the conversation JSON file.
type ConversationFile struct {
	System   string           `json:"system"`
	Settings TopLevelSettings `json:"settings"`
	Messages []Message        `json:"messages"`
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
	var builder strings.Builder

	// --- Usage ---
	builder.WriteString(fmt.Sprintf("%snvidia-chat (go)%s\n", bold, normal))
	builder.WriteString("Usage: nvidia-chat [OPTIONS] [CONVERSATION_FILE]\n\n")
	builder.WriteString(fmt.Sprintf("If CONVERSATION_FILE is omitted, one will be created at:\n  %s/conversation-<timestamp>.json\nand its path will be printed.\n\n", cfg["HISTORY_DIR"]))

	// --- General Options ---
	builder.WriteString(fmt.Sprintf("%sGeneral Options:%s\n", bold, normal))
	builder.WriteString(fmt.Sprintf("  -m, --model NAME      Model ID to use (default: %s)\n", defaultModel))
	builder.WriteString("  -s, --sys-prompt-file PATH\n                        Path to system prompt text file (content used for this run).\n")
	builder.WriteString("  -S                    Persist the -s content into the conversation file's 'system' field.\n")
	builder.WriteString("  --save-settings       Persist current model settings into the conversation file.\n")
	builder.WriteString("  -k, --access-token KEY\n                        Provide API key (overrides environment variables).\n")
	builder.WriteString("  --prompt TEXT|FILE|-\n                        Non-interactive mode: provide a prompt and print the response.\n")
	builder.WriteString("  -l, --list            List supported models and exit.\n")
	builder.WriteString("  --modelinfo NAME      Show detailed settings for a specific model and exit.\n")
	builder.WriteString("  -h, --help            Show this help.\n\n")

	// --- Model Setting Options (Dynamic) ---
	builder.WriteString(fmt.Sprintf("%sModel Setting Options:%s\n", bold, normal))
	builder.WriteString("These flags override settings for the current session. For model-specific ranges and defaults, use `/modelinfo <model_name>`.\n\n")

	// Collect all unique parameters from all models
	allParams := make(map[string]ModelParameter)
	paramOrder := []string{}
	for _, modelDef := range ModelDefinitions {
		for name, param := range modelDef.Parameters {
			if _, exists := allParams[name]; !exists {
				allParams[name] = param
				paramOrder = append(paramOrder, name)
			}
		}
	}
	sort.Strings(paramOrder)

	// Add global settings to the list
	paramOrder = append([]string{"stream", "history_limit"}, paramOrder...)
	allParams["stream"] = ModelParameter{Type: Bool, Default: true, Description: "Enable or disable streaming responses."}
	allParams["history_limit"] = ModelParameter{Type: Int, Default: defaultHistoryLimit, Description: "Maximum number of messages in conversation history."}

	for _, name := range paramOrder {
		param := allParams[name]
		flagName := strings.ReplaceAll(name, "_", "-")
		builder.WriteString(fmt.Sprintf("  --%s VALUE\n", flagName))
		builder.WriteString(fmt.Sprintf("      %s\n", param.Description))
		builder.WriteString(fmt.Sprintf("      To unset, use the interactive command: /%s unset\n\n", name))
	}

	// --- Interactive Commands ---
	builder.WriteString(fmt.Sprintf("%sInteractive Commands:%s\n", bold, normal))
	builder.WriteString("  /help                 Show this help message.\n")
	builder.WriteString("  /exit, /quit          Exit the program.\n")
	builder.WriteString("  /history              Print full conversation JSON.\n")
	builder.WriteString("  /clear                Clear conversation messages.\n")
	builder.WriteString("  /save <file>          Save conversation to a new file.\n")
	builder.WriteString("  /model <model_name>   Switch model for the session.\n")
	builder.WriteString("  /modelinfo <name>     List settings for a specific model.\n")
	builder.WriteString("  /persist-settings     Save the current session's settings to the conversation file.\n")
	builder.WriteString("  /persist-system <file>\n                        Persist a system prompt from a file.\n")
	builder.WriteString("  /exportlast [-t] <file>\n                        Export last AI response to a markdown file (-t filters thinking).\n")
	builder.WriteString("  /exportlastn [-t] <n> <file>\n                        Export last n AI responses.\n")
	builder.WriteString("  /exportn [-t] <n> <file>\n                        Export the Nth-to-last AI response.\n")
	builder.WriteString("  /randomodel           Switch to a random supported model.\n\n")
	builder.WriteString("For any model setting, you can use `/setting_name <value>` or `/setting_name unset`.\n")
	builder.WriteString("For example: `/temperature 0.8`, `/stop unset`\n\n")

	fmt.Print(builder.String())
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
		limit, _ := strconv.Atoi(cfg["HISTORY_LIMIT"])

		// Create default settings based on the generic model definition
		defaultSettings := make(ModelSettings)
		genericDef := GetModelDefinition("others")
		for name, param := range genericDef.Parameters {
			defaultSettings[name] = param.Default
		}

		s := TopLevelSettings{
			Stream:       stream,
			HistoryLimit: limit,
			Default:      defaultSettings,
			Models:       make(map[string]ModelSettings),
		}
		// Add the specific default model to the models map
		s.Models[defaultModel] = ModelSettings{
			"temperature":       mustParseFloat(defaultTemperature, 1.0),
			"top_p":             mustParseFloat(defaultTopP, 1.0),
			"frequency_penalty": mustParseFloat(defaultFrequency, 0),
			"presence_penalty":  mustParseFloat(defaultPresence, 0),
			"max_tokens":        mustAtoi(defaultMaxTokens, 4096),
			"reasoning_effort":  defaultReasoning,
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
	var cf ConversationFile
	if err := json.Unmarshal(data, &cf); err != nil {
		// back up and recreate
		backup := path + ".bak." + strconv.FormatInt(time.Now().Unix(), 10)
		_ = os.Rename(path, backup)
		fmt.Fprintf(os.Stderr, "Warning: Conversation file at %s was malformed. Backed up to %s and creating a new one.\n", path, backup)
		return ensureHistoryFileStructure(path, cfg)
	}

	// Basic validation of structure
	if cf.Messages == nil || cf.Settings.Default == nil || cf.Settings.Models == nil {
		backup := path + ".bak." + strconv.FormatInt(time.Now().Unix(), 10)
		_ = os.Rename(path, backup)
		fmt.Fprintf(os.Stderr, "Warning: Conversation file at %s was missing required fields. Backed up to %s and creating a new one.\n", path, backup)
		return ensureHistoryFileStructure(path, cfg)
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

	modelName := cfg["MODEL"]
	modelDef := GetModelDefinition(modelName)

	// Get current model settings or initialize if not present
	modelSettings, ok := cf.Settings.Models[modelName]
	if !ok {
		modelSettings = make(ModelSettings)
	}

	// Update settings for the current model from the session config (cfg)
	for key, paramDef := range modelDef.Parameters {
		if valStr, ok := cfg[strings.ToUpper(key)]; ok {
			// Convert string value from cfg to the correct type
			switch paramDef.Type {
			case Float:
				val, err := strconv.ParseFloat(valStr, 64)
				if err == nil {
					modelSettings[key] = val
				}
			case Int:
				val, err := strconv.Atoi(valStr)
				if err == nil {
					modelSettings[key] = val
				}
			case String, StringA:
				modelSettings[key] = valStr
			case Bool:
				val, err := strconv.ParseBool(valStr)
				if err == nil {
					modelSettings[key] = val
				}
			}
		}
	}

	// Save the updated model-specific settings
	cf.Settings.Models[modelName] = modelSettings

	// Also save global settings
	cf.Settings.Stream = cfg["STREAM"] == "true"
	cf.Settings.HistoryLimit = mustAtoi(cfg["HISTORY_LIMIT"], defaultHistoryLimit)

	return writeConversation(path, cf)
}

func applyFileSettingsAsDefaults(path string, cfg map[string]string, provided map[string]bool) error {
	cf, err := readConversation(path)
	if err != nil {
		return err
	}

	modelName := cfg["MODEL"]

	// Get the settings for the current model, falling back to default settings.
	settings, ok := cf.Settings.Models[modelName]
	if !ok {
		settings = cf.Settings.Default
	}

	// Apply model-specific settings if they were not provided via CLI flags.
	modelDef := GetModelDefinition(modelName)
	for key, paramDef := range modelDef.Parameters {
		configKey := strings.ToUpper(key)
		if !provided[configKey] {
			if value, exists := settings[key]; exists {
				// Convert the loaded value to a string for the cfg map
				switch paramDef.Type {
				case Float:
					if v, ok := value.(float64); ok {
						cfg[configKey] = fmt.Sprintf("%g", v)
					}
				case Int:
					// JSON unmarshals numbers into float64 by default
					if v, ok := value.(float64); ok {
						cfg[configKey] = fmt.Sprintf("%d", int(v))
					} else if v, ok := value.(int); ok {
						cfg[configKey] = fmt.Sprintf("%d", v)
					}
				case String, StringA:
					if v, ok := value.(string); ok {
						cfg[configKey] = v
					}
				case Bool:
					if v, ok := value.(bool); ok {
						cfg[configKey] = strconv.FormatBool(v)
					}
				}
			}
		}
	}

	// Apply global settings
	if !provided["STREAM"] {
		cfg["STREAM"] = strconv.FormatBool(cf.Settings.Stream)
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

// buildPayload constructs the JSON payload for the API call based on the current model's definition.
func buildPayload(cfg map[string]string, messages []Message) ([]byte, error) {
	modelName := cfg["MODEL"]
	modelDef := GetModelDefinition(modelName)

	payload := map[string]interface{}{
		"model":    modelName,
		"messages": messages,
		"stream":   cfg["STREAM"] == "true",
	}

	for key, paramDef := range modelDef.Parameters {
		// Skip parameters that are not part of the API payload (e.g., internal 'thinking' flag)
		if paramDef.APIKey == "" {
			continue
		}

		configKey := strings.ToUpper(key)
		valStr, ok := cfg[configKey]
		if !ok {
			continue // Should not happen if cfg is populated correctly from defaults
		}

		// Convert value and add to payload
		switch paramDef.Type {
		case Float:
			if val, err := strconv.ParseFloat(valStr, 64); err == nil {
				payload[paramDef.APIKey] = val
			}
		case Int:
			if val, err := strconv.Atoi(valStr); err == nil {
				// Special handling for seed=0, which usually means "omit"
				if key == "seed" && val == 0 {
					if modelName != "deepseek-ai/deepseek-v3.1" {
						continue // Omit for other models
					}
				}
				payload[paramDef.APIKey] = val
			}
		case String, StringA:
			// Don't send empty stop strings
			if key == "stop" && valStr == "" {
				continue
			}
			payload[paramDef.APIKey] = valStr
		case Bool:
			if val, err := strconv.ParseBool(valStr); err == nil {
				payload[paramDef.APIKey] = val
			}
		}
	}

	// Handle special payload structures like chat_template_kwargs
	if modelDef.ChatTemplateKwargsThinking {
		if thinking, err := strconv.ParseBool(cfg["THINKING"]); err == nil {
			payload["chat_template_kwargs"] = map[string]interface{}{"thinking": thinking}
		}
	}

	// Handle deepseek seed nil case. If seed wasn't in cfg, it won't be in payload yet.
	if modelName == "deepseek-ai/deepseek-v3.1" {
		if _, exists := payload["seed"]; !exists {
			payload["seed"] = nil
		}
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

	// Handle special thinking-related system messages
	modelDef := GetModelDefinition(cfg["MODEL"])
	if modelDef.PrependedSystemMessageOnThinking != "" {
		thinkingEnabled, _ := strconv.ParseBool(cfg["THINKING"])
		if thinkingEnabled {
			messages = append(messages, Message{Role: "system", Content: modelDef.PrependedSystemMessageOnThinking})
		} else if cfg["MODEL"] == "nvidia/llama-3.3-nemotron-super-49b-v1.5" { // Special case for disabling
			messages = append(messages, Message{Role: "system", Content: "/no_think"})
		}
	}

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
	MODEL_INFO_FLAG := "" // for --modelinfo

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
		case "--modelinfo":
			if val == "" {
				v, err := nextArg(&i)
				if err != nil {
					fmt.Fprintf(os.Stderr, "%s%s%s\n", red, err.Error(), normal)
					os.Exit(1)
				}
				val = v
			}
			MODEL_INFO_FLAG = val
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

	// If model info requested
	if MODEL_INFO_FLAG != "" {
		printModelInfo(MODEL_INFO_FLAG)
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
	fmt.Fprintf(os.Stderr, "%sNVIDIA chat (go)%s model=%s temperature=%s top_p=%s max_tokens=%s stream=%s freq_penalty=%s pres_penalty=%s reasoning=%s stop=%q\n\n", bold, normal, cfg["MODEL"], cfg["TEMPERATURE"], cfg["TOP_P"], cfg["MAX_TOKENS"], cfg["STREAM"], cfg["FREQUENCY_PENALTY"], cfg["PRESENCE_PENALTY"], cfg["REASONING_EFFORT"], cfg["STOP"])
	fmt.Fprintf(os.Stderr, "Conversation file: %s\n\n", convFile)
	fmt.Fprintln(os.Stderr, "Type your message and end it by Ctrl+D. See /help for commands")

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

func getModelInfoString(modelName string, modelDef ModelDefinition) string {
	var builder strings.Builder

	builder.WriteString(fmt.Sprintf("%sModel: %s%s\n\n", bold, modelName, normal))
	builder.WriteString(fmt.Sprintf("%sParameters:%s\n", bold, normal))

	paramNames := make([]string, 0, len(modelDef.Parameters))
	for name := range modelDef.Parameters {
		paramNames = append(paramNames, name)
	}
	sort.Strings(paramNames)

	for _, name := range paramNames {
		param := modelDef.Parameters[name]
		builder.WriteString(fmt.Sprintf("  %s%s%s\n", blue, name, normal))
		builder.WriteString(fmt.Sprintf("    Description: %s\n", param.Description))
		builder.WriteString(fmt.Sprintf("    Type: %s\n", param.Type))

		defaultStr := "Not set"
		if param.Default != nil {
			// Handle float formatting
			if f, ok := param.Default.(float64); ok {
				defaultStr = fmt.Sprintf("%g", f)
			} else {
				defaultStr = fmt.Sprintf("%v", param.Default)
			}
		} else {
			// Special case for deepseek seed
			if modelName == "deepseek-ai/deepseek-v3.1" && name == "seed" {
				defaultStr = "null (omitted)"
			}
		}

		builder.WriteString(fmt.Sprintf("    Default: %s\n", defaultStr))

		if param.Type == Float || param.Type == Int {
			hasMin := param.Min != 0 || (param.Type == Float && param.Min == 0.0)
			hasMax := param.Max != 0
			if hasMin && hasMax {
				builder.WriteString(fmt.Sprintf("    Range: %g to %g\n", param.Min, param.Max))
			} else if hasMin {
				builder.WriteString(fmt.Sprintf("    Range: >= %g\n", param.Min))
			} else if hasMax {
				builder.WriteString(fmt.Sprintf("    Range: <= %g\n", param.Max))
			}
		}

		if len(param.Options) > 0 {
			builder.WriteString(fmt.Sprintf("    Options: %s\n", strings.Join(param.Options, ", ")))
		}
		builder.WriteString("\n")
	}

	if modelDef.PrependedSystemMessageOnThinking != "" || modelDef.ChatTemplateKwargsThinking {
		builder.WriteString(fmt.Sprintf("%sSpecial Behavior:%s\n", bold, normal))
		if modelDef.PrependedSystemMessageOnThinking != "" {
			builder.WriteString(fmt.Sprintf("  - This model uses a system message to control thinking. Use `/thinking true` to enable.\n"))
		}
		if modelDef.ChatTemplateKwargsThinking {
			builder.WriteString(fmt.Sprintf("  - This model uses 'chat_template_kwargs' to control thinking. Use `/thinking true` to enable.\n"))
		}
	}
	return builder.String()
}

func printModelInfo(modelName string) {
	modelDef, exists := ModelDefinitions[modelName]
	if !exists {
		fmt.Fprintf(os.Stderr, "%sError: Model '%s' not found.%s\n", red, modelName, normal)
		fmt.Fprintf(os.Stderr, "Use the -l flag to list all supported models.\n")
		os.Exit(1)
	}

	info := getModelInfoString(modelName, modelDef)
	fmt.Print(info)
}

// validateParameter checks if a given value string is valid for a parameter.
func validateParameter(paramName, value string, modelDef ModelDefinition) error {
	param, ok := modelDef.Parameters[paramName]
	if !ok {
		// Also check global settings like stream/history_limit
		if paramName == "stream" {
			_, err := strconv.ParseBool(value)
			if err != nil {
				return fmt.Errorf("invalid boolean value for stream: %s", value)
			}
			return nil
		}
		if paramName == "history_limit" {
			v, err := strconv.Atoi(value)
			if err != nil || v < 0 {
				return fmt.Errorf("invalid non-negative integer for history_limit: %s", value)
			}
			return nil
		}
		return fmt.Errorf("unknown parameter: %s", paramName)
	}

	switch param.Type {
	case Float:
		v, err := strconv.ParseFloat(value, 64)
		if err != nil {
			return fmt.Errorf("invalid float value: %s", value)
		}
		if (param.Min != 0 || param.Max != 0) && (v < param.Min || v > param.Max) {
			return fmt.Errorf("value out of range [%g, %g]: %g", param.Min, param.Max, v)
		}
	case Int:
		v, err := strconv.Atoi(value)
		if err != nil {
			return fmt.Errorf("invalid integer value: %s", value)
		}
		if (param.Min != 0 || param.Max != 0) && (float64(v) < param.Min || float64(v) > param.Max) {
			return fmt.Errorf("value out of range [%d, %d]: %d", int(param.Min), int(param.Max), v)
		}
	case String:
		if len(param.Options) > 0 {
			found := false
			for _, opt := range param.Options {
				if value == opt {
					found = true
					break
				}
			}
			if !found {
				return fmt.Errorf("invalid option. Must be one of: %s", strings.Join(param.Options, ", "))
			}
		}
	case Bool:
		if _, err := strconv.ParseBool(value); err != nil {
			return fmt.Errorf("invalid boolean value (true/false): %s", value)
		}
	case StringA:
		// No specific validation for string arrays, any string is fine.
	}
	return nil
}

func handleInteractiveInput(userInput, convFile string, cfg map[string]string) bool {
	trimmed := strings.TrimSpace(userInput)
	parts := strings.Fields(trimmed)
	if len(parts) == 0 {
		return false
	}
	command := parts[0]
	if !strings.HasPrefix(command, "/") {
		return false
	}
	commandName := strings.TrimPrefix(command, "/")

	// --- Static commands ---
	switch commandName {
	case "exit", "quit":
		fmt.Fprintln(os.Stderr, "Bye.")
		os.Exit(0)
		return true
	case "history":
		b, err := ioutil.ReadFile(convFile)
		if err != nil {
			fmt.Fprintf(os.Stderr, "%sFailed reading conversation: %v%s\n", red, err, normal)
		} else {
			fmt.Fprintf(os.Stderr, "%s:\n%s\n", convFile, string(b))
		}
		return true
	case "clear":
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
	case "save":
		if len(parts) < 2 {
			fmt.Fprintln(os.Stderr, "Usage: /save <path>")
			return true
		}
		if err := copyFile(convFile, parts[1]); err != nil {
			fmt.Fprintf(os.Stderr, "%sFailed to save: %v%s\n", red, err, normal)
		} else {
			fmt.Fprintf(os.Stderr, "Saved to %s\n", parts[1])
		}
		return true
	case "persist-system":
		if len(parts) < 2 {
			fmt.Fprintln(os.Stderr, "Usage: /persist-system <file>")
			return true
		}
		path := parts[1]
		content, err := ioutil.ReadFile(path)
		if err != nil {
			fmt.Fprintf(os.Stderr, "%sFailed to read file: %v%s\n", red, err, normal)
			return true
		}
		if err := persistSystemToFile(convFile, string(content)); err != nil {
			fmt.Fprintf(os.Stderr, "%sFailed to persist system prompt: %v%s\n", red, err, normal)
		} else {
			fmt.Fprintf(os.Stderr, "%sPersisted system prompt from %s%s\n", green, path, normal)
		}
		return true
	case "persist-settings":
		if err := persistSettingsToFile(convFile, cfg); err != nil {
			fmt.Fprintf(os.Stderr, "%sFailed to persist settings: %v%s\n", red, err, normal)
		} else {
			fmt.Fprintf(os.Stderr, "%sPersisted current settings to %s%s\n", green, convFile, normal)
		}
		return true
	case "exportlast", "exportn", "exportlastn":
		filterThinking, newParts := parseTFlag(parts)
		var err error
		switch commandName {
		case "exportlast":
			if len(newParts) < 2 {
				fmt.Fprintln(os.Stderr, "Usage: /exportlast [-t] <file>")
				return true
			}
			err = exportLastN(1, convFile, newParts[1], filterThinking)
		case "exportn":
			if len(newParts) < 3 {
				fmt.Fprintln(os.Stderr, "Usage: /exportn [-t] <n> <file>")
				return true
			}
			n, _ := strconv.Atoi(newParts[1])
			err = exportNth(n, convFile, newParts[2], filterThinking)
		case "exportlastn":
			if len(newParts) < 3 {
				fmt.Fprintln(os.Stderr, "Usage: /exportlastn [-t] <n> <file>")
				return true
			}
			n, _ := strconv.Atoi(newParts[1])
			err = exportLastN(n, convFile, newParts[2], filterThinking)
		}
		if err != nil {
			fmt.Fprintf(os.Stderr, "%sFailed to export: %v%s\n", red, err, normal)
		} else {
			fmt.Fprintf(os.Stderr, "%sExport successful%s\n", green, normal)
		}
		return true
	case "randomodel":
		newModel := modelsList[rand.Intn(len(modelsList))]
		cfg["MODEL"] = newModel
		fmt.Fprintf(os.Stderr, "%sSwitched model to %s%s\n", green, newModel, normal)
		return true
	case "help":
		printHelp(cfg)
		return true
	case "model":
		if len(parts) < 2 {
			fmt.Fprintln(os.Stderr, "Usage: /model <model_name>")
			return true
		}
		modelName := parts[1]
		if _, exists := ModelDefinitions[modelName]; !exists {
			// Check if it's in the master list even if not in our detailed defs
			found := false
			for _, m := range modelsList {
				if m == modelName {
					found = true
					break
				}
			}
			if !found {
				fmt.Fprintf(os.Stderr, "%sModel '%s' not found in the list of supported models.%s\n", red, modelName, normal)
				return true
			}
		}
		cfg["MODEL"] = modelName
		fmt.Fprintf(os.Stderr, "%sModel set to %s%s\n", green, modelName, normal)
		return true
	case "modelinfo":
		if len(parts) < 2 {
			fmt.Fprintln(os.Stderr, "Usage: /modelinfo <model_name>")
			return true
		}
		modelName := parts[1]
		modelDef, exists := ModelDefinitions[modelName]
		if !exists {
			fmt.Fprintf(os.Stderr, "%sError: Model '%s' not found.%s\n", red, modelName, normal)
			return true
		}
		info := getModelInfoString(modelName, modelDef)
		fmt.Fprint(os.Stderr, info)
		return true
	}

	// --- Dynamic parameter setting commands ---
	modelDef := GetModelDefinition(cfg["MODEL"])
	if _, ok := modelDef.Parameters[commandName]; ok || commandName == "stream" || commandName == "history_limit" {
		if len(parts) < 2 {
			fmt.Fprintf(os.Stderr, "Usage: /%s <value> or /%s unset\n", commandName, commandName)
			return true
		}
		value := parts[1]
		configKey := strings.ToUpper(commandName)

		if value == "unset" {
			// Find the default value from the model definition and set it
			param, exists := modelDef.Parameters[commandName]
			if !exists {
				// Handle global settings
				if commandName == "stream" {
					cfg["STREAM"] = strconv.FormatBool(true)
				} else if commandName == "history_limit" {
					cfg["HISTORY_LIMIT"] = fmt.Sprintf("%d", defaultHistoryLimit)
				}
			} else {
				// Convert default value to string and set it in cfg
				defaultValStr := ""
				if f, ok := param.Default.(float64); ok {
					defaultValStr = fmt.Sprintf("%g", f)
				} else {
					defaultValStr = fmt.Sprintf("%v", param.Default)
				}
				cfg[configKey] = defaultValStr
			}
			fmt.Fprintf(os.Stderr, "%s%s unset (reverted to default)%s\n", green, commandName, normal)
		} else {
			// Validate and set the new value
			if err := validateParameter(commandName, value, modelDef); err != nil {
				fmt.Fprintf(os.Stderr, "%sError: %v%s\n", red, err, normal)
				return true
			}
			cfg[configKey] = value
			fmt.Fprintf(os.Stderr, "%s%s set to %s%s\n", green, commandName, value, normal)
		}
		return true
	}

	return false
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
