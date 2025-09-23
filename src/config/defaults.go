// Path: src/config/defaults.go
// Rationale: This file contains the default configuration values for the application.

package main

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
		"deepseek-v3.1",
		"deepseek-r1-distill-qwen-32b",
		"deepseek-r1-distill-llama-8b",
		"deepseek-r1-0528",
	}
	apiEnvNames = []string{"NVIDIA_BUILD_AI_ACCESS_TOKEN", "NVIDIA_ACCESS_TOKEN", "ACCESS_TOKEN", "NVIDIA_API_KEY", "API_KEY"}
)
