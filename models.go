package main

import (
	"fmt"
	"strings"
)

// ParameterType defines the type of a model parameter.
type ParameterType string

const (
	Float   ParameterType = "float"
	Int     ParameterType = "int"
	String  ParameterType = "string"
	Bool    ParameterType = "bool"
	StringA ParameterType = "string_array"
)

// ModelParameter defines the schema for a single model setting.
type ModelParameter struct {
	Type        ParameterType `json:"type"`
	Default     interface{}   `json:"default"`
	Min         float64       `json:"min,omitempty"`
	Max         float64       `json:"max,omitempty"`
	Options     []string      `json:"options,omitempty"`
	Description string        `json:"description"`
	APIKey      string        `json:"api_key"` // The key to use in the JSON payload for the API call.
}

// ModelDefinition holds all the parameters for a specific model.
type ModelDefinition struct {
	// Special properties for some models
	PrependedSystemMessageOnThinking string `json:"prepended_system_message_on_thinking,omitempty"`
	ChatTemplateKwargsThinking       bool   `json:"chat_template_kwargs_thinking,omitempty"`

	Parameters map[string]ModelParameter `json:"parameters"`
}

// ModelDefinitions is a map of all supported model definitions.
var ModelDefinitions = map[string]ModelDefinition{
	"openai/gpt-oss-120b": {
		Parameters: map[string]ModelParameter{
			"temperature":      {Type: Float, Default: 1.0, Min: 0, Max: 1, Description: "The sampling temperature to use for text generation. The higher the temperature value is, the less deterministic the output text will be. It is not recommended to modify both temperature and top_p in the same call.", APIKey: "temperature"},
			"top_p":            {Type: Float, Default: 1.0, Min: 0.01, Max: 1, Description: "The top-p sampling mass used for text generation. The top-p value determines the probability mass that is sampled at sampling time. For example, if top_p = 0.2, only the most likely tokens (summing to 0.2 cumulative probability) will be sampled. It is not recommended to modify both temperature and top_p in the same call.", APIKey: "top_p"},
			"frequency_penalty": {Type: Float, Default: 0.0, Min: -2, Max: 2, Description: "Indicates how much to penalize new tokens based on their existing frequency in the text so far, decreasing model likelihood to repeat the same line verbatim.", APIKey: "frequency_penalty"},
			"presence_penalty": {Type: Float, Default: 0.0, Min: -2, Max: 2, Description: "Positive values penalize new tokens based on whether they appear in the text so far, increasing model likelihood to talk about new topics.", APIKey: "presence_penalty"},
			"max_tokens":       {Type: Int, Default: 4096, Min: 1, Max: 4096, Description: "The maximum number of tokens to generate in any given call. Note that the model is not aware of this value, and generation will simply stop at the number of tokens specified.", APIKey: "max_tokens"},
			"stop":             {Type: StringA, Default: "", Description: "A string or a list of strings where the API will stop generating further tokens. The returned text will not contain the stop sequence.", APIKey: "stop"},
			"reasoning_effort": {Type: String, Default: "medium", Options: []string{"low", "medium", "high"}, Description: "Controls the effort level for reasoning in reasoning-capable models. 'low' provides basic reasoning, 'medium' provides balanced reasoning, and 'high' provides detailed step-by-step reasoning.", APIKey: "reasoning_effort"},
		},
	},
	"bytedance/seed-oss-36b-instruct": {
		Parameters: map[string]ModelParameter{
			"temperature":      {Type: Float, Default: 1.1, Min: 0, Max: 2, Description: "The sampling temperature to use for text generation.", APIKey: "temperature"},
			"top_p":            {Type: Float, Default: 0.95, Min: 0.01, Max: 1, Description: "The top-p sampling mass used for text generation.", APIKey: "top_p"},
			"max_tokens":       {Type: Int, Default: 4096, Min: 1, Description: "The maximum number of tokens to generate.", APIKey: "max_tokens"},
			"thinking_budget":  {Type: Int, Default: -1, Min: -1, Max: 16384, Description: "Controls the token budget for the model's internal reasoning. Set to -1 for unlimited thinking (default), O for no thinking, or a positive integer to limit thinking tokens. Recommended values are multiples of 512. Must be less than max_tokens.", APIKey: "thinking_budget"},
			"frequency_penalty": {Type: Float, Default: 0.0, Min: -2, Max: 2, Description: "Indicates how much to penalize new tokens based on their existing frequency.", APIKey: "frequency_penalty"},
			"presence_penalty": {Type: Float, Default: 0.0, Min: -2, Max: 2, Description: "Positive values penalize new tokens based on whether they appear in the text so far.", APIKey: "presence_penalty"},
			"stop":             {Type: StringA, Default: "", Description: "Stop sequences.", APIKey: "stop"},
			"seed":             {Type: Int, Default: 0, Description: "Seed for reproducibility. Default 0 means not included.", APIKey: "seed"},
		},
	},
	"qwen/qwen3-coder-480b-a35b-instruct": {
		Parameters: map[string]ModelParameter{
			"temperature":      {Type: Float, Default: 0.7, Min: 0, Max: 1, Description: "Sampling temperature.", APIKey: "temperature"},
			"top_p":            {Type: Float, Default: 0.8, Min: 0.01, Max: 1, Description: "Top-p sampling.", APIKey: "top_p"},
			"frequency_penalty": {Type: Float, Default: 0.0, Min: -2, Max: 2, Description: "Frequency penalty.", APIKey: "frequency_penalty"},
			"presence_penalty": {Type: Float, Default: 0.0, Min: -2, Max: 2, Description: "Presence penalty.", APIKey: "presence_penalty"},
			"max_tokens":       {Type: Int, Default: 4096, Min: 1, Max: 16384, Description: "Maximum tokens to generate.", APIKey: "max_tokens"},
			"stop":             {Type: StringA, Default: "", Description: "Stop sequences.", APIKey: "stop"},
		},
	},
	"nvidia/nvidia-nemotron-nano-9b-v2": {
		PrependedSystemMessageOnThinking: "/think",
		Parameters: map[string]ModelParameter{
			"temperature":         {Type: Float, Default: 0.6, Min: 0, Max: 1, Description: "Sampling temperature.", APIKey: "temperature"},
			"top_p":               {Type: Float, Default: 0.95, Min: 0.01, Max: 1, Description: "Top-p sampling.", APIKey: "top_p"},
			"max_tokens":          {Type: Int, Default: 2048, Min: 1, Max: 8192, Description: "Maximum tokens to generate.", APIKey: "max_tokens"},
			"min_thinking_tokens": {Type: Int, Default: 1024, Min: 1, Max: 4096, Description: "The minimum number of tokens the model should use for internal reasoning. Must be less than max_thinking_tokens. Ignored when '/no_think' is in the system message.", APIKey: "min_thinking_tokens"},
			"max_thinking_tokens": {Type: Int, Default: 2048, Min: 1, Max: 4096, Description: "The maximum number of tokens the model can use for internal reasoning. Must be greater than min_thinking_tokens. Ignored when '/no_think' is in the system message.", APIKey: "max_thinking_tokens"},
			"frequency_penalty":   {Type: Float, Default: 0.0, Min: -2, Max: 2, Description: "Frequency penalty.", APIKey: "frequency_penalty"},
			"presence_penalty":    {Type: Float, Default: 0.0, Min: -2, Max: 2, Description: "Presence penalty.", APIKey: "presence_penalty"},
			"stop":                {Type: StringA, Default: "", Description: "Stop sequences.", APIKey: "stop"},
			"seed":                {Type: Int, Default: 0, Description: "Seed for reproducibility. Default 0 means not included.", APIKey: "seed"},
		},
	},
	"nvidia/llama-3.3-nemotron-super-49b-v1.5": {
		PrependedSystemMessageOnThinking: "/think",
		Parameters: map[string]ModelParameter{
			"temperature":      {Type: Float, Default: 0.6, Min: 0, Max: 1, Description: "Sampling temperature.", APIKey: "temperature"},
			"top_p":            {Type: Float, Default: 0.95, Min: 0.01, Max: 1, Description: "Top-p sampling.", APIKey: "top_p"},
			"max_tokens":       {Type: Int, Default: 65536, Min: 1, Description: "Maximum tokens to generate.", APIKey: "max_tokens"},
			"frequency_penalty": {Type: Float, Default: 0.0, Min: -2, Max: 2, Description: "Frequency penalty.", APIKey: "frequency_penalty"},
			"presence_penalty": {Type: Float, Default: 0.0, Min: -2, Max: 2, Description: "Presence penalty.", APIKey: "presence_penalty"},
			"stop":             {Type: StringA, Default: "", Description: "Stop sequences.", APIKey: "stop"},
			"seed":             {Type: Int, Default: 0, Description: "Seed for reproducibility. Default 0 means not included.", APIKey: "seed"},
			"thinking":         {Type: Bool, Default: false, Description: "Enable thinking mode. Prepends a system message to enable/disable thinking.", APIKey: ""}, // Not a direct API key
		},
	},
	"mistralai/mistral-nemotron": {
		Parameters: map[string]ModelParameter{
			"temperature":      {Type: Float, Default: 0.6, Min: 0, Max: 1, Description: "Sampling temperature.", APIKey: "temperature"},
			"top_p":            {Type: Float, Default: 0.7, Min: 0.01, Max: 1, Description: "Top-p sampling.", APIKey: "top_p"},
			"frequency_penalty": {Type: Float, Default: 0.0, Min: -2, Max: 2, Description: "Frequency penalty.", APIKey: "frequency_penalty"},
			"presence_penalty": {Type: Float, Default: 0.0, Min: -2, Max: 2, Description: "Presence penalty.", APIKey: "presence_penalty"},
			"max_tokens":       {Type: Int, Default: 4096, Min: 1, Max: 4096, Description: "Maximum tokens to generate.", APIKey: "max_tokens"},
			"stop":             {Type: StringA, Default: "", Description: "Stop sequences.", APIKey: "stop"},
		},
	},
	"mistralai/mistral-small-24b-instruct": {
		Parameters: map[string]ModelParameter{
			"temperature":      {Type: Float, Default: 0.2, Min: 0, Max: 1, Description: "Sampling temperature.", APIKey: "temperature"},
			"top_p":            {Type: Float, Default: 0.7, Min: 0.01, Max: 1, Description: "Top-p sampling.", APIKey: "top_p"},
			"frequency_penalty": {Type: Float, Default: 0.0, Min: -2, Max: 2, Description: "Frequency penalty.", APIKey: "frequency_penalty"},
			"presence_penalty": {Type: Float, Default: 0.0, Min: -2, Max: 2, Description: "Presence penalty.", APIKey: "presence_penalty"},
			"max_tokens":       {Type: Int, Default: 1024, Min: 1, Max: 8192, Description: "Maximum tokens to generate.", APIKey: "max_tokens"},
			"stop":             {Type: StringA, Default: "", Description: "Stop sequences.", APIKey: "stop"},
		},
	},
	"deepseek-ai/deepseek-v3.1": {
		ChatTemplateKwargsThinking: true,
		Parameters: map[string]ModelParameter{
			"temperature": {Type: Float, Default: 0.2, Min: 0.01, Max: 1, Description: "Sampling temperature.", APIKey: "temperature"},
			"top_p":       {Type: Float, Default: 0.7, Min: 0.01, Max: 1, Description: "Top-p sampling.", APIKey: "top_p"},
			"max_tokens":  {Type: Int, Default: 8192, Min: 1, Max: 16384, Description: "Maximum tokens to generate.", APIKey: "max_tokens"},
			"stop":        {Type: StringA, Default: "", Description: "Stop sequences.", APIKey: "stop"},
			"seed":        {Type: Int, Default: nil, Description: "Seed for reproducibility. Omitted if not set.", APIKey: "seed"},
			"thinking":    {Type: Bool, Default: true, Description: "Enable thinking mode via chat_template_kwargs.", APIKey: ""}, // Not a direct API key
		},
	},
	"deepseek-ai/deepseek-r1-distill-qwen-32b": {
		Parameters: map[string]ModelParameter{
			"temperature":      {Type: Float, Default: 0.6, Min: 0, Max: 1, Description: "Sampling temperature.", APIKey: "temperature"},
			"top_p":            {Type: Float, Default: 0.7, Min: 0.01, Max: 1, Description: "Top-p sampling.", APIKey: "top_p"},
			"frequency_penalty": {Type: Float, Default: 0.0, Min: -2, Max: 2, Description: "Frequency penalty.", APIKey: "frequency_penalty"},
			"presence_penalty": {Type: Float, Default: 0.0, Min: -2, Max: 2, Description: "Presence penalty.", APIKey: "presence_penalty"},
			"max_tokens":       {Type: Int, Default: 4096, Min: 1, Max: 4096, Description: "Maximum tokens to generate.", APIKey: "max_tokens"},
			"stop":             {Type: StringA, Default: "", Description: "Stop sequences.", APIKey: "stop"},
		},
	},
	"deepseek-ai/deepseek-r1-distill-llama-8b": {
		Parameters: map[string]ModelParameter{
			"temperature":      {Type: Float, Default: 0.6, Min: 0, Max: 1, Description: "Sampling temperature.", APIKey: "temperature"},
			"top_p":            {Type: Float, Default: 0.7, Min: 0.01, Max: 1, Description: "Top-p sampling.", APIKey: "top_p"},
			"frequency_penalty": {Type: Float, Default: 0.0, Min: -2, Max: 2, Description: "Frequency penalty.", APIKey: "frequency_penalty"},
			"presence_penalty": {Type: Float, Default: 0.0, Min: -2, Max: 2, Description: "Presence penalty.", APIKey: "presence_penalty"},
			"max_tokens":       {Type: Int, Default: 4096, Min: 1, Max: 4096, Description: "Maximum tokens to generate.", APIKey: "max_tokens"},
			"stop":             {Type: StringA, Default: "", Description: "Stop sequences.", APIKey: "stop"},
		},
	},
	"deepseek-ai/deepseek-r1-0528": {
		Parameters: map[string]ModelParameter{
			"temperature":      {Type: Float, Default: 0.6, Min: 0, Max: 1, Description: "Sampling temperature.", APIKey: "temperature"},
			"top_p":            {Type: Float, Default: 0.7, Min: 0.01, Max: 1, Description: "Top-p sampling.", APIKey: "top_p"},
			"frequency_penalty": {Type: Float, Default: 0.0, Min: -2, Max: 2, Description: "Frequency penalty.", APIKey: "frequency_penalty"},
			"presence_penalty": {Type: Float, Default: 0.0, Min: -2, Max: 2, Description: "Presence penalty.", APIKey: "presence_penalty"},
			"max_tokens":       {Type: Int, Default: 4096, Min: 1, Max: 4096, Description: "Maximum tokens to generate.", APIKey: "max_tokens"},
			"stop":             {Type: StringA, Default: "", Description: "Stop sequences.", APIKey: "stop"},
		},
	},
	"qwen/qwen3-next-80b-a3b-instruct": {
		Parameters: map[string]ModelParameter{
			"temperature":      {Type: Float, Default: 0.6, Min: 0, Max: 1, Description: "Sampling temperature.", APIKey: "temperature"},
			"top_p":            {Type: Float, Default: 0.7, Min: 0.01, Max: 1, Description: "Top-p sampling.", APIKey: "top_p"},
			"frequency_penalty": {Type: Float, Default: 0.0, Min: -2, Max: 2, Description: "Frequency penalty.", APIKey: "frequency_penalty"},
			"presence_penalty": {Type: Float, Default: 0.0, Min: -2, Max: 2, Description: "Presence penalty.", APIKey: "presence_penalty"},
			"max_tokens":       {Type: Int, Default: 4096, Min: 1, Max: 4096, Description: "Maximum tokens to generate.", APIKey: "max_tokens"},
			"stop":             {Type: StringA, Default: "", Description: "Stop sequences.", APIKey: "stop"},
		},
	},
	"qwen/qwen3-next-80b-a3b-thinking": {
		Parameters: map[string]ModelParameter{
			"temperature":      {Type: Float, Default: 0.6, Min: 0, Max: 1, Description: "Sampling temperature.", APIKey: "temperature"},
			"top_p":            {Type: Float, Default: 0.7, Min: 0.01, Max: 1, Description: "Top-p sampling.", APIKey: "top_p"},
			"frequency_penalty": {Type: Float, Default: 0.0, Min: -2, Max: 2, Description: "Frequency penalty.", APIKey: "frequency_penalty"},
			"presence_penalty": {Type: Float, Default: 0.0, Min: -2, Max: 2, Description: "Presence penalty.", APIKey: "presence_penalty"},
			"max_tokens":       {Type: Int, Default: 4096, Min: 1, Max: 4096, Description: "Maximum tokens to generate.", APIKey: "max_tokens"},
			"stop":             {Type: StringA, Default: "", Description: "Stop sequences.", APIKey: "stop"},
		},
	},
	"moonshotai/kimi-k2-instruct-0905": {
		Parameters: map[string]ModelParameter{
			"temperature": {Type: Float, Default: 0.6, Min: 0, Max: 1, Description: "Sampling temperature.", APIKey: "temperature"},
			"top_p":       {Type: Float, Default: 0.9, Min: 0.01, Max: 1, Description: "Top-p sampling.", APIKey: "top_p"},
			"max_tokens":  {Type: Int, Default: 4096, Min: 1, Max: 16384, Description: "Maximum tokens to generate.", APIKey: "max_tokens"},
			"stop":        {Type: StringA, Default: "", Description: "Stop sequences.", APIKey: "stop"},
		},
	},
	"google/codegemma-7b": {
		Parameters: map[string]ModelParameter{
			"temperature": {Type: Float, Default: 0.5, Min: 0, Max: 1, Description: "Sampling temperature.", APIKey: "temperature"},
			"top_p":       {Type: Float, Default: 1.0, Min: 0, Max: 1, Description: "Top-p sampling.", APIKey: "top_p"},
			"max_tokens":  {Type: Int, Default: 1024, Min: 1, Max: 1024, Description: "Maximum tokens to generate.", APIKey: "max_tokens"},
			"stop":        {Type: StringA, Default: "", Description: "Stop sequences.", APIKey: "stop"},
			"seed":        {Type: Int, Default: 0, Description: "Seed for reproducibility. Default 0 means not included.", APIKey: "seed"},
		},
	},
	"google/gemma-7b": {
		Parameters: map[string]ModelParameter{
			"temperature": {Type: Float, Default: 0.5, Min: 0, Max: 1, Description: "Sampling temperature.", APIKey: "temperature"},
			"top_p":       {Type: Float, Default: 1.0, Min: 0, Max: 1, Description: "Top-p sampling.", APIKey: "top_p"},
			"max_tokens":  {Type: Int, Default: 1024, Min: 1, Max: 1024, Description: "Maximum tokens to generate.", APIKey: "max_tokens"},
			"stop":        {Type: StringA, Default: "", Description: "Stop sequences.", APIKey: "stop"},
		},
	},
	"mistralai/mixtral-8x22b-instruct-v0.1": {
		Parameters: map[string]ModelParameter{
			"temperature": {Type: Float, Default: 0.5, Min: 0, Max: 1, Description: "Sampling temperature.", APIKey: "temperature"},
			"top_p":       {Type: Float, Default: 1.0, Min: 0, Max: 1, Description: "Top-p sampling.", APIKey: "top_p"},
			"max_tokens":  {Type: Int, Default: 1024, Min: 1, Max: 1024, Description: "Maximum tokens to generate.", APIKey: "max_tokens"},
			"stop":        {Type: StringA, Default: "", Description: "Stop sequences.", APIKey: "stop"},
			"seed":        {Type: Int, Default: 0, Description: "Seed for reproducibility. Default 0 means not included.", APIKey: "seed"},
		},
	},
	"others": { // Generic model for fallback
		Parameters: map[string]ModelParameter{
			"temperature":      {Type: Float, Default: 0.5, Min: 0, Max: 1, Description: "Sampling temperature.", APIKey: "temperature"},
			"top_p":            {Type: Float, Default: 1.0, Min: 0, Max: 1, Description: "Top-p sampling.", APIKey: "top_p"},
			"max_tokens":       {Type: Int, Default: 1024, Min: 1, Description: "Maximum tokens to generate.", APIKey: "max_tokens"},
			"frequency_penalty": {Type: Float, Default: 0.0, Min: -2, Max: 2, Description: "Frequency penalty.", APIKey: "frequency_penalty"},
			"presence_penalty": {Type: Float, Default: 0.0, Min: -2, Max: 2, Description: "Presence penalty.", APIKey: "presence_penalty"},
			"stop":             {Type: StringA, Default: "", Description: "Stop sequences.", APIKey: "stop"},
		},
	},
}

// GetModelDefinition returns the definition for a given model, or the generic definition if not found.
func GetModelDefinition(modelName string) ModelDefinition {
	if def, ok := ModelDefinitions[modelName]; ok {
		return def
	}
	return ModelDefinitions["others"]
}

// Format a description of a model's parameters for help text.
func (md ModelDefinition) FormatForHelp() string {
	var builder strings.Builder
	for name, param := range md.Parameters {
		builder.WriteString(fmt.Sprintf("  --%s ", name))
		switch param.Type {
		case Float:
			builder.WriteString(fmt.Sprintf("<%.2f..%.2f>", param.Min, param.Max))
		case Int:
			if param.Max > 0 {
				builder.WriteString(fmt.Sprintf("<%d..%d>", int(param.Min), int(param.Max)))
			} else {
				builder.WriteString(fmt.Sprintf("<%d..>", int(param.Min)))
			}
		case String:
			if len(param.Options) > 0 {
				builder.WriteString(fmt.Sprintf("<%s>", strings.Join(param.Options, "|")))
			} else {
				builder.WriteString("<string>")
			}
		case Bool:
			builder.WriteString("<true|false>")
		case StringA:
			builder.WriteString("<string>")
		}

		builder.WriteString(fmt.Sprintf(" (default: %v)\n", param.Default))
		builder.WriteString(fmt.Sprintf("    %s\n", param.Description))
	}
	return builder.String()
}