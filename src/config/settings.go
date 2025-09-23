// Path: src/config/settings.go
// Rationale: This file contains the Settings struct and related functions for managing application settings.

package main

import (
	"fmt"
	"strconv"
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
