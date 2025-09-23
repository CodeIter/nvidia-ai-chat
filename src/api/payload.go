// Path: src/api/payload.go
// Rationale: This file contains the buildPayload function, which is responsible for creating the JSON payload for API requests.

package main

import (
	"encoding/json"
	"strconv"
)

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
