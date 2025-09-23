// Path: src/api/client.go
// Rationale: This file contains functions responsible for making API calls.

package main

import (
	"bytes"
	"fmt"
	"io/ioutil"
	"net/http"
	"os"
	"strconv"
)

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
