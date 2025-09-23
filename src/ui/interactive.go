// Path: src/ui/interactive.go
// Rationale: This file contains functions for handling interactive commands.

package main

import (
	"fmt"
	"io/ioutil"
	"math/rand"
	"os"
	"strconv"
	"strings"
)

// Returns true if the command was a special/handled interactive command
// handleInteractiveInput returns true if the input was a special command that was handled here.
// Otherwise returns false so the caller will continue normal message processing.
func exportLastN(n int, convFile, targetFile string) error {
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

	content := strings.Join(aiResponses, "\n\n---\n\n")
	return ioutil.WriteFile(targetFile, []byte(content), 0o644)
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
		if len(parts) < 2 {
			fmt.Fprintln(os.Stderr, "Usage: /exportlast <file>")
			return true
		}
		targetFile := parts[1]
		if err := exportLastN(1, convFile, targetFile); err != nil {
			fmt.Fprintf(os.Stderr, "%sFailed to export: %v%s\n", red, err, normal)
		} else {
			fmt.Fprintf(os.Stderr, "%sExported last response to %s%s\n", green, targetFile, normal)
		}
		return true
	case "/exportn":
		if len(parts) < 3 {
			fmt.Fprintln(os.Stderr, "Usage: /exportn <n> <file>")
			return true
		}
		n, err := strconv.Atoi(parts[1])
		if err != nil || n <= 0 {
			fmt.Fprintf(os.Stderr, "%sInvalid number: %s%s\n", red, parts[1], normal)
			return true
		}
		targetFile := parts[2]
		if err := exportLastN(n, convFile, targetFile); err != nil {
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
		fmt.Fprintln(os.Stderr, "  /exportlast <file>: Export last AI response to a markdown file")
		fmt.Fprintln(os.Stderr, "  /exportn <n> <file>: Export last n AI responses to a markdown file")
		fmt.Fprintln(os.Stderr, "  /randomodel: Switch to a random model")
		fmt.Fprintln(os.Stderr, "  /help: Show this help message")
		return true

	default:
		return false
	}
}
