// Path: src/api/stream.go
// Rationale: This file contains structs and functions for handling API stream responses.

package main

import (
	"bufio"
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"strings"
)

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
				fmt.Printf("\n%s\n", green+"[/End of Assistant Reasoning]"+normal)
				assistantTextBuf.WriteString("\n[/End of Assistant Reasoning]\n")
				inReasoning = false
			}
			fmt.Print(content)
			assistantTextBuf.WriteString(content)
		}
	}

	if inReasoning {
		fmt.Printf("\n%s\n", green+"[/End of Assistant Reasoning]"+normal)
		assistantTextBuf.WriteString("\n[/End of Assistant Reasoning]\n")
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
		fmt.Printf("\n%s\n", green+"[/End of Assistant Reasoning]"+normal)
		outBuf.WriteString("[Begin of Assistant Reasoning]")
		outBuf.WriteString(reasoning)
		outBuf.WriteString("[End of Assistant Reasoning]")
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
