// Path: src/conversation/conversation.go
// Rationale: This file contains the ConversationFile and Message structs, along with functions for managing conversation files.

package main

import (
	"encoding/json"
	"io/ioutil"
	"os"
	"path/filepath"
	"strconv"
	"time"
)

type Message struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type ConversationFile struct {
	System   string    `json:"system"`
	Settings Settings  `json:"settings"`
	Messages []Message `json:"messages"`
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
