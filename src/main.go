// Path: src/main.go
// Rationale: This file contains the main function, which is the entry point of the application.

package main

import (
	"bytes"
	"fmt"
	"io"
	"io/ioutil"
	"math/rand"
	"net/http"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"
)

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

	// interactive loop
	for {
		fmt.Fprintf(os.Stderr, "\n%s: ", blue+"You"+normal)

		// open /dev/tty for per-iteration multi-line input terminated by Ctrl+D
		tty, err := openTTY()
		if err != nil {
			// fallback: read from stdin (non-interactive)
			inputBytes, err := ioutil.ReadAll(os.Stdin)
			if err != nil && err != io.EOF {
				fmt.Fprintf(os.Stderr, "%sFailed reading input: %v%s\n", red, err, normal)
				return
			}
			userInput := strings.TrimRight(string(inputBytes), "\r\n")
			if userInput == "" {
				// EOF with no input -> exit
				fmt.Println()
				return
			}
			// handle commands and requests below
			if handleInteractiveInput(userInput, convFile, cfg) {
				// if it returned true, continue loop
				continue
			}
			// proceed to send message
			if err := processMessage(userInput, convFile, cfg, sysPromptContent, ACCESS_TOKEN); err != nil {
				fmt.Fprintf(os.Stderr, "%sError: %v%s\n", red, err, normal)
			}
			return
		}

		// read all from tty until EOF (Ctrl+D)
		inputBytes, err := ioutil.ReadAll(tty)
		tty.Close()
		if err != nil && err != io.EOF {
			fmt.Fprintf(os.Stderr, "%sFailed reading input: %v%s\n", red, err, normal)
			return
		}
		userInput := strings.TrimRight(string(inputBytes), "\r\n")
		if userInput == "" {
			// if user immediately sends EOF without typing -> exit
			fmt.Println()
			return
		}
		// handle commands
		if handled := handleInteractiveInput(userInput, convFile, cfg); handled {
			continue
		}
		// append user message
		if err := appendMessage(convFile, "user", userInput); err != nil {
			fmt.Fprintf(os.Stderr, "%sFailed appending message: %v%s\n", red, err, normal)
			continue
		}
		// re-check limit
		count, _ = messageCount(convFile)
		limit, _ = strconv.Atoi(cfg["HISTORY_LIMIT"])
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
