# Makefile for nvidia-ai-chat

APP_NAME := nvidia-ai-chat
BIN := $(APP_NAME)

# Detect OS for Windows binary name
ifeq ($(OS),Windows_NT)
	BIN := $(APP_NAME).exe
endif

.PHONY: all build run clean

all: build

build:
	@echo "Building $(BIN)..."
	go build -o $(BIN) .

run: build
	@echo "Running $(BIN)..."
	./$(BIN)

clean:
	@echo "Cleaning up..."
	rm -f $(APP_NAME) $(APP_NAME).exe nvidia-ai-chat-*-amd64 nvidia-ai-chat-*-amd64.exe

.PHONY: build-linux build-windows build-macos

build-linux:
	@echo "Building for Linux (amd64)..."
	GOOS=linux GOARCH=amd64 go build -o nvidia-ai-chat-linux-amd64 .

build-windows:
	@echo "Building for Windows (amd64)..."
	GOOS=windows GOARCH=amd64 go build -o nvidia-ai-chat-windows-amd64.exe .

build-macos:
	@echo "Building for macOS (amd64)..."
	GOOS=darwin GOARCH=amd64 go build -o nvidia-ai-chat-darwin-amd64 .

