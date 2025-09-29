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
	rm -f $(APP_NAME) $(APP_NAME).exe

