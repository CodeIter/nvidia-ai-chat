#!/bin/bash
set -ex
go build -o nvidia-chat-refactored ./src/api/client.go ./src/api/payload.go ./src/api/stream.go ./src/config/defaults.go ./src/config/settings.go ./src/conversation/conversation.go ./src/main.go ./src/ui/help.go ./src/ui/interactive.go ./src/ui/terminal.go ./src/utils/converters.go ./src/utils/files.go
