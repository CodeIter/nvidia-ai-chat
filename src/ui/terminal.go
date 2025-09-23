// Path: src/ui/terminal.go
// Rationale: This file contains terminal-related functions and variables for colored output and user input.

package main

import (
	"os"
	"os/exec"
)

func tput(name string) string {
	out, err := exec.Command("tput", name).Output()
	if err != nil {
		return ""
	}
	return string(out)
}

var (
	bold   = tput("bold")
	normal = tput("sgr0")
	blue   = tput("setaf 4")
	green  = tput("setaf 2")
	red    = tput("setaf 1")
)

func openTTY() (*os.File, error) {
	// Opens /dev/tty to read user input per loop iteration (so reading until Ctrl+D doesn't break loops)
	return os.Open("/dev/tty")
}
