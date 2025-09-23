// Path: src/utils/converters.go
// Rationale: This file contains utility functions for string conversions.

package main

import "strconv"

// helpers
func mustAtoi(s string, def int) int {
	if v, err := strconv.Atoi(s); err == nil {
		return v
	}
	return def
}
func mustParseFloat(s string, def float64) float64 {
	if v, err := strconv.ParseFloat(s, 64); err == nil {
		return v
	}
	return def
}
