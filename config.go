// Package gollm provides a high-level interface for interacting with various Language Learning Models (LLMs).
// This file re-exports configuration types and functions from the config package to provide
// a clean, centralized API for configuring LLM interactions.
package gollm

import (
	"github.com/teilomillet/gollm/config"
	"github.com/teilomillet/gollm/utils"
)

// Re-export core configuration types for easier access
type (
	// Config represents the complete configuration for LLM interactions.
	// It includes settings for model selection, API endpoints, generation parameters,
	// and runtime behavior. See config.Config for detailed field documentation.
	//
	// Example usage:
	//   cfg := NewConfig()
	//   cfg = ApplyOptions(cfg, SetProvider("openai"), SetModel("gpt-3.5-turbo"))
	Config = config.Config

	// ConfigOption is a function type that modifies a Config instance.
	// It enables a builder pattern for configuration, allowing for clean
	// and flexible configuration updates.
	//
	// Example usage:
	//   cfg := NewConfig()
	//   cfg = ApplyOptions(cfg, SetTemperature(0.7), SetMaxTokens(2048))
	ConfigOption = config.ConfigOption

	// LogLevel defines the verbosity of logging output.
	// Available levels are defined as constants: LogLevelOff through LogLevelDebug.
	//
	// Example usage:
	//   cfg := NewConfig()
	//   cfg = ApplyOptions(cfg, SetLogLevel(LogLevelInfo))
	LogLevel = utils.LogLevel

	// Logger is the interface for custom loggers.
	// Implement this interface to use your own logging solution.
	Logger = utils.Logger

	// MemoryOption configures the memory settings for conversation history.
	// It controls how much context is retained between interactions.
	//
	// Example usage:
	//   cfg := NewConfig()
	//   cfg = ApplyOptions(cfg, SetMemory(MemoryOption{MaxHistory: 10}))
	MemoryOption = config.MemoryOption
)

// Re-export core configuration functions
var (
	// LoadConfig loads configuration from environment variables and returns a new Config instance.
	// It automatically detects and loads API keys from environment variables matching the pattern
	// *_API_KEY (e.g., OPENAI_API_KEY, ANTHROPIC_API_KEY).
	//
	// Example usage:
	//   cfg, err := LoadConfig()
	//   if err != nil {
	//       log.Fatal(err)
	//   }
	LoadConfig = config.LoadConfig

	// ApplyOptions applies a series of ConfigOption functions to a Config instance.
	// This enables fluent configuration updates using the builder pattern.
	//
	// Example usage:
	//   cfg := NewConfig()
	//   cfg = ApplyOptions(cfg, SetProvider("openai"), SetModel("gpt-3.5-turbo"))
	ApplyOptions = config.ApplyOptions
)

// Re-export ConfigOption functions for configuration modification.
// Each function returns a ConfigOption that can be used with ApplyOptions
// to modify configuration settings.
var (
	// Provider configuration
	SetProvider       = config.SetProvider       // Sets the LLM provider (e.g., "openai", "anthropic")
	SetModel          = config.SetModel          // Sets the model name for the selected provider
	SetOllamaEndpoint = config.SetOllamaEndpoint // Sets the endpoint URL for Ollama local deployment
	SetVLLMEndpoint   = config.SetVLLMEndpoint   // Sets the endpoint URL for vLLM local deployment
	SetAPIKey         = config.SetAPIKey         // Sets the API key for the current provider

	// Generation parameters
	SetTemperature      = config.SetTemperature      // Controls randomness in generation (0.0-1.0)
	SetMaxTokens        = config.SetMaxTokens        // Sets maximum tokens to generate
	SetTopP             = config.SetTopP             // Controls nucleus sampling
	SetFrequencyPenalty = config.SetFrequencyPenalty // Penalizes frequent token usage
	SetPresencePenalty  = config.SetPresencePenalty  // Penalizes repeated tokens
	SetSeed             = config.SetSeed             // Sets random seed for reproducible generation

	// Advanced generation parameters
	SetMinP          = config.SetMinP          // Sets minimum probability threshold
	SetRepeatPenalty = config.SetRepeatPenalty // Controls repetition penalty
	SetRepeatLastN   = config.SetRepeatLastN   // Sets context window for repetition
	SetMirostat      = config.SetMirostat      // Enables Mirostat sampling
	SetMirostatEta   = config.SetMirostatEta   // Sets Mirostat learning rate
	SetMirostatTau   = config.SetMirostatTau   // Sets Mirostat target entropy
	SetTfsZ          = config.SetTfsZ          // Sets tail-free sampling parameter

	// Runtime configuration
	SetTimeout       = config.SetTimeout       // Sets request timeout duration
	SetMaxRetries    = config.SetMaxRetries    // Sets maximum retry attempts
	SetRetryDelay    = config.SetRetryDelay    // Sets base delay for retry backoff
	SetMaxRetryDelay = config.SetMaxRetryDelay // Caps exponential retry backoff
	SetLogLevel      = config.SetLogLevel      // Sets logging verbosity
	SetExtraHeaders  = config.SetExtraHeaders  // Sets additional HTTP headers

	// Feature toggles
	SetEnableCaching = config.SetEnableCaching // Enables/disables response caching
	SetMemory        = config.SetMemory        // Configures conversation memory

	// Validation configuration
	SetCustomValidator = config.SetCustomValidator // Sets a custom validation function to override default validation

	// Logging
	SetLogger = config.SetLogger // Sets a custom logger implementation

	// Configuration creation
	NewConfig = config.NewConfig // Creates a new Config with default values

	// Logger utilities
	NewNopLogger = utils.NewNopLogger // Creates a no-op logger that discards all output
)

// LogLevel constants define available logging verbosity levels
const (
	LogLevelOff   = utils.LogLevelOff   // Disables all logging
	LogLevelError = utils.LogLevelError // Logs only errors
	LogLevelWarn  = utils.LogLevelWarn  // Logs warnings and errors
	LogLevelInfo  = utils.LogLevelInfo  // Logs info, warnings, and errors
	LogLevelDebug = utils.LogLevelDebug // Logs all messages including debug
)
