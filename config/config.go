// Package config provides the core configuration system for the gollm library,
// enabling flexible and type-safe configuration of Language Learning Model (LLM)
// interactions through environment variables and programmatic options.
package config

import (
	"os"
	"strings"
	"time"

	"github.com/caarlos0/env/v11"
	"github.com/teilomillet/gollm/utils"
)

// MemoryOption configures conversation memory settings, controlling how much
// context is retained between interactions with the LLM.
type MemoryOption struct {
	// MaxTokens specifies the maximum number of tokens to retain in memory
	// for context in subsequent interactions.
	MaxTokens int
}

// Config represents the complete configuration for LLM interactions.
// It supports configuration through environment variables, with sensible defaults
// for most settings. API keys are automatically loaded from environment variables
// matching the pattern *_API_KEY (e.g., OPENAI_API_KEY, ANTHROPIC_API_KEY).
//
// Environment Variables:
//   - LLM_PROVIDER: LLM provider name (default: "anthropic")
//   - LLM_MODEL: Model name (default: "claude-3-opus-20240229")
//   - OLLAMA_ENDPOINT: Ollama API endpoint (default: "http://localhost:11434")
//   - LLM_TEMPERATURE: Generation temperature (default: 0.7)
//   - LLM_MAX_TOKENS: Maximum tokens to generate (default: 100)
//   - LLM_TOP_P: Top-p sampling parameter (default: 0.9)
//   - LLM_FREQUENCY_PENALTY: Token frequency penalty (default: 0.0)
//   - LLM_PRESENCE_PENALTY: Token presence penalty (default: 0.0)
//   - LLM_TIMEOUT: Request timeout duration (default: 30s)
//   - LLM_MAX_RETRIES: Maximum retry attempts (default: 3)
//   - LLM_RETRY_DELAY: Delay between retries (default: 2s)
//   - LLM_LOG_LEVEL: Logging verbosity (default: "WARN")
//   - LLM_SEED: Random seed for reproducible generation
//   - LLM_ENABLE_CACHING: Enable response caching (default: false)
//   - LLM_ENABLE_STREAMING: Enable streaming responses (default: false)
//
// Advanced Parameters:
//   - LLM_MIN_P: Minimum token probability threshold
//   - LLM_REPEAT_PENALTY: Penalty for repeated tokens
//   - LLM_REPEAT_LAST_N: Context window for repetition checking
//   - LLM_MIROSTAT: Mirostat sampling mode
//   - LLM_MIROSTAT_ETA: Mirostat learning rate
//   - LLM_MIROSTAT_TAU: Mirostat target entropy
//   - LLM_TFS_Z: Tail-free sampling parameter
type Config struct {
	Provider              string            `env:"LLM_PROVIDER" envDefault:"anthropic" validate:"required"`
	Model                 string            `env:"LLM_MODEL" envDefault:"claude-3-5-haiku-latest" validate:"required"`
	OllamaEndpoint        string            `env:"OLLAMA_ENDPOINT" envDefault:"http://localhost:11434"`
	VLLMEndpoint          string            `env:"VLLM_ENDPOINT" envDefault:"http://localhost:8000"`
	Temperature           float64           `env:"LLM_TEMPERATURE" envDefault:"0.7" validate:"gte=0,lte=1"`
	MaxTokens             int               `env:"LLM_MAX_TOKENS" envDefault:"100"`
	TopP                  float64           `env:"LLM_TOP_P" envDefault:"0.9" validate:"gte=0,lte=1"`
	FrequencyPenalty      float64           `env:"LLM_FREQUENCY_PENALTY" envDefault:"0.0"`
	PresencePenalty       float64           `env:"LLM_PRESENCE_PENALTY" envDefault:"0.0"`
	Timeout               time.Duration     `env:"LLM_TIMEOUT" envDefault:"30s"`
	MaxRetries            int               `env:"LLM_MAX_RETRIES" envDefault:"3"`
	RetryDelay            time.Duration     `env:"LLM_RETRY_DELAY" envDefault:"2s"`
	MaxRetryDelay         time.Duration     `env:"LLM_MAX_RETRY_DELAY" envDefault:"60s"`
	APIKeys               map[string]string `validate:"required,apikey"`
	LogLevel              utils.LogLevel    `env:"LLM_LOG_LEVEL" envDefault:"WARN"`
	Seed                  *int              `env:"LLM_SEED"`
	MinP                  *float64          `env:"LLM_MIN_P" envDefault:"0.05"`
	RepeatPenalty         *float64          `env:"LLM_REPEAT_PENALTY" envDefault:"1.1"`
	RepeatLastN           *int              `env:"LLM_REPEAT_LAST_N" envDefault:"64"`
	Mirostat              *int              `env:"LLM_MIROSTAT" envDefault:"0"`
	MirostatEta           *float64          `env:"LLM_MIROSTAT_ETA" envDefault:"0.1"`
	MirostatTau           *float64          `env:"LLM_MIROSTAT_TAU" envDefault:"5.0"`
	TfsZ                  *float64          `env:"LLM_TFS_Z" envDefault:"1"`
	SystemPrompt          string
	SystemPromptCacheType string
	ExtraHeaders          map[string]string
	EnableCaching         bool `env:"LLM_ENABLE_CACHING" envDefault:"false"`
	EnableStreaming       bool `env:"LLM_ENABLE_STREAMING" envDefault:"false"`
	MemoryOption          *MemoryOption
	CustomValidator       func(interface{}) error // Custom validation function to override default validation
	Logger                utils.Logger            // Custom logger, if nil uses default
}

// LoadConfig creates a new Config instance, loading values from environment
// variables and automatically detecting API keys. It returns an error if
// environment variable parsing fails.
//
// Example usage:
//
//	cfg, err := LoadConfig()
//	if err != nil {
//	    log.Fatalf("Failed to load config: %v", err)
//	}
//
//	// Access configuration values
//	fmt.Printf("Using provider: %s\n", cfg.Provider)
//	fmt.Printf("Model: %s\n", cfg.Model)
func LoadConfig() (*Config, error) {
	cfg := &Config{
		APIKeys: make(map[string]string),
	}
	if err := env.Parse(cfg); err != nil {
		return nil, err
	}

	loadAPIKeys(cfg)
	return cfg, nil
}

// loadAPIKeys automatically detects and loads API keys from environment variables
// matching the pattern *_API_KEY. It ensures the default provider has an API key
// available.
func loadAPIKeys(cfg *Config) {
	for _, envVar := range os.Environ() {
		key, value, found := strings.Cut(envVar, "=")
		if found && strings.HasSuffix(strings.ToUpper(key), "_API_KEY") {
			provider := strings.TrimSuffix(strings.ToUpper(key), "_API_KEY")
			cfg.APIKeys[strings.ToLower(provider)] = value
		}
	}

	// Ensure the default provider has an API key
	if apiKey, exists := cfg.APIKeys[strings.ToUpper(cfg.Provider)]; exists {
		cfg.APIKeys[cfg.Provider] = apiKey
	}
}

// ConfigOption is a function type that modifies a Config instance.
// It enables a builder pattern for configuration, allowing for clean
// and flexible configuration updates.
type ConfigOption func(*Config)

// NewConfig creates a new Config instance with default values suitable
// for most use cases. The defaults can be overridden using ConfigOption
// functions.
//
// Example usage:
//
//	cfg := NewConfig()
//	ApplyOptions(cfg,
//	    SetProvider("openai"),
//	    SetModel("gpt-4"),
//	    SetTemperature(0.8),
//	    SetMaxTokens(500),
//	)
func NewConfig() *Config {
	return &Config{
		Provider:      "openai",
		Model:         "gpt-4o-mini",
		Temperature:   0.7,
		MaxTokens:     300,
		Timeout:       30 * time.Second,
		MaxRetries:    3,
		RetryDelay:    2 * time.Second,
		MaxRetryDelay: 60 * time.Second,
		APIKeys:       make(map[string]string),
		LogLevel:      utils.LogLevelWarn,
		ExtraHeaders:  make(map[string]string),
	}
}

// SetEnableCaching sets the EnableCaching flag.
func SetEnableCaching(enableCaching bool) ConfigOption {
	return func(c *Config) {
		c.EnableCaching = enableCaching
	}
}

// SetProvider sets the LLM provider.
func SetProvider(provider string) ConfigOption {
	return func(c *Config) {
		c.Provider = provider
	}
}

// SetModel sets the LLM model.
func SetModel(model string) ConfigOption {
	return func(c *Config) {
		c.Model = model
	}
}

// SetOllamaEndpoint sets the Ollama API endpoint.
func SetOllamaEndpoint(endpoint string) ConfigOption {
	return func(c *Config) {
		c.OllamaEndpoint = endpoint
	}
}

// SetVLLMEndpoint sets the vLLM API endpoint.
func SetVLLMEndpoint(endpoint string) ConfigOption {
	return func(c *Config) {
		c.VLLMEndpoint = endpoint
	}
}

// SetTemperature sets the generation temperature.
func SetTemperature(temperature float64) ConfigOption {
	return func(c *Config) {
		c.Temperature = temperature
	}
}

// SetMaxTokens sets the maximum number of tokens to generate.
func SetMaxTokens(maxTokens int) ConfigOption {
	return func(c *Config) {
		if maxTokens < 1 {
			maxTokens = 1
		}
		c.MaxTokens = maxTokens
	}
}

// SetTimeout sets the request timeout duration.
func SetTimeout(timeout time.Duration) ConfigOption {
	return func(c *Config) {
		c.Timeout = timeout
	}
}

// SetAPIKey sets the API key for the specified provider.
func SetAPIKey(apiKey string) ConfigOption {
	return func(c *Config) {
		if c.APIKeys == nil {
			c.APIKeys = make(map[string]string)
		}
		// For Ollama, we don't need a real API key, but we need to set something
		// to satisfy the validation
		if c.Provider == "ollama" && apiKey == "" {
			c.APIKeys[c.Provider] = "ollama-local"
			return
		}
		c.APIKeys[c.Provider] = apiKey
	}
}

// SetMaxRetries sets the maximum number of retry attempts.
func SetMaxRetries(maxRetries int) ConfigOption {
	return func(c *Config) {
		c.MaxRetries = maxRetries
	}
}

// SetRetryDelay sets the base delay for retry backoff. The retry loop grows
// this exponentially per attempt (base × 2^attempt) with full jitter, so it is
// the floor of the backoff curve rather than a fixed inter-attempt delay.
func SetRetryDelay(retryDelay time.Duration) ConfigOption {
	return func(c *Config) {
		c.RetryDelay = retryDelay
	}
}

// SetMaxRetryDelay caps the exponential backoff between retries. Zero means
// uncapped growth from the base RetryDelay.
func SetMaxRetryDelay(maxRetryDelay time.Duration) ConfigOption {
	return func(c *Config) {
		c.MaxRetryDelay = maxRetryDelay
	}
}

// SetLogLevel sets the logging verbosity.
func SetLogLevel(level utils.LogLevel) ConfigOption {
	return func(c *Config) {
		c.LogLevel = level
	}
}

// SetMemory sets the conversation memory settings.
func SetMemory(maxTokens int) ConfigOption {
	return func(c *Config) {
		c.MemoryOption = &MemoryOption{
			MaxTokens: maxTokens,
		}
	}
}

// SetExtraHeaders sets additional HTTP headers.
func SetExtraHeaders(headers map[string]string) ConfigOption {
	return func(c *Config) {
		if c.ExtraHeaders == nil {
			c.ExtraHeaders = make(map[string]string)
		}
		for k, v := range headers {
			c.ExtraHeaders[k] = v
		}
	}
}

// WithStream enables or disables streaming responses.
func WithStream(enableStreaming bool) ConfigOption {
	return func(c *Config) {
		c.EnableStreaming = enableStreaming
	}
}

// SetTopP sets the top-p sampling parameter.
func SetTopP(topP float64) ConfigOption {
	return func(c *Config) {
		c.TopP = topP
	}
}

// SetFrequencyPenalty sets the token frequency penalty.
func SetFrequencyPenalty(penalty float64) ConfigOption {
	return func(c *Config) {
		c.FrequencyPenalty = penalty
	}
}

// SetPresencePenalty sets the token presence penalty.
func SetPresencePenalty(penalty float64) ConfigOption {
	return func(c *Config) {
		c.PresencePenalty = penalty
	}
}

// SetSeed sets the random seed for reproducible generation.
func SetSeed(seed int) ConfigOption {
	return func(c *Config) {
		c.Seed = &seed
	}
}

// SetMinP sets the minimum token probability threshold.
func SetMinP(minP float64) ConfigOption {
	return func(c *Config) {
		c.MinP = &minP
	}
}

// SetRepeatPenalty sets the penalty for repeated tokens.
func SetRepeatPenalty(penalty float64) ConfigOption {
	return func(c *Config) {
		c.RepeatPenalty = &penalty
	}
}

// SetRepeatLastN sets the context window for repetition checking.
func SetRepeatLastN(n int) ConfigOption {
	return func(c *Config) {
		c.RepeatLastN = &n
	}
}

// SetMirostat sets the Mirostat sampling mode.
func SetMirostat(mode int) ConfigOption {
	return func(c *Config) {
		c.Mirostat = &mode
	}
}

// SetMirostatEta sets the Mirostat learning rate.
func SetMirostatEta(eta float64) ConfigOption {
	return func(c *Config) {
		c.MirostatEta = &eta
	}
}

// SetMirostatTau sets the Mirostat target entropy.
func SetMirostatTau(tau float64) ConfigOption {
	return func(c *Config) {
		c.MirostatTau = &tau
	}
}

// SetTfsZ sets the tail-free sampling parameter.
func SetTfsZ(z float64) ConfigOption {
	return func(c *Config) {
		c.TfsZ = &z
	}
}

// SetCustomValidator sets a custom validation function that overrides the default validation behavior.
// This is particularly useful for bypassing API key validation for specific providers like Google/Gemini
// or implementing custom validation logic.
//
// The custom validator is scoped to the NewLLM call and does not affect other goroutines,
// making it safe for concurrent use.
//
// Parameters:
//   - fn: A custom validation function that takes an interface{} and returns an error.
//     If fn is nil, default validation is used.
//
// Example usage:
//
//	// Skip all validation (useful for Google/Gemini)
//	llm, err := gollm.NewLLM(
//	    gollm.SetProvider("google"),
//	    gollm.SetModel("gemini-1.5-pro-latest"),
//	    gollm.SetAPIKey(os.Getenv("GEMINI_API_KEY")),
//	    gollm.SetCustomValidator(func(v interface{}) error {
//	        return nil // Skip all validation
//	    }),
//	)
//
//	// Or implement provider-specific logic with fallback to default validation
//	llm, err := gollm.NewLLM(
//	    gollm.SetProvider("google"),
//	    gollm.SetCustomValidator(func(v interface{}) error {
//	        if config, ok := v.(*gollm.Config); ok && config.Provider == "google" {
//	            return nil // Skip validation for Google
//	        }
//	        return gollm.DefaultValidate(v) // Use default for others
//	    }),
//	)
func SetCustomValidator(fn func(interface{}) error) ConfigOption {
	return func(c *Config) {
		c.CustomValidator = fn
	}
}

// SetLogger sets a custom logger implementation.
// If not set, a default logger writing to stderr is used.
// Pass nil to disable logging entirely.
func SetLogger(logger utils.Logger) ConfigOption {
	return func(c *Config) {
		c.Logger = logger
	}
}

// ApplyOptions applies a series of ConfigOption functions to a Config instance.
// This enables fluent configuration updates using the builder pattern.
//
// Example usage:
//
//	cfg := NewConfig()
//	ApplyOptions(cfg,
//	    SetProvider("anthropic"),
//	    SetModel("claude-3-opus-20240229"),
//	    SetTemperature(0.7),
//	    SetMaxTokens(2000),
//	    SetLogLevel(LogLevelDebug),
//	)
func ApplyOptions(cfg *Config, options ...ConfigOption) {
	for _, option := range options {
		option(cfg)
	}
}
