// Package providers implements various Language Learning Model (LLM) provider interfaces
// and their concrete implementations. It supports multiple providers including OpenAI,
// Anthropic, Groq, Ollama, and Mistral, providing a unified interface for interacting
// with different LLM services.
package providers

import (
	"fmt"
	"sync"

	"github.com/teilomillet/gollm/config"
	"github.com/teilomillet/gollm/types"
	"github.com/teilomillet/gollm/utils"
)

// Provider defines the interface that all LLM providers must implement.
// This interface abstracts the common operations needed to interact with
// different LLM services, allowing for a unified approach to LLM integration.
type Provider interface {
	// Name returns the provider's identifier (e.g., "openai", "anthropic").
	Name() string

	// Endpoint returns the API endpoint URL for the provider.
	Endpoint() string

	// Headers returns the HTTP headers required for API requests.
	// This typically includes authentication and content-type headers.
	Headers() map[string]string

	// PrepareRequest creates the request body for an API call.
	// It takes a prompt string and additional options, returning the serialized request body.
	PrepareRequest(prompt string, options map[string]interface{}) ([]byte, error)

	// PrepareRequestWithSchema creates a request body that includes JSON schema validation.
	// This is used for providers that support structured output validation.
	PrepareRequestWithSchema(prompt string, options map[string]interface{}, schema interface{}) ([]byte, error)

	// PrepareRequestWithMessages creates a request body using structured message objects
	// rather than a flattened prompt string. This enables more efficient caching and
	// better preserves conversation structure for the LLM API.
	//
	// Parameters:
	//   - messages: Slice of MemoryMessage objects representing the conversation
	//   - options: Additional options for the request
	//
	// Returns:
	//   - Serialized JSON request body
	//   - Any error encountered during preparation
	PrepareRequestWithMessages(messages []types.MemoryMessage, options map[string]interface{}) ([]byte, error)

	// ParseResponse extracts the generated text from the API response.
	// It handles provider-specific response formats and error cases.
	ParseResponse(body []byte) (string, error)

	// SetExtraHeaders configures additional HTTP headers for API requests.
	// This is useful for provider-specific features or authentication methods.
	SetExtraHeaders(extraHeaders map[string]string)

	// HandleFunctionCalls processes function calling capabilities.
	// This is particularly relevant for providers that support function/tool calling.
	HandleFunctionCalls(body []byte) ([]byte, error)

	// SupportsJSONSchema indicates whether the provider supports native JSON schema validation.
	SupportsJSONSchema() bool

	// SetDefaultOptions configures provider-specific defaults from the global configuration.
	SetDefaultOptions(config *config.Config)

	// SetOption sets a specific option for the provider (e.g., temperature, max_tokens).
	SetOption(key string, value interface{})

	// SetLogger configures the logger for the provider instance.
	SetLogger(logger utils.Logger)

	// SupportsStreaming indicates whether the provider supports streaming responses.
	SupportsStreaming() bool

	// PrepareStreamRequest creates a request body for streaming API calls.
	// It's similar to PrepareRequest but includes streaming-specific options.
	PrepareStreamRequest(prompt string, options map[string]interface{}) ([]byte, error)

	// ParseStreamResponse processes a single chunk from a streaming response.
	// It returns the token text and any error encountered.
	ParseStreamResponse(chunk []byte) (string, error)
}

// ProviderType represents the general type of LLM API
type ProviderType string

const (
	// TypeOpenAI is for providers using the OpenAI API format
	TypeOpenAI ProviderType = "openai"

	// TypeAnthropic is for providers using the Anthropic API format
	TypeAnthropic ProviderType = "anthropic"

	// TypeClaude is an alias for Anthropic
	TypeClaude ProviderType = "claude"

	// TypeCustom is for completely custom providers
	TypeCustom ProviderType = "custom"
)

// ProviderConfig holds the configuration for a provider
type ProviderConfig struct {
	// Name is the provider identifier
	Name string

	// Type is the API format this provider uses (e.g., "openai", "anthropic")
	Type ProviderType

	// Endpoint is the API endpoint URL
	Endpoint string

	// BaseURL is the public API base URL used with RequestPath.
	// Endpoint takes precedence when both fields are set.
	BaseURL string

	// RequestPath is appended to BaseURL to build the request endpoint.
	RequestPath string

	// AuthHeader is the header key used for authentication
	AuthHeader string

	// AuthPrefix is the prefix to use before the API key (e.g., "Bearer ")
	AuthPrefix string

	// RequiredHeaders are additional headers always needed
	RequiredHeaders map[string]string

	// EndpointParams are URL parameters to add to the endpoint
	EndpointParams map[string]string

	// ResponseFormat defines how to parse the response
	// If empty, uses the default parser for the provider type
	ResponseFormat string

	// SupportsSchema indicates if JSON schema validation is supported
	SupportsSchema bool

	// SupportsStreaming indicates if streaming is supported
	SupportsStreaming bool
}

// ProviderConstructor defines a function type for creating new provider instances.
// Each provider implementation must provide a constructor function of this type.
type ProviderConstructor func(apiKey, model string, extraHeaders map[string]string) Provider

// ProviderRegistry manages the registration and retrieval of LLM providers.
// It provides thread-safe access to provider constructors and supports
// dynamic provider registration.
type ProviderRegistry struct {
	providers map[string]ProviderConstructor
	configs   map[string]ProviderConfig
	mutex     sync.RWMutex
}

// NewProviderRegistry creates a new provider registry with the specified providers.
// If no providers are specified, all known providers are registered by default.
//
// Supported providers:
//   - "openai": OpenAI's GPT models
//   - "anthropic": Anthropic's Claude models
//   - "groq": Groq's LLM services
//   - "ollama": Local LLM deployment
//   - "mistral": Mistral AI's models
//   - "cohere": Cohere's models
//   - "deepseek": DeepSeek's models
//   - "google-openai": Google's Gemini models using OpenAI compatible API
//   - "minimax": MiniMax models using the global chat completions API
//
// Example usage:
//
//	// Register all providers
//	registry := NewProviderRegistry()
//
//	// Register specific providers
//	registry := NewProviderRegistry("openai", "anthropic")
func NewProviderRegistry(providerNames ...string) *ProviderRegistry {
	registry := &ProviderRegistry{
		providers: make(map[string]ProviderConstructor),
		configs:   make(map[string]ProviderConfig),
	}

	// Register all known providers
	knownProviders := map[string]ProviderConstructor{
		"openai":              NewOpenAIProvider,
		"anthropic":           NewAnthropicProvider,
		"groq":                NewGroqProvider,
		"ollama":              NewOllamaProvider,
		"mistral":             NewMistralProvider,
		"cohere":              NewCohereProvider,
		"deepseek":            NewDeepSeekProvider,
		"google-openai":       NewGoogleProvider,
		"azure-openai":        NewAzureOpenAIProvider,
		"aliyun":              NewAliyunProvider,
		"lmstudio":            NewLMStudioProvider,
		"openrouter":          NewOpenRouterProvider,
		"lambda":              NewLambdaProvider,
		"bedrock":             NewBedrockProvider,
		"vllm":                NewVLLMProvider,
		"minimax":             NewMiniMaxProvider,
		"minimax-cn":          NewMiniMaxCNProvider,
		"minimax-messages":    NewMiniMaxMessagesProvider,
		"minimax-messages-cn": NewMiniMaxMessagesCNProvider,
	}

	// Standard provider configurations
	standardConfigs := map[string]ProviderConfig{
		"openai": {
			Name:              "openai",
			Type:              TypeOpenAI,
			Endpoint:          "https://api.openai.com/v1/chat/completions",
			AuthHeader:        "Authorization",
			AuthPrefix:        "Bearer ",
			RequiredHeaders:   map[string]string{"Content-Type": "application/json"},
			SupportsSchema:    true,
			SupportsStreaming: true,
		},
		"azure-openai": {
			Name: "azure-openai",
			Type: TypeOpenAI,
			// Endpoint should be configured by the user
			AuthHeader:        "api-key",
			AuthPrefix:        "",
			RequiredHeaders:   map[string]string{"Content-Type": "application/json"},
			SupportsSchema:    true,
			SupportsStreaming: true,
		},
		"anthropic": {
			Name:              "anthropic",
			Type:              TypeAnthropic,
			Endpoint:          "https://api.anthropic.com/v1/messages",
			AuthHeader:        "x-api-key",
			AuthPrefix:        "",
			RequiredHeaders:   map[string]string{"Content-Type": "application/json", "anthropic-version": "2023-06-01"},
			SupportsSchema:    true,
			SupportsStreaming: true,
		},
		"groq": {
			Name:              "groq",
			Type:              TypeOpenAI,
			Endpoint:          "https://api.groq.com/openai/v1/chat/completions",
			AuthHeader:        "Authorization",
			AuthPrefix:        "Bearer ",
			RequiredHeaders:   map[string]string{"Content-Type": "application/json"},
			SupportsSchema:    true,
			SupportsStreaming: true,
		},
		"ollama": {
			Name:              "ollama",
			Type:              TypeCustom,
			Endpoint:          "http://localhost:11434/api/generate",
			AuthHeader:        "", // Ollama doesn't require authentication
			AuthPrefix:        "",
			RequiredHeaders:   map[string]string{"Content-Type": "application/json"},
			SupportsSchema:    false,
			SupportsStreaming: true,
		},
		"deepseek": {
			Name:              "deepseek",
			Type:              TypeOpenAI,
			Endpoint:          "https://api.deepseek.com/chat/completions",
			AuthHeader:        "Authorization",
			AuthPrefix:        "Bearer ",
			RequiredHeaders:   map[string]string{"Content-Type": "application/json"},
			SupportsSchema:    true,
			SupportsStreaming: true,
		},
		"google-openai": {
			Name:              "google-openai",
			Type:              TypeOpenAI,
			Endpoint:          "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
			AuthHeader:        "Authorization",
			AuthPrefix:        "Bearer ",
			RequiredHeaders:   map[string]string{"Content-Type": "application/json"},
			SupportsSchema:    true,
			SupportsStreaming: true,
		},
		"aliyun": {
			Name:              "aliyun",
			Type:              TypeOpenAI,
			Endpoint:          "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
			AuthHeader:        "Authorization",
			AuthPrefix:        "Bearer ",
			RequiredHeaders:   map[string]string{"Content-Type": "application/json"},
			SupportsSchema:    true,
			SupportsStreaming: true,
		},
		"lmstudio": {
			Name:              "lmstudio",
			Type:              TypeOpenAI,
			Endpoint:          "http://localhost:1234/v1/chat/completions",
			AuthHeader:        "Authorization",
			AuthPrefix:        "Bearer ",
			RequiredHeaders:   map[string]string{"Content-Type": "application/json"},
			SupportsSchema:    true,
			SupportsStreaming: false, // LM Studio streaming works differently
		},
		"openrouter": {
			Name:              "openrouter",
			Type:              TypeOpenAI,
			Endpoint:          "https://openrouter.ai/api/v1/chat/completions",
			AuthHeader:        "Authorization",
			AuthPrefix:        "Bearer ",
			RequiredHeaders:   map[string]string{"Content-Type": "application/json"},
			SupportsSchema:    true,
			SupportsStreaming: true,
		},
		"lambda": {
			Name:              "lambda",
			Type:              TypeOpenAI,
			Endpoint:          "https://api.lambdalabs.com/v1/chat/completions",
			AuthHeader:        "Authorization",
			AuthPrefix:        "Bearer ",
			RequiredHeaders:   map[string]string{"Content-Type": "application/json"},
			SupportsSchema:    true,
			SupportsStreaming: true,
		},
		"bedrock": {
			Name:              "bedrock",
			Type:              TypeCustom,
			Endpoint:          "", // Configured dynamically based on region and model
			AuthHeader:        "", // Uses AWS Signature V4
			AuthPrefix:        "",
			RequiredHeaders:   map[string]string{"Content-Type": "application/json"},
			SupportsSchema:    true,
			SupportsStreaming: true,
		},
		"vllm": {
			Name:              "vllm",
			Type:              TypeOpenAI,
			Endpoint:          "", // Configured by user for local vLLM server
			AuthHeader:        "", // No authentication required
			AuthPrefix:        "",
			RequiredHeaders:   map[string]string{"Content-Type": "application/json"},
			SupportsSchema:    true,
			SupportsStreaming: true,
		},
		"minimax": {
			Name:              "minimax",
			Type:              TypeOpenAI,
			BaseURL:           "https://api.minimax.io/v1",
			RequestPath:       "chat/completions",
			AuthHeader:        "Authorization",
			AuthPrefix:        "Bearer ",
			RequiredHeaders:   map[string]string{"Content-Type": "application/json"},
			SupportsSchema:    true,
			SupportsStreaming: true,
		},
		"minimax-cn": {
			Name:              "minimax-cn",
			Type:              TypeOpenAI,
			BaseURL:           "https://api.minimaxi.com/v1",
			RequestPath:       "chat/completions",
			AuthHeader:        "Authorization",
			AuthPrefix:        "Bearer ",
			RequiredHeaders:   map[string]string{"Content-Type": "application/json"},
			SupportsSchema:    true,
			SupportsStreaming: true,
		},
		"minimax-messages": {
			Name:              "minimax-messages",
			Type:              TypeAnthropic,
			BaseURL:           "https://api.minimax.io/anthropic",
			RequestPath:       "v1/messages",
			AuthHeader:        "Authorization",
			AuthPrefix:        "Bearer ",
			RequiredHeaders:   map[string]string{"Content-Type": "application/json", "anthropic-version": "2023-06-01"},
			SupportsSchema:    true,
			SupportsStreaming: true,
		},
		"minimax-messages-cn": {
			Name:              "minimax-messages-cn",
			Type:              TypeAnthropic,
			BaseURL:           "https://api.minimaxi.com/anthropic",
			RequestPath:       "v1/messages",
			AuthHeader:        "Authorization",
			AuthPrefix:        "Bearer ",
			RequiredHeaders:   map[string]string{"Content-Type": "application/json", "anthropic-version": "2023-06-01"},
			SupportsSchema:    true,
			SupportsStreaming: true,
		},
	}

	// Store standard configs
	for name, config := range standardConfigs {
		registry.configs[name] = config
	}

	if len(providerNames) == 0 {
		// If no specific providers are requested, register all known providers
		for name, constructor := range knownProviders {
			registry.providers[name] = constructor
		}
	} else {
		// Otherwise, register only the requested providers
		for _, name := range providerNames {
			if constructor, ok := knownProviders[name]; ok {
				registry.providers[name] = constructor
			}
		}
	}

	return registry
}

// GetProviderConfig returns the configuration for a named provider
// Returns the config and a boolean indicating whether the provider was found
func (r *ProviderRegistry) GetProviderConfig(name string) (ProviderConfig, bool) {
	r.mutex.RLock()
	defer r.mutex.RUnlock()

	config, exists := r.configs[name]
	return config, exists
}

// RegisterProviderConfig registers a new provider configuration
func (r *ProviderRegistry) RegisterProviderConfig(name string, config ProviderConfig) {
	r.mutex.Lock()
	defer r.mutex.Unlock()

	r.configs[name] = config
}

// defaultRegistry is a singleton instance of the provider registry
var defaultRegistry *ProviderRegistry
var defaultRegistryOnce sync.Once

// GetDefaultRegistry returns the default provider registry
// This registry contains all known providers and is created lazily
func GetDefaultRegistry() *ProviderRegistry {
	defaultRegistryOnce.Do(func() {
		defaultRegistry = NewProviderRegistry()
	})
	return defaultRegistry
}

// RegisterGenericProvider creates a constructor for a generic provider
// with the specified name and configuration
func RegisterGenericProvider(name string, config ProviderConfig) {
	registry := GetDefaultRegistry()
	registry.RegisterProviderConfig(name, config)

	// Register a constructor that creates a GenericProvider with this config
	registry.providers[name] = func(apiKey, model string, extraHeaders map[string]string) Provider {
		return NewGenericProvider(apiKey, model, name, extraHeaders)
	}
}

// Register adds a new provider constructor to the registry.
// This method is thread-safe and can be used to dynamically add new providers.
//
// Parameters:
//   - name: The identifier for the provider (e.g., "openai")
//   - constructor: A function that creates new instances of the provider
func (pr *ProviderRegistry) Register(name string, constructor ProviderConstructor) {
	pr.mutex.Lock()
	defer pr.mutex.Unlock()
	pr.providers[name] = constructor
}

// Get retrieves a provider instance by name.
// It creates a new provider instance using the registered constructor.
//
// Parameters:
//   - name: The provider identifier
//   - apiKey: The API key for authentication
//   - model: The specific model to use
//   - extraHeaders: Additional HTTP headers for requests
//
// Returns:
//   - A configured Provider instance
//   - An error if the provider is not found
//
// Example:
//
//	provider, err := registry.Get("openai", "sk-...", "gpt-4", nil)
func (pr *ProviderRegistry) Get(name, apiKey, model string, extraHeaders map[string]string) (Provider, error) {
	pr.mutex.RLock()
	constructor, exists := pr.providers[name]
	pr.mutex.RUnlock()

	if !exists {
		return nil, fmt.Errorf("unknown provider: %s", name)
	}

	return constructor(apiKey, model, extraHeaders), nil
}
