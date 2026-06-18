// Package providers implements LLM provider interfaces and implementations.
package providers

import (
	"github.com/teilomillet/gollm/config"
)

// AtlasCloudProvider implements the Provider interface for Atlas Cloud's API.
// It inherits from OpenAIProvider since Atlas Cloud uses an OpenAI-compatible API.
type AtlasCloudProvider struct {
	OpenAIProvider
}

// NewAtlasCloudProvider creates a new Atlas Cloud provider instance.
// It initializes the provider with the given API key, model, and optional headers.
//
// Parameters:
//   - apiKey: Atlas Cloud API key for authentication (set via ATLASCLOUD_API_KEY env var)
//   - model: The model to use (e.g., "deepseek-v3", "claude-3-5-sonnet-20241022")
//   - extraHeaders: Additional HTTP headers for requests
//
// Returns:
//   - A configured Atlas Cloud Provider instance
func NewAtlasCloudProvider(apiKey, model string, extraHeaders map[string]string) Provider {
	provider := &AtlasCloudProvider{
		OpenAIProvider: *NewOpenAIProvider(apiKey, model, extraHeaders).(*OpenAIProvider),
	}
	return provider
}

// Name returns "atlascloud" as the provider identifier.
// This is used to identify the provider in the system.
func (p *AtlasCloudProvider) Name() string {
	return "atlascloud"
}

// Endpoint returns the Atlas Cloud API endpoint URL.
// Atlas Cloud provides an OpenAI-compatible API with access to 300+ frontier models.
func (p *AtlasCloudProvider) Endpoint() string {
	return "https://api.atlascloud.ai/v1/chat/completions"
}

// SetDefaultOptions configures standard options from the global configuration.
// This includes setting options like temperature and max tokens based on the provided config.
//
// Parameters:
//   - config: The global configuration containing options to set
func (p *AtlasCloudProvider) SetDefaultOptions(config *config.Config) {
	p.SetOption("temperature", config.Temperature)
	p.SetOption("max_tokens", config.MaxTokens)
	if config.Seed != nil {
		p.SetOption("seed", *config.Seed)
	}
	p.logger.Debug("Default options set", "temperature", config.Temperature, "max_tokens", config.MaxTokens)
}
