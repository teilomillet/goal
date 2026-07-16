# Provider System

The Gollm provider system allows you to connect to different LLM APIs using a consistent interface. This document explains how the provider system works and how to use it effectively.

## Overview

Gollm's provider system has been enhanced with a configuration-driven approach that makes it easier to:

1. Use existing providers with minimal code
2. Add new providers without writing custom code
3. Configure providers with flexible options
4. Switch between providers easily

This enhancement builds on top of the existing provider system, maintaining full backward compatibility while adding new capabilities.

## Provider Types

Providers in Gollm are categorized by their API format:

- **OpenAI-compatible**: Providers that use an API format similar to OpenAI's chat completion API 
  - Examples: OpenAI, Azure OpenAI, Groq, DeepSeek
- **Anthropic-compatible**: Providers that use an API format similar to Anthropic's Claude API
  - Examples: Anthropic, Claude
- **Custom**: Providers with unique API formats that require custom implementation

## Using Existing Providers

For built-in providers, usage remains the same as before:

```go
import (
    "context"
    "github.com/teilomillet/gollm"
    "github.com/teilomillet/gollm/config"
)

// Create an LLM instance
llm, err := gollm.NewLLM(
    config.SetProvider("openai"),
    config.SetAPIKey("your-api-key"),
    config.SetModel("gpt-4"),
)

// Use it as before
ctx := context.Background()
prompt := gollm.NewPrompt("Explain quantum computing in simple terms")
response, err := llm.Generate(ctx, prompt)
```

## Using Azure OpenAI

Azure OpenAI is now supported through the enhanced provider system:

```go
import (
    "context"
    "fmt"
    "github.com/teilomillet/gollm"
    "github.com/teilomillet/gollm/config"
)

// Get Azure details
resourceName := "your-resource-name"
deploymentName := "your-deployment-name"
apiVersion := "2023-05-15"

// Create the endpoint URL
endpoint := fmt.Sprintf(
    "https://%s.openai.azure.com/openai/deployments/%s/chat/completions?api-version=%s", 
    resourceName, deploymentName, apiVersion
)

// Use the built-in azure-openai provider
llm, err := gollm.NewLLM(
    config.SetProvider("azure-openai"),
    config.SetAPIKey("your-azure-api-key"),
    config.SetModel(deploymentName), // Azure uses deployment name as model
    config.SetExtraHeaders(map[string]string{
        "azure_endpoint": endpoint,
    }),
)

// Use it like any other provider
ctx := context.Background()
prompt := gollm.NewPrompt("What are the key features of Go?")
response, err := llm.Generate(ctx, prompt)
```

## Adding New Providers

### Method 1: One-time Registration (Recommended for Most Cases)

For providers not built into Gollm, you can register them at the start of your program:

```go
import (
    "github.com/teilomillet/gollm/providers"
)

// Define a configuration for the new provider
newProvider := providers.ProviderConfig{
    Name:       "new-provider",
    Type:       providers.TypeOpenAI, // If API is OpenAI-compatible
    Endpoint:   "https://api.newprovider.com/v1/chat/completions",
    AuthHeader: "Authorization",
    AuthPrefix: "Bearer ",
    RequiredHeaders: map[string]string{
        "Content-Type": "application/json",
    },
    SupportsSchema:    true,
    SupportsStreaming: true,
}

// Register it with the system
providers.RegisterGenericProvider("new-provider", newProvider)

// Now use it like any built-in provider
llm, err := gollm.NewLLM(
    config.SetProvider("new-provider"),
    config.SetAPIKey("your-api-key"),
    config.SetModel("model-name"),
)
```

### Method 2: Application Configuration

For applications that need to support multiple providers based on configuration:

```go
// Define provider configurations in your app config
type AppConfig struct {
    LLMProviders map[string]providers.ProviderConfig `json:"llm_providers"`
    // other app config...
}

// During app initialization
func initLLM(appConfig *AppConfig) {
    // Register all configured providers
    for name, providerConfig := range appConfig.LLMProviders {
        providers.RegisterGenericProvider(name, providerConfig)
    }
}
```

## Provider Configuration Options

A `ProviderConfig` includes these fields:

| Field | Description | Example |
|-------|-------------|---------|
| `Name` | Identifier for the provider | `"openai"` |
| `Type` | API format (OpenAI, Anthropic, etc.) | `providers.TypeOpenAI` |
| `Endpoint` | API endpoint URL | `"https://api.openai.com/v1/chat/completions"` |
| `BaseURL` | Public API base URL used with `RequestPath` | `"https://api.example.com/v1"` |
| `RequestPath` | Request path appended to `BaseURL` | `"chat/completions"` |
| `AuthHeader` | Header key for authentication | `"Authorization"` |
| `AuthPrefix` | Prefix for auth token | `"Bearer "` |
| `RequiredHeaders` | Additional required headers | `{"Content-Type": "application/json"}` |
| `EndpointParams` | URL parameters to add | `{"api-version": "2023-05-15"}` |
| `SupportsSchema` | Supports JSON schema validation | `true` |
| `SupportsStreaming` | Supports streaming responses | `true` |

## Advanced Usage

### Dynamic Endpoint Configuration

You can dynamically adjust provider endpoints:

```go
// Get the provider from the registry
registry := providers.GetDefaultRegistry()
provider, err := registry.Get("azure-openai", apiKey, model, extraHeaders)

// Check if it's a GenericProvider and set the endpoint
if genericProvider, ok := provider.(*providers.GenericProvider); ok {
    genericProvider.SetEndpoint("https://custom-endpoint.com/api")
}
```

### Extending the System

For providers with truly unique APIs, you can still implement the `Provider` interface directly:

```go
// Create a new provider implementation
type CustomProvider struct {
    // Your implementation
}

// Implement all required methods of the Provider interface
func (p *CustomProvider) Name() string { return "custom" }
// ... implement other methods

// Register your provider constructor
func init() {
    providers.GetDefaultRegistry().Register("custom", NewCustomProvider)
}

// Constructor function
func NewCustomProvider(apiKey, model string, extraHeaders map[string]string) providers.Provider {
    // Initialize and return your provider
}
```

## How It Works

The enhanced provider system works alongside the existing one:

1. The standard provider interface remains unchanged
2. A new `GenericProvider` type implements this interface for all compatible providers
3. Provider configurations are stored in the registry
4. When you request a provider, the system either:
   - Uses an existing provider implementation, or
   - Creates a `GenericProvider` instance with the right configuration

This approach allows the system to be both backward-compatible and forward-looking.

## Benefits

This enhanced provider system offers several advantages:

1. **Reduced code duplication**: Similar providers share implementation
2. **Flexible configuration**: Providers can be configured at runtime
3. **Easy extensibility**: New providers can be added with minimal code
4. **Consistent behavior**: All providers follow the same patterns

## When to Use Each Approach

- **Built-in providers**: Continue using them as before
- **Azure OpenAI**: Use the built-in `azure-openai` provider with endpoint configuration
- **API-compatible providers**: Use `RegisterGenericProvider` with appropriate configuration
- **Unique APIs**: Implement the `Provider` interface directly

By choosing the right approach for your needs, you can minimize code while maximizing flexibility.
