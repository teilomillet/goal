// Package providers implements LLM provider interfaces and implementations.
package providers

import (
	"bytes"
	"encoding/json"
	"fmt"
	"net/url"
	"strings"

	"github.com/teilomillet/gollm/config"
	"github.com/teilomillet/gollm/types"
	"github.com/teilomillet/gollm/utils"
)

// GenericProvider is a flexible provider implementation that can adapt to
// different LLM APIs based on configuration. It supports OpenAI-compatible
// and Anthropic-compatible APIs out of the box, and can be extended for
// custom implementations.
type GenericProvider struct {
	apiKey        string                 // API key for authentication
	model         string                 // Model identifier
	config        ProviderConfig         // Provider configuration
	extraHeaders  map[string]string      // Additional HTTP headers
	options       map[string]interface{} // Model-specific options
	logger        utils.Logger           // Logger instance
	extraEndpoint string                 // Optional override for endpoint
}

// NewGenericProvider creates a new provider instance based on the provided configuration.
// It can be used to create providers for any service with minimal code.
//
// Parameters:
//   - apiKey: API key for authentication
//   - model: The model to use
//   - providerName: The name of the provider configuration to use
//   - extraHeaders: Additional HTTP headers for requests
//
// Returns:
//   - A configured Provider instance
func NewGenericProvider(apiKey, model, providerName string, extraHeaders map[string]string) Provider {
	if extraHeaders == nil {
		extraHeaders = make(map[string]string)
	}

	// Get default registry to access configurations
	registry := GetDefaultRegistry()
	config, exists := registry.GetProviderConfig(providerName)
	if !exists {
		panic(fmt.Sprintf("Provider configuration for '%s' not found", providerName))
	}

	// Check for azure_endpoint in extraHeaders (used for Azure OpenAI)
	var customEndpoint string
	if endpoint, ok := extraHeaders["azure_endpoint"]; ok {
		customEndpoint = endpoint
		delete(extraHeaders, "azure_endpoint") // Remove so it's not sent as HTTP header
	}

	return &GenericProvider{
		apiKey:        apiKey,
		model:         model,
		config:        config,
		extraHeaders:  extraHeaders,
		options:       make(map[string]interface{}),
		logger:        utils.NewLogger(utils.LogLevelInfo),
		extraEndpoint: customEndpoint,
	}
}

// Name returns the provider's identifier from its configuration.
func (p *GenericProvider) Name() string {
	return p.config.Name
}

// Endpoint returns the API endpoint URL.
// It handles template replacement for endpoint parameters.
func (p *GenericProvider) Endpoint() string {
	// If a custom endpoint has been set, use it
	if p.extraEndpoint != "" {
		return p.extraEndpoint
	}

	// If endpoint contains parameters like {model}, replace them.
	endpoint := p.config.Endpoint
	if endpoint == "" && p.config.BaseURL != "" {
		endpoint = strings.TrimRight(p.config.BaseURL, "/")
		if p.config.RequestPath != "" {
			endpoint += "/" + strings.TrimLeft(p.config.RequestPath, "/")
		}
	}

	// Replace {model} placeholder if present
	endpoint = strings.Replace(endpoint, "{model}", p.model, -1)

	// Add additional endpoint parameters if specified
	if len(p.config.EndpointParams) > 0 {
		parsedURL, err := url.Parse(endpoint)
		if err != nil {
			p.logger.Error("Failed to parse endpoint URL", "error", err)
			return endpoint
		}

		q := parsedURL.Query()
		for k, v := range p.config.EndpointParams {
			q.Set(k, v)
		}
		parsedURL.RawQuery = q.Encode()
		endpoint = parsedURL.String()
	}

	return endpoint
}

// SetEndpoint allows overriding the endpoint
func (p *GenericProvider) SetEndpoint(endpoint string) {
	p.extraEndpoint = endpoint
}

// Headers returns the HTTP headers required for API requests.
func (p *GenericProvider) Headers() map[string]string {
	headers := make(map[string]string)

	// Add required headers from configuration
	for k, v := range p.config.RequiredHeaders {
		headers[k] = v
	}

	// Add authentication header
	if p.apiKey != "" {
		headers[p.config.AuthHeader] = p.config.AuthPrefix + p.apiKey
	}

	// Add extra headers
	for k, v := range p.extraHeaders {
		headers[k] = v
	}

	p.logger.Debug("Headers prepared", "headers", headers)
	return headers
}

// PrepareRequest creates the request body for an API call.
func (p *GenericProvider) PrepareRequest(prompt string, options map[string]interface{}) ([]byte, error) {
	switch p.config.Type {
	case TypeOpenAI:
		return p.prepareOpenAIRequest(prompt, options, nil)
	case TypeAnthropic, TypeClaude:
		return p.prepareAnthropicRequest(prompt, options)
	case TypeCustom:
		// For custom types, we would need a custom implementation
		return nil, fmt.Errorf("custom API type requires specialized implementation")
	default:
		return nil, fmt.Errorf("unsupported provider type: %s", p.config.Type)
	}
}

// PrepareRequestWithSchema creates a request with JSON schema validation.
func (p *GenericProvider) PrepareRequestWithSchema(prompt string, options map[string]interface{}, schema interface{}) ([]byte, error) {
	if !p.config.SupportsSchema {
		return nil, fmt.Errorf("provider %s does not support JSON schema validation", p.config.Name)
	}

	switch p.config.Type {
	case TypeOpenAI:
		return p.prepareOpenAIRequest(prompt, options, schema)
	case TypeAnthropic, TypeClaude:
		// Anthropic has a different approach to structured output
		return p.prepareAnthropicStructuredRequest(prompt, options, schema)
	default:
		return nil, fmt.Errorf("JSON schema not supported for provider type: %s", p.config.Type)
	}
}

// ParseResponse extracts the generated text from the API response.
func (p *GenericProvider) ParseResponse(body []byte) (string, error) {
	switch p.config.Type {
	case TypeOpenAI:
		return p.parseOpenAIResponse(body)
	case TypeAnthropic, TypeClaude:
		return p.parseAnthropicResponse(body)
	default:
		return "", fmt.Errorf("unsupported provider type: %s", p.config.Type)
	}
}

// SetExtraHeaders configures additional HTTP headers for API requests.
func (p *GenericProvider) SetExtraHeaders(extraHeaders map[string]string) {
	if extraHeaders == nil {
		extraHeaders = make(map[string]string)
	}
	p.extraHeaders = extraHeaders
}

// HandleFunctionCalls processes function calling capabilities.
func (p *GenericProvider) HandleFunctionCalls(body []byte) ([]byte, error) {
	switch p.config.Type {
	case TypeOpenAI:
		return p.handleOpenAIFunctionCalls(body)
	default:
		// Most providers don't support function calls yet
		return nil, fmt.Errorf("function calls not supported for provider: %s", p.config.Name)
	}
}

// SupportsJSONSchema indicates whether the provider supports native JSON schema validation.
func (p *GenericProvider) SupportsJSONSchema() bool {
	return p.config.SupportsSchema
}

// SetDefaultOptions configures provider-specific defaults from the global configuration.
func (p *GenericProvider) SetDefaultOptions(config *config.Config) {
	// Common options
	p.SetOption("temperature", config.Temperature)
	p.SetOption("max_tokens", config.MaxTokens)

	if config.Seed != nil {
		p.SetOption("seed", *config.Seed)
	}

	p.logger.Debug("Default options set", "temperature", config.Temperature, "max_tokens", config.MaxTokens)
}

// SetOption sets a specific option for the provider.
func (p *GenericProvider) SetOption(key string, value interface{}) {
	p.options[key] = value
}

// SetLogger configures the logger for the provider instance.
func (p *GenericProvider) SetLogger(logger utils.Logger) {
	p.logger = logger
}

// SupportsStreaming indicates whether the provider supports streaming responses.
func (p *GenericProvider) SupportsStreaming() bool {
	return p.config.SupportsStreaming
}

// PrepareStreamRequest creates a request body for streaming API calls.
func (p *GenericProvider) PrepareStreamRequest(prompt string, options map[string]interface{}) ([]byte, error) {
	if !p.SupportsStreaming() {
		return nil, fmt.Errorf("provider %s does not support streaming", p.config.Name)
	}

	// Add streaming flag to options
	streamOptions := make(map[string]interface{})
	for k, v := range options {
		streamOptions[k] = v
	}
	streamOptions["stream"] = true

	return p.PrepareRequest(prompt, streamOptions)
}

// ParseStreamResponse processes a single chunk from a streaming response.
func (p *GenericProvider) ParseStreamResponse(chunk []byte) (string, error) {
	switch p.config.Type {
	case TypeOpenAI:
		return p.parseOpenAIStreamResponse(chunk)
	case TypeAnthropic, TypeClaude:
		return p.parseAnthropicStreamResponse(chunk)
	default:
		return "", fmt.Errorf("streaming not implemented for provider type: %s", p.config.Type)
	}
}

// OpenAI implementation methods
func (p *GenericProvider) prepareOpenAIRequest(prompt string, options map[string]interface{}, schema interface{}) ([]byte, error) {
	requestOptions := make(map[string]interface{})

	// Copy default options
	for k, v := range p.options {
		requestOptions[k] = v
	}

	// Override with passed options
	for k, v := range options {
		requestOptions[k] = v
	}

	// Set model
	requestOptions["model"] = p.model
	images, hasImages := requestOptions["images"].([]types.ContentPart)
	delete(requestOptions, "images")

	// Handle messages format
	if _, ok := requestOptions["messages"]; !ok {
		content := interface{}(prompt)
		if hasImages && len(images) > 0 {
			multimodalContent := []map[string]interface{}{{"type": "text", "text": prompt}}
			multimodalContent = append(multimodalContent, ConvertImagesToOpenAIContent(images)...)
			content = multimodalContent
		}
		requestOptions["messages"] = []map[string]interface{}{
			{
				"role":    "user",
				"content": content,
			},
		}
	}

	// Handle JSON schema if provided
	if schema != nil {
		// Set response format for JSON
		requestOptions["response_format"] = map[string]string{
			"type": "json_object",
		}

		// Add function calling for schema
		requestOptions["functions"] = []map[string]interface{}{
			{
				"name":        "output_formatter",
				"description": "Format the output according to the schema",
				"parameters":  schema,
			},
		}
		requestOptions["function_call"] = map[string]string{
			"name": "output_formatter",
		}
	}

	return json.Marshal(requestOptions)
}

func (p *GenericProvider) parseOpenAIResponse(body []byte) (string, error) {
	var response struct {
		Choices []struct {
			Message struct {
				Content      string `json:"content"`
				FunctionCall struct {
					Arguments string `json:"arguments"`
				} `json:"function_call"`
			} `json:"message"`
			FinishReason string `json:"finish_reason"`
		} `json:"choices"`
		Error struct {
			Message string `json:"message"`
		} `json:"error"`
	}

	if err := json.Unmarshal(body, &response); err != nil {
		return "", fmt.Errorf("failed to parse response: %v", err)
	}

	if response.Error.Message != "" {
		return "", fmt.Errorf("API error: %s", response.Error.Message)
	}

	if len(response.Choices) == 0 {
		return "", fmt.Errorf("empty response from API")
	}

	// Check for function calling response
	if response.Choices[0].Message.FunctionCall.Arguments != "" {
		return response.Choices[0].Message.FunctionCall.Arguments, nil
	}

	return response.Choices[0].Message.Content, nil
}

func (p *GenericProvider) handleOpenAIFunctionCalls(body []byte) ([]byte, error) {
	// Implementation for handling OpenAI function calls
	return body, nil // Simplified for now
}

func (p *GenericProvider) parseOpenAIStreamResponse(chunk []byte) (string, error) {
	// Skip empty chunks
	if len(chunk) == 0 {
		return "", nil
	}

	// Remove "data: " prefix
	data := bytes.TrimPrefix(chunk, []byte("data: "))

	// Handle "[DONE]" marker
	if bytes.Equal(data, []byte("[DONE]")) {
		return "", nil
	}

	var response struct {
		Choices []struct {
			Delta struct {
				Content string `json:"content"`
			} `json:"delta"`
		} `json:"choices"`
	}

	if err := json.Unmarshal(data, &response); err != nil {
		return "", err
	}

	if len(response.Choices) == 0 {
		return "", nil
	}

	return response.Choices[0].Delta.Content, nil
}

// Anthropic implementation methods
func (p *GenericProvider) prepareAnthropicRequest(prompt string, options map[string]interface{}) ([]byte, error) {
	requestOptions := make(map[string]interface{})

	// Copy default options
	for k, v := range p.options {
		requestOptions[k] = v
	}

	// Override with passed options
	for k, v := range options {
		requestOptions[k] = v
	}

	// Set model
	requestOptions["model"] = p.model
	images, hasImages := requestOptions["images"].([]types.ContentPart)
	delete(requestOptions, "images")

	// Handle messages format
	if _, ok := requestOptions["messages"]; !ok {
		content := interface{}(prompt)
		if hasImages && len(images) > 0 {
			multimodalContent := ConvertImagesToAnthropicContent(images)
			multimodalContent = append(multimodalContent, map[string]interface{}{"type": "text", "text": prompt})
			content = multimodalContent
		}
		requestOptions["messages"] = []map[string]interface{}{
			{
				"role":    "user",
				"content": content,
			},
		}
	}

	// Set max tokens if needed
	if maxTokens, ok := requestOptions["max_tokens"]; ok {
		requestOptions["max_tokens"] = maxTokens
	} else {
		requestOptions["max_tokens"] = 1024 // Default
	}

	return json.Marshal(requestOptions)
}

func (p *GenericProvider) prepareAnthropicStructuredRequest(prompt string, options map[string]interface{}, schema interface{}) ([]byte, error) {
	requestOptions := make(map[string]interface{})

	// Copy original options
	for k, v := range options {
		requestOptions[k] = v
	}

	// Add schema to the prompt
	schemaBytes, err := json.Marshal(schema)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal schema: %v", err)
	}

	enhancedPrompt := fmt.Sprintf("%s\n\nPlease provide a response in the following JSON format: %s",
		prompt, string(schemaBytes))

	// Create a new request with the enhanced prompt
	return p.prepareAnthropicRequest(enhancedPrompt, requestOptions)
}

func (p *GenericProvider) parseAnthropicResponse(body []byte) (string, error) {
	var response struct {
		Content []struct {
			Text string `json:"text"`
		} `json:"content"`
		Error struct {
			Message string `json:"message"`
		} `json:"error"`
	}

	if err := json.Unmarshal(body, &response); err != nil {
		return "", fmt.Errorf("failed to parse response: %v", err)
	}

	if response.Error.Message != "" {
		return "", fmt.Errorf("API error: %s", response.Error.Message)
	}

	if len(response.Content) == 0 {
		return "", fmt.Errorf("empty response from API")
	}

	return response.Content[0].Text, nil
}

func (p *GenericProvider) parseAnthropicStreamResponse(chunk []byte) (string, error) {
	// Skip empty chunks
	if len(chunk) == 0 {
		return "", nil
	}

	// Remove "data: " prefix
	data := bytes.TrimPrefix(chunk, []byte("data: "))

	// Handle end event
	if bytes.Equal(data, []byte("[DONE]")) {
		return "", nil
	}

	var response struct {
		Type    string `json:"type"`
		Content []struct {
			Text string `json:"text"`
		} `json:"content"`
	}

	if err := json.Unmarshal(data, &response); err != nil {
		return "", err
	}

	if response.Type == "content_block_delta" && len(response.Content) > 0 {
		return response.Content[0].Text, nil
	}

	return "", nil
}

// PrepareRequestWithMessages creates a request body using structured message objects
// rather than a flattened prompt string. This enables more efficient caching.
//
// Parameters:
//   - messages: Slice of MemoryMessage objects representing the conversation
//   - options: Additional options for the request
//
// Returns:
//   - Serialized JSON request body
//   - Any error encountered during preparation
func (p *GenericProvider) PrepareRequestWithMessages(messages []types.MemoryMessage, options map[string]interface{}) ([]byte, error) {
	switch p.config.Type {
	case TypeOpenAI:
		return p.prepareOpenAIRequestWithMessages(messages, options)
	case TypeAnthropic, TypeClaude:
		return p.prepareAnthropicRequestWithMessages(messages, options)
	case TypeCustom:
		// For custom types, we would need a custom implementation
		return nil, fmt.Errorf("custom API type requires specialized implementation")
	default:
		return nil, fmt.Errorf("unsupported provider type: %s", p.config.Type)
	}
}

// prepareOpenAIRequestWithMessages creates a request for OpenAI APIs using structured messages
func (p *GenericProvider) prepareOpenAIRequestWithMessages(messages []types.MemoryMessage, options map[string]interface{}) ([]byte, error) {
	requestOptions := make(map[string]interface{})

	// Copy default options
	for k, v := range p.options {
		requestOptions[k] = v
	}

	// Override with passed options
	for k, v := range options {
		requestOptions[k] = v
	}

	// Set model
	requestOptions["model"] = p.model

	// Handle messages
	openAIMessages := []map[string]interface{}{}

	// Add system message first if present
	if systemPrompt, ok := options["system_prompt"].(string); ok && systemPrompt != "" {
		openAIMessages = append(openAIMessages, map[string]interface{}{
			"role":    "system",
			"content": systemPrompt,
		})
	}

	// Add all other messages
	for _, msg := range messages {
		message := map[string]interface{}{
			"role":    msg.Role,
			"content": msg.Content,
		}
		openAIMessages = append(openAIMessages, message)
	}

	requestOptions["messages"] = openAIMessages

	return json.Marshal(requestOptions)
}

// prepareAnthropicRequestWithMessages creates a request for Anthropic APIs using structured messages
func (p *GenericProvider) prepareAnthropicRequestWithMessages(messages []types.MemoryMessage, options map[string]interface{}) ([]byte, error) {
	requestOptions := make(map[string]interface{})

	// Copy default options
	for k, v := range p.options {
		requestOptions[k] = v
	}

	// Override with passed options
	for k, v := range options {
		requestOptions[k] = v
	}

	// Set model
	requestOptions["model"] = p.model

	// Set system prompt if provided
	if systemPrompt, ok := options["system_prompt"].(string); ok && systemPrompt != "" {
		requestOptions["system"] = systemPrompt
	}

	// Format messages for Anthropic
	anthropicMessages := []map[string]interface{}{}
	for _, msg := range messages {
		content := []map[string]interface{}{
			{
				"type": "text",
				"text": msg.Content,
			},
		}

		// Add cache_control if specified
		if msg.CacheControl != "" {
			content[0]["cache_control"] = map[string]string{"type": msg.CacheControl}
		}

		message := map[string]interface{}{
			"role":    msg.Role,
			"content": content,
		}

		anthropicMessages = append(anthropicMessages, message)
	}

	requestOptions["messages"] = anthropicMessages

	// Set max tokens if needed
	if maxTokens, ok := requestOptions["max_tokens"]; ok {
		requestOptions["max_tokens"] = maxTokens
	} else {
		requestOptions["max_tokens"] = 1024 // Default
	}

	return json.Marshal(requestOptions)
}
