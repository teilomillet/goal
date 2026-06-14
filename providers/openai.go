// Package providers implements LLM provider interfaces and implementations.
package providers

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"strings"

	"github.com/teilomillet/gollm/config"
	"github.com/teilomillet/gollm/types"
	"github.com/teilomillet/gollm/utils"
)

// OpenAIProvider implements the Provider interface for OpenAI's API.
// It supports GPT models and provides access to OpenAI's language model capabilities,
// including function calling, JSON mode, and structured output validation.
type OpenAIProvider struct {
	apiKey       string                 // API key for authentication
	model        string                 // Model identifier (e.g., "gpt-4", "gpt-4o-mini")
	extraHeaders map[string]string      // Additional HTTP headers
	options      map[string]interface{} // Model-specific options
	logger       utils.Logger           // Logger instance
}

// NewOpenAIProvider creates a new OpenAI provider instance.
// It initializes the provider with the given API key, model, and optional headers.
//
// Parameters:
//   - apiKey: OpenAI API key for authentication
//   - model: The model to use (e.g., "gpt-4", "gpt-3.5-turbo")
//   - extraHeaders: Additional HTTP headers for requests
//
// Returns:
//   - A configured OpenAI Provider instance
func NewOpenAIProvider(apiKey, model string, extraHeaders map[string]string) Provider {
	if extraHeaders == nil {
		extraHeaders = make(map[string]string)
	}
	return &OpenAIProvider{
		apiKey:       apiKey,
		model:        model,
		extraHeaders: extraHeaders,
		options:      make(map[string]interface{}),
		logger:       utils.NewLogger(utils.LogLevelInfo),
	}
}

// SetLogger configures the logger for the OpenAI provider.
// This is used for debugging and monitoring API interactions.
func (p *OpenAIProvider) SetLogger(logger utils.Logger) {
	p.logger = logger
}

// needsMaxCompletionTokens checks if the model requires max_completion_tokens instead of max_tokens
func (p *OpenAIProvider) needsMaxCompletionTokens() bool {
	// Check for models that start with "o"
	if strings.HasPrefix(p.model, "o") {
		return true
	}

	// Check for gpt-4o and similar models
	if strings.Contains(p.model, "4o") || strings.Contains(p.model, "-o") {
		return true
	}

	// Check for gpt-5 and newer chat models, which require max_completion_tokens
	// and reject max_tokens with HTTP 400 (e.g. gpt-5, gpt-5-mini, gpt-5.4-2026-03-05)
	if strings.HasPrefix(p.model, "gpt-5") {
		return true
	}

	return false
}

// SetOption sets a specific option for the OpenAI provider.
// Supported options include:
//   - temperature: Controls randomness (0.0 to 2.0)
//   - max_tokens: Maximum tokens in the response (automatically converted to max_completion_tokens for "o" models)
//   - top_p: Nucleus sampling parameter
//   - frequency_penalty: Repetition reduction
//   - presence_penalty: Topic steering
//   - seed: Deterministic sampling seed
func (p *OpenAIProvider) SetOption(key string, value interface{}) {
	// Handle max_tokens conversion for "o" models
	if key == "max_tokens" {
		if p.needsMaxCompletionTokens() {
			// For models requiring max_completion_tokens, use that instead
			key = "max_completion_tokens"
			// Delete max_tokens if it was previously set
			delete(p.options, "max_tokens")
		} else {
			// For models using max_tokens, make sure max_completion_tokens is not set
			delete(p.options, "max_completion_tokens")
		}
	} else if key == "max_completion_tokens" {
		// If explicitly setting max_completion_tokens, remove max_tokens to avoid conflicts
		delete(p.options, "max_tokens")
	}

	p.options[key] = value
	p.logger.Debug("Option set", "key", key, "value", value)
}

// SetDefaultOptions configures standard options from the global configuration.
// This includes temperature, max tokens, and sampling parameters.
func (p *OpenAIProvider) SetDefaultOptions(config *config.Config) {
	p.SetOption("temperature", config.Temperature)
	p.SetOption("max_tokens", config.MaxTokens)
	if config.Seed != nil {
		p.SetOption("seed", *config.Seed)
	}
	p.logger.Debug("Default options set", "temperature", config.Temperature, "max_tokens", config.MaxTokens, "seed", config.Seed)
}

// Name returns "openai" as the provider identifier.
func (p *OpenAIProvider) Name() string {
	return "openai"
}

// Endpoint returns the OpenAI API endpoint URL.
// For API version 1, this is "https://api.openai.com/v1/chat/completions".
func (p *OpenAIProvider) Endpoint() string {
	return "https://api.openai.com/v1/chat/completions"
}

// SupportsJSONSchema indicates that OpenAI supports native JSON schema validation
// through its function calling and JSON mode capabilities.
func (p *OpenAIProvider) SupportsJSONSchema() bool {
	return true
}

// Headers returns the required HTTP headers for OpenAI API requests.
// This includes:
//   - Authorization: Bearer token using the API key
//   - Content-Type: application/json
//   - Any additional headers specified via SetExtraHeaders
func (p *OpenAIProvider) Headers() map[string]string {
	headers := map[string]string{
		"Content-Type":  "application/json",
		"Authorization": "Bearer " + p.apiKey,
	}

	for key, value := range p.extraHeaders {
		headers[key] = value
	}

	p.logger.Debug("Headers prepared", "headers", headers)
	return headers
}

// PrepareRequest creates the request body for an OpenAI API call.
// It handles:
//   - Message formatting
//   - System messages
//   - Function/tool definitions
//   - Model-specific options
//
// Parameters:
//   - prompt: The input text or conversation
//   - options: Additional parameters for the request
//
// Returns:
//   - Serialized JSON request body
//   - Any error encountered during preparation
func (p *OpenAIProvider) PrepareRequest(prompt string, options map[string]interface{}) ([]byte, error) {
	request := map[string]interface{}{
		"model":    p.model,
		"messages": []map[string]interface{}{},
	}

	// Handle system prompt as developer message
	if systemPrompt, ok := options["system_prompt"].(string); ok && systemPrompt != "" {
		request["messages"] = append(request["messages"].([]map[string]interface{}), map[string]interface{}{
			"role":    "developer",
			"content": systemPrompt,
		})
	}

	// Check if we have images to include
	images, hasImages := options["images"].([]types.ContentPart)

	// Add user message (with or without images)
	if hasImages && len(images) > 0 {
		// Build multimodal content array using shared helper
		content := []map[string]interface{}{
			{"type": "text", "text": prompt},
		}
		content = append(content, ConvertImagesToOpenAIContent(images)...)
		request["messages"] = append(request["messages"].([]map[string]interface{}), map[string]interface{}{
			"role":    "user",
			"content": content,
		})
	} else {
		// Simple text-only message
		request["messages"] = append(request["messages"].([]map[string]interface{}), map[string]interface{}{
			"role":    "user",
			"content": prompt,
		})
	}

	// Handle tool_choice
	if toolChoice, ok := options["tool_choice"].(string); ok {
		request["tool_choice"] = toolChoice
	}

	// Handle tools
	if tools, ok := options["tools"].([]utils.Tool); ok && len(tools) > 0 {
		openAITools := make([]map[string]interface{}, len(tools))
		for i, tool := range tools {
			openAITools[i] = map[string]interface{}{
				"type": "function",
				"function": map[string]interface{}{
					"name":        tool.Function.Name,
					"description": tool.Function.Description,
					"parameters":  tool.Function.Parameters,
				},
				"strict": true, // Add this if you want strict mode
			}
		}
		request["tools"] = openAITools
	}

	// Create a merged copy of options to handle token parameters properly
	mergedOptions := make(map[string]interface{})

	// First add options from provider (p.options)
	for k, v := range p.options {
		if k != "tools" && k != "tool_choice" && k != "system_prompt" {
			mergedOptions[k] = v
		}
	}

	// Then add options from the function parameters (may override provider options)
	for k, v := range options {
		if k != "tools" && k != "tool_choice" && k != "system_prompt" && k != "images" {
			mergedOptions[k] = v
		}
	}

	// Handle max_tokens/max_completion_tokens conflict
	// For models that need max_completion_tokens, ensure we use that and not max_tokens
	if p.needsMaxCompletionTokens() {
		if _, hasMaxTokens := mergedOptions["max_tokens"]; hasMaxTokens {
			// Move max_tokens value to max_completion_tokens
			mergedOptions["max_completion_tokens"] = mergedOptions["max_tokens"]
			delete(mergedOptions, "max_tokens")
		}
	} else {
		// For other models, ensure we use max_tokens and not max_completion_tokens
		if _, hasMaxCompletionTokens := mergedOptions["max_completion_tokens"]; hasMaxCompletionTokens {
			// Move max_completion_tokens value to max_tokens
			mergedOptions["max_tokens"] = mergedOptions["max_completion_tokens"]
			delete(mergedOptions, "max_completion_tokens")
		}
	}

	// Add merged options to the request
	for k, v := range mergedOptions {
		request[k] = v
	}

	return json.Marshal(request)
}

// PrepareRequestWithSchema creates a request that includes JSON schema validation.
// This uses OpenAI's function calling feature to enforce response structure.
//
// Parameters:
//   - prompt: The input text or conversation
//   - options: Additional request parameters
//   - schema: JSON schema for response validation
//
// Returns:
//   - Serialized JSON request body
//   - Any error encountered during preparation
func (p *OpenAIProvider) PrepareRequestWithSchema(prompt string, options map[string]interface{}, schema interface{}) ([]byte, error) {
	p.logger.Debug("Preparing request with schema", "prompt", prompt, "schema", schema)

	// First, ensure we have a proper object for the schema
	var schemaObj interface{}
	switch s := schema.(type) {
	case string:
		if err := json.Unmarshal([]byte(s), &schemaObj); err != nil {
			return nil, fmt.Errorf("failed to unmarshal schema string: %w", err)
		}
	case []byte:
		if err := json.Unmarshal(s, &schemaObj); err != nil {
			return nil, fmt.Errorf("failed to unmarshal schema bytes: %w", err)
		}
	case map[string]interface{}:
		schemaObj = s
	default:
		// Try to marshal and unmarshal to ensure we have a proper object
		schemaBytes, err := json.Marshal(schema)
		if err != nil {
			return nil, fmt.Errorf("failed to marshal schema: %w", err)
		}
		if err := json.Unmarshal(schemaBytes, &schemaObj); err != nil {
			return nil, fmt.Errorf("failed to unmarshal schema: %w", err)
		}
	}

	// Clean the schema for OpenAI by removing unsupported validation rules
	cleanSchema := cleanSchemaForOpenAI(schemaObj)

	// Debug log the cleaned schema
	cleanSchemaJSON, _ := json.MarshalIndent(cleanSchema, "", "  ")
	p.logger.Debug("Cleaned schema for OpenAI", "schema", string(cleanSchemaJSON))

	request := map[string]interface{}{
		"model": p.model,
		"messages": []map[string]interface{}{
			{"role": "user", "content": prompt},
		},
		"response_format": map[string]interface{}{
			"type": "json_schema",
			"json_schema": map[string]interface{}{
				"name":   "structured_response",
				"schema": cleanSchema,
				"strict": true,
			},
		},
	}

	// Handle system prompt as system message
	if systemPrompt, ok := options["system_prompt"].(string); ok && systemPrompt != "" {
		request["messages"] = append([]map[string]interface{}{
			{"role": "system", "content": systemPrompt},
		}, request["messages"].([]map[string]interface{})...)
	}

	// Create a merged copy of options to handle token parameters properly
	mergedOptions := make(map[string]interface{})

	// First add options from provider (p.options)
	for k, v := range p.options {
		if k != "system_prompt" {
			mergedOptions[k] = v
		}
	}

	// Then add options from the function parameters (may override provider options)
	for k, v := range options {
		if k != "system_prompt" {
			mergedOptions[k] = v
		}
	}

	// Handle max_tokens/max_completion_tokens conflict
	// For models that need max_completion_tokens, ensure we use that and not max_tokens
	if p.needsMaxCompletionTokens() {
		if _, hasMaxTokens := mergedOptions["max_tokens"]; hasMaxTokens {
			// Move max_tokens value to max_completion_tokens
			mergedOptions["max_completion_tokens"] = mergedOptions["max_tokens"]
			delete(mergedOptions, "max_tokens")
		}
	} else {
		// For other models, ensure we use max_tokens and not max_completion_tokens
		if _, hasMaxCompletionTokens := mergedOptions["max_completion_tokens"]; hasMaxCompletionTokens {
			// Move max_completion_tokens value to max_tokens
			mergedOptions["max_tokens"] = mergedOptions["max_completion_tokens"]
			delete(mergedOptions, "max_completion_tokens")
		}
	}

	// Add merged options to the request
	for k, v := range mergedOptions {
		request[k] = v
	}

	reqJSON, err := json.Marshal(request)
	if err != nil {
		p.logger.Error("Failed to marshal request with schema", "error", err)
		return nil, err
	}

	p.logger.Debug("Full request to OpenAI", "request", string(reqJSON))
	return reqJSON, nil
}

// cleanSchemaForOpenAI removes validation rules that OpenAI doesn't support
func cleanSchemaForOpenAI(schema interface{}) interface{} {
	if schemaMap, ok := schema.(map[string]interface{}); ok {
		result := make(map[string]interface{})
		for k, v := range schemaMap {
			switch k {
			case "type", "properties", "required", "items":
				if k == "properties" {
					props := make(map[string]interface{})
					if propsMap, ok := v.(map[string]interface{}); ok {
						for propName, propSchema := range propsMap {
							props[propName] = cleanSchemaForOpenAI(propSchema)
						}
					}
					result[k] = props
				} else if k == "items" {
					result[k] = cleanSchemaForOpenAI(v)
				} else {
					result[k] = v
				}
			}
		}
		// Add additionalProperties: false at each object level
		if schemaMap["type"] == "object" {
			result["additionalProperties"] = false
		}
		return result
	}
	return schema
}

// ParseResponse extracts the generated text from the OpenAI API response.
// It handles various response formats and error cases.
//
// Parameters:
//   - body: Raw API response body
//
// Returns:
//   - Generated text content
//   - Any error encountered during parsing
func (p *OpenAIProvider) ParseResponse(body []byte) (string, error) {
	var response struct {
		Choices []struct {
			Message struct {
				Content   string `json:"content"`
				ToolCalls []struct {
					ID       string `json:"id"`
					Type     string `json:"type"`
					Function struct {
						Name      string          `json:"name"`
						Arguments json.RawMessage `json:"arguments"`
					} `json:"function"`
				} `json:"tool_calls"`
			} `json:"message"`
		} `json:"choices"`
	}

	if err := json.Unmarshal(body, &response); err != nil {
		return "", err
	}

	if len(response.Choices) == 0 {
		return "", fmt.Errorf("empty response from API")
	}

	message := response.Choices[0].Message
	if message.Content != "" {
		return message.Content, nil
	}

	if len(message.ToolCalls) > 0 {
		var functionCalls []string
		for _, call := range message.ToolCalls {
			// Parse arguments as raw JSON to preserve the exact format
			var args interface{}
			if err := json.Unmarshal(call.Function.Arguments, &args); err != nil {
				return "", fmt.Errorf("error parsing function arguments: %w", err)
			}

			functionCall, err := utils.FormatFunctionCall(call.Function.Name, args)
			if err != nil {
				return "", fmt.Errorf("error formatting function call: %w", err)
			}
			functionCalls = append(functionCalls, functionCall)
		}
		return strings.Join(functionCalls, "\n"), nil
	}

	return "", fmt.Errorf("no content or tool calls in response")
}

// HandleFunctionCalls processes function calling in the response.
// This supports OpenAI's function calling and JSON mode features.
func (p *OpenAIProvider) HandleFunctionCalls(body []byte) ([]byte, error) {
	response := string(body)
	functionCalls, err := utils.ExtractFunctionCalls(response)
	if err != nil {
		return nil, fmt.Errorf("error extracting function calls: %w", err)
	}

	if len(functionCalls) == 0 {
		return nil, fmt.Errorf("no function calls found in response")
	}

	p.logger.Debug("Function calls to handle", "calls", functionCalls)
	return json.Marshal(functionCalls)
}

// SetExtraHeaders configures additional HTTP headers for API requests.
// This allows for custom headers needed for specific features or requirements.
func (p *OpenAIProvider) SetExtraHeaders(extraHeaders map[string]string) {
	p.extraHeaders = extraHeaders
	p.logger.Debug("Extra headers set", "headers", extraHeaders)
}

// SupportsStreaming indicates whether streaming is supported
func (p *OpenAIProvider) SupportsStreaming() bool {
	return true
}

// PrepareStreamRequest creates a request body for streaming API calls
func (p *OpenAIProvider) PrepareStreamRequest(prompt string, options map[string]interface{}) ([]byte, error) {
	// Start with regular request preparation
	requestBody := map[string]interface{}{
		"model": p.model,
		"messages": []map[string]string{
			{"role": "user", "content": prompt},
		},
		"stream": true,
	}

	// Create a merged copy of options to handle token parameters properly
	mergedOptions := make(map[string]interface{})

	// First add options from provider (p.options)
	for k, v := range p.options {
		if k != "stream" { // Don't override stream setting
			mergedOptions[k] = v
		}
	}

	// Then add options from the function parameters (may override provider options)
	for k, v := range options {
		if k != "stream" { // Don't override stream setting
			mergedOptions[k] = v
		}
	}

	// Handle max_tokens/max_completion_tokens conflict
	// For models that need max_completion_tokens, ensure we use that and not max_tokens
	if p.needsMaxCompletionTokens() {
		if _, hasMaxTokens := mergedOptions["max_tokens"]; hasMaxTokens {
			// Move max_tokens value to max_completion_tokens
			mergedOptions["max_completion_tokens"] = mergedOptions["max_tokens"]
			delete(mergedOptions, "max_tokens")
		}
	} else {
		// For other models, ensure we use max_tokens and not max_completion_tokens
		if _, hasMaxCompletionTokens := mergedOptions["max_completion_tokens"]; hasMaxCompletionTokens {
			// Move max_completion_tokens value to max_tokens
			mergedOptions["max_tokens"] = mergedOptions["max_completion_tokens"]
			delete(mergedOptions, "max_completion_tokens")
		}
	}

	// Add merged options to the request
	for k, v := range mergedOptions {
		requestBody[k] = v
	}

	return json.Marshal(requestBody)
}

// ParseStreamResponse processes a single chunk from a streaming response
func (p *OpenAIProvider) ParseStreamResponse(chunk []byte) (string, error) {
	// Skip empty lines
	if len(bytes.TrimSpace(chunk)) == 0 {
		return "", fmt.Errorf("empty chunk")
	}

	// Check for [DONE] marker
	if bytes.Equal(bytes.TrimSpace(chunk), []byte("[DONE]")) {
		return "", io.EOF
	}

	// Parse the chunk
	var response struct {
		Choices []struct {
			Delta struct {
				Role    string `json:"role"`
				Content string `json:"content"`
			} `json:"delta"`
			FinishReason string `json:"finish_reason"`
		} `json:"choices"`
	}

	if err := json.Unmarshal(chunk, &response); err != nil {
		return "", fmt.Errorf("malformed response: %w", err)
	}

	if len(response.Choices) == 0 {
		return "", fmt.Errorf("no choices in response")
	}

	// Handle finish reason
	if response.Choices[0].FinishReason != "" {
		return "", io.EOF
	}

	// Skip role-only messages
	if response.Choices[0].Delta.Role != "" && response.Choices[0].Delta.Content == "" {
		return "", fmt.Errorf("skip token")
	}

	return response.Choices[0].Delta.Content, nil
}

// PrepareRequestWithMessages creates a request body using structured message objects
// rather than a flattened prompt string. This enables more efficient caching and
// better preserves conversation structure for the OpenAI API.
//
// Parameters:
//   - messages: Slice of MemoryMessage objects representing the conversation
//   - options: Additional options for the request
//
// Returns:
//   - Serialized JSON request body
//   - Any error encountered during preparation
func (p *OpenAIProvider) PrepareRequestWithMessages(messages []types.MemoryMessage, options map[string]interface{}) ([]byte, error) {
	request := map[string]interface{}{
		"model":    p.model,
		"messages": []map[string]interface{}{},
	}

	// Handle system prompt as system message
	if systemPrompt, ok := options["system_prompt"].(string); ok && systemPrompt != "" {
		request["messages"] = append(request["messages"].([]map[string]interface{}), map[string]interface{}{
			"role":    "system",
			"content": systemPrompt,
		})
	}

	// Convert MemoryMessage objects to OpenAI messages format
	for _, msg := range messages {
		message := map[string]interface{}{
			"role": msg.Role,
		}

		// Handle tool result messages (role=tool)
		if msg.Role == "tool" && msg.ToolCallID != "" {
			message["tool_call_id"] = msg.ToolCallID
			message["content"] = msg.Content
		} else if len(msg.ToolCalls) > 0 {
			// Handle assistant messages with tool calls
			// OpenAI expects tool_calls array for assistant messages
			toolCalls := make([]map[string]interface{}, len(msg.ToolCalls))
			for i, tc := range msg.ToolCalls {
				toolCalls[i] = map[string]interface{}{
					"id":   tc.ID,
					"type": tc.Type,
					"function": map[string]interface{}{
						"name":      tc.Function.Name,
						"arguments": string(tc.Function.Arguments),
					},
				}
			}
			message["tool_calls"] = toolCalls
			// Content can be empty or contain text alongside tool calls
			if msg.Content != "" {
				message["content"] = msg.Content
			}
		} else if msg.HasMultiContent() {
			// Handle multimodal content (text + images)
			message["content"] = BuildOpenAIContentFromParts(msg.MultiContent)
		} else {
			// Regular text message
			message["content"] = msg.Content
		}

		// Add metadata if present (excluding tool-related fields already handled)
		if len(msg.Metadata) > 0 {
			for k, v := range msg.Metadata {
				if k != "tool_calls" && k != "tool_call_id" {
					message[k] = v
				}
			}
		}

		request["messages"] = append(request["messages"].([]map[string]interface{}), message)
	}

	// Handle tool_choice
	if toolChoice, ok := options["tool_choice"].(string); ok {
		request["tool_choice"] = toolChoice
	}

	// Handle tools
	if tools, ok := options["tools"].([]utils.Tool); ok && len(tools) > 0 {
		openAITools := make([]map[string]interface{}, len(tools))
		for i, tool := range tools {
			openAITools[i] = map[string]interface{}{
				"type": "function",
				"function": map[string]interface{}{
					"name":        tool.Function.Name,
					"description": tool.Function.Description,
					"parameters":  tool.Function.Parameters,
				},
				"strict": true, // Add this if you want strict mode
			}
		}
		request["tools"] = openAITools
	}

	// Create a merged copy of options to handle token parameters properly
	mergedOptions := make(map[string]interface{})

	// First add options from provider (p.options)
	for k, v := range p.options {
		if k != "tools" && k != "tool_choice" && k != "system_prompt" && k != "structured_messages" {
			mergedOptions[k] = v
		}
	}

	// Then add options from the function parameters (may override provider options)
	for k, v := range options {
		if k != "tools" && k != "tool_choice" && k != "system_prompt" && k != "structured_messages" && k != "images" {
			mergedOptions[k] = v
		}
	}

	// Handle max_tokens/max_completion_tokens conflict
	// For models that need max_completion_tokens, ensure we use that and not max_tokens
	if p.needsMaxCompletionTokens() {
		if _, hasMaxTokens := mergedOptions["max_tokens"]; hasMaxTokens {
			// Move max_tokens value to max_completion_tokens
			mergedOptions["max_completion_tokens"] = mergedOptions["max_tokens"]
			delete(mergedOptions, "max_tokens")
		}
	} else {
		// For other models, ensure we use max_tokens and not max_completion_tokens
		if _, hasMaxCompletionTokens := mergedOptions["max_completion_tokens"]; hasMaxCompletionTokens {
			// Move max_completion_tokens value to max_tokens
			mergedOptions["max_tokens"] = mergedOptions["max_completion_tokens"]
			delete(mergedOptions, "max_completion_tokens")
		}
	}

	// Add merged options to the request
	for k, v := range mergedOptions {
		request[k] = v
	}

	return json.Marshal(request)
}
