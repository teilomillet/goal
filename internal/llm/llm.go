// File: internal/llm/llm.go

package llm

import (
	"bytes"
	"context"
	"fmt"
	"io"
	"net/http"
	"time"
)

// LLM interface defines the methods that our internal language model should implement
type LLM interface {
	Generate(ctx context.Context, prompt *Prompt) (response string, fullPrompt string, err error)
	SetOption(key string, value interface{})
	SetDebugLevel(level LogLevel)
	SetEndpoint(endpoint string)
}

type Provider interface {
	Name() string
	Endpoint() string
	Headers() map[string]string
	PrepareRequest(prompt string, options map[string]interface{}) ([]byte, error)
	ParseResponse(body []byte) (string, error)
}

// LLMImpl is our implementation of the internal LLM interface
type LLMImpl struct {
	Provider   Provider
	Options    map[string]interface{}
	client     *http.Client
	logger     Logger
	config     *Config
	MaxRetries int
	RetryDelay time.Duration
}

func NewLLM(config *Config, logger Logger, registry *ProviderRegistry) (LLM, error) {
	provider, err := registry.Get(config.Provider, config.APIKeys[config.Provider], config.Model)
	if err != nil {
		return nil, err
	}

	logger.SetLevel(config.LogLevel)

	// Special handling for Ollama provider
	if config.Provider == "ollama" {
		ollamaProvider, ok := provider.(*OllamaProvider)
		if !ok {
			return nil, fmt.Errorf("unexpected provider type for Ollama")
		}
		if config.OllamaEndpoint != "" {
			ollamaProvider.SetEndpoint(config.OllamaEndpoint)
		}
		return ollamaProvider, nil
	}

	llmClient := &LLMImpl{
		Provider:   provider,
		Options:    make(map[string]interface{}),
		client:     &http.Client{Timeout: config.Timeout},
		logger:     logger,
		config:     config,
		MaxRetries: config.MaxRetries,
		RetryDelay: config.RetryDelay,
	}
	llmClient.SetOption("temperature", config.Temperature)
	llmClient.SetOption("max_tokens", config.MaxTokens)

	return llmClient, nil
}

func (l *LLMImpl) SetOption(key string, value interface{}) {
	l.Options[key] = value
	l.logger.Debug("Option set", key, value)
}

func (l *LLMImpl) SetEndpoint(endpoint string) {
	// This is a no-op for non-Ollama providers
	l.logger.Debug("SetEndpoint called on non-Ollama provider", "endpoint", endpoint)
}

// SetDebugLevel updates the debug level for the internal LLM
func (l *LLMImpl) SetDebugLevel(level LogLevel) {
	l.logger.Debug("Setting internal LLM debug level", "new_level", level)
	l.logger.SetLevel(level)
}

func (l *LLMImpl) Generate(ctx context.Context, prompt *Prompt) (string, string, error) {
	promptString := prompt.String()
	var result string
	var lastErr error

	for attempt := 0; attempt <= l.MaxRetries; attempt++ {
		l.logger.Debug("Generating text", "provider", l.Provider.Name(), "prompt", promptString, "attempt", attempt+1)

		result, lastErr = l.attemptGenerate(ctx, prompt.String())
		if lastErr == nil {
			return result, promptString, nil
		}

		l.logger.Warn("Generation attempt failed", "error", lastErr, "attempt", attempt+1)

		if attempt < l.MaxRetries {
			l.logger.Debug("Retrying", "delay", l.RetryDelay)
			select {
			case <-ctx.Done():
				return "", promptString, ctx.Err()
			case <-time.After(l.RetryDelay):
				// Continue to next attempt
			}
		}
	}

	return "", promptString, fmt.Errorf("failed to generate after %d attempts: %w", l.MaxRetries+1, lastErr)
}

func (l *LLMImpl) attemptGenerate(ctx context.Context, prompt string) (string, error) {
	reqBody, err := l.Provider.PrepareRequest(prompt, l.Options)
	if err != nil {
		return "", NewLLMError(ErrorTypeRequest, "failed to prepare request", err)
	}
	l.logger.Debug("Request body", "provider", l.Provider.Name(), "body", string(reqBody))

	req, err := http.NewRequestWithContext(ctx, "POST", l.Provider.Endpoint(), bytes.NewReader(reqBody))
	if err != nil {
		return "", NewLLMError(ErrorTypeRequest, "failed to create request", err)
	}

	for k, v := range l.Provider.Headers() {
		req.Header.Set(k, v)
		l.logger.Debug("Request header", "provider", l.Provider.Name(), "key", k, "value", v)
	}

	resp, err := l.client.Do(req)
	if err != nil {
		return "", NewLLMError(ErrorTypeRequest, "failed to send request", err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", NewLLMError(ErrorTypeResponse, "failed to read response body", err)
	}

	if resp.StatusCode != http.StatusOK {
		l.logger.Error("API error", "provider", l.Provider.Name(), "status", resp.StatusCode, "body", string(body))
		return "", NewLLMError(ErrorTypeAPI, fmt.Sprintf("API error: status code %d", resp.StatusCode), nil)
	}

	result, err := l.Provider.ParseResponse(body)
	if err != nil {
		return "", NewLLMError(ErrorTypeResponse, "failed to parse response", err)
	}

	l.logger.Debug("Text generated successfully", "result", result)
	return result, nil
}
