// File: internal/llm/ollama.go

package llm

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
)

// OllamaProvider implements both the Provider and LLM interfaces for Ollama
type OllamaProvider struct {
	model    string
	endpoint string
	logger   Logger
	options  map[string]interface{}
}

// NewOllamaProvider creates a new OllamaProvider
func NewOllamaProvider(apiKey, model string) Provider {
	return &OllamaProvider{
		model:    model,
		endpoint: "http://localhost:11434", // Default endpoint
		options:  make(map[string]interface{}),
	}
}

// Implement LLM interface methods
func (p *OllamaProvider) Generate(ctx context.Context, prompt *Prompt) (string, string, error) {
	promptString := prompt.String()
	reqBody, err := p.PrepareRequest(promptString, p.options)
	if err != nil {
		return "", "", err
	}

	req, err := http.NewRequestWithContext(ctx, "POST", p.Endpoint(), bytes.NewReader(reqBody))
	if err != nil {
		return "", "", err
	}

	for k, v := range p.Headers() {
		req.Header.Set(k, v)
	}

	client := &http.Client{}
	resp, err := client.Do(req)
	if err != nil {
		return "", "", err
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", "", err
	}

	result, err := p.ParseResponse(body)
	if err != nil {
		return "", "", err
	}

	return result, promptString, nil
}

func (p *OllamaProvider) SetOption(key string, value interface{}) {
	p.options[key] = value
	if p.logger != nil {
		p.logger.Debug("Setting option for Ollama", "key", key, "value", value)
	}
}

// Add the missing SetDebugLevel method
func (p *OllamaProvider) SetDebugLevel(level LogLevel) {
	if p.logger != nil {
		p.logger.SetLevel(level)
	}
}

func (p *OllamaProvider) SetEndpoint(endpoint string) {
	p.endpoint = endpoint
}

// Existing Provider interface methods

func (p *OllamaProvider) Endpoint() string {
	return p.endpoint + "/api/generate"
}

func (p *OllamaProvider) Name() string {
	return "ollama"
}

func (p *OllamaProvider) Headers() map[string]string {
	return map[string]string{
		"Content-Type": "application/json",
	}
}

func (p *OllamaProvider) PrepareRequest(prompt string, options map[string]interface{}) ([]byte, error) {
	requestBody := map[string]interface{}{
		"model":  p.model,
		"prompt": prompt,
	}

	for k, v := range options {
		requestBody[k] = v
	}

	return json.Marshal(requestBody)
}

func (p *OllamaProvider) ParseResponse(body []byte) (string, error) {
	var fullResponse strings.Builder
	decoder := json.NewDecoder(bytes.NewReader(body))

	for decoder.More() {
		var response struct {
			Model    string `json:"model"`
			Response string `json:"response"`
			Done     bool   `json:"done"`
		}
		if err := decoder.Decode(&response); err != nil {
			return "", fmt.Errorf("error parsing Ollama response: %w", err)
		}
		fullResponse.WriteString(response.Response)
		if response.Done {
			break
		}
	}

	return fullResponse.String(), nil
}
