package providers

import (
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/assert"
)

// TestNeedsMaxCompletionTokens verifies that the needsMaxCompletionTokens function
// correctly identifies models that require max_completion_tokens instead of max_tokens
func TestNeedsMaxCompletionTokens(t *testing.T) {
	testCases := []struct {
		modelName      string
		expectedResult bool
		description    string
	}{
		{"gpt-4", false, "Standard GPT-4 model should not use max_completion_tokens"},
		{"gpt-4-turbo", false, "GPT-4 Turbo model should not use max_completion_tokens"},
		{"gpt-3.5-turbo", false, "GPT-3.5 Turbo model should not use max_completion_tokens"},
		{"o1-preview", true, "o1-preview model should use max_completion_tokens"},
		{"o-preview", true, "o-preview model should use max_completion_tokens"},
		{"gpt-4o", true, "GPT-4o model should use max_completion_tokens"},
		{"gpt-4o-mini", true, "GPT-4o mini model should use max_completion_tokens"},
		{"gpt-5", true, "GPT-5 model should use max_completion_tokens"},
		{"gpt-5-mini", true, "GPT-5 mini model should use max_completion_tokens"},
		{"gpt-5.4-2026-03-05", true, "Dated GPT-5 model should use max_completion_tokens"},
	}

	for _, tc := range testCases {
		t.Run(tc.description, func(t *testing.T) {
			// Create a provider with the test model
			provider := NewOpenAIProvider("fake-api-key", tc.modelName, nil)

			// Get the OpenAIProvider concrete type from the Provider interface
			openAIProvider, ok := provider.(*OpenAIProvider)
			assert.True(t, ok, "Provider should be of type *OpenAIProvider")

			// Test the needsMaxCompletionTokens function
			result := openAIProvider.needsMaxCompletionTokens()
			assert.Equal(t, tc.expectedResult, result, "needsMaxCompletionTokens returned unexpected result for model %s", tc.modelName)
		})
	}
}

// TestPrepareRequestGPT5UsesMaxCompletionTokens verifies that a gpt-5 request
// carries max_completion_tokens and never max_tokens — the parameter gpt-5-class
// models reject with HTTP 400.
func TestPrepareRequestGPT5UsesMaxCompletionTokens(t *testing.T) {
	provider, ok := NewOpenAIProvider("fake-api-key", "gpt-5.4-2026-03-05", nil).(*OpenAIProvider)
	assert.True(t, ok, "Provider should be of type *OpenAIProvider")

	body, err := provider.PrepareRequest("hello", map[string]interface{}{"max_tokens": 4096})
	assert.NoError(t, err)

	var req map[string]interface{}
	assert.NoError(t, json.Unmarshal(body, &req))

	_, hasMaxTokens := req["max_tokens"]
	maxCompletion, hasMaxCompletion := req["max_completion_tokens"]
	assert.False(t, hasMaxTokens, "gpt-5 request must not send max_tokens (OpenAI rejects it with HTTP 400)")
	assert.True(t, hasMaxCompletion, "gpt-5 request must send max_completion_tokens instead")
	assert.Equal(t, float64(4096), maxCompletion, "token budget must be preserved across the conversion")
}
