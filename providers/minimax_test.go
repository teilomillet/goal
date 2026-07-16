package providers

import (
	"bytes"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"github.com/teilomillet/gollm/types"
	"github.com/teilomillet/gollm/utils"
)

func TestMiniMaxProviderVariants(t *testing.T) {
	tests := []struct {
		name         string
		providerName string
		baseURL      string
		requestPath  string
		endpoint     string
		providerType ProviderType
	}{
		{
			name:         "global chat completions",
			providerName: "minimax",
			baseURL:      "https://api.minimax.io/v1",
			requestPath:  "chat/completions",
			endpoint:     "https://api.minimax.io/v1/chat/completions",
			providerType: TypeOpenAI,
		},
		{
			name:         "China chat completions",
			providerName: "minimax-cn",
			baseURL:      "https://api.minimaxi.com/v1",
			requestPath:  "chat/completions",
			endpoint:     "https://api.minimaxi.com/v1/chat/completions",
			providerType: TypeOpenAI,
		},
		{
			name:         "global messages",
			providerName: "minimax-messages",
			baseURL:      "https://api.minimax.io/anthropic",
			requestPath:  "v1/messages",
			endpoint:     "https://api.minimax.io/anthropic/v1/messages",
			providerType: TypeAnthropic,
		},
		{
			name:         "China messages",
			providerName: "minimax-messages-cn",
			baseURL:      "https://api.minimaxi.com/anthropic",
			requestPath:  "v1/messages",
			endpoint:     "https://api.minimaxi.com/anthropic/v1/messages",
			providerType: TypeAnthropic,
		},
	}

	registry := GetDefaultRegistry()
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			providerConfig, exists := registry.GetProviderConfig(tt.providerName)
			require.True(t, exists)
			assert.Equal(t, tt.baseURL, providerConfig.BaseURL)
			assert.Equal(t, tt.requestPath, providerConfig.RequestPath)
			assert.Equal(t, tt.providerType, providerConfig.Type)

			for _, model := range []string{MiniMaxModelM3, MiniMaxModelM27} {
				t.Run(model, func(t *testing.T) {
					provider, err := registry.Get(tt.providerName, "test-key", model, nil)
					require.NoError(t, err)
					assert.Equal(t, tt.providerName, provider.Name())
					assert.Equal(t, tt.endpoint, provider.Endpoint())
					assert.Equal(t, "Bearer test-key", provider.Headers()["Authorization"])

					body, err := provider.PrepareRequest("Hello", nil)
					require.NoError(t, err)
					assert.Contains(t, string(body), `"model":"`+model+`"`)
				})
			}
		})
	}
}

func TestMiniMaxM3ImageRequests(t *testing.T) {
	image := types.NewImageURLContent("https://example.com/image.png", "auto")
	registry := GetDefaultRegistry()

	tests := []struct {
		providerName string
		expected     string
	}{
		{providerName: "minimax", expected: `"type":"image_url"`},
		{providerName: "minimax-messages", expected: `"type":"image"`},
	}

	for _, tt := range tests {
		t.Run(tt.providerName, func(t *testing.T) {
			provider, err := registry.Get(tt.providerName, "test-key", MiniMaxModelM3, nil)
			require.NoError(t, err)

			body, err := provider.PrepareRequest("Describe this image", map[string]interface{}{
				"images": []types.ContentPart{image},
			})
			require.NoError(t, err)
			assert.Contains(t, string(body), tt.expected)
			assert.NotContains(t, string(body), `"images"`)
		})
	}
}

func TestMiniMaxMessagesRequestPathCapture(t *testing.T) {
	requestPath := make(chan string, 1)
	requestHeaders := make(chan http.Header, 1)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		requestPath <- r.URL.Path
		requestHeaders <- r.Header.Clone()
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"content":[{"type":"text","text":"ok"}]}`)
	}))
	defer server.Close()

	provider := &GenericProvider{
		apiKey: "test-key",
		model:  MiniMaxModelM3,
		config: ProviderConfig{
			Name:            "minimax-messages",
			Type:            TypeAnthropic,
			BaseURL:         server.URL + "/anthropic",
			RequestPath:     "v1/messages",
			AuthHeader:      "Authorization",
			AuthPrefix:      "Bearer ",
			RequiredHeaders: map[string]string{"anthropic-version": "2023-06-01"},
		},
		extraHeaders: map[string]string{},
		options:      map[string]interface{}{},
		logger:       utils.NewLogger(utils.LogLevelInfo),
	}

	body, err := provider.PrepareRequest("Hello", nil)
	require.NoError(t, err)
	request, err := http.NewRequest(http.MethodPost, provider.Endpoint(), bytes.NewReader(body))
	require.NoError(t, err)
	for key, value := range provider.Headers() {
		request.Header.Set(key, value)
	}
	response, err := http.DefaultClient.Do(request)
	require.NoError(t, err)
	defer response.Body.Close()

	assert.Equal(t, "/anthropic/v1/messages", <-requestPath)
	headers := <-requestHeaders
	assert.Equal(t, "Bearer test-key", headers.Get("Authorization"))
	assert.Equal(t, "2023-06-01", headers.Get("anthropic-version"))
	assert.Contains(t, string(body), `"model":"MiniMax-M3"`)
}
