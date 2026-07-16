// Package providers implements LLM provider interfaces and implementations.
package providers

const (
	MiniMaxModelM3  = "MiniMax-M3"
	MiniMaxModelM27 = "MiniMax-M2.7"
)

// NewMiniMaxProvider creates a MiniMax provider for the global chat completions endpoint.
func NewMiniMaxProvider(apiKey, model string, extraHeaders map[string]string) Provider {
	return NewGenericProvider(apiKey, model, "minimax", extraHeaders)
}

// NewMiniMaxCNProvider creates a MiniMax provider for the China chat completions endpoint.
func NewMiniMaxCNProvider(apiKey, model string, extraHeaders map[string]string) Provider {
	return NewGenericProvider(apiKey, model, "minimax-cn", extraHeaders)
}

// NewMiniMaxMessagesProvider creates a MiniMax provider for the global messages endpoint.
func NewMiniMaxMessagesProvider(apiKey, model string, extraHeaders map[string]string) Provider {
	return NewGenericProvider(apiKey, model, "minimax-messages", extraHeaders)
}

// NewMiniMaxMessagesCNProvider creates a MiniMax provider for the China messages endpoint.
func NewMiniMaxMessagesCNProvider(apiKey, model string, extraHeaders map[string]string) Provider {
	return NewGenericProvider(apiKey, model, "minimax-messages-cn", extraHeaders)
}
