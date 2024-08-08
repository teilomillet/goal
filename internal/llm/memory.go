// File: internal/llm/memory.go

package llm

import (
	"context"
	"fmt"
	"sync"

	"github.com/pkoukk/tiktoken-go"
)

type Memory struct {
	messages    []Message
	mutex       sync.Mutex
	totalTokens int
	maxTokens   int
	encoding    *tiktoken.Tiktoken
	logger      Logger
}

type Message struct {
	Role    string
	Content string
	Tokens  int
}

func NewMemory(maxTokens int, model string, logger Logger) (*Memory, error) {
	encoding, err := tiktoken.EncodingForModel(model)
	if err != nil {
		return nil, fmt.Errorf("failed to get encoding for model: %v", err)
	}

	return &Memory{
		messages:  []Message{},
		maxTokens: maxTokens,
		encoding:  encoding,
		logger:    logger,
	}, nil
}

func (m *Memory) Add(role, content string) {
	m.mutex.Lock()
	defer m.mutex.Unlock()

	tokens := m.encoding.Encode(content, nil, nil)
	message := Message{Role: role, Content: content, Tokens: len(tokens)}
	m.messages = append(m.messages, message)
	m.totalTokens += len(tokens)

	m.truncate()
	m.logger.Debug("Added message to memory", "role", role, "tokens", len(tokens), "total_tokens", m.totalTokens)
}

func (m *Memory) truncate() {
	for m.totalTokens > m.maxTokens && len(m.messages) > 1 {
		removed := m.messages[0]
		m.messages = m.messages[1:]
		m.totalTokens -= removed.Tokens
		m.logger.Debug("Removed message from memory", "role", removed.Role, "tokens", removed.Tokens, "total_tokens", m.totalTokens)
	}
}

func (m *Memory) GetPrompt() string {
	m.mutex.Lock()
	defer m.mutex.Unlock()

	var prompt string
	for _, msg := range m.messages {
		prompt += fmt.Sprintf("%s: %s\n", msg.Role, msg.Content)
	}
	return prompt
}

func (m *Memory) Clear() {
	m.mutex.Lock()
	defer m.mutex.Unlock()

	m.messages = []Message{}
	m.totalTokens = 0
	m.logger.Debug("Cleared memory")
}

func (m *Memory) GetMessages() []Message {
	m.mutex.Lock()
	defer m.mutex.Unlock()

	return append([]Message(nil), m.messages...)
}

type LLMWithMemory struct {
	LLM
	memory *Memory
}

func NewLLMWithMemory(baseLLM LLM, maxTokens int, model string, logger Logger) (*LLMWithMemory, error) {
	memory, err := NewMemory(maxTokens, model, logger)
	if err != nil {
		return nil, err
	}

	return &LLMWithMemory{
		LLM:    baseLLM,
		memory: memory,
	}, nil
}

func (l *LLMWithMemory) Generate(ctx context.Context, prompt *Prompt) (string, string, error) {
	l.memory.Add("user", prompt.String())
	fullPrompt := l.memory.GetPrompt()

	response, _, err := l.LLM.Generate(ctx, NewPrompt(fullPrompt))
	if err != nil {
		return "", fullPrompt, err
	}

	l.memory.Add("assistant", response)
	return response, fullPrompt, nil
}

func (l *LLMWithMemory) ClearMemory() {
	l.memory.Clear()
}

func (l *LLMWithMemory) GetMemory() []Message {
	return l.memory.GetMessages()
}
