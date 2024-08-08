// File: internal/llm/prompt.go

package llm

import (
	"strings"
)

type PromptComponent struct {
	Content string
	Type    string
}

type Prompt struct {
	Components []PromptComponent
}

func NewPrompt(input string) *Prompt {
	return &Prompt{
		Components: []PromptComponent{
			{Content: input, Type: "instruction"},
		},
	}
}

func (p *Prompt) WithContext(context string) *Prompt {
	p.Components = append(p.Components, PromptComponent{Content: context, Type: "context"})
	return p
}

func (p *Prompt) WithDirective(directive string) *Prompt {
	p.Components = append(p.Components, PromptComponent{Content: directive, Type: "directive"})
	return p
}

func (p *Prompt) WithOutput(output string) *Prompt {
	p.Components = append(p.Components, PromptComponent{Content: output, Type: "output"})
	return p
}

func (p *Prompt) WithExample(example string) *Prompt {
	p.Components = append(p.Components, PromptComponent{Content: example, Type: "example"})
	return p
}

func (p *Prompt) String() string {
	var parts []string
	for _, component := range p.Components {
		switch component.Type {
		case "instruction":
			parts = append(parts, component.Content)
		case "context":
			parts = append(parts, "Context: "+component.Content)
		case "directive":
			parts = append(parts, "Directive: "+component.Content)
		case "output":
			parts = append(parts, "Expected Output: "+component.Content)
		case "example":
			parts = append(parts, "Example: "+component.Content)
		default:
			parts = append(parts, component.Type+": "+component.Content)
		}
	}
	return strings.Join(parts, "\n\n")
}
