// File: internal/llm/compare.go

package llm

import (
	"context"
	"fmt"
	"strings"
	"sync"
)

type ComparisonResult struct {
	Provider string
	Model    string
	Response string
	Error    error
}

func CompareProviders(ctx context.Context, prompt *Prompt, registry *ProviderRegistry, logger Logger, configs ...*Config) []ComparisonResult {
	var results []ComparisonResult
	var wg sync.WaitGroup
	resultChan := make(chan ComparisonResult, len(configs))

	for _, config := range configs {
		wg.Add(1)
		go func(cfg *Config) {
			defer wg.Done()

			llm, err := NewLLM(cfg, logger, registry)
			if err != nil {
				resultChan <- ComparisonResult{Provider: cfg.Provider, Model: cfg.Model, Error: err}
				return
			}

			response, _, err := llm.Generate(ctx, prompt)
			resultChan <- ComparisonResult{
				Provider: cfg.Provider,
				Model:    cfg.Model,
				Response: response,
				Error:    err,
			}
		}(config)
	}

	go func() {
		wg.Wait()
		close(resultChan)
	}()

	for result := range resultChan {
		results = append(results, result)
	}

	return results
}

func PrintComparisonResults(results []ComparisonResult) {
	for _, result := range results {
		fmt.Printf("Provider: %s, Model: %s\n", result.Provider, result.Model)
		if result.Error != nil {
			fmt.Printf("Error: %v\n", result.Error)
		} else {
			fmt.Printf("Response: %s\n", result.Response)
		}
		fmt.Println(strings.Repeat("-", 40))
	}
}
