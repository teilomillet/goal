// File: internal/llm/prompt_optimizer.go

package llm

import (
	"context"
	"encoding/json"
	"fmt"
	"strconv"
	"strings"
	"time"

	"github.com/go-playground/validator/v10"
)

type Metric struct {
	Name        string  `json:"name"`
	Description string  `json:"description"`
	Value       float64 `json:"value" validate:"required,min=0,max=20"`
	Reasoning   string  `json:"reasoning"`
}

type Strength struct {
	Point   string `json:"point"`
	Example string `json:"example"`
}

type Weakness struct {
	Point   string `json:"point"`
	Example string `json:"example"`
}

type Suggestion struct {
	Description    string  `json:"description"`
	ExpectedImpact float64 `json:"expectedImpact" validate:"required,min=0,max=20"`
	Reasoning      string  `json:"reasoning"`
}

type PromptAssessment struct {
	Metrics           []Metric     `json:"metrics" validate:"required,min=1"`
	Strengths         []Strength   `json:"strengths" validate:"required,min=1"`
	Weaknesses        []Weakness   `json:"weaknesses" validate:"required,min=1"`
	Suggestions       []Suggestion `json:"suggestions" validate:"required,min=1"`
	OverallScore      float64      `json:"overallScore" validate:"required,min=0,max=20"`
	OverallGrade      string       `json:"overallGrade" validate:"required,validGrade"`
	EfficiencyScore   float64      `json:"efficiencyScore" validate:"required,min=0,max=20"`
	AlignmentWithGoal float64      `json:"alignmentWithGoal" validate:"required,min=0,max=20"`
}

type PromptOptimizer struct {
	llm               LLM
	debugManager      *DebugManager
	initialPrompt     *Prompt
	taskDesc          string
	customMetrics     []Metric
	optimizationGoal  string
	history           []OptimizationEntry
	ratingSystem      string
	threshold         float64
	iterationCallback IterationCallback
	maxRetries        int
	retryDelay        time.Duration
}

type OptimizationEntry struct {
	Prompt     *Prompt          `json:"prompt"`
	Assessment PromptAssessment `json:"assessment"`
}

type OptimizerOption func(*PromptOptimizer)

type OptimizationRating interface {
	IsGoalMet() bool
	String() string
}

type NumericalRating struct {
	Score float64
	Max   float64
}

type IterationCallback func(iteration int, entry OptimizationEntry)

func (nr NumericalRating) IsGoalMet() bool {
	return nr.Score >= nr.Max*0.9 // Consider goal met if score is 90% or higher
}

func (nr NumericalRating) String() string {
	return fmt.Sprintf("%.1f/%.1f", nr.Score, nr.Max)
}

type LetterRating struct {
	Grade string
}

func (lr LetterRating) IsGoalMet() bool {
	return lr.Grade == "A" || lr.Grade == "A+" || lr.Grade == "S"
}

func (lr LetterRating) String() string {
	return lr.Grade
}

func (po *PromptOptimizer) WithCustomMetrics(metrics ...Metric) {
	po.customMetrics = metrics
}

func (po *PromptOptimizer) WithOptimizationGoal(goal string) {
	po.optimizationGoal = goal
}

func (po *PromptOptimizer) WithRatingSystem(system string) {
	po.ratingSystem = system
}

func (po *PromptOptimizer) WithThreshold(threshold float64) {
	po.threshold = threshold
}

func (po *PromptOptimizer) WithIterationCallback(callback IterationCallback) {
	po.iterationCallback = callback
}

func (po *PromptOptimizer) WithMaxRetries(maxRetries int) {
	po.maxRetries = maxRetries
}

func (po *PromptOptimizer) WithRetryDelay(delay time.Duration) {
	po.retryDelay = delay
}

func validGrade(fl validator.FieldLevel) bool {
	grade := fl.Field().String()
	validGrades := map[string]bool{
		"A+": true, "A": true, "A-": true,
		"B+": true, "B": true, "B-": true,
		"C+": true, "C": true, "C-": true,
		"D+": true, "D": true, "D-": true,
		"F": true,
	}
	return validGrades[grade]
}

func NewPromptOptimizer(llm LLM, debugManager *DebugManager, initialPrompt *Prompt, taskDesc string, opts ...OptimizerOption) *PromptOptimizer {
	optimizer := &PromptOptimizer{
		llm:           llm,
		debugManager:  debugManager,
		initialPrompt: initialPrompt,
		taskDesc:      taskDesc,
		history:       []OptimizationEntry{},
		threshold:     0.8,
		maxRetries:    3,
		retryDelay:    time.Second * 2,
	}

	for _, opt := range opts {
		opt(optimizer)
	}

	return optimizer
}

func (po *PromptOptimizer) assessPrompt(ctx context.Context, prompt *Prompt) (OptimizationEntry, error) {
	assessPrompt := NewPrompt(fmt.Sprintf(`
    Assess the following prompt for the task: %s

    Full Prompt Structure:
    %s

    Custom Metrics: %v

    Optimization Goal: %s

    Provide your assessment as a JSON object with the following structure:
    {
        "metrics": [{"name": string, "value": number, "reasoning": string}, ...],
        "strengths": [{"point": string, "example": string}, ...],
        "weaknesses": [{"point": string, "example": string}, ...],
        "suggestions": [{"description": string, "expectedImpact": number, "reasoning": string}, ...],
        "overallScore": number,
        "overallGrade": string,
        "efficiencyScore": number,
        "alignmentWithGoal": number
    }

    IMPORTANT: 
    - Do not use any markdown formatting, code blocks, or backticks in your response.
    - Return only the raw JSON object.
    - For numerical ratings, use a scale of 0 to 20 (inclusive).
    - For the overallGrade:
      - If using letter grades, use one of: F, D, C, B, A, A+
      - If using numerical grades, use the same value as overallScore (0-20)
    - Include at least one item in each array (metrics, strengths, weaknesses, suggestions).
    - Provide specific examples and reasoning for each point.
    - Rate the prompt's efficiency and alignment with the optimization goal.
    - Rank suggestions by their expected impact (20 being highest impact).
    - Use clear, jargon-free language in your assessment.
    - Double-check that your response is valid JSON before submitting.
`, po.taskDesc, prompt.String(), po.customMetrics, po.optimizationGoal))

	response, _, err := po.llm.Generate(ctx, assessPrompt)
	if err != nil {
		return OptimizationEntry{}, fmt.Errorf("failed to assess prompt: %w", err)
	}

	// Clean the response
	cleanedResponse := cleanJSONResponse(response)
	var assessment PromptAssessment
	err = json.Unmarshal([]byte(cleanedResponse), &assessment)
	if err != nil {
		po.debugManager.LogResponse(fmt.Sprintf("Raw response: %s", response))
		return OptimizationEntry{}, fmt.Errorf("failed to parse assessment response: %w", err)
	}

	// Validate the unmarshaled struct
	if err := validate.Struct(assessment); err != nil {
		return OptimizationEntry{}, fmt.Errorf("invalid assessment structure: %w", err)
	}

	// Normalize the OverallGrade
	assessment.OverallGrade, err = normalizeGrade(assessment.OverallGrade, assessment.OverallScore)
	if err != nil {
		return OptimizationEntry{}, fmt.Errorf("invalid overall grade: %w", err)
	}

	return OptimizationEntry{
		Prompt:     prompt,
		Assessment: assessment,
	}, nil
}

func normalizeGrade(grade string, score float64) (string, error) {
	validGrades := map[string]bool{
		"A+": true, "A": true, "A-": true,
		"B+": true, "B": true, "B-": true,
		"C+": true, "C": true, "C-": true,
		"D+": true, "D": true, "D-": true,
		"F": true,
	}

	if validGrades[grade] {
		return grade, nil
	}

	// If it's not a recognized letter grade, try to parse as a number
	numericGrade, err := strconv.ParseFloat(grade, 64)
	if err != nil {
		return "", fmt.Errorf("invalid grade format: %s", grade)
	}

	// Convert numerical grade to letter grade
	switch {
	case numericGrade >= 19:
		return "A+", nil
	case numericGrade >= 17:
		return "A", nil
	case numericGrade >= 15:
		return "A-", nil
	case numericGrade >= 13:
		return "B+", nil
	case numericGrade >= 11:
		return "B", nil
	case numericGrade >= 9:
		return "B-", nil
	case numericGrade >= 7:
		return "C+", nil
	case numericGrade >= 5:
		return "C", nil
	case numericGrade >= 3:
		return "C-", nil
	case numericGrade >= 2:
		return "D+", nil
	case numericGrade >= 1:
		return "D", nil
	default:
		return "F", nil
	}
}

func cleanJSONResponse(response string) string {
	// Remove any markdown code block syntax
	response = strings.TrimPrefix(response, "```json")
	response = strings.TrimSuffix(response, "```")
	response = strings.TrimSpace(response)

	// If the response starts with a {, assume it's already JSON
	if strings.HasPrefix(response, "{") {
		return response
	}

	// Otherwise, try to find JSON object in the text
	start := strings.Index(response, "{")
	end := strings.LastIndex(response, "}")

	if start != -1 && end != -1 && end > start {
		return response[start : end+1]
	}

	// If we can't find a JSON object, return the original response
	// This will likely cause a JSON parsing error, which we can catch and handle
	return response
}

func (po *PromptOptimizer) generateImprovedPrompt(ctx context.Context, prevEntry OptimizationEntry) (*Prompt, error) {
	improvePrompt := NewPrompt(fmt.Sprintf(`
    Based on the following assessment, generate an improved version of the entire prompt structure:

    Previous prompt:
    %s

    Assessment: %+v

    Task Description: %s
    Optimization Goal: %s

    Provide two versions of the improved prompt:
    1. An incremental improvement
    2. A bold reimagining

    IMPORTANT: Respond ONLY with a raw JSON object. Do not use any markdown formatting, code blocks, or backticks.
    The JSON object should have this structure:
    {
        "incrementalImprovement": {
            "input": "improved prompt text",
            "directives": ["directive1", "directive2", ...],
            "examples": ["example1", "example2", ...],
            "reasoning": "explanation of changes and their link to the assessment"
        },
        "boldRedesign": {
            "input": "reimagined prompt text",
            "directives": ["directive1", "directive2", ...],
            "examples": ["example1", "example2", ...],
            "reasoning": "explanation of the new approach and its potential benefits"
        },
        "expectedImpact": {
            "incremental": number,
            "bold": number
        }
    }

    For each improvement:
    - Directly address weaknesses identified in the assessment.
    - Build upon identified strengths.
    - Ensure alignment with the task description and optimization goal.
    - Strive for efficiency in language use.
    - Use clear, jargon-free language.
    - Provide a brief reasoning for major changes.
    - Rate the expected impact of each version on a scale of 0 to 20.

    Double-check that your response is valid JSON before submitting.
`, prevEntry.Prompt.String(), prevEntry.Assessment))

	po.debugManager.LogPrompt(improvePrompt.String())

	response, _, err := po.llm.Generate(ctx, improvePrompt)
	if err != nil {
		return nil, fmt.Errorf("failed to generate improved prompt: %w", err)
	}

	po.debugManager.LogResponse(response)

	// Clean the response
	cleanedResponse := cleanJSONResponse(response)

	var improvedPrompts struct {
		IncrementalImprovement Prompt `json:"incrementalImprovement"`
		BoldRedesign           Prompt `json:"boldRedesign"`
		ExpectedImpact         struct {
			Incremental float64 `json:"incremental"`
			Bold        float64 `json:"bold"`
		} `json:"expectedImpact"`
	}

	err = json.Unmarshal([]byte(cleanedResponse), &improvedPrompts)
	if err != nil {
		return nil, fmt.Errorf("failed to parse improved prompts: %w", err)
	}

	// Choose the prompt with the higher expected impact
	if improvedPrompts.ExpectedImpact.Bold > improvedPrompts.ExpectedImpact.Incremental {
		return &improvedPrompts.BoldRedesign, nil
	}
	return &improvedPrompts.IncrementalImprovement, nil
}

func (po *PromptOptimizer) OptimizePrompt(ctx context.Context, iterations int) (*Prompt, error) {
	currentPrompt := po.initialPrompt
	var bestPrompt *Prompt
	var bestScore float64

	for i := 0; i < iterations; i++ {
		var entry OptimizationEntry
		var err error

		for attempt := 0; attempt < po.maxRetries; attempt++ {
			entry, err = po.assessPrompt(ctx, currentPrompt)
			if err == nil {
				break
			}

			po.debugManager.LogResponse(fmt.Sprintf("Error in iteration %d, attempt %d: %v", i+1, attempt+1, err))
			if attempt < po.maxRetries-1 {
				po.debugManager.LogResponse(fmt.Sprintf("Retrying in %v...", po.retryDelay))
				time.Sleep(po.retryDelay)
			}
		}

		if err != nil {
			return bestPrompt, fmt.Errorf("optimization failed at iteration %d after %d attempts: %w", i+1, po.maxRetries, err)
		}

		po.history = append(po.history, entry)

		if po.iterationCallback != nil {
			po.iterationCallback(i+1, entry)
		}

		if entry.Assessment.OverallScore > bestScore {
			bestScore = entry.Assessment.OverallScore
			bestPrompt = currentPrompt
		}

		goalMet, err := po.isOptimizationGoalMet(entry.Assessment)
		if err != nil {
			po.debugManager.LogResponse(fmt.Sprintf("Error checking optimization goal: %v", err))
		} else if goalMet {
			po.debugManager.LogResponse(fmt.Sprintf("Optimization complete after %d iterations. Goal achieved.", i+1))
			return currentPrompt, nil
		}

		improvedPrompt, err := po.generateImprovedPrompt(ctx, entry)
		if err != nil {
			po.debugManager.LogResponse(fmt.Sprintf("Failed to generate improved prompt at iteration %d: %v", i+1, err))
			continue
		}

		currentPrompt = improvedPrompt
		po.debugManager.LogResponse(fmt.Sprintf("Iteration %d complete. New prompt: %s", i+1, currentPrompt.String()))
	}

	return bestPrompt, nil
}

func (po *PromptOptimizer) isOptimizationGoalMet(assessment PromptAssessment) (bool, error) {
	if po.ratingSystem == "" {
		return false, nil
	}

	switch po.ratingSystem {
	case "numerical":
		return assessment.OverallScore >= 20*po.threshold, nil
	case "letter":
		gradeValues := map[string]float64{
			"A+": 4.3, "A": 4.0, "A-": 3.7,
			"B+": 3.3, "B": 3.0, "B-": 2.7,
			"C+": 2.3, "C": 2.0, "C-": 1.7,
			"D+": 1.3, "D": 1.0, "D-": 0.7,
			"F": 0.0,
		}
		gradeValue, exists := gradeValues[assessment.OverallGrade]
		if !exists {
			return false, fmt.Errorf("invalid grade: %s", assessment.OverallGrade)
		}
		return gradeValue >= 3.7, nil // Equivalent to A- or better
	default:
		return false, fmt.Errorf("unknown rating system: %s", po.ratingSystem)
	}
}

func (po *PromptOptimizer) GetOptimizationHistory() []OptimizationEntry {
	return po.history
}

