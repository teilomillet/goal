package llm

import (
	"fmt"
	"time"

	"github.com/teilomillet/gollm/utils"
)

// ErrorType represents the category of an LLM error.
// It helps classify errors for appropriate handling and logging.
type ErrorType int

const (
	// ErrorTypeUnknown represents an unclassified error
	ErrorTypeUnknown ErrorType = iota

	// ErrorTypeProvider indicates an error from the LLM provider
	ErrorTypeProvider

	// ErrorTypeRequest indicates an error in preparing or sending the request
	ErrorTypeRequest

	// ErrorTypeResponse indicates an error in processing the response
	ErrorTypeResponse

	// ErrorTypeAPI indicates an error returned by the provider's API
	ErrorTypeAPI

	// ErrorTypeRateLimit indicates the provider's rate limit has been exceeded
	ErrorTypeRateLimit

	// ErrorTypeAuthentication indicates an authentication or authorization failure
	ErrorTypeAuthentication

	// ErrorTypeInvalidInput indicates invalid input parameters or prompt
	ErrorTypeInvalidInput

	// ErrorTypeUnsupported indicates a requested feature is not supported
	ErrorTypeUnsupported
)

// LLMError represents a structured error in the LLM package.
// It implements the error interface and provides additional context
// about the error type and underlying cause.
type LLMError struct {
	Type    ErrorType // The category of the error
	Message string    // A human-readable error message
	Err     error     // The underlying error, if any

	// Retryable reports whether retrying the request may succeed. It is set
	// by the classifier for transient network failures and retryable HTTP
	// statuses (408, 409, 429 rate limits, 5xx); permanent failures (most
	// 4xx, exhausted quota) leave it false. The retry loop reads this field.
	Retryable bool
	// RetryAfter carries a server-provided minimum wait before the next
	// attempt, parsed from Retry-After / retry-after-ms / rate-limit reset
	// headers. Zero means "no server hint — use computed backoff".
	RetryAfter time.Duration
	// StatusCode is the HTTP status of an API error, or 0 when the error did
	// not originate from an HTTP response.
	StatusCode int
}

// LoggableFields returns a slice of interface{} containing error information
// in a format suitable for structured logging. The classification fields
// (status_code, retryable, retry_after) are appended only when meaningful, so
// non-HTTP errors do not emit noisy zero-values.
func (e *LLMError) LoggableFields() []interface{} {
	fields := []interface{}{
		"error_type", e.TypeString(),
		"message", e.Message,
		"error", e.Err,
	}
	if e.StatusCode != 0 {
		fields = append(fields, "status_code", e.StatusCode)
	}
	if e.Retryable {
		fields = append(fields, "retryable", e.Retryable)
	}
	if e.RetryAfter > 0 {
		fields = append(fields, "retry_after", e.RetryAfter.String())
	}
	return fields
}

// Error implements the error interface.
// It returns a formatted string containing the error type, message,
// and underlying error (if present).
func (e *LLMError) Error() string {
	if e.Err != nil {
		return fmt.Sprintf("%s (%s): %v", e.TypeString(), e.Message, e.Err)
	}
	return fmt.Sprintf("%s: %s", e.TypeString(), e.Message)
}

// Unwrap returns the underlying error.
// This implements the Go 1.13+ error unwrapping interface.
func (e *LLMError) Unwrap() error {
	return e.Err
}

// TypeString returns a string representation of the error type.
// This is used for logging and error messages.
func (e *LLMError) TypeString() string {
	switch e.Type {
	case ErrorTypeProvider:
		return "ProviderError"
	case ErrorTypeRequest:
		return "RequestError"
	case ErrorTypeResponse:
		return "ResponseError"
	case ErrorTypeAPI:
		return "APIError"
	case ErrorTypeRateLimit:
		return "RateLimitError"
	case ErrorTypeAuthentication:
		return "AuthenticationError"
	case ErrorTypeInvalidInput:
		return "InvalidInputError"
	case ErrorTypeUnsupported:
		return "UnsupportedError"
	default:
		return "UnknownError"
	}
}

// NewLLMError creates a new LLMError with the specified type, message,
// and underlying error.
//
// Parameters:
//   - errType: The category of the error
//   - message: A human-readable error message
//   - err: The underlying error, if any
//
// Returns:
//   - A new LLMError instance
func NewLLMError(errType ErrorType, message string, err error) *LLMError {
	return &LLMError{
		Type:    errType,
		Message: message,
		Err:     err,
	}
}

// HandleError processes an error based on its severity.
// It logs the error appropriately and can optionally terminate the program
// if the error is considered fatal.
//
// Parameters:
//   - err: The error to handle
//   - fatal: If true, the program will panic after logging
//   - logger: The logger to use for error reporting
func HandleError(err error, fatal bool, logger utils.Logger) {
	if err == nil {
		return
	}

	if llmErr, ok := err.(*LLMError); ok {
		logger.Error(llmErr.Message, "error_type", llmErr.TypeString(), "error", llmErr.Err)
	} else {
		logger.Error("An error occurred", "error", err)
	}

	if fatal {
		// Consider using os.Exit(1) here or returning an error to let the caller decide
		panic(err)
	}
}
