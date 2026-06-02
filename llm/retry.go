package llm

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"math/rand"
	"net/http"
	"strconv"
	"strings"
	"time"
)

// classifyAPIError builds an LLMError for a non-2xx HTTP response, annotating
// it with retryability and any server-provided Retry-After delay. The
// classification mirrors the official Anthropic and OpenAI SDKs:
//
//   - 408, 409, 5xx (including Anthropic's 529 overloaded): retryable-transient.
//   - 429: rate-limited and retryable, honoring the server's wait hint, unless
//     the body reports OpenAI's insufficient_quota (out of credits) which is
//     permanent.
//   - any other 4xx: permanent.
func classifyAPIError(statusCode int, header http.Header, body []byte) *LLMError {
	le := &LLMError{
		Type:       ErrorTypeAPI,
		Message:    fmt.Sprintf("API error: status code %d", statusCode),
		StatusCode: statusCode,
		RetryAfter: parseRetryAfter(header),
	}

	switch {
	case statusCode == http.StatusTooManyRequests: // 429
		le.Type = ErrorTypeRateLimit
		// insufficient_quota is a 429 that means the account is out of
		// credits — retrying never recovers it.
		if errType, errCode := parseProviderError(body); errType == "insufficient_quota" || errCode == "insufficient_quota" {
			le.Message = "API error: insufficient quota"
			le.Retryable = false
		} else {
			le.Retryable = true
		}
	case statusCode == http.StatusRequestTimeout, // 408
		statusCode == http.StatusConflict, // 409
		statusCode >= 500:                 // 5xx incl. Anthropic 529
		le.Retryable = true
	default:
		le.Retryable = false
	}

	return le
}

// classifyNetworkError wraps a transport-level failure from http.Client.Do.
// Genuine network errors (connection reset, DNS, refused) are retryable;
// context cancellation and deadline expiry are not — the deadline is shared
// across attempts, so retrying cannot beat it.
func classifyNetworkError(err error) *LLMError {
	if errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) {
		return &LLMError{Type: ErrorTypeRequest, Message: "request cancelled", Err: err, Retryable: false}
	}
	return &LLMError{Type: ErrorTypeRequest, Message: "failed to send request", Err: err, Retryable: true}
}

// parseProviderError reads the provider error envelope to extract the typed
// error string. It tolerates both the Anthropic shape ({"error":{"type":...}})
// and the OpenAI shape ({"error":{"type":...,"code":...}}). Missing or
// unparseable bodies yield empty strings.
func parseProviderError(body []byte) (errType, errCode string) {
	if len(body) == 0 {
		return "", ""
	}
	var env struct {
		Error struct {
			Type string `json:"type"`
			Code string `json:"code"`
		} `json:"error"`
	}
	if err := json.Unmarshal(body, &env); err != nil {
		return "", ""
	}
	return env.Error.Type, env.Error.Code
}

// parseRetryAfter extracts a server-provided wait hint from rate-limit
// response headers, in priority order:
//
//  1. retry-after-ms — OpenAI's millisecond-precision hint.
//  2. retry-after — integer seconds (Anthropic) or an HTTP-date.
//  3. x-ratelimit-reset-{requests,tokens} — OpenAI duration strings ("6s",
//     "1m30s") describing when a bucket refills; used only as a fallback.
//
// It returns 0 when no usable hint is present, leaving the caller to apply
// computed backoff.
func parseRetryAfter(h http.Header) time.Duration {
	if h == nil {
		return 0
	}

	if ms := strings.TrimSpace(h.Get("retry-after-ms")); ms != "" {
		if v, err := strconv.Atoi(ms); err == nil && v >= 0 {
			return time.Duration(v) * time.Millisecond
		}
	}

	if ra := strings.TrimSpace(h.Get("retry-after")); ra != "" {
		if secs, err := strconv.Atoi(ra); err == nil && secs >= 0 {
			return time.Duration(secs) * time.Second
		}
		if t, err := http.ParseTime(ra); err == nil {
			if d := time.Until(t); d > 0 {
				return d
			}
			return 0
		}
	}

	// Fallback: the soonest a constrained bucket says it will refill.
	var reset time.Duration
	for _, name := range []string{"x-ratelimit-reset-requests", "x-ratelimit-reset-tokens"} {
		if d, ok := parseResetDuration(h.Get(name)); ok && d > reset {
			reset = d
		}
	}
	return reset
}

// parseResetDuration parses OpenAI's x-ratelimit-reset-* header values, which
// are Go-style duration strings ("6s", "1m30s", "880ms") rather than epoch
// seconds. It reports ok=false for empty, malformed, or negative values.
func parseResetDuration(s string) (time.Duration, bool) {
	s = strings.TrimSpace(s)
	if s == "" {
		return 0, false
	}
	d, err := time.ParseDuration(s)
	if err != nil || d < 0 {
		return 0, false
	}
	return d, true
}

// retryDelay computes the wait before the next attempt: full jitter over an
// exponential backoff (RetryDelay × 2^attempt, capped at MaxRetryDelay),
// floored by any server-provided delay. attempt is zero-based.
func (l *LLMImpl) retryDelay(attempt int, serverDelay time.Duration) time.Duration {
	base := l.RetryDelay
	if base <= 0 {
		base = 2 * time.Second
	}

	// Cap the base first so attempt 0 honors MaxRetryDelay even when base
	// alone exceeds it; the in-loop cap then both clamps growth and prevents
	// the doubling from overflowing int64 at high attempt counts.
	backoff := base
	if l.MaxRetryDelay > 0 && backoff > l.MaxRetryDelay {
		backoff = l.MaxRetryDelay
	}
	for i := 0; i < attempt; i++ {
		backoff *= 2
		if l.MaxRetryDelay > 0 && backoff > l.MaxRetryDelay {
			backoff = l.MaxRetryDelay
			break
		}
	}

	// Full jitter: a uniformly random point in [0, backoff].
	jittered := backoff
	if backoff > 0 {
		jittered = time.Duration(rand.Int63n(int64(backoff) + 1))
	}

	// An explicit server hint is authoritative as a floor.
	if serverDelay > jittered {
		return serverDelay
	}
	return jittered
}

// shouldRetry decides, after a failed attempt, whether the loop should make
// another one and how long to wait first. It returns proceed=false when the
// error is non-retryable or attempts are exhausted. When proceed is true it
// has already waited (honoring context cancellation); a non-nil error means
// the context was cancelled during the wait.
func (l *LLMImpl) shouldRetry(ctx context.Context, attempt int, err error) (proceed bool, ctxErr error) {
	if attempt >= l.MaxRetries {
		return false, nil
	}
	var le *LLMError
	if !errors.As(err, &le) || !le.Retryable {
		return false, nil
	}

	delay := l.retryDelay(attempt, le.RetryAfter)
	l.logger.Debug("Retrying generation", "attempt", attempt+1, "delay", delay.String(), "error", err)
	if werr := l.waitFor(ctx, delay); werr != nil {
		return false, werr
	}
	return true, nil
}

// waitFor sleeps for d, returning early with the context error if the context
// is cancelled. A non-positive d still observes an already-cancelled context.
func (l *LLMImpl) waitFor(ctx context.Context, d time.Duration) error {
	if d <= 0 {
		select {
		case <-ctx.Done():
			return ctx.Err()
		default:
			return nil
		}
	}
	t := time.NewTimer(d)
	defer t.Stop()
	select {
	case <-ctx.Done():
		return ctx.Err()
	case <-t.C:
		return nil
	}
}
