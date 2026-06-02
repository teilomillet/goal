package llm

import (
	"context"
	"errors"
	"io"
	"net/http"
	"strings"
	"testing"
	"time"

	"github.com/teilomillet/gollm/providers"
	"github.com/teilomillet/gollm/utils"
)

func TestParseResetDuration(t *testing.T) {
	cases := []struct {
		in   string
		want time.Duration
		ok   bool
	}{
		{"6s", 6 * time.Second, true},
		{"1m30s", 90 * time.Second, true},
		{"880ms", 880 * time.Millisecond, true},
		{" 2s ", 2 * time.Second, true},
		{"", 0, false},
		{"soon", 0, false},
		{"-3s", 0, false},
	}
	for _, c := range cases {
		got, ok := parseResetDuration(c.in)
		if ok != c.ok || got != c.want {
			t.Errorf("parseResetDuration(%q) = (%v, %v), want (%v, %v)", c.in, got, ok, c.want, c.ok)
		}
	}
}

func TestParseRetryAfter(t *testing.T) {
	cases := []struct {
		name string
		hdr  map[string]string
		want time.Duration
	}{
		{"openai retry-after-ms wins", map[string]string{"retry-after-ms": "1500", "retry-after": "9"}, 1500 * time.Millisecond},
		{"anthropic retry-after seconds", map[string]string{"retry-after": "12"}, 12 * time.Second},
		{"reset fallback takes the larger", map[string]string{"x-ratelimit-reset-requests": "2s", "x-ratelimit-reset-tokens": "5s"}, 5 * time.Second},
		{"explicit beats reset", map[string]string{"retry-after": "1", "x-ratelimit-reset-tokens": "30s"}, 1 * time.Second},
		{"nothing usable", map[string]string{}, 0},
		{"garbage ignored", map[string]string{"retry-after": "later"}, 0},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			h := http.Header{}
			for k, v := range c.hdr {
				h.Set(k, v)
			}
			if got := parseRetryAfter(h); got != c.want {
				t.Errorf("parseRetryAfter(%v) = %v, want %v", c.hdr, got, c.want)
			}
		})
	}

	if got := parseRetryAfter(nil); got != 0 {
		t.Errorf("parseRetryAfter(nil) = %v, want 0", got)
	}
}

func TestParseProviderError(t *testing.T) {
	cases := []struct {
		name     string
		body     string
		wantType string
		wantCode string
	}{
		{"anthropic rate limit", `{"type":"error","error":{"type":"rate_limit_error","message":"slow down"}}`, "rate_limit_error", ""},
		{"openai insufficient quota", `{"error":{"message":"You exceeded your quota","type":"insufficient_quota","code":"insufficient_quota"}}`, "insufficient_quota", "insufficient_quota"},
		{"empty body", ``, "", ""},
		{"non-json body", `Bad Gateway`, "", ""},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			gotType, gotCode := parseProviderError([]byte(c.body))
			if gotType != c.wantType || gotCode != c.wantCode {
				t.Errorf("parseProviderError(%q) = (%q, %q), want (%q, %q)", c.body, gotType, gotCode, c.wantType, c.wantCode)
			}
		})
	}
}

func TestClassifyAPIError(t *testing.T) {
	cases := []struct {
		name          string
		status        int
		body          string
		wantRetryable bool
		wantType      ErrorType
	}{
		{"429 rate limit", 429, `{"error":{"type":"rate_limit_error"}}`, true, ErrorTypeRateLimit},
		{"429 insufficient quota", 429, `{"error":{"type":"insufficient_quota","code":"insufficient_quota"}}`, false, ErrorTypeRateLimit},
		{"408 timeout", 408, ``, true, ErrorTypeAPI},
		{"409 conflict", 409, ``, true, ErrorTypeAPI},
		{"500 server error", 500, ``, true, ErrorTypeAPI},
		{"529 anthropic overloaded", 529, ``, true, ErrorTypeAPI},
		{"400 bad request", 400, ``, false, ErrorTypeAPI},
		{"401 unauthorized", 401, ``, false, ErrorTypeAPI},
		{"404 not found", 404, ``, false, ErrorTypeAPI},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			le := classifyAPIError(c.status, http.Header{}, []byte(c.body))
			if le.Retryable != c.wantRetryable {
				t.Errorf("status %d: Retryable = %v, want %v", c.status, le.Retryable, c.wantRetryable)
			}
			if le.Type != c.wantType {
				t.Errorf("status %d: Type = %v, want %v", c.status, le.Type, c.wantType)
			}
			if le.StatusCode != c.status {
				t.Errorf("status %d: StatusCode = %v, want %v", c.status, le.StatusCode, c.status)
			}
		})
	}
}

func TestClassifyNetworkError(t *testing.T) {
	if le := classifyNetworkError(errors.New("connection reset by peer")); !le.Retryable {
		t.Error("generic network error should be retryable")
	}
	if le := classifyNetworkError(context.Canceled); le.Retryable {
		t.Error("context.Canceled must not be retryable")
	}
	if le := classifyNetworkError(context.DeadlineExceeded); le.Retryable {
		t.Error("context.DeadlineExceeded must not be retryable")
	}
}

func TestRetryDelay(t *testing.T) {
	l := &LLMImpl{RetryDelay: 2 * time.Second, MaxRetryDelay: 10 * time.Second}

	// Backoff upper bound grows as base*2^attempt, capped at MaxRetryDelay.
	// Full jitter keeps each sample within [0, bound].
	bounds := map[int]time.Duration{0: 2 * time.Second, 1: 4 * time.Second, 2: 8 * time.Second, 3: 10 * time.Second, 4: 10 * time.Second}
	for attempt, bound := range bounds {
		for i := 0; i < 200; i++ {
			got := l.retryDelay(attempt, 0)
			if got < 0 || got > bound {
				t.Fatalf("retryDelay(%d) = %v, want within [0, %v]", attempt, got, bound)
			}
		}
	}

	// A server-provided delay is an authoritative floor, even past the cap.
	if got := l.retryDelay(0, 30*time.Second); got != 30*time.Second {
		t.Errorf("server floor: retryDelay(0, 30s) = %v, want 30s", got)
	}

	// Default config (the common caller path): RetryDelay <= 0 falls back to a
	// 2s base, and MaxRetryDelay == 0 leaves backoff uncapped — so the bound
	// grows as 2s*2^attempt with no ceiling.
	t.Run("default config", func(t *testing.T) {
		def := &LLMImpl{}
		bounds := map[int]time.Duration{
			0: 2 * time.Second,
			1: 4 * time.Second,
			2: 8 * time.Second,
			3: 16 * time.Second,
			4: 32 * time.Second,
		}
		for attempt, bound := range bounds {
			for i := 0; i < 200; i++ {
				if got := def.retryDelay(attempt, 0); got < 0 || got > bound {
					t.Fatalf("default retryDelay(%d) = %v, want within [0, %v]", attempt, got, bound)
				}
			}
		}
	})
}

// --- loop integration: transient retries, permanent fails fast ---

// stubProvider implements just the methods attemptGenerate calls; the embedded
// interface satisfies the rest (and panics loudly if anything else is hit).
type stubProvider struct {
	providers.Provider
}

func (stubProvider) Name() string               { return "stub" }
func (stubProvider) Endpoint() string           { return "http://stub.local/v1/generate" }
func (stubProvider) Headers() map[string]string { return map[string]string{} }
func (stubProvider) PrepareRequest(string, map[string]interface{}) ([]byte, error) {
	return []byte(`{}`), nil
}
func (stubProvider) ParseResponse(body []byte) (string, error) { return string(body), nil }

// scriptedRT returns canned responses/errors in sequence and records call count.
type scriptedRT struct {
	steps []func() (*http.Response, error)
	calls int
}

func (s *scriptedRT) RoundTrip(*http.Request) (*http.Response, error) {
	i := s.calls
	s.calls++
	if i >= len(s.steps) {
		return nil, errors.New("scriptedRT: unexpected extra call")
	}
	return s.steps[i]()
}

func resp(status int, body string) func() (*http.Response, error) {
	return func() (*http.Response, error) {
		return &http.Response{
			StatusCode: status,
			Header:     http.Header{},
			Body:       io.NopCloser(strings.NewReader(body)),
		}, nil
	}
}

func newTestLLM(rt http.RoundTripper) *LLMImpl {
	return &LLMImpl{
		Provider:      stubProvider{},
		client:        &http.Client{Transport: rt},
		logger:        utils.NewLogger(utils.LogLevelOff),
		Options:       make(map[string]interface{}),
		MaxRetries:    4,
		RetryDelay:    time.Millisecond,
		MaxRetryDelay: 5 * time.Millisecond,
	}
}

func TestGenerateRetriesTransientThenSucceeds(t *testing.T) {
	rt := &scriptedRT{steps: []func() (*http.Response, error){
		func() (*http.Response, error) { return nil, errors.New("connection reset by peer") },
		resp(503, `boom`),
		resp(200, `ok`),
	}}
	l := newTestLLM(rt)

	got, err := l.Generate(context.Background(), &Prompt{Input: "hi"})
	if err != nil {
		t.Fatalf("expected success after retries, got %v", err)
	}
	if got != "ok" {
		t.Errorf("result = %q, want %q", got, "ok")
	}
	if rt.calls != 3 {
		t.Errorf("calls = %d, want 3", rt.calls)
	}
}

func TestGeneratePermanentFailsFast(t *testing.T) {
	rt := &scriptedRT{steps: []func() (*http.Response, error){
		resp(400, `{"error":{"type":"invalid_request_error"}}`),
	}}
	l := newTestLLM(rt)

	_, err := l.Generate(context.Background(), &Prompt{Input: "hi"})
	if err == nil {
		t.Fatal("expected permanent error, got nil")
	}
	var le *LLMError
	if !errors.As(err, &le) || le.StatusCode != 400 {
		t.Errorf("want underlying 400 LLMError, got %v", err)
	}
	if rt.calls != 1 {
		t.Errorf("calls = %d, want 1 (no retry on permanent)", rt.calls)
	}
}

func TestGenerateInsufficientQuotaFailsFast(t *testing.T) {
	rt := &scriptedRT{steps: []func() (*http.Response, error){
		resp(429, `{"error":{"type":"insufficient_quota","code":"insufficient_quota"}}`),
	}}
	l := newTestLLM(rt)

	if _, err := l.Generate(context.Background(), &Prompt{Input: "hi"}); err == nil {
		t.Fatal("expected insufficient_quota to fail without retry")
	}
	if rt.calls != 1 {
		t.Errorf("calls = %d, want 1 (insufficient_quota is permanent)", rt.calls)
	}
}

func TestGenerateRetriesRateLimitHonorsRetryAfter(t *testing.T) {
	rt := &scriptedRT{steps: []func() (*http.Response, error){
		func() (*http.Response, error) {
			h := http.Header{}
			h.Set("retry-after-ms", "60")
			return &http.Response{
				StatusCode: http.StatusTooManyRequests,
				Header:     h,
				Body:       io.NopCloser(strings.NewReader(`{"error":{"type":"rate_limit_error"}}`)),
			}, nil
		},
		resp(200, `ok`),
	}}
	l := newTestLLM(rt)
	// Tiny base backoff so the server-provided 60ms Retry-After dominates.
	l.RetryDelay = time.Millisecond
	l.MaxRetryDelay = 10 * time.Second

	start := time.Now()
	got, err := l.Generate(context.Background(), &Prompt{Input: "hi"})
	elapsed := time.Since(start)
	if err != nil {
		t.Fatalf("expected success after rate-limit retry, got %v", err)
	}
	if got != "ok" {
		t.Errorf("result = %q, want %q", got, "ok")
	}
	if rt.calls != 2 {
		t.Errorf("calls = %d, want 2", rt.calls)
	}
	// The retry must have waited at least the header-derived 60ms, not the 1ms
	// base backoff — proves the server hint is honored as the floor.
	if elapsed < 60*time.Millisecond {
		t.Errorf("elapsed %v, want >= 60ms (Retry-After honored over base backoff)", elapsed)
	}
}

func TestWaitForCancellation(t *testing.T) {
	l := &LLMImpl{}
	ctx, cancel := context.WithCancel(context.Background())
	cancel() // cancelled before the wait begins

	if err := l.waitFor(ctx, time.Hour); !errors.Is(err, context.Canceled) {
		t.Errorf("waitFor(1h) on cancelled ctx = %v, want context.Canceled", err)
	}
	// The non-positive fast path still observes cancellation.
	if err := l.waitFor(ctx, 0); !errors.Is(err, context.Canceled) {
		t.Errorf("waitFor(0) on cancelled ctx = %v, want context.Canceled", err)
	}
}

func TestGenerateStopsRetryingOnCancel(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	rt := &scriptedRT{steps: []func() (*http.Response, error){
		func() (*http.Response, error) {
			cancel() // cancel after the first retryable failure, before the wait
			return resp(503, `boom`)()
		},
		resp(200, `ok`), // must never be reached
	}}
	l := newTestLLM(rt)
	// Long backoff: without cancellation the wait would block well past the test.
	l.RetryDelay = time.Hour
	l.MaxRetryDelay = time.Hour

	_, err := l.Generate(ctx, &Prompt{Input: "hi"})
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("Generate = %v, want context.Canceled", err)
	}
	if rt.calls != 1 {
		t.Errorf("calls = %d, want 1 (cancelled before retry)", rt.calls)
	}
}
