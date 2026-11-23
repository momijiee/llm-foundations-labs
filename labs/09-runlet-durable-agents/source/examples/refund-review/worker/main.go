// The refund-review reference Agent. The ReAct loop, model and tools are all
// user code; Runlet records each model call, tool call and supervisor approval
// as a durable position so the task survives waits and process crashes.
//
// Skeleton constraints from today's API: tools run sequentially, and every
// Step key carries its turn and call index because keys must be unique per Run.
package main

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log"
	"os"
	"os/signal"
	"syscall"
	"time"

	"runlet"
	"runlet/examples/refund-review/refund"
)

const maxTurns = 8

const systemPrompt = `你是电商售后退款审核助手。先调用只读工具核实订单、物流与用户历史，再决定是否调用 issue_refund。
退款金额以分为单位，不得超过订单金额。完成后用一句话说明结论。`

// AgentConfig is read once inside a Step, so a Run keeps the prompt, model and
// approval policy it started with even if the Worker's environment changes.
type AgentConfig struct {
	ReexecuteRefund   bool   `json:"reexecute_refund,omitempty"`
	PromptVersion     string `json:"prompt_version"`
	SystemPrompt      string `json:"system_prompt"`
	Model             string `json:"model"`               // e.g. fake:refund or openai:gpt-4o-mini
	ApprovalThreshold int64  `json:"approval_threshold"`  // 分
	ApprovalTimeoutMS int64  `json:"approval_timeout_ms"` // 0 waits forever
}

// agent is the per-Run state. It is rebuilt by Replay on every invocation;
// receipt is set only after the refund Step has a committed result.
type agent struct {
	e       *runlet.Execution
	cfg     AgentConfig
	req     refund.Request
	model   Model
	gateway *refund.Gateway
	receipt *refund.Receipt
}

func refundReview(ctx context.Context, e *runlet.Execution, req refund.Request) (refund.Result, error) {
	if req.ApplicationID == "" || req.OrderID == "" || req.UserID == "" {
		return refund.Result{}, runlet.Fail("INVALID_REQUEST", "application_id, order_id and user_id are required")
	}
	cfg, err := step(ctx, e, "load-config", struct{}{}, runlet.Reexecute,
		func(context.Context, struct{}) (AgentConfig, error) { return loadConfig() })
	if err != nil {
		return refund.Result{}, err
	}
	model, err := newModel(cfg.Model)
	if err != nil {
		return refund.Result{}, runlet.Fail("INVALID_CONFIG", err.Error())
	}
	a := &agent{e: e, cfg: cfg, req: req, model: model, gateway: refund.NewGateway(envOr("REFUND_GATEWAY_URL", "http://127.0.0.1:8095"))}

	rationale, err := a.run(ctx)
	if errors.Is(err, runlet.ErrWaitTimeout) {
		return refund.Result{}, runlet.Fail("APPROVAL_TIMEOUT", "no supervisor decision before the deadline")
	}
	if err != nil {
		return refund.Result{}, err
	}

	result := refund.Result{Outcome: "rejected", Rationale: rationale}
	if a.receipt != nil {
		result = refund.Result{Outcome: "refunded", Amount: a.receipt.Amount, RefundID: a.receipt.RefundID, Rationale: rationale}
	}
	if _, err := step(ctx, e, "notify-user", result, runlet.Idempotent, notifyUser(req.UserID)); err != nil {
		return refund.Result{}, err
	}
	return result, nil
}

// run is the ReAct loop: model -> tools in order -> model, until the model
// answers without tool calls. It returns the model's final message.
func (a *agent) run(ctx context.Context) (string, error) {
	application, err := json.Marshal(a.req)
	if err != nil {
		return "", err
	}
	msgs := []Message{{Role: "system", Content: a.cfg.SystemPrompt}, {Role: "user", Content: string(application)}}
	for turn := 1; turn <= maxTurns; turn++ {
		prefix := fmt.Sprintf("turn-%d", turn)
		reply, err := step(ctx, a.e, prefix+"/model",
			ModelRequest{Model: a.cfg.Model, Messages: msgs, Tools: toolSpecs}, runlet.Reexecute, a.model.Complete)
		if err != nil {
			return "", err
		}
		msgs = append(msgs, reply)
		if len(reply.ToolCalls) == 0 {
			return reply.Content, nil
		}
		for i, call := range reply.ToolCalls {
			result, err := a.callTool(ctx, fmt.Sprintf("%s/call-%d/%s", prefix, i+1, call.Name), call)
			if err != nil {
				return "", err
			}
			content, err := json.Marshal(result)
			if err != nil {
				return "", err
			}
			msgs = append(msgs, Message{Role: "tool", ToolCallID: call.ID, Name: call.Name, Content: string(content)})
		}
	}
	return "", runlet.Fail("TURN_LIMIT", fmt.Sprintf("no final answer within %d turns", maxTurns))
}

// notifyUser is a fake notification channel. A real one would deduplicate by
// the Step's idempotency key, which is why the Step uses Idempotent.
func notifyUser(userID string) func(context.Context, refund.Result) (string, error) {
	return func(ctx context.Context, result refund.Result) (string, error) {
		attempt, _ := runlet.StepAttempt(ctx)
		log.Printf("notify user=%s outcome=%s refund=%s message_id=%s", userID, result.Outcome, result.RefundID, attempt.IdempotencyKey)
		return attempt.IdempotencyKey, nil
	}
}

func loadConfig() (AgentConfig, error) {
	cfg := AgentConfig{ReexecuteRefund: os.Getenv("REFUND_EXPERIMENT_REEXECUTE") == "1", PromptVersion: "refund-review/v1", SystemPrompt: systemPrompt,
		Model: envOr("REFUND_MODEL", "fake:refund"), ApprovalThreshold: 50000}
	if value := os.Getenv("REFUND_APPROVAL_THRESHOLD"); value != "" {
		if _, err := fmt.Sscan(value, &cfg.ApprovalThreshold); err != nil {
			return AgentConfig{}, fmt.Errorf("REFUND_APPROVAL_THRESHOLD: %w", err)
		}
	}
	timeout := 48 * time.Hour
	if value := os.Getenv("REFUND_APPROVAL_TIMEOUT"); value != "" {
		var err error
		if timeout, err = time.ParseDuration(value); err != nil {
			return AgentConfig{}, fmt.Errorf("REFUND_APPROVAL_TIMEOUT: %w", err)
		}
	}
	cfg.ApprovalTimeoutMS = timeout.Milliseconds()
	return cfg, nil
}

func envOr(key, fallback string) string {
	if value := os.Getenv(key); value != "" {
		return value
	}
	return fallback
}

func main() {
	log.SetFlags(log.LstdFlags | log.Lmicroseconds)
	ctx, stop := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM)
	defer stop()
	if err := runlet.Run(ctx, runlet.Task(refund.Workflow, refundReview), runlet.Task("journal-measure", measureJournal)); err != nil && !errors.Is(err, context.Canceled) {
		log.Fatal(err)
	}
}
