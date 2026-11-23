package sdk

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"time"

	"runlet/protocol"
)

var ErrSuspended = errors.New("execution suspended")
var ErrWaitTimeout = errors.New("durable wait timed out")

type AwaitOptions struct{ Timeout time.Duration }

// Await persists the request before suspending. Return its error from the task;
// a later invocation replays the prefix and returns the recorded response.
func Await[T any](ctx context.Context, run *Run, key string, request any, options AwaitOptions) (T, error) {
	var result T
	if run == nil || run.aborted || key == "" {
		return result, errors.New("Await requires an active execution and key")
	}
	if err := ctx.Err(); err != nil {
		run.aborted = true
		return result, err
	}
	if options.Timeout < 0 || (options.Timeout > 0 && options.Timeout < time.Millisecond) {
		run.aborted = true
		return result, errors.New("wait timeout must be zero or at least one millisecond")
	}
	encoded, err := json.Marshal(request)
	if err != nil {
		run.aborted = true
		return result, err
	}
	run.position++
	var out protocol.AwaitResponse
	err = run.client.call(ctx, http.MethodPost, "/runs/"+run.ID+"/waits/resolve", protocol.AwaitRequest{LeaseRequest: protocol.LeaseRequest{Owner: run.owner, Epoch: run.epoch}, Position: run.position, Key: key, CodecVersion: protocol.CodecVersion, Request: encoded, TimeoutMS: options.Timeout.Milliseconds()}, &out)
	if err != nil {
		run.aborted = true
		return result, err
	}
	if out.State == "suspended" {
		run.suspended = true
		run.aborted = true
		return result, ErrSuspended
	}
	if out.State != "completed" || out.Outcome == nil {
		run.aborted = true
		return result, errors.New("invalid Await response")
	}
	switch out.Outcome.Kind {
	case "timeout":
		return result, ErrWaitTimeout
	case "input":
		if err = json.Unmarshal(out.Outcome.Value, &result); err != nil {
			run.aborted = true
			return result, fmt.Errorf("decode recorded wait response: %w", err)
		}
		return result, nil
	default:
		run.aborted = true
		return result, errors.New("invalid wait outcome")
	}
}
