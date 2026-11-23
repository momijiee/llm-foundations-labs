package sdk

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"time"

	"runlet/protocol"
)

type RetryMode = protocol.RetryMode

const (
	Reexecute   = protocol.Reexecute
	Idempotent  = protocol.Idempotent
	NoAutoRetry = protocol.NoAutoRetry
)

// Failure is a definite application outcome that can be recorded and replayed.
// Ordinary errors from a Step callback are ambiguous operation failures.
type Failure struct{ Code, Message string }

func (e *Failure) Error() string      { return e.Code + ": " + e.Message }
func Fail(code, message string) error { return &Failure{Code: code, Message: message} }

type Attempt struct {
	IdempotencyKey string
	Number         int
}

type StepSpec[I any] struct {
	Key       string
	Operation string
	Retry     RetryMode
	Input     I
}

type Run struct {
	ID        string
	Input     json.RawMessage
	client    *Client
	owner     string
	epoch     int64
	position  int
	aborted   bool
	suspended bool
}

func (r *Run) DecodeInput(value any) error { return json.Unmarshal(r.Input, value) }

func Step[I, O any](ctx context.Context, run *Run, spec StepSpec[I], fn func(context.Context, I, Attempt) (O, error)) (O, error) {
	var zero O
	if run == nil || run.aborted || fn == nil {
		return zero, errors.New("run is not active")
	}
	if err := ctx.Err(); err != nil {
		run.aborted = true
		return zero, err
	}
	input, snapshot, err := snapshotStepInput(spec.Input)
	if err != nil {
		run.aborted = true
		return zero, err
	}

	hash := sha256.Sum256(input)
	run.position++
	position := run.position
	// Resolve either records STARTED or returns a previously committed result.
	// The callback must not run before this request is confirmed.
	var response protocol.StepResponse
	err = run.client.call(ctx, http.MethodPost, "/runs/"+run.ID+"/steps/resolve", protocol.StepRequest{
		Owner: run.owner, Epoch: run.epoch, Position: position, Key: spec.Key,
		Operation: spec.Operation, Retry: spec.Retry, CodecVersion: protocol.CodecVersion,
		Input: input, InputHash: hex.EncodeToString(hash[:]),
	}, &response)
	if err != nil {
		run.aborted = true
		return zero, err
	}

	if response.State == "completed" { // Replay Step
		if response.Error != nil {
			return zero, &Failure{Code: response.Error.Code, Message: response.Error.Message}
		}
		if err := json.Unmarshal(response.Result, &zero); err != nil {
			run.aborted = true
			return zero, fmt.Errorf("decode recorded step result: %w", err)
		}
		return zero, nil
	}

	if response.State != "execute" {
		run.aborted = true
		return zero, errors.New("unexpected step state")
	}

	// Execute user fn callback
	result, callbackErr := fn(ctx, snapshot, Attempt{IdempotencyKey: response.IdempotencyKey, Number: response.Attempt})
	if err := ctx.Err(); err != nil {
		run.aborted = true
		return zero, err
	}
	return finishStepAttempt(ctx, run, spec.Retry, position, result, callbackErr)
}

// snapshotStepInput gives the callback the value represented by the exact
// JSON bytes sent to the Journal, avoiding a mutable closure/input mismatch.
func snapshotStepInput[I any](value I) ([]byte, I, error) {
	var snapshot I
	input, err := json.Marshal(value)
	if err != nil {
		return nil, snapshot, err
	}
	if len(input) > 1<<20 {
		return nil, snapshot, errors.New("step input exceeds 1 MiB")
	}
	if err := json.Unmarshal(input, &snapshot); err != nil {
		return nil, snapshot, err
	}
	return input, snapshot, nil
}

func finishStepAttempt[O any](ctx context.Context, run *Run, retry RetryMode, position int, result O, callbackErr error) (O, error) {
	var zero O
	complete := protocol.CompleteStepRequest{LeaseRequest: protocol.LeaseRequest{Owner: run.owner, Epoch: run.epoch}, Position: position}
	if callbackErr != nil {
		var failure *Failure
		if errors.As(callbackErr, &failure) && failure.Code != "" {
			complete.Error = &protocol.BusinessError{Code: failure.Code, Message: failure.Message}
		} else {
			run.aborted = true
			if retry == NoAutoRetry {
				run.waitOutcomeUnknown(ctx, position)
			}
			return zero, callbackErr
		}
	} else {
		encoded, err := json.Marshal(result)
		if err != nil || len(encoded) > 1<<20 {
			run.aborted = true
			if err == nil {
				err = errors.New("step result exceeds 1 MiB")
			}
			if retry == NoAutoRetry {
				run.waitOutcomeUnknown(ctx, position)
			}
			return zero, err
		}
		complete.Result = encoded
	}
	if err := run.client.call(ctx, http.MethodPost, "/runs/"+run.ID+"/steps/complete", complete, nil); err != nil {
		run.aborted = true
		return zero, err
	}
	if callbackErr != nil {
		return zero, callbackErr
	}
	return result, nil
}

// waitOutcomeUnknown is best effort. If its request fails, the STARTED record
// still prevents an automatic NoAutoRetry callback on the next claim.
func (run *Run) waitOutcomeUnknown(ctx context.Context, position int) {
	waitCtx, cancel := context.WithTimeout(context.WithoutCancel(ctx), 5*time.Second)
	defer cancel()
	_ = run.client.call(waitCtx, http.MethodPost, "/runs/"+run.ID+"/wait", protocol.WaitRequest{
		LeaseRequest: protocol.LeaseRequest{Owner: run.owner, Epoch: run.epoch}, Position: position, Reason: "OUTCOME_UNKNOWN",
	}, nil)
}
