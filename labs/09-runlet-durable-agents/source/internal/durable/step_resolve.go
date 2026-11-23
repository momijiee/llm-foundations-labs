package durable

import (
	"context"
	"database/sql"
	"encoding/json"
	"errors"

	"runlet/protocol"
)

// Resolve decides whether this call replays a committed Step or starts an
// attempt. It locks the Run row before reading or changing any Step row.
func (s *Store) Resolve(ctx context.Context, runID string, req protocol.StepRequest) (protocol.StepResponse, error) {
	if !validStepRequest(req) {
		return protocol.StepResponse{}, ErrConflict
	}
	tx, err := s.db.BeginTx(ctx, nil)
	if err != nil {
		return protocol.StepResponse{}, err
	}
	defer tx.Rollback()

	o, err := lockOwned(ctx, tx, runID, protocol.LeaseRequest{Owner: req.Owner, Epoch: req.Epoch})
	if err != nil {
		return protocol.StepResponse{}, err
	}

	if req.Position > maxSteps {
		return commitWaitingRun(ctx, tx, runID, "STEP_LIMIT", 0, ErrLimit)
	}

	step, err := loadJournalStep(ctx, tx, runID, req.Position)
	if errors.Is(err, sql.ErrNoRows) {
		if req.Position != o.last+1 {
			return commitWaitingRun(ctx, tx, runID, "REPLAY_MISMATCH", 0, ErrReplay)
		}

		var unfinished int
		if err = tx.QueryRowContext(ctx, `SELECT COUNT(*) FROM durable_steps WHERE run_id=? AND state!='completed'`, runID).Scan(&unfinished); err != nil {
			return protocol.StepResponse{}, err
		}
		if unfinished != 0 {
			return protocol.StepResponse{}, ErrConflict
		}
		idem, err := recordStepIntent(ctx, tx, runID, req) // Set Step to 'Started'

		if err != nil {
			// A key reused at another position is a replay mismatch. Other SQL
			// errors remain infrastructure errors and must not mutate the Run.
			var used int
			if qerr := tx.QueryRowContext(ctx, `SELECT COUNT(*) FROM durable_steps WHERE run_id = ? AND step_key = ?`, runID, req.Key).Scan(&used); qerr == nil && used > 0 {
				return commitWaitingRun(ctx, tx, runID, "REPLAY_MISMATCH", 0, ErrReplay)
			}
			return protocol.StepResponse{}, err
		}
		if _, err = tx.ExecContext(ctx, `UPDATE durable_runs SET last_position = ? WHERE id = ?`, req.Position, runID); err != nil {
			return protocol.StepResponse{}, err
		}
		if err = tx.Commit(); err != nil {
			return protocol.StepResponse{}, err
		}
		return protocol.StepResponse{State: "execute", Attempt: 1, IdempotencyKey: idem}, nil // Happy Path to execute new step
	}

	if err != nil {
		return protocol.StepResponse{}, err
	}
	if !step.matches(req) {
		return commitWaitingRun(ctx, tx, runID, "REPLAY_MISMATCH", 0, ErrReplay)
	}
	if step.state == "completed" {
		if err = tx.Commit(); err != nil {
			return protocol.StepResponse{}, err
		}
		return step.completedResponse(), nil // Happy Path to replay step
	}

	if step.state == "outcome_unknown" {
		var retryPosition sql.NullInt64
		if err = tx.QueryRowContext(ctx, `SELECT retry_position FROM durable_reconciliations
			WHERE run_id = ? FOR UPDATE`, runID).Scan(&retryPosition); err != nil {
			return protocol.StepResponse{}, err
		}
		if !retryPosition.Valid || int(retryPosition.Int64) != req.Position {
			return protocol.StepResponse{}, ErrUnknown
		}
		step.attempt++
		if _, err = tx.ExecContext(ctx, `UPDATE durable_steps SET state = 'started', attempt = ?, attempt_epoch = ?
			WHERE run_id = ? AND position = ?`, step.attempt, req.Epoch, runID, req.Position); err != nil {
			return protocol.StepResponse{}, err
		}
		if _, err = tx.ExecContext(ctx, `UPDATE durable_reconciliations SET retry_position = NULL WHERE run_id = ?`, runID); err != nil {
			return protocol.StepResponse{}, err
		}
		if err = tx.Commit(); err != nil {
			return protocol.StepResponse{}, err
		}
		return protocol.StepResponse{State: "execute", Attempt: step.attempt, IdempotencyKey: step.idempotencyKey}, nil
	}

	if step.state != "started" {
		return protocol.StepResponse{}, ErrUnknown
	}

	if step.attemptEpoch != req.Epoch {
		if req.Retry == protocol.NoAutoRetry || step.attempt >= maxAttempts {
			reason := "OUTCOME_UNKNOWN"
			if req.Retry == protocol.Reexecute {
				reason = "RETRY_EXHAUSTED"
			}
			return commitWaitingRun(ctx, tx, runID, reason, req.Position, ErrUnknown)
		}
		step.attempt++
		if _, err = tx.ExecContext(ctx, `UPDATE durable_steps SET attempt = ?, attempt_epoch = ?
			WHERE run_id = ? AND position = ?`, step.attempt, req.Epoch, runID, req.Position); err != nil {
			return protocol.StepResponse{}, err
		}
	}
	if err = tx.Commit(); err != nil {
		return protocol.StepResponse{}, err
	}
	return protocol.StepResponse{State: "execute", Attempt: step.attempt, IdempotencyKey: step.idempotencyKey}, nil // Auto retry
}

func validStepRequest(req protocol.StepRequest) bool {
	return req.Position >= 1 && req.Key != "" && len(req.Key) <= 128 &&
		req.Operation != "" && len(req.Operation) <= 128 &&
		(req.Retry == protocol.Reexecute || req.Retry == protocol.Idempotent || req.Retry == protocol.NoAutoRetry) &&
		req.CodecVersion == protocol.CodecVersion && len(req.Input) <= 1<<20 &&
		json.Valid(req.Input) && digest(req.Input) == req.InputHash
}

type journalStep struct {
	kind           string
	key            string
	operation      string
	retry          string
	codec          string
	inputHash      string
	state          string
	idempotencyKey string
	attempt        int
	attemptEpoch   int64
	result         []byte
	code           sql.NullString
	message        sql.NullString
}

func loadJournalStep(ctx context.Context, tx *sql.Tx, runID string, position int) (journalStep, error) {
	var step journalStep
	err := tx.QueryRowContext(ctx, `SELECT kind, step_key, operation_name, COALESCE(retry_mode, ''), codec_version,
		input_hash, state, attempt, attempt_epoch, COALESCE(idempotency_key, ''), result_json, error_code, error_message
		FROM durable_steps WHERE run_id = ? AND position = ? FOR UPDATE`, runID, position).Scan(
		&step.kind, &step.key, &step.operation, &step.retry, &step.codec, &step.inputHash, &step.state,
		&step.attempt, &step.attemptEpoch, &step.idempotencyKey, &step.result, &step.code, &step.message)
	return step, err
}

func (step journalStep) matches(req protocol.StepRequest) bool {
	return step.kind == "call" && step.key == req.Key && step.operation == req.Operation &&
		step.retry == string(req.Retry) && step.codec == req.CodecVersion &&
		step.inputHash == req.InputHash
}

func (step journalStep) completedResponse() protocol.StepResponse {
	response := protocol.StepResponse{State: "completed", Result: step.result,
		Attempt: step.attempt, IdempotencyKey: step.idempotencyKey}
	if step.code.Valid {
		response.Error = &protocol.BusinessError{Code: step.code.String, Message: step.message.String}
	}
	return response
}

func recordStepIntent(ctx context.Context, tx *sql.Tx, runID string, req protocol.StepRequest) (string, error) {
	idempotencyKey := digest([]byte(runID + ":" + req.Key))
	_, err := tx.ExecContext(ctx, `INSERT INTO durable_steps
		(run_id, position, step_key, operation_name, retry_mode, codec_version,
		input_json, input_hash, state, attempt, attempt_epoch, idempotency_key)
		VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'started', 1, ?, ?)`,
		runID, req.Position, req.Key, req.Operation, req.Retry, req.CodecVersion,
		string(req.Input), req.InputHash, req.Epoch, idempotencyKey)
	if err != nil {
		return "", err
	}
	return idempotencyKey, nil
}

// A waiting transition must commit before its error is returned to the Worker;
// rolling back would make the Run claimable again without its waiting reason.
func commitWaitingRun(ctx context.Context, tx *sql.Tx, runID, reason string, position int, outcome error) (protocol.StepResponse, error) {
	if err := wait(ctx, tx, runID, reason, position); err != nil {
		return protocol.StepResponse{}, err
	}
	if err := tx.Commit(); err != nil {
		return protocol.StepResponse{}, err
	}
	return protocol.StepResponse{}, outcome
}
