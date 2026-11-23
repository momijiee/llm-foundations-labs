package durable

import (
	"context"
	"database/sql"
	"encoding/json"
	"errors"

	"runlet/protocol"
)

func validResolution(req protocol.ResolveReconciliationRequest) bool {
	if req.DecisionID == "" || len(req.DecisionID) > 128 || req.ExpectedVersion < 1 || req.Position < 1 ||
		req.Evidence == "" || len(req.Evidence) > 4096 || len(req.Reason) > 4096 {
		return false
	}
	switch req.Decision {
	case "record_result":
		return req.Reason == "" && ((req.Error == nil && len(req.Result) > 0 && len(req.Result) <= 1<<20 && json.Valid(req.Result)) ||
			(req.Error != nil && req.Error.Code != "" && len(req.Error.Code) <= 128 && len(req.Error.Message) <= 4096 && len(req.Result) == 0))
	case "retry_once":
		return req.Reason == "" && req.Error == nil && len(req.Result) == 0
	case "fail_run":
		return req.Reason != "" && req.Error == nil && len(req.Result) == 0
	}
	return false
}

// ResolveReconciliation applies one operator decision atomically with its audit
// record. This transaction does not use Worker fencing: the waiting Run has no
// owner, and the operator instead supplies an expected reconciliation version.
func (s *Store) ResolveReconciliation(ctx context.Context, runID, actor string, req protocol.ResolveReconciliationRequest) (protocol.ReconciliationDecision, error) {
	var empty protocol.ReconciliationDecision
	if !validResolution(req) || actor == "" || len(actor) > 128 {
		return empty, ErrConflict
	}
	encoded, err := json.Marshal(req)
	if err != nil {
		return empty, ErrConflict
	}
	hash := digest(encoded)
	tx, err := s.db.BeginTx(ctx, nil)
	if err != nil {
		return empty, err
	}
	defer tx.Rollback()

	// Lock the Run before its reconciliation and Step rows. A concurrent
	// decision on the same Run must see this transaction's committed state.
	status, reason, err := lockReconciliationRun(ctx, tx, runID)
	if errors.Is(err, sql.ErrNoRows) {
		return empty, ErrNotFound
	}
	if err != nil {
		return empty, err
	}
	// Check the decision ID before checking current status/version. A retry of a
	// committed decision must return its original answer after the Run moved on.
	storedRun, storedHash, decision, err := findReconciliationDecision(ctx, tx, req.DecisionID)
	if err == nil {
		if storedRun != runID || storedHash != hash {
			return empty, ErrConflict
		}
		return decision, nil
	}
	if !errors.Is(err, sql.ErrNoRows) {
		return empty, err
	}
	if status != "waiting" || reason.String != "OUTCOME_UNKNOWN" {
		return empty, ErrConflict
	}
	version, err := lockReconciliationVersion(ctx, tx, runID)
	if err != nil {
		return empty, err
	}
	if version != req.ExpectedVersion {
		return empty, ErrConflict
	}
	mode, stepState, err := lockReconciliationStep(ctx, tx, runID, req.Position)
	if err != nil {
		if errors.Is(err, sql.ErrNoRows) {
			return empty, ErrConflict
		}
		return empty, err
	}
	if stepState != "outcome_unknown" || (mode != string(protocol.NoAutoRetry) && mode != string(protocol.Idempotent)) {
		return empty, ErrConflict
	}
	// Keep the three possible state transitions visible here. SQL helpers only
	// perform individual writes; none of them may commit this transaction.
	switch req.Decision {
	case "record_result":
		err = recordReconciledResult(ctx, tx, runID, req)
		if err == nil {
			err = markRunPending(ctx, tx, runID)
		}
	case "retry_once":
		err = authorizeStepRetry(ctx, tx, runID, req.Position)
		if err == nil {
			err = markRunPending(ctx, tx, runID)
		}
	case "fail_run":
		err = failUnresolvedRun(ctx, tx, runID, req.Reason)
	}
	if err != nil {
		return empty, err
	}
	// Audit and state change commit together. A failed insert rolls back both.
	decision, err = insertReconciliationDecision(ctx, tx, runID, actor, hash, version, req)
	if err != nil {
		return empty, err
	}
	if err := tx.Commit(); err != nil {
		return empty, err
	}
	return decision, nil
}
