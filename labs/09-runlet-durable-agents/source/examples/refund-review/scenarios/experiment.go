package main

import (
	"context"
	"encoding/json"
	"fmt"
	"time"

	"runlet/examples/refund-review/refund"
)

func reexecuteAfterRefund(ctx context.Context, h *harness) error {
	run, app, err := h.submit(ctx, "ORD-1001", "U-01")
	if err != nil {
		return err
	}
	code, err := h.waitWorkerExit(20 * time.Second)
	if code != 17 || err != nil {
		return fmt.Errorf("expected exit 17, got %d: %v", code, err)
	}
	if err := h.expectLedger(ctx, app, 1); err != nil {
		return err
	}
	if err := h.restartWorker(); err != nil {
		return err
	}
	return expectRepeatedRefund(ctx, h, run, app)
}

func reexecuteDroppedResponse(ctx context.Context, h *harness) error {
	if err := h.gateway.SetFaults(ctx, refund.Faults{DropResponse: true}); err != nil {
		return err
	}
	defer h.gateway.SetFaults(ctx, h.baseFaults)
	run, app, err := h.submit(ctx, "ORD-1001", "U-01")
	if err != nil {
		return err
	}
	deadline := time.Now().Add(15 * time.Second)
	for {
		entries, err := h.ledgerFor(ctx, app)
		if err != nil {
			return err
		}
		if len(entries) == 1 {
			break
		}
		if time.Now().After(deadline) {
			return fmt.Errorf("first refund did not reach ledger")
		}
		time.Sleep(50 * time.Millisecond)
	}
	if err := h.gateway.SetFaults(ctx, h.baseFaults); err != nil {
		return err
	}
	return expectRepeatedRefund(ctx, h, run, app)
}

func expectRepeatedRefund(ctx context.Context, h *harness, run, app string) error {
	done, err := h.waitStatus(ctx, run, "completed", "", 60*time.Second)
	if err != nil {
		return err
	}
	var result refund.Result
	if err := json.Unmarshal(done.Result, &result); err != nil {
		return err
	}
	entries, err := h.ledgerFor(ctx, app)
	if err != nil {
		return err
	}
	if result.Outcome != "refunded" || len(entries) != 2 || entries[0].RefundID != result.RefundID || entries[1].RefundID != result.RefundID {
		return fmt.Errorf("expected two ledger entries for the same refund, got %+v, %+v", result, entries)
	}
	fmt.Printf("E3 duplicate ledger_count=%d\n", len(entries))
	return nil
}

func measureJournal(ctx context.Context, h *harness) error {
	const task = "journal-measure"
	if err := h.app.Activate(ctx, h.adminToken, task, h.buildID); err != nil {
		return err
	}
	if err := h.startWorker(nil); err != nil {
		return err
	}
	defer h.stopWorker()
	fmt.Println("steps,iteration,direct_ns,normal_ns,replay_ns,replay_callbacks")
	for iteration := 0; iteration <= 15; iteration++ {
		for _, n := range []int{8, 32, 64, 128} {
			run, err := h.app.Submit(ctx, task, fmt.Sprintf("%s-%d-%d", h.runID, iteration, n), n)
			if err != nil {
				return err
			}
			waiting, err := h.waitStatus(ctx, run.ID, "waiting", "AWAIT_INPUT", 20*time.Second)
			if err != nil {
				return err
			}
			if _, err = h.app.RespondWait(ctx, run.ID, waiting.CurrentWait.ID, fmt.Sprintf("measure-%s", run.ID), true); err != nil {
				return err
			}
			done, err := h.waitStatus(ctx, run.ID, "completed", "", 20*time.Second)
			if err != nil {
				return err
			}
			var m struct {
				NormalNS, ReplayNS, DirectNS int64
				Callbacks                    int
			}
			if err := json.Unmarshal(done.Result, &m); err != nil {
				return err
			}
			if m.Callbacks != 0 || m.NormalNS <= 0 || m.ReplayNS <= 0 {
				return fmt.Errorf("unexpected measurement: %+v", m)
			}
			if iteration > 0 {
				fmt.Printf("%d,%d,%d,%d,%d,%d\n", n, iteration, m.DirectNS, m.NormalNS, m.ReplayNS, m.Callbacks)
			}
		}
	}
	return nil
}
