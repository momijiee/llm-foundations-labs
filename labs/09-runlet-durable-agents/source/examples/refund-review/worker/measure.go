package main

import (
	"context"
	"fmt"
	"time"

	"runlet"
)

type measurement struct {
	NormalNS  int64
	ReplayNS  int64
	DirectNS  int64
	Callbacks int
}

func measureJournal(ctx context.Context, e *runlet.Execution, n int) (measurement, error) {
	if n < 1 || n > 128 {
		return measurement{}, fmt.Errorf("invalid step count: %d", n)
	}
	callback := func(_ context.Context, input int) (int, error) { return input + 1, nil }
	start := time.Now()
	direct := 0
	for i := 0; i < n; i++ {
		v, _ := callback(ctx, i)
		direct += v
	}
	directNS := time.Since(start).Nanoseconds()
	start = time.Now()
	actual, total := 0, 0
	for i := 0; i < n; i++ {
		v, err := runlet.Step(ctx, e, fmt.Sprintf("measure-%d", i), i, runlet.Reexecute, func(ctx context.Context, in int) (int, error) {
			actual++
			return callback(ctx, in)
		})
		if err != nil {
			return measurement{}, err
		}
		total += v
	}
	elapsed := time.Since(start).Nanoseconds()
	if total != direct {
		return measurement{}, fmt.Errorf("different prefix results")
	}
	// Store the first pass timing once. On replay the cached timing is returned.
	normal, err := runlet.Step(ctx, e, "normal-time", struct{}{}, runlet.Reexecute, func(context.Context, struct{}) ([2]int64, error) {
		return [2]int64{elapsed, directNS}, nil
	})
	if err != nil {
		return measurement{}, err
	}
	_, err = runlet.Await[bool](ctx, e, "continue", struct{}{}, runlet.AwaitOptions{Timeout: time.Minute})
	if err != nil {
		return measurement{}, err
	}
	return measurement{normal[0], elapsed, normal[1], actual}, nil
}
