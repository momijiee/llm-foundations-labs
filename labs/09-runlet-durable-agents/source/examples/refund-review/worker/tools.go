package main

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"time"

	"runlet"
	"runlet/examples/refund-review/refund"
)

// In-process read-only fake data. Amounts are in 分.
type orderInfo struct {
	OrderID  string `json:"order_id"`
	UserID   string `json:"user_id"`
	Item     string `json:"item"`
	Amount   int64  `json:"amount"`
	Refunded bool   `json:"refunded"`
}

type logisticsInfo struct {
	OrderID string `json:"order_id"`
	Status  string `json:"status"`
	Note    string `json:"note"`
}

type userHistory struct {
	UserID        string `json:"user_id"`
	Orders        int    `json:"orders"`
	RecentRefunds int    `json:"recent_refunds"`
}

var orders = map[string]orderInfo{
	"ORD-1001": {OrderID: "ORD-1001", UserID: "U-01", Item: "蓝牙耳机", Amount: 8900},
	"ORD-2001": {OrderID: "ORD-2001", UserID: "U-02", Item: "机械键盘", Amount: 128000},
	"ORD-3001": {OrderID: "ORD-3001", UserID: "U-03", Item: "保温杯", Amount: 5900, Refunded: true},
}

var logistics = map[string]logisticsInfo{
	"ORD-1001": {OrderID: "ORD-1001", Status: "delivered", Note: "签收时外包装破损"},
	"ORD-2001": {OrderID: "ORD-2001", Status: "lost", Note: "中转站超过 7 天无更新"},
	"ORD-3001": {OrderID: "ORD-3001", Status: "delivered"},
}

var histories = map[string]userHistory{
	"U-01": {UserID: "U-01", Orders: 23, RecentRefunds: 0},
	"U-02": {UserID: "U-02", Orders: 5, RecentRefunds: 1},
	"U-03": {UserID: "U-03", Orders: 2, RecentRefunds: 2},
}

var toolSpecs = []ToolSpec{
	{Name: "get_order", Description: "查询订单金额、商品与退款状态", Parameters: json.RawMessage(`{"type":"object","properties":{"order_id":{"type":"string"}},"required":["order_id"]}`)},
	{Name: "get_logistics", Description: "查询订单物流状态", Parameters: json.RawMessage(`{"type":"object","properties":{"order_id":{"type":"string"}},"required":["order_id"]}`)},
	{Name: "get_user_history", Description: "查询用户历史订单与近期退款次数", Parameters: json.RawMessage(`{"type":"object","properties":{"user_id":{"type":"string"}},"required":["user_id"]}`)},
	{Name: "issue_refund", Description: "发起退款。金额单位为分；超过审批阈值时需要主管审批", Parameters: json.RawMessage(`{"type":"object","properties":{"order_id":{"type":"string"},"amount":{"type":"integer"},"reason":{"type":"string"}},"required":["order_id","amount","reason"]}`)},
}

type refundArgs struct {
	OrderID string `json:"order_id"`
	Amount  int64  `json:"amount"`
	Reason  string `json:"reason"`
}

// toolResult is the tool message content returned to the model.
type toolResult struct {
	Result json.RawMessage       `json:"result,omitempty"`
	Error  *runlet.BusinessError `json:"error,omitempty"`
}

// callTool runs one model-requested tool under key prefix (for example
// "turn-2/call-1/issue_refund"). A *runlet.Failure is a business answer for
// the model; any other error (suspension, transport, timeout) stops the task.
func (a *agent) callTool(ctx context.Context, key string, call ToolCall) (toolResult, error) {
	value, err := a.dispatch(ctx, key, call)
	var failure *runlet.Failure
	if errors.As(err, &failure) {
		return toolResult{Error: &runlet.BusinessError{Code: failure.Code, Message: failure.Message}}, nil
	}
	if err != nil {
		return toolResult{}, err
	}
	data, err := json.Marshal(value)
	return toolResult{Result: data}, err
}

func (a *agent) dispatch(ctx context.Context, key string, call ToolCall) (any, error) {
	var args struct {
		OrderID string `json:"order_id"`
		UserID  string `json:"user_id"`
	}
	switch call.Name {
	case "get_order", "get_logistics", "get_user_history":
		if err := json.Unmarshal([]byte(call.Arguments), &args); err != nil {
			return nil, runlet.Fail("INVALID_ARGUMENTS", err.Error())
		}
	}
	switch call.Name {
	case "get_order":
		return step(ctx, a.e, key, args.OrderID, runlet.Reexecute, getOrder)
	case "get_logistics":
		return step(ctx, a.e, key, args.OrderID, runlet.Reexecute, func(_ context.Context, id string) (logisticsInfo, error) {
			if info, ok := logistics[id]; ok {
				return info, nil
			}
			return logisticsInfo{}, runlet.Fail("NOT_FOUND", "no logistics for "+id)
		})
	case "get_user_history":
		return step(ctx, a.e, key, args.UserID, runlet.Reexecute, func(_ context.Context, id string) (userHistory, error) {
			if h, ok := histories[id]; ok {
				return h, nil
			}
			return userHistory{}, runlet.Fail("NOT_FOUND", "unknown user "+id)
		})
	case "issue_refund":
		var refundCall refundArgs
		if err := json.Unmarshal([]byte(call.Arguments), &refundCall); err != nil {
			return nil, runlet.Fail("INVALID_ARGUMENTS", err.Error())
		}
		return a.issueRefund(ctx, key, refundCall)
	}
	return nil, runlet.Fail("UNKNOWN_TOOL", "no tool named "+call.Name)
}

func getOrder(_ context.Context, id string) (orderInfo, error) {
	if order, ok := orders[id]; ok {
		return order, nil
	}
	return orderInfo{}, runlet.Fail("NOT_FOUND", "unknown order "+id)
}

// issueRefund is a tool with several durable positions: read the order,
// validate in code, Await a supervisor above the threshold, then call the
// non-idempotent gateway exactly once under NoAutoRetry.
func (a *agent) issueRefund(ctx context.Context, key string, args refundArgs) (refund.Receipt, error) {
	order, err := step(ctx, a.e, key+"/load-order", args.OrderID, runlet.Reexecute, getOrder)
	if err != nil {
		return refund.Receipt{}, err
	}
	switch {
	case args.Amount <= 0 || args.Amount > order.Amount:
		return refund.Receipt{}, runlet.Fail("INVALID_AMOUNT", fmt.Sprintf("amount must be in 1..%d", order.Amount))
	case order.Refunded:
		return refund.Receipt{}, runlet.Fail("ALREADY_REFUNDED", "order "+order.OrderID+" was already refunded")
	}

	if args.Amount > a.cfg.ApprovalThreshold {
		approval, err := runlet.Await[refund.Approval](ctx, a.e, key+"/approval",
			refund.ApprovalRequest{ApplicationID: a.req.ApplicationID, OrderID: order.OrderID, Amount: args.Amount, Reason: args.Reason},
			runlet.AwaitOptions{Timeout: time.Duration(a.cfg.ApprovalTimeoutMS) * time.Millisecond})
		if err != nil {
			return refund.Receipt{}, err // ErrSuspended, ErrWaitTimeout or transport errors stop the task
		}
		if !approval.Approved {
			return refund.Receipt{}, runlet.Fail("APPROVAL_REJECTED", "主管 "+approval.Approver+" 拒绝："+approval.Comment)
		}
	}

	cmd := refund.Command{ApplicationID: a.req.ApplicationID, OrderID: order.OrderID, Amount: args.Amount}
	retry := runlet.NoAutoRetry
	if a.cfg.ReexecuteRefund {
		retry = runlet.Reexecute
	}
	receipt, err := step(ctx, a.e, key+"/gateway", cmd, retry,
		func(ctx context.Context, cmd refund.Command) (refund.Receipt, error) {
			attempt, ok := runlet.StepAttempt(ctx)
			if !ok {
				return refund.Receipt{}, errors.New("missing Step attempt")
			}
			// The stable idempotency key is the refund number, so an operator can
			// find exactly this request in the gateway when the outcome is unknown.
			return a.gateway.Refund(ctx, refund.GatewayRefund{
				RefundID: attempt.IdempotencyKey, ApplicationID: cmd.ApplicationID, OrderID: cmd.OrderID, Amount: cmd.Amount,
			})
		})
	if err != nil {
		return refund.Receipt{}, err
	}
	a.receipt = &receipt
	return receipt, nil
}
