# Runlet 智能体持久执行实验

以合成退款审核为例，验证审批跨进程恢复、非幂等工具结果核验和步骤日志重放。基础版本为 `82c7a7f`，实验源码取自 `fc35b3d`。

- [实验 Notebook](experiment_record.ipynb) 包含采集命令、对照方法、实测统计与图表。
- [SDK 源码](source/sdk/) 展示 Step 结果复用和 Await 持久等待。
- [服务端源码](source/internal/durable/) 展示历史匹配、未确认步骤处理和管理员核验。
- [实验源码](source/examples/refund-review/) 展示工具循环、退款审批、策略对照和重放计时。
- [对照事件](comparison.log)、[重放样本](journal.csv) 和[内存基线](memory.jsonl) 支持复算结果。

在本目录用 Jupyter 打开 Notebook，需要 Python 和 Matplotlib。Notebook 已保存运行输出，分析单元直接读取本目录的数据。完整采集使用 Go 1.26.5、MySQL 8.0.46 和 WSL 2。源码为课程快照，独立运行故障注入需要完整 Runlet 工作区。实验使用脚本化模型和模拟支付网关。
