# 白模转写实：纯 CPU 深度控制实验

参考教师代码附录自行补全的 Blender → Mist 深度图 → SD-Turbo + SD2.1 深度 ControlNet 流水线。
实验过程与 A/B/C 三组推理入口在 [experiment_record.ipynb](experiment_record.ipynb)。三组运行及后续分析已完成，Notebook 保留真实日志、图片、耗时与思考题解答。

## 本机使用

在 VS Code 打开 Notebook，选择已安装依赖的 **Windows Python** 内核，从上到下运行并保存。三组推理分别放在 3.1、3.2、3.3 单元，第四节生成耗时表、对比图和 CSV。现有分析对应已保存的运行会话；复现得到新结果后，应同步核对分析文字中的数字与观察。

准备完成的程序、权重、场景、中间图和原始日志存放在仓库内本地私有 `lesson7/runtime/`，不会进入版本控制。其他机器默认使用实验目录下已忽略的 `.runtime/`；也可在启动 Notebook 前设置 `HW7_RUNTIME` 指定其他目录。`BLENDER_EXE`、`SD_CLI` 可覆盖可执行程序位置。Notebook 不需要填写个人绝对路径。

## 从零准备

以下命令均在本实验目录的 Windows 终端执行；选择同一个 Python 运行准备脚本及 Notebook。

```powershell
python -m pip install -r requirements.txt
python scripts/prepare_resources.py
python scripts/run_pipeline.py --prepare --threads 12
```

资源脚本只下载、校验、解压及转换 GGUF，不执行图像推理。原始权重与输出合计需要十余 GB 空间；原始主模型下载约 5.21 GB。网络中断后可重跑资源脚本续传其 `.part` 文件，下载后校验作者 SHA256。模型和后端保持本地，不随代码上传。遵守各上游项目与模型的许可。

固定资源：

| 资源 | 来源与固定版本 |
|---|---|
| Blender | [官方 4.5 发布目录](https://download.blender.org/release/Blender4.5/)，4.5.13 Windows x64 ZIP |
| stable-diffusion.cpp | [官方 CPU 发布包](https://github.com/leejet/stable-diffusion.cpp/releases/tag/master-1017-be48649)，提交 `be48649` |
| SD-Turbo | [stabilityai/sd-turbo](https://huggingface.co/stabilityai/sd-turbo)，修订 `b261bac6fd2cf515557d5d0707481eafa0485ec2` |
| SD2.1 depth ControlNet | [thibaud/controlnet-sd21](https://huggingface.co/thibaud/controlnet-sd21)，修订 `ba159a8ce47185aa1821db7fe10137cf2f5e04dd` |

按[官方转换接口](https://github.com/leejet/stable-diffusion.cpp/blob/be48649/docs/quantization_and_gguf.md)将原始 safetensors 转为 `sd_turbo-q4_0.gguf` 和 `control_depth_sd21-fp16.gguf`。主模型先转换 F16，再转换 Q4_0，降低非量化张量保留原始 F32 所带来的额外体积；部分嵌入、偏置等张量仍由后端保留原精度，以实际文件为准。转换命令、输入与输出校验和保存在本地；不能用 SD1.5 ControlNet 代替 SD2.1 版本。

## 实验设计与记录

| 组别 | 分辨率 | 步数 | CFG | ControlNet |
|---|---:|---:|---:|---:|
| A | 512² | 4 | 1.2 | 0.9 |
| B | 512² | 8 | 1.2 | 0.9 |
| C | 512² | 4 | 1.2 | 0.5 |

共同设置：CPU、12 线程、种子 42、`euler_a`、`discrete` 调度器、CPU RNG。保持场景、提示词及控制图一致。线程数可在 Notebook 中修改，修改后须重新导图并对所有组保持一致。

本次实测：

附件见 [results/](results/)：包含 clay、depth、normal、A/B/C 生成图、[对比图](results/comparison.png)及[耗时 CSV](results/experiment_results.csv)。公开 PNG 与 Notebook 内嵌 PNG 已移除文本及 EXIF 元数据，像素保持不变；Notebook 中的原始文件校验和仍对应本地运行文件。

| 组别 | 采样(s) | VAE(s) | 总耗时(s) |
|---|---:|---:|---:|
| A | 43.52 | 17.83 | 69.77 |
| B | 82.45 | 15.47 | 106.89 |
| C | 38.13 | 12.85 | 58.75 |

较强控制保留了主要布局，但 A/B 的中央方体均被生成圆口容器；C 的弱控制产生明显场景偏移。八步增加了耗时，未解决几何误变问题。每组仅运行一次，结果仅说明本场景及配置下的表现。

一次 Cycles 渲染同时导出 clay、depth、normal，共同计时。推理总耗时为 sd-cli 子进程墙钟耗时，包含模型加载与写图，共享 Blender 准备单列。采样/VAE 耗时从后端真实日志提取，无法识别时留空待核对。进程树 RSS 每 0.25 秒采样，仅表示采样观察到的峰值。

每个 Notebook 会话和每次尝试有独立目录。每组保留 `record.json`、原始 `sd-cli.log`、控制图副本及最终图片；失败或中断留存状态，重跑不会覆盖旧尝试。中断 Notebook 单元时会停止正在运行的子进程。

控制图范围遵循附录的视锥顶点分位数方法；视锥内顶点不保证无遮挡，且不同网格顶点密度会影响分位数。应观察深度饱和、物体相互遮挡等局限，按真实图片分析几何保持度，不能将像素差直接当成几何精度。

当前未实现可选头像融合。报告、教师材料、个人信息、模型及本地原始日志不公开。发布 Notebook 前检查输出和元数据中的个人路径；本实验代码会对控制台中的本地路径做替换，但异常回溯仍需检查。
