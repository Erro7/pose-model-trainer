# Pose Model Trainer

魔搭 Notebook 图片 / 视频 LoRA 快速训练器，带姿势图库按需导入功能。

[进入魔搭 Notebook](https://modelscope.cn/my/mynotebook)

## 推荐 Notebook 环境

新建魔搭 GPU Notebook 时，推荐选择：

```text
modelscope:1.40.1-pytorch2.13.0-gpu-py312-cu130-ubuntu22.04
```

推荐理由：

- Python 3.12
- PyTorch 2.13.0
- CUDA 13.0
- 与 H3 EntroPack 的 PyTorch 2.10+ 要求兼容
- H3 使用 `entropack[cuda13]`
- 不再推荐旧的 torch 2.0 / CUDA 11.7 WebUI 镜像

### AMD ROCm：H3 不支持，Wan 视频实验支持

`ubuntu22.04-rocm7.2.3-py312-torch2.12.0-1.40.1` 不能运行本项目的 H3 工作流。H3 EntroPack 只提供 `entropack[cuda12]` / `entropack[cuda13]` 扩展，启动时强制检查 `torch.version.cuda` 与 `torch.cuda`；AMD ROCm 提供的 `torch.version.hip` 不会被当成 CUDA 使用。

MiniMax-H3 家族另有一支视频模型（FL2VA / Ref2VA，输出视频与原生音频），上游 DiffSynth-Studio 已提供示例和 LoRA 训练脚本；但本项目尚未把 H3 视频接入一键流程，也没有可在当前 ROCm 镜像上验证的 H3 视频量化/训练路径，因此不会把“上游存在模型”误报成“AMD 可用”。

| 工作流 | NVIDIA CUDA 12.8+ | AMD ROCm |
|---|---|---|
| H3-Image + EntroPack（本项目一键 H3 流程） | 支持 | 不支持 |
| H3 视频 FL2VA / Ref2VA（上游 DiffSynth） | 上游可用，本项目未接入 | 不支持 / 未验证 |
| Wan 2.1 / 2.2 视频 LoRA（本项目训练页） | 支持 | 实验支持 |

一键安装和启动脚本会检查镜像名、`torch.version.hip`、`rocm-smi`、`ROCM_PATH`、`HIP_PATH` 与 `/opt/rocm`。检测到 ROCm 且没有 `nvidia-smi` 时仍会启动 WebUI，但 H3 生成、H3 环境准备和 H3 训练入口会直接禁用；Wan 视频训练会自动关闭 FP8、T5 FP8、cuDNN 加速和 `adamw8bit`，改用 bf16 + AdamW。该 ROCm 路径属于实验支持，建议先用小分辨率、少帧数验证。要使用 H3，请切换到推荐的 NVIDIA 镜像：

```text
modelscope:1.40.1-pytorch2.13.0-gpu-py312-cu130-ubuntu22.04
```

启动器使用两个完全隔离的持久环境：`.venv` 只运行 Gradio WebUI，`.venv-h3` 专门安装 PyTorch、DiffSynth、EntroPack 和训练依赖。安装或修复 H3 环境不会再修改正在运行的 WebUI。更换容器导致 Python 版本变化时会自动重建两个虚拟环境；模型、Pose 图库、生成数据和 LoRA 输出不会删除。

启动前会自动执行环境预检：

- Python 3.10 以下或没有可用 GPU：停止，不下载模型，并显示推荐镜像。
- 驱动 CUDA 13.0+：选择 `cu130` 配置继续。
- 驱动 CUDA 12.8–12.x：提示推荐环境，选择 `cu128` 兼容配置继续。
- 驱动 CUDA 低于 12.8：停止并推荐更换容器。
- AMD ROCm：启动 WebUI，禁用 H3，保留 Wan 视频实验路径。
- 检查通过后才更新仓库、构建环境和下载模型。


## 一键启动

安装器默认使用阿里云 PyPI 镜像：

```text
https://mirrors.aliyun.com/pypi/simple
```

普通 Python 依赖走该镜像；PyTorch CUDA wheel 仍从对应的 PyTorch 官方专用源获取，并把阿里云源作为补充依赖源。DSW 内网环境仍可通过 `PIP_INDEX_URL` 覆盖为内网镜像，`H3_TORCH_INDEX` 可覆盖 PyTorch 源。

在推荐容器的终端中运行：

```bash
bash <(curl -fsSL https://raw.githubusercontent.com/Erro7/pose-model-trainer/main/bootstrap.sh)
```

如果 DSW 访问 GitHub 或 Raw 较慢，可使用 `ghfast.top` 代理版一键安装：

```bash
bash <(curl -fsSL https://ghfast.top/https://raw.githubusercontent.com/Erro7/pose-model-trainer/main/bootstrap.sh)
```

启动器默认使用 `https://ghfast.top/`，本次启动中的主仓库 clone/pull，以及后续 DiffSynth-Studio 等 GitHub 仓库下载都会统一使用代理。代理地址只作用于 GitHub Git/Raw 下载；PyPI 仍使用阿里云内网镜像，ModelScope 模型仍从魔搭下载。

已下载仓库也可以这样通过代理更新并启动：

```bash
cd /mnt/workspace/pose-model-trainer
bash bootstrap.sh
```

默认环境配置：

```text
TRAINER_BASE_IMAGE=modelscope:1.40.1-pytorch2.13.0-gpu-py312-cu130-ubuntu22.04
TRAINER_HOME=/mnt/workspace/pose-model-trainer-data
TRAINER_TOTAL_BUDGET_GB=100
TRAINER_MODEL_BUDGET_GB=80
```

启动器默认关闭 Gradio `share=True`，因此不需要下载 `frpc`。新版魔搭 DSW 使用独立的 `实例号-proxy-7860` 子域名，子域名根目录已经直接映射到 7860，所以 Gradio 不设置额外的 `root_path`。终端保留标准的 `Running on local URL: http://0.0.0.0:7860` 输出；请从魔搭 Notebook 提供的 7860 端口入口打开页面。只有确实需要 Gradio 公网临时链接时才设置 `GRADIO_SHARE=1`。

也可以下载仓库内的 `launch_modelscope.ipynb`，上传到魔搭后依次运行两个单元格。

面向单卡 24GB 显存的 Gradio 启动器。支持：

- 图片：SDXL、FLUX.1（基于 `kohya-ss/sd-scripts`）
- 视频：Wan 2.1 / 2.2（基于 `kohya-ss/musubi-tuner`）
- AMD ROCm：Wan 视频实验路径；H3 Image 与 H3 视频均禁用
- ModelScope 模型 ID 下载、已有本地权重路径
- 按需同步 `anhao/ai-pose-gallery-zh`，筛选姿势并生成图片/描述训练集
- MiniMax-H3-Image 人物 + 服装 + Pose 多参考编辑 LoRA，两阶段缓存与训练
- MiniMax-H3-Image EntroPack 量化训练默认配置：7-bit DiT + 4-bpp 文本编码器 + INT8 VAE，总占用低于 80GB
- H3 一键自动检查缓存、断点下载缺失权重、启动训练并提供 LoRA `.safetensors` 下载
- 生成目标图时人物、服装和 Pose 均从完整图库随机抽样，也可上传任意自定义素材覆盖对应图库来源；筛选只是可选缩小随机池，不填关键词即可直接生成
- 人物策略支持每批固定一人、每条独立随机、多人均分随机和同一人物多图参考；Pose 与服装从完整候选池按 Seed 抽样，不再按 JSON 顺序取前几条
- 人物、Pose 和服装范围支持多个关键词（英文/中文逗号、分号、竖线或换行分隔），任一关键词命中即可入选；留空时全库参与随机
- H3 提示词会区分人物参考是否包含肢体比例：头像/发型参考不会硬复制不存在的身体比例，而是构造自然的亚洲成年女性体型；全身参考以人物自身可见的体型比例为基准，再收敛为自然比例
- 人物参考的脸型、五官、发型和可识别特征必须保留；亚洲女性风格表现为白皙清透肤色、纤细身形、柔和轮廓。Pose 可提供淡妆和上镜感参考，但不得替换人物脸部身份或参与身材比例
- Pose 参考只提供关节位置、肢体摆放和视角，绝不参与人物身材比例；同时不沿用 Pose 图的身份、发型、肤色和服装
- H3 服装与 Pose 只通过参考图传递，不把中文服装/Pose 元数据写进提示词正文；输出强制禁止字幕、标题、标签、水印和排版文字，避免训练图被文案污染
- H3 生成页支持可编辑默认提示词模板和原版 Prompt 切换：默认模板保留人物身份、亚洲女性身材与无文字约束；原版 Prompt 使用旧版写法，便于对比或恢复旧效果，也可继续手动编辑模板占位符
- 生成页默认只露出人物策略、数量和画面尺寸；自定义素材、关键词、Seed 与提示词模板收进可展开区域
- 审核页支持“创建训练集并开始训练”一键衔接，自动使用推荐训练参数；训练页仅用于单独重跑和查看日志
- 多张自定义服装可选“同一套多图参考”或“每张独立随机”，人物支持逐条随机、均分随机和同一人多图参考
- 素材与生成 Seed 默认 `-1`，每批自动换随机种子；填写固定整数即可复现同一组人物、Pose 和服装组合
- H3 成果库持久保存生成批次：刷新或重启后可预览、载入筛选、直接创建训练集，并批量下载 ZIP；历史 LoRA 输出也可一键打包
- 手动数据只强制目标成品图；人物、服装、Pose 参考均可留空，留空时自动生成纯目标图文生图训练集
- 自动生成数据集 TOML、训练命令预览、后台训练和停止任务
- 基于 `loguru` 的训练日志写入/轮转，WebUI 自动刷新、自动滚动、复制和完整日志下载
- 启动器、清理脚本与 WebUI 控制台统一为 `YYYY-MM-DD HH:mm:ss | LEVEL | [模块] 消息`，启动日志写入 `logs/launcher.log`（20MB 轮转、保留 3 份），pip 正常输出降噪，失败时打印尾部诊断
- 任务状态持久化到 `logs/task-state.json`：生成或训练期间禁用所有启动按钮，开放对应暂停/继续和停止按钮；刷新页面或 WebUI 重启后会恢复运行/暂停状态、按钮、任务目录、命令和最近日志
- H3 训练显式传入 `accelerate` 的单进程、单机、`mixed_precision=no` 和 `dynamo_backend=no`，避免默认值警告；进程被信号终止时记录 `SIGTERM/SIGKILL` 等信号名，不再只显示负退出码
- H3 目标图生成和 LoRA 训练共用 `${TRAINER_HOME}/logs/h3-gpu.lock`：同一时间只允许一个 H3 GPU 任务，后续任务会立即提示“已有 H3 GPU 任务正在运行”，避免同卡并发争抢显存
- 低显存默认值：bf16、gradient checkpointing、8-bit AdamW、缓存 latent / 文本编码器输出

## 手动启动

1. 使用 **GPU Notebook**，建议 24GB 显存、32GB 以上内存和足够的数据盘。
2. 上传并解压此项目。
3. 在终端执行：

```bash
cd modelscope-trainer
bash start.sh
```

`start.sh` 会自行检查驱动 CUDA 版本并选择 `cu130` 或 `cu128`；已有 WebUI 依赖满足要求时不会重复联网安装。

4. 打开输出的 Gradio 链接。
5. 进入“首页”点击“统一检查并准备完整环境”。系统、存储、WebUI、H3、CUDA、图库、权重和流程状态会在同一份报告中检查；H3 依赖与模型均在首次任务中按需安装/下载，无需使用独立模型下载页。
6. 如需姿势资源，在“姿势图库”页同步资源，生成数据集后把返回路径填入“训练数据目录”。
7. H3 常规流程简化为 `1 · 生成目标图 → 2 · 审核与训练`：生成页只要求人物策略、数量和画面尺寸，默认全库随机；需要时再展开自定义素材或高级设置。审核页确认图片后点击“创建训练集并开始训练”，会自动重建 metadata 并使用推荐的量化、低显存训练参数启动。`3 · 训练与日志` 只用于单独重跑或查看日志，`4 · 手动数据` 和 `5 · 成果库` 是旁路入口。候选图以 `#0001` 格式显示原任务序号，点选图片后可一键排除或恢复，也可继续手填序号/范围；精修图仍按顺序替换前 N 张。生成批次保存在 `${TRAINER_HOME}/datasets`，刷新页面后可在成果库继续查看、复用或批量下载。

## 清理与强制启动

`start.sh` 支持强制重启：

```bash
bash start.sh --force
```

`--force` 会先停止当前 WebUI、H3 环境安装和其他训练进程，清理残留 PID/锁文件，然后启动新进程。需要同时删除旧版 `modelscope-cache` 时使用：

```bash
bash start.sh --clean
```

`--clean` 会先停止旧进程，把旧版 `${TRAINER_HOME}/modelscope-cache` 中缺失的 H3 文件迁移到 `${TRAINER_HOME}/models`，确认量化权重完整后再删除旧缓存，并清除没有目标图、也没有 `metadata.json` 的空生成批次。含 `targets/`、`metadata.json` 的批次、图库、模型和 `outputs/` 中的 LoRA 结果都不会删除。也可以单独运行清理脚本，默认只做 dry-run：

```bash
bash cleanup.sh
bash cleanup.sh --empty-batches
bash cleanup.sh --yes --empty-batches
bash cleanup.sh --yes --stop-processes --legacy-cache
```

`bootstrap.sh` 会透传相同参数，因此远程一键安装也可以使用 `--force` 或 `--clean`。

## H3 默认模型与参数

H3 没有独立“模型”页。点击“生成目标图”或“开始 LoRA 训练”时，只下载下列必要文件：

| 用途 | ModelScope ID | 选取文件 |
| --- | --- | --- |
| 量化 DiT / VAE | `DiffSynth-Studio/MiniMax-H3-Image-EntroPack` | `models/dit_7bit.safetensors`、对应 JSON、`models/vae_int8.safetensors` |
| 文本编码器 | `DiffSynth-Studio/EntroPackPreQuants` | `models/MiniMax-H3/text_encoder_4bpp.safetensors`、对应 JSON |
| 输出头 / Processor | `MiniMax/MiniMax-H3-Image` | `pdd_fused_heads.safetensors`、`processor/` |

默认模型目录：`${TRAINER_HOME}/models`。H3 预下载与 DiffSynth 训练共用这一目录，避免同一权重保存两份。魔搭一键启动时通常为：

```text
/mnt/workspace/pose-model-trainer-data/models
```

顶层只保留 `${TRAINER_HOME}/models`。ModelScope 内部元数据缓存在 `models/.modelscope-cache`；旧版本使用过的顶层 `${TRAINER_HOME}/modelscope-cache` 会由 `--clean` 一次性迁移并删除。

模型下载完成后会写入 `models/.h3-download-receipts.json`。后续任务只校验本地文件是否完整，命中缓存时不会连接 ModelScope，也不会再显示 “Downloading … files”。需要重新检查远端模型更新时执行：

```bash
.venv-h3/bin/python download_h3.py --profile quantized \
  --models-dir "${TRAINER_HOME:-/mnt/workspace/pose-model-trainer-data}/models" --refresh
```

首轮验证默认值：

| 参数 | 默认值 |
| --- | ---: |
| 批量生成数量 | 4 |
| 目标图尺寸 | 按显存与系统内存动态推荐（24GB 默认 768 × 768） |
| 素材与生成 Seed | -1（每批自动随机；固定整数可复现） |
| 量化配置 | 7-bit DiT + 4-bpp Text Encoder + INT8 VAE |
| LoRA Rank | 16 |
| 学习率 | 1e-4 |
| 数据重复 | 5 |
| Epoch | 1 |
| 最大训练像素 | 589,824 |
| 低显存 CPU Offload | 开启 |

24GB GPU 上 `1024 × 1024` 多参考生成可能 OOM；默认已降到 `768 × 768` 并预留约 2.5GB 显存。仍 OOM 时把生成尺寸降到 `512 × 512`、训练最大像素降到 `262144`。确认生成与训练运行正常后，可逐步提高生成数量、数据重复、Epoch 和 Rank。不要在“模型”页手动下载完整 H3 仓库，否则可能额外占用大量磁盘。

H3 生成页会根据 `nvidia-smi` 显存和系统运行内存动态列出常用安全尺寸，并按常用比例优先显示：`1:1 / 2:3 / 3:2 / 3:4 / 4:3 / 9:16 / 16:9`。当前 24GB A10 可选 `512×512`、`512×768`、`768×512`、`576×768`、`768×576`、`576×1024`、`1024×576` 和 `768×768`，其中 `768×768` 为推荐项。自定义宽高必须为 16 的倍数且不能超过安全预算，避免误选高分辨率再次 OOM。“18+ 安全过滤”位于可选的“图库筛选”区域并默认开启，只过滤图库候选，不影响已上传的自定义素材。

一键启动的默认持久目录是 `/mnt/workspace/pose-model-trainer-data`，可通过环境变量修改：

```bash
export TRAINER_HOME=/mnt/workspace/trainer-data
bash start.sh
```

## 数据集格式

图片数据目录：图片与同名 `.txt` 描述文件放在一起。

```text
dataset/images/001.png
dataset/images/001.txt
```

视频数据目录：视频与同名 `.txt` 描述文件放在一起。

```text
dataset/videos/001.mp4
dataset/videos/001.txt
```

视频会按设定的分辨率与帧数缓存。24GB 显存建议先从 `512×288 / 49 帧 / batch 1` 开始；Wan 14B 仍可能受系统内存和具体模型影响。AMD ROCm 下请先使用 Wan 2.1 T2V 1.3B，并保持小分辨率、少帧数；当前实验路径使用 bf16、AdamW 和 `--disable_cudnn_backend`，不启用 FP8。

FLUX 需要分别指定主模型、CLIP-L、T5XXL、AE；Wan 需要分别指定 DiT、VAE、UMT5。页面会校验这些路径。

## 注意

- 模型许可证和数据授权由使用者确认。
- `ai-pose-gallery-zh` 的 MIT 许可仅覆盖代码，不覆盖图库图片与原始数据；启动器不会把第三方资源打包进本仓库。
- 默认量化配置只自动下载训练所需的几个文件，使用 DiffSynth 官方 EntroPack LoRA 配方，控制在 80GB 以内。模型不会打包进 GitHub 仓库；首次训练下载到 `${TRAINER_HOME}/models`，支持断点和缓存复用。原始 BF16 模式约 130GB，作为高级选项保留。
- 24GB GPU 建议保持“低显存 CPU Offload”开启；训练速度取决于系统内存、磁盘和 PCIe 传输速度。
- 三合一任务属于监督式编辑 LoRA：目标成品图是期望输出，人物/服装/Pose 是参考输入；仅有参考图无法训练出正确映射。
- 第一次安装与下载可能耗时较久；Notebook 重启后，只要数据盘保留，无需重新下载。
- “停止”会终止当前训练进程；断点续训需在额外参数中加入训练后端对应的 resume 参数。
- 启动器会显示最终命令，遇到版本差异时可复制到终端调整。

