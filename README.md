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

启动器使用两个完全隔离的持久环境：`.venv` 只运行 Gradio WebUI，`.venv-h3` 专门安装 PyTorch、DiffSynth、EntroPack 和训练依赖。安装或修复 H3 环境不会再修改正在运行的 WebUI。更换容器导致 Python 版本变化时会自动重建两个虚拟环境；模型、Pose 图库、生成数据和 LoRA 输出不会删除。

启动前会自动执行环境预检：

- Python 3.10 以下或没有 NVIDIA GPU：停止，不下载模型，并显示推荐镜像。
- 驱动 CUDA 13.0+：选择 `cu130` 配置继续。
- 驱动 CUDA 12.8–12.x：提示推荐环境，选择 `cu128` 兼容配置继续。
- 驱动 CUDA 低于 12.8：停止并推荐更换容器。
- 检查通过后才更新仓库、构建环境和下载模型。


## 一键启动

安装器默认使用魔搭 DSW 的阿里云内网 PyPI 镜像：

```text
https://mirrors.cloud.aliyuncs.com/pypi/simple
```

普通 Python 依赖走内网镜像；PyTorch CUDA wheel 仍从对应的 PyTorch 官方专用源获取，并把阿里内网源作为补充依赖源。可通过 `PIP_INDEX_URL` 和 `H3_TORCH_INDEX` 覆盖。

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
- ModelScope 模型 ID 下载、已有本地权重路径
- 按需同步 `anhao/ai-pose-gallery-zh`，筛选姿势并生成图片/描述训练集
- MiniMax-H3-Image 人物 + 服装 + Pose 多参考编辑 LoRA，两阶段缓存与训练
- MiniMax-H3-Image EntroPack 量化训练默认配置：7-bit DiT + 4-bpp 文本编码器 + INT8 VAE，总占用低于 80GB
- H3 一键自动检查缓存、断点下载缺失权重、启动训练并提供 LoRA `.safetensors` 下载
- 首轮只需上传人物：自动从三合一仓库配对服装图与 Pose 图，读取服装名称/英文关键词及姿势体系/分类/名称/角度，构造多参考提示词并生成候选目标图
- 自动生成数据集 TOML、训练命令预览、后台训练、实时日志和停止任务
- 低显存默认值：bf16、gradient checkpointing、8-bit AdamW、缓存 latent / 文本编码器输出

## 手动启动

1. 使用 **GPU Notebook**，建议 24GB 显存、32GB 以上内存和足够的数据盘。
2. 上传并解压此项目。
3. 在终端执行：

```bash
cd modelscope-trainer
bash start.sh
```

4. 打开输出的 Gradio 链接。
5. 进入“首页”点击“检查完整环境与流程”。H3 依赖与模型均在首次任务中按需安装/下载，无需使用独立模型下载页。
6. 如需姿势资源，在“姿势图库”页同步资源，生成数据集后把返回路径填入“训练数据目录”。
7. H3 使用分步工作流：`1 · 生成目标图 → 2 · 筛选整理 → 3 · 训练 LoRA`。人物、服装和 Pose 默认都可从三合一图库选择；自定义上传放在高级区域。首次任务自动准备环境和权重，之后复用持久缓存。

## H3 默认模型与参数

H3 没有独立“模型”页。点击“生成目标图”或“开始 LoRA 训练”时，只下载下列必要文件：

| 用途 | ModelScope ID | 选取文件 |
| --- | --- | --- |
| 量化 DiT / VAE | `DiffSynth-Studio/MiniMax-H3-Image-EntroPack` | `models/dit_7bit.safetensors`、对应 JSON、`models/vae_int8.safetensors` |
| 文本编码器 | `DiffSynth-Studio/EntroPackPreQuants` | `models/MiniMax-H3/text_encoder_4bpp.safetensors`、对应 JSON |
| 输出头 / Processor | `MiniMax/MiniMax-H3-Image` | `pdd_fused_heads.safetensors`、`processor/` |

默认缓存目录：`${TRAINER_HOME}/modelscope-cache`。魔搭一键启动时通常为：

```text
/mnt/workspace/pose-model-trainer-data/modelscope-cache
```

首轮验证默认值：

| 参数 | 默认值 |
| --- | ---: |
| 批量生成数量 | 4 |
| 目标图尺寸 | 1024 × 1024 |
| Seed | 42 |
| 量化配置 | 7-bit DiT + 4-bpp Text Encoder + INT8 VAE |
| LoRA Rank | 16 |
| 学习率 | 1e-4 |
| 数据重复 | 5 |
| Epoch | 1 |
| 最大训练像素 | 1,048,576 |
| 低显存 CPU Offload | 开启 |

确认生成与训练运行正常后，可逐步提高生成数量、数据重复、Epoch 和 Rank。不要在“模型”页手动下载完整 H3 仓库，否则可能额外占用大量磁盘。

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

视频会按设定的分辨率与帧数缓存。24GB 显存建议先从 `512×288 / 49 帧 / batch 1` 开始；Wan 14B 仍可能受系统内存和具体模型影响。

FLUX 需要分别指定主模型、CLIP-L、T5XXL、AE；Wan 需要分别指定 DiT、VAE、UMT5。页面会校验这些路径。

## 注意

- 模型许可证和数据授权由使用者确认。
- `ai-pose-gallery-zh` 的 MIT 许可仅覆盖代码，不覆盖图库图片与原始数据；启动器不会把第三方资源打包进本仓库。
- 默认量化配置只自动下载训练所需的几个文件，使用 DiffSynth 官方 EntroPack LoRA 配方，控制在 80GB 以内。模型不会打包进 GitHub 仓库；首次训练下载到 `${TRAINER_HOME}/modelscope-cache`，支持断点和缓存复用。原始 BF16 模式约 130GB，作为高级选项保留。
- 24GB GPU 建议保持“低显存 CPU Offload”开启；训练速度取决于系统内存、磁盘和 PCIe 传输速度。
- 三合一任务属于监督式编辑 LoRA：目标成品图是期望输出，人物/服装/Pose 是参考输入；仅有参考图无法训练出正确映射。
- 第一次安装与下载可能耗时较久；Notebook 重启后，只要数据盘保留，无需重新下载。
- “停止”会终止当前训练进程；断点续训需在额外参数中加入训练后端对应的 resume 参数。
- 启动器会显示最终命令，遇到版本差异时可复制到终端调整。

