# Pose Model Trainer

魔搭 Notebook 的图片 / 视频 LoRA 批量训练工作台。

- [GitHub 仓库](https://github.com/Erro7/pose-model-trainer)
- [魔搭 Notebook](https://modelscope.cn/my/mynotebook)
- [第三方 Pose 图库](https://github.com/anhao/ai-pose-gallery-zh)

## 环境

推荐镜像：

```text
modelscope:1.40.1-pytorch2.13.0-gpu-py312-cu130-ubuntu22.04
```

| 工作流 | NVIDIA CUDA 12.8+ | AMD ROCm |
|---|---|---|
| H3-Image + EntroPack | 支持 | 不支持 |
| Wan 2.1 / 2.2 视频 LoRA | 支持 | 实验支持 |

H3 依赖 CUDA 版 EntroPack。AMD ROCm 可启动 WebUI 和 Wan 视频实验路径，但 H3 生成、H3 环境准备与 H3 训练会禁用。

WebUI 与训练依赖分别使用 `.venv`、`.venv-h3`，ComfyUI 使用 `.venv-comfy`；安装训练依赖不会影响正在运行的 WebUI。

## 启动

```bash
bash <(curl -fsSL https://raw.githubusercontent.com/Erro7/pose-model-trainer/main/bootstrap.sh)
```

GitHub 较慢时：

```bash
bash <(curl -fsSL https://ghfast.top/https://raw.githubusercontent.com/Erro7/pose-model-trainer/main/bootstrap.sh)
```

本地仓库：

```bash
cd /mnt/workspace/pose-model-trainer
bash start.sh
```

默认目录：

```text
TRAINER_HOME=/mnt/workspace/pose-model-trainer-data
模型：${TRAINER_HOME}/models
生成数据：${TRAINER_HOME}/datasets
LoRA 输出：${TRAINER_HOME}/outputs
日志：${TRAINER_HOME}/logs
```

需要强制重启时：

```bash
bash start.sh --force
```

## WebUI

顶部入口：`图片 / 视频 / 结果 / 环境`。

图片页二级入口：

- `生成`：模型、批量规模、Seed、尺寸、人物风格、身材、Prompt 和素材筛选
- `结果`：自动刷新画廊、点选审核、排除/恢复、创建训练集
- `LoRA 训练`：H3 LoRA 训练、训练日志、LoRA 输出和手动数据

结果画廊每 4 秒检查一次更新，切入结果 Tab 时立即刷新。图片批量生成和 H3 LoRA 训练共用同一套暂停/继续控制。

环境页可检查容器、GPU、CUDA/ROCm、PyTorch、DiffSynth、EntroPack、ComfyUI 和模型兼容性，并配置存储上限。

## H3 图片

默认使用 MiniMax-H3-Image EntroPack 量化训练：7-bit DiT、4-bpp 文本编码器和 INT8 VAE。模型按需下载到 `${TRAINER_HOME}/models`，支持断点与缓存复用。

工作流为“人物 + 服装 + Pose 多参考编辑 LoRA”。批量生成目标图后，在结果页审核建集，再启动训练。常规配置保持默认即可。

## Pose 图库

第三方仓库：[anhao/ai-pose-gallery-zh](https://github.com/anhao/ai-pose-gallery-zh)

标准位置：

```text
${TRAINER_HOME}/repos/ai-pose-gallery-zh
${TRAINER_HOME}/datasets/pose_gallery  # 兼容软链接
```

首次缺少图库时 WebUI 会提示自动同步，也可在图片生成页的“其余参数”中手动更新。仅需迁移到其他磁盘时使用：

```bash
export POSE_GALLERY_DIR=/path/to/ai-pose-gallery-zh
```

`ai-pose-gallery-zh` 的 MIT 许可仅覆盖代码，不覆盖图库图片和原始数据；本项目不会把第三方图库打包进仓库。

## 数据格式

图片：图片与同名 `.txt` 描述放在同一目录。

```text
dataset/images/001.png
dataset/images/001.txt
```

视频：视频与同名 `.txt` 描述放在同一目录。

```text
dataset/videos/001.mp4
dataset/videos/001.txt
```

视频建议从 `512×288 / 49 帧 / batch 1` 开始。Wan 14B 仍可能受显存、内存和具体模型影响。

## 清理

```bash
bash cleanup.sh --yes --stop-processes
```

`--clean` 会额外迁移旧 H3 缓存并清理空生成批次；模型、图库、训练数据和 LoRA 输出不会删除。

## 注意

- 模型许可证和数据授权由使用者确认。
- H3 原始 BF16 模式约需 130GB 模型空间；24GB GPU 建议保持低显存 CPU Offload。
- 停止会终止当前任务；暂停后可在同一容器内点击“继续”恢复。
- 容器或进程退出后无法从原进程继续，需使用训练后端支持的 resume 参数重新启动。
