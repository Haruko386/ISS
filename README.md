# StitchDiff：几何引导的扩散图像拼接

这是一个可以直接运行的“两阶段”图像拼接原型：

1. SIFT/ORB + RANSAC 单应性把两张输入图像 warp 到同一个全景画布；
2. 条件 latent diffusion 消除接缝、曝光差异和小范围重影，同时尽量保持单图覆盖区域不变。

它参考了 [Haruko386/ApDepth](https://github.com/Haruko386/ApDepth) 中“扩展 Stable Diffusion UNet 输入层并拼接多路 latent 条件”的思路，但针对拼接任务使用正确的扩散训练目标：

```text
14 通道默认模型：
[noisy_target(4), left_canvas(4), right_canvas(4), left_mask(1), right_mask(1)]

12 通道消融模型：
[noisy_target(4), left_canvas(4), right_canvas(4)]
```

扩展预训练 `conv_in` 时只复制前四个 noisy-target 权重，新增条件通道为零初始化。训练时冻结 VAE、使用预训练 CLIP 的空文本 embedding，只更新 UNet。

## 现在能做什么

- 对未对齐的图片执行特征匹配、RANSAC、单应性 warp 和 feather blend；
- 保存左右独立画布、有效区域 mask、接缝 mask 和粗拼接图；
- 从完整全景图自动合成带曝光、色彩与少量残余偏移的训练三元组；
- 使用标准随机时间步噪声预测训练 12/14 通道扩散网络；
- 同时优化 diffusion、重建、接缝、梯度和内容保持损失；
- 定期验证 MAE、接缝 MAE、重叠区 MAE、PSNR，并保存 `best` checkpoint；
- 保存 optimizer、GradScaler 和随机数状态，可从 checkpoint 继续训练；
- 以 tiny 后端离线跑通 CPU smoke test；
- 切换 Stable Diffusion 1.5/2.1 VAE 和 UNet 做正式微调；
- 推理后把单一来源区域原样贴回，只让生成模型主要处理重叠区。

`tiny` 是验证数据流和研究假设的轻量真实扩散模型，不使用预训练生成先验，因此少量训练不会产出高质量全景图。视觉质量实验应使用 `stable-diffusion` 后端和足够的数据。

## 安装

建议使用 Python 3.10～3.12 的新环境：

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

如果要微调 Stable Diffusion：

```bash
pip install -r requirements-sd.txt
```

依赖将 Diffusers 限制在已验证的 `<0.36` 范围；实测 0.39 的新版 attention dispatch 无法被 Torch 2.4 的自定义算子解析器加载。项目已用 Diffusers 0.35.2 做真实 14 通道前向和反向验证。

检查环境：

```bash
python main.py doctor
```

## 一条命令跑通

仓库自带 `res/1.jpg` 和 `res/2.jpg`。下面的命令会依次完成对齐、生成小数据集、训练 tiny 模型并推理：

```bash
python main.py demo
```

CPU 默认只训练 4 步，用于确认整个工程可运行。主要输出：

```text
outputs/demo/
├── alignment/             # 几何阶段的双画布、mask、粗拼接
├── data/                  # 自动生成的训练三元组
├── training/final/        # tiny checkpoint
├── stitched.png           # 扩散结果
└── stitched_coarse.png    # 传统几何基线
```

## 分步使用

### 1. 几何对齐

```bash
python main.py align \
  --left res/1.jpg \
  --right res/2.jpg \
  --output outputs/alignment
```

若可靠匹配不足，程序会明确报错，不会静默写出错误的巨大画布。对于重复纹理、重叠过少或大视差场景，应先换用更强的几何模块/光流，再把结果整理成同样的左右画布与 mask。

### 2. 从完整全景图构造数据

```bash
python main.py prepare \
  --panoramas path/to/panorama_folder \
  --output data/panoramas \
  --width 1024 \
  --height 512 \
  --samples-per-image 16
```

生成的 `manifest.jsonl` 每行包括：

```json
{
  "left": "left/000000.png",
  "right": "right/000000.png",
  "target": "target/000000.png",
  "left_mask": "left_mask/000000.png",
  "right_mask": "right_mask/000000.png",
  "seam_mask": "seam_mask/000000.png",
  "split": "train"
}
```

真实数据也可以直接按这个格式组织。所有图片必须对应同一个目标画布；原始未对齐双图应先做几何预处理。

### 3. 训练离线 tiny 后端

```bash
python main.py train \
  --config configs/tiny.yaml \
  --data data/panoramas \
  --output outputs/tiny \
  --steps 1000
```

训练日志会逐行写入 `metrics.jsonl`；验证预览按“左图、右图、粗拼接、预测、GT”的顺序保存到 `validation/`。`best/` 以最低接缝 MAE 为准，`final/` 是最后一步。

从中断处恢复，并把总训练步数提高到 2000：

```bash
python main.py train \
  --resume outputs/tiny/checkpoint-001000 \
  --steps 2000
```

恢复会读取 checkpoint 附近的配置，也可以显式传入 `--config`。`--steps` 表示目标总 optimizer step，而不是在原 checkpoint 之后额外运行的步数。梯度累积现在也按 optimizer step 计数。

运行 12 通道消融实验：

```bash
python main.py train --config configs/tiny.yaml --data data/panoramas --no-masks
```

### 4. 正式微调 Stable Diffusion

编辑 `configs/sd15.yaml` 中的模型名、分辨率和数据路径，然后运行：

```bash
python main.py train --config configs/sd15.yaml
```

在下载完整 SD 1.5 前，可以先用 safetensors 格式的 17.9 MB 测试模型验证本机 Diffusers 路径：

```bash
python main.py prepare \
  --panoramas res/output.jpg \
  --output data/sd-tiny \
  --width 64 --height 64 \
  --samples-per-image 2 \
  --validation-fraction 0.5

python main.py train --config configs/sd-tiny.yaml
```

该配置会真实加载 VAE、CLIP 和 `UNet2DConditionModel` 并执行反向传播，但测试权重不具备生成质量，仅用于依赖和结构 smoke test。

推荐先用 256×512 确认显存和收敛，再提高到 512×1024。默认配置面向 SD 1.5/2.1 的 `AutoencoderKL + UNet2DConditionModel`，没有实现 SDXL 的额外文本/尺寸条件。下载受限时，可把 `pretrained_model` 改成本地 Diffusers 模型目录。

`configs/sd15.yaml` 默认启用 gradient checkpointing 和 channels-last。xFormers 默认关闭；确认本机 PyTorch/CUDA 与 xFormers 版本兼容后，可设置 `enable_xformers: true`。这些选项遵循 Diffusers 的 [模型 API](https://huggingface.co/docs/diffusers/api/models/overview) 和 [内存优化说明](https://huggingface.co/docs/diffusers/optimization/memory)。

### 5. 推理

```bash
python main.py infer \
  --checkpoint outputs/tiny/final \
  --left res/1.jpg \
  --right res/2.jpg \
  --output outputs/result.png
```

默认启用 `--preserve-known`：保留粗拼接的非接缝区域，只在接缝带软融合扩散结果。若要观察纯生成结果，可传 `--no-preserve-known`。

推理默认还会把粗拼接作为 img2img 初值（`strength` 见配置）：`0` 完全保留粗拼接，`1` 从纯随机噪声生成。未充分训练的 tiny smoke test 建议使用 0.05～0.1；充分训练的 Stable Diffusion 模型可逐步提高，或使用 `--strength 1.0` 验证纯条件生成能力。

### 6. 评估 checkpoint

```bash
python main.py evaluate \
  --checkpoint outputs/tiny/best \
  --data data/panoramas \
  --split val \
  --batches 8 \
  --output outputs/tiny/evaluation.json
```

命令会写出 JSON 指标和同名 PNG 对比图。PSNR 和 MAE 使用 `[0,1]` 像素尺度；模型选择重点看 `seam_mae`，同时观察 `overlap_mae`，避免只修接缝却破坏整个重叠区域。

## 损失函数

```text
L = L_diff
  + λ_rec      L_reconstruction
  + λ_seam     L_seam
  + λ_grad     L_gradient
  + λ_preserve L_content-preservation
```

- `L_diff`：标准噪声预测 MSE；
- `L_reconstruction`：预测 `x0` 解码后与 GT 的 masked L1；
- `L_seam`：只在接缝带计算的 L1；
- `L_gradient`：水平/垂直梯度连续性；
- `L_preserve`：只在单图覆盖区约束输出接近输入。

权重位于 YAML 配置的 `loss` 段。高分辨率 SD 训练若显存不足，可先把像素损失权重设为 0，仅训练 diffusion loss，再在第二阶段打开区域损失。

## 测试

```bash
pip install -r requirements-dev.txt
pytest
```

测试覆盖已知单应性画布、合成数据、14 通道训练/采样、12 通道消融、预训练 `conv_in` 的零初始化规则、SD 组件接线、显存优化开关，以及训练恢复/验证/best checkpoint。

## 已知边界

- 单应性适用于相机中心接近、场景近似平面或远景为主的情况；大视差需要光流或网格 warp。
- 扩散模型可能生成视觉自然但不真实的结构，文字、建筑直线和移动物体尤其需要内容保持与后处理。
- 当前像素精修采用“单来源区域硬保持”，尚未加入可训练的 Restormer/残差 refinement 网络。
- 合成裁剪数据不能代替真实多视角训练对；正式实验应混入真实相机数据。
