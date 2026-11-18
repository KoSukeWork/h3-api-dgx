# LightX2V Ref2VA：本地接入前检查（2026-10-21）

## 结论与边界

**可以进入 DGX 实机验证阶段，但不是替换 LoRA 文件即可上线。** 本文记录最初的静态检查快照：
该阶段没有修改现有 API、连接 DGX、安装 torch/CUDA、下载完整权重或运行推理。
后续的独立实机测试工具见 [Ref2VA 测试说明](../benchmark/ref2va/README.md)。
后续单次实机生成的结果与边界见 [DGX 测试记录](REF2VA_DGX_TEST.md)。
后续正式服务升级与串行切换已完成，见 [正式验收记录](PROFILES_DGX_ACCEPTANCE.md)；下文保留最初静态检查时的状态。
本报告的静态源码/权重头检查本身不代表 ARM64 安装、GPU 加载、速度或画质已经通过。

现有 Larry FL2VA profile 保持原样。参考图需另一个 Ref2VA profile，并加载匹配的 Ref2VA 主模型。

## 固定检查对象

- 本仓库基线：`6283198716960b1ec9f67d43eb2b185712fd704e`。
- SGLang：`70b5b03e78612c94f86ac98eb4d2d8d19ceda738`，保持已验证的 QKV 与 LoRA 补丁。
- [LightX2V 推理源码](https://github.com/ModelTC/Minimax-H3-Turbo/tree/02e26d591f7a04d5d1a074c9566d5dd4f22f6225)：
  `02e26d591f7a04d5d1a074c9566d5dd4f22f6225`。
- [模型文件仓库](https://huggingface.co/lightx2v/Minimax-h3-Turbo/tree/30f485ed874aa44c8183348ad361c5eabad318da)：
  `30f485ed874aa44c8183348ad361c5eabad318da`。

首轮候选是原始 PEFT/Diffusers 格式的
`minimax_h3_ref2v_turbo_4step_v0.1_bf16.safetensors`：

- 大小：1,383,677,768 bytes（约 1.38 GB）。
- HF LFS 声明 SHA256：`9e642fc8749c74f8da5e2382877ab5c7aa37b9a73b7fd0d6d457bd1b3cb1ae99`。
- 未读取全部文件，**尚未计算并核验整文件 SHA256**；下载到 DGX 后必须核验。
- HF 文件列表还存在 Ref2VA 768p v1.0/v1.1 等变体；本次固定 README 的明确 Ref2VA 配方为 v0.1。
  不把其他文件的版本号当成可直接套用相同 alpha/shift/步数的依据，也不宣称 v0.1 是最新或画质最佳版本。

## 已完成的静态检查

| 检查 | 结果 |
| --- | --- |
| 原始 LoRA 头信息 | 624 个 BF16 张量，312 对 A/B，rank=128，metadata alpha=8 |
| 目标模块匹配 | 50 个主 Transformer block + 2 个 token-refiner block；映射到 208 个原生线性模块 |
| QKV | 52 组 split Q/K/V 的 A/B 齐全；SGLang 有融合映射路径 |
| FFN | SGLang 有 PEFT `ff.net.0.proj` 到原生 `mlp.fc1` 的 SwiGLU B 行交换逻辑 |
| 现有补丁 | 固定源码能依次通过现有 QKV 和 LoRA capability-guard 补丁校验 |
| 读取范围 | 原始头 81,736 bytes + Comfy 转换版头 73,640 bytes；合计 155,376 bytes，不含权重主体 |

映射检查解析上游配置的 AST 字面量，不导入或执行上游 torch 模块。
它核对名字、shape、A/B 配对和 QKV 分组，**不证明数值变换或 CUDA 执行正确**。

### 为什么优先原始 PEFT 文件

Comfy 转换版约 1.96 GB，QKV 融合后 rank=384，含单独 alpha 标量；其 metadata 的 `base_model`
还写有 FL2VA 名称，与 Ref2VA 文件名存在不一致。不能仅凭这个字段判断训练底模，需向发布方核实。
当前 SGLang 已有原始 PEFT 的名称规范化、alpha 读取和 QKV/FFN 变换路径，因此优先验证约 1.38 GB 原始文件，
避免再叠加一次转换和歧义。不能因当前主模型是 Comfy 导出就直接认定必须用 Comfy 版 LoRA。

## 不能省略的接入工作

1. **alpha/rank**：v0.1 有效增量系数为 `1.0 × 8 / 128 = 0.0625`。
   当前 SGLang 能读取该 metadata；新 profile 应显式锁定 alpha=8，并核对加载日志。
   不要把运行时强度也设成 0.0625，避免重复缩放；也不能漏掉 alpha 而使用 rank 作为默认值。
2. **采样**：4 次 NFE，内部 `num_inference_steps=5`，video/audio shift=12/3，guidance=1。
   LightX2V v0.1 训练尺度为 544p。当前网关仅开放 short_edge 480/768，严格配方对照需独立测试配置；
   用 480p 做接口冒烟可以，但不应称为训练尺度复现。不得对所有 profile 一律沿用 Larry 的可调 1～49 次。
3. **参考图缩放**：LightX2V 建议 `match`，按目标面积缩小且不主动放大小图，尺寸对齐 32。
   固定 SGLang `reference_encoding.py` 默认将参考图短边放大到 2048，且 admission 阶段已计算布局。
   仅在客户端预缩小会被后端再次放大；正确接入须同步 admission/resolved plan 与编码阶段，
   并保留原始 Ref2VA 行为，避免影响其他 profile。
4. **网关保护**：当前 `schemas.py`/`backend.py` 明确禁止 Turbo+Ref2VA，`turbo.py` 固定 Larry SHA。
   这些限制仍在，不能删除保护后直接复用 Larry。应增加白名单 profile，绑定 task、主模型、LoRA、
   alpha、shift、步数、缩放策略和预期层数；禁止客户端传任意服务器文件路径。
5. **加载验收**：以 208 个原生模块作为静态预期，实机检查全部应匹配的模块，不只检查“应用层数 > 0”。
   保留主模型 Comfy BF16 QKV 布局校验；不要对原始 LoRA 再执行主模型行重排。

## 多套 profile 与单 GPU

- 保留现有 Larry/T2VA/FL2VA 默认行为，老请求不加 profile 时仍保持兼容。
- 共享一个持久队列和一个模型 worker。profile/主模型改变时，先结束旧推理进程组再加载下一套。
- 初版采用明确的冷切换，暂不承诺热插拔；验证后才优化同底模 LoRA 的动态切换。
- 后端缓存标识需包含 variant、profile、adapter revision/hash、alpha、采样与预处理策略；
  不能仍只用当前 `variant` 判断是否复用。
- 所有权重可以共存磁盘，不能同时启动多个 benchmark/API 各自占用 GB10。
- 首轮不加入 PDD/PTD：它们有专用加载/采样路径，不能当普通 LoRA 塞入这个 profile。

## DGX 到手后先做什么

1. 只读检查 aarch64、GB10/sm_121、驱动、容器 GPU、Compose 实际配置与当前任务队列。
   SSH 使用受限账户/密钥优先；不要将凭据写进代码、报告、命令参数或公开仓库。
2. 核对运行镜像中的 SGLang revision、两项补丁、torch/CUDA；不得从历史记录假定完全一致。
3. 检查 prepared manifest 和 Ref2VA BF16 主权重。准备脚本原来已包括该权重，可能可复用，
   但此前的 178 GB 目录统计不能证明当前文件仍完整。若已存在并通过校验，不重下约 66.28 GB。
4. 核对磁盘空间、API 空闲状态后，再下载固定的约 1.38 GB LoRA 并验证 SHA256。
5. 独立派生镜像和数据目录验证，保留原镜像/模型/结果，不改驱动、不重装宿主依赖、不做全局 prune。
   试跑占用 GPU 前先协调现有服务停用推理；只读盘点不需要停服务。
6. 先小规模加载检查，再跑单参考图样例。记录冷启动、生成耗时、峰值统一内存、MP4 音视频流、
   人物一致性、双手操作、箱盖动作、回 idle、脚滑与相机稳定性。
7. 通过后再接入正式 API profile，并验证 FL2VA→Ref2VA→FL2VA 的顺序切换与失败回收。

私人图片和原始提示词仅存放本地被忽略的 `dist/ref2va-audit/private/`，不进入发布包。
图像是人物灰模加方块示意，不能把缺少纹理/铰链结构导致的模型补全当成 LoRA 加载失败。
先保持原始提示词作对照；另设 `<Picture 1>` 明确绑定参考图的版本，不静默替换原提示词。
五秒内完成全部动作比较紧，若遗漏动作，再单独比较八秒；每次只改一个变量。

## 重复检查

```bash
# 只获取固定 LoRA 的 safetensors 头，不下载主体，不导入 torch。
uv run python -m scripts.audit_ref2va_adapter
uv run pytest -q
uv run ruff check h3_api scripts production bootstrap tests examples comfyui_client
node --experimental-vm-modules --test tests/comfyui_preview.test.mjs
```

报告中的源码检查对应上述固定 revision，不代表任意新版本已兼容。当前仍须完成受控接入补丁与 DGX 实测，
不能仅凭静态检查把 Ref2VA 开放到正式服务。
