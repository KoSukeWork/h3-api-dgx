# H3 API for DGX Spark GB10

独立的 MiniMax H3 **Docker Compose API 服务**，不是 ComfyUI 推理部署。
ComfyUI 和其他开发工具均通过 HTTP 调用；模型在 DGX 的 ARM64/GB10 上运行。

这是个人维护的社区部署方案，非 NVIDIA 或 MiniMax 官方项目，不提供官方支持承诺。
面向单机、可信内网，不是开箱即用的多租户公网服务。参见 [安全说明](SECURITY.md)。

默认方案：BF16 主模型 + Larry Turbo v4 LoRA，非量化、非裁剪 Transformer。
视频 VAE 沿用已验证的 FP16 文件、音频 VAE 为 FP32；不宣称完全等价于官方全套发布流程。

## 从哪里开始

| 场景 | 文档 |
| --- | --- |
| 当前 DGX 已有成功运行的 BF16/Turbo 镜像和模型 | [正式 API 部署](production/README.md) |
| 新 DGX，没有之前的镜像或安装目录 | [从零部署](docs/FRESH_INSTALL.md) |
| 换机、备份、回退，希望尽量复现已跑通环境 | [备份与恢复](docs/BACKUP_RESTORE.md) |
| 想核对参数、测速及兼容补丁 | [部署记录](docs/DEPLOYMENT_RECORD.md) |
| 在另一台 ComfyUI 上远程调用，无需本地 H3 权重 | [远程节点安装与配置](comfyui_client/README.md) |

默认 8 次推理，支持按请求切换，不重新加载模型：

```json
{"task":"t2va","prompt":"A red toy car drives across a white table.","preset":"fast"}
```

`preset: fast` 为 4 次、`preset: quality` 为 8 次；也可换成 `steps: 12`。
两者二选一。自定义范围 1～49；增加步数不保证画质更好，4/8 之外需自行验证。
外层 `steps` 是实际推理次数，内部自动转换为 SGLang 的 `num_inference_steps=steps+1`。
当前 Turbo 配方支持 t2va/fl2va，不接受 ref2va；实测记录来自 t2va。

## 仓库内容

- `h3_api/`：鉴权、文件上传、串行任务队列、SGLang 进程管理及采样参数。
- `production/`：正式服务镜像、Compose、准备/只读盘点脚本。
- `Dockerfile`、`bootstrap/`：从 CUDA 基础镜像重建运行环境和准备模型。
- `examples/client.py`、`comfyui_client/`：Python CLI 与 ComfyUI 远程客户端。
- `scripts/`：证书导入、环境检查、QKV 补丁、下载、输出检查和打包。
- `benchmark/`：这次独立横评的工具和 LoRA 兼容补丁。量化对照仅留档，不是默认部署。
- `tests/`：无需 torch/CUDA 的 API/编排/补丁回归测试。
- `docs/`：重建、备份说明，以及实际依赖版本和 Turbo 测速记录。

不包含模型文件、视频、Docker 镜像、API key、HF token 或公司 CA；这些不得提交 Git。
`certs/README.md` 说明如何单独放入 IT 提供的可信公共 CA。

## 可复用性与验证边界

SGLang commit、CUDA 基础镜像摘要、模型 revision 和 SHA256 固定；网关依赖由 `uv.lock` 固定。
从零构建额外使用本次成功运行的包版本约束 `inference-constraints.txt`，避免任意升级。
这不是包含所有 wheel/系统包的离线、逐字节可复现构建；第三方资源未来也可能下架。
**换机最稳妥的是备份并恢复验收后的正式 Docker 镜像 + 模型**。

已有 GB10 测速：768p、约 5 秒视频，Turbo 8 次 440.46 秒，4 次 248.10 秒。
正式 API 与新从零构建配方仍须在 DGX 上验收；本机只执行 CPU 测试/配置校验，不做 x86 GPU 测试。

## 开发与打包

```bash
uv sync --extra test --frozen
uv run pytest -q
uv run ruff check h3_api scripts production bootstrap tests examples comfyui_client
```

Windows 打包：

```powershell
powershell -ExecutionPolicy Bypass -File scripts/package.ps1
```

输出 `dist/h3-api-dgx.tar.gz`，只包含代码/文档；不会把本地证书、密钥、权重或 Git 历史打进去。
打包须从 Git 工作目录执行，文件清单遵循 Git 忽略规则和打包白名单；内容仍需人工审查。
对外发布前请按 [公开发布检查清单](docs/PUBLIC_RELEASE.md) 审查暂存区、历史和发布附件。

仅打包 ComfyUI 客户端：`python scripts/package_comfyui.py`，生成 `dist/h3-comfyui-client.zip`。
本地 `comfyui_client/config.json` 保存地址/密钥，不提交、不打包；只分发 `config.example.json`。

软件与上游来源说明见 [NOTICE.md](NOTICE.md)；模型、CUDA 和第三方依赖受各自许可证约束。
