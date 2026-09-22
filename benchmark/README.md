# H3 / GB10 独立横评

只在 DGX Spark ARM64 上运行。复用已经修好 QKV 布局、能正常出视频的
`h3-api:gb10-bf16` 镜像和 `/srv/h3/models/prepared/manifest.json`。
不覆盖主部署的 Compose、配置或 BF16 权重；测试通过 SGLang HTTP API 生成，不依赖 ComfyUI。
此包仅完成 CPU 编排检查，GB10 的实际兼容性、质量、速度仍需实测。

Turbo 使用派生镜像 `h3-bench:gb10-turbo`：只补丁修复固定版 SGLang 中
LoRA 包装层缺少 `quant_method` 时的 MXFP8 能力判断，不安装依赖、不改权重。
补丁让包装层继续使用普通张量的 BF16 + LoRA 路径，不禁用 LoRA，不改原 API 镜像。
已有的 QKV 布局修复由源文件 SHA256 校验保护。

## 对照内容

| 候选 | 新增下载 | 实际 DiT 推理次数 | 性质 |
| --- | --- | --- | --- |
| 现有 BF16 | 无 | 49 | 使用之前 2103.52 秒的单次结果作历史参考 |
| BF16 + Larry Turbo v4 | 0.78 GB | 8、4 | 社区蒸馏 LoRA，非量化；4 次更激进 |
| 完整层数 INT8 ConvRot | 34.04 GB | 49 | 仅量化 Transformer；其余组件复用 |

固定红色玩具车提示词、seed=42、1344×768、124 帧、24fps、请求时长 5 秒。
不启用额外缓存近似、稀疏注意力或低分辨率上采样。不把这些社区候选称为官方满血版。
8 次对应 API `num_inference_steps=9`，4 次对应 5；多出的一个是终点 sigma。
Turbo 用强度 1.0、动态 LoRA，避免修改 mmap 基础权重。它不是融合 LoRA 的极限速度测试。

## 先跑 Turbo

从本仓库根目录运行。以下以 `~/h3-service` 为例；需要先有 BF16 基础镜像和 prepared 模型配置，
新机器按 [从零部署](../docs/FRESH_INSTALL.md) 完成模型准备。不需要另外下载旧 benchmark 压缩包。

```bash
cd ~/h3-service
if [ ! -d /srv/h3/data ]; then
  sudo install -d -m 750 -o 10001 -g 10001 /srv/h3/data
fi
sudo docker compose -f benchmark/compose.yaml run --rm download
sudo docker compose -f benchmark/compose.yaml build turbo
```

下载完成、SHA256 校验成功后，确保当前没有生成任务，再停止原 API 并测试：

```bash
sudo docker compose --env-file .env -f production/compose.yaml stop h3-api
sudo docker compose -f benchmark/compose.yaml run --rm turbo
sudo docker compose --env-file .env -f production/compose.yaml start h3-api
```

若已经下载过 LoRA，只需更新此测试包、执行 `build turbo` 再重新测试，不必重新下载。
构建不需要 `--no-cache`，也不需要重新构建根目录的生产镜像。

三个命令按顺序执行；若尚未部署正式 API，跳过 stop/start，仅运行 benchmark 命令。
仅当测试前服务在运行时，测试结束再恢复它；脚本不会替你重启生产服务。
不要在另一个终端同时启动 API 或其他 GPU 任务。
如中途按 Ctrl+C，等容器退出、GPU 进程清理完后再恢复 API。

脚本先加载模型，然后做一次 4 次推理的预热，再正式生成 `turbo-8.mp4` 和 `turbo-4.mp4`。
加载期间输出较少，可以在另一终端查看最新日志：

```bash
sudo find /srv/h3/data/benchmark/runs -name sglang-fl2va.log
# 将上面输出中这次测试的完整路径粘到 tail -f 后查看。
```

请求执行时每 20 秒报告耗时。加载超时 45 分钟，单次请求超时 2 小时。
首次创建 spill/cache 可能慢；预热单独计时，不算进正式生成耗时。
下载容器不需要 GPU，它的 NVIDIA Driver 未检测到警告可以忽略；生成容器不可以忽略。

## 然后跑 INT8

原 API 可在下载和构建时保持运行；约 34 GB 下载完成后再停服务。

```bash
cd ~/h3-service
sudo docker compose -f benchmark/compose.yaml run --rm download /opt/h3-api/app/.venv/bin/python /bench/assets.py int8
sudo docker compose -f benchmark/compose.yaml build int8
sudo docker compose --env-file .env -f production/compose.yaml stop h3-api
sudo docker compose -f benchmark/compose.yaml run --rm int8
sudo docker compose --env-file .env -f production/compose.yaml start h3-api
```

派生镜像只添加固定 SHA 的 ARM64 comfy-kitchen 0.2.35 wheel，不更换 torch 或生产镜像。
不传在线 `--quantization`，因为下载的权重已包含 INT8 ConvRot 元数据。
运行前先用三个 H3 矩阵尺寸检查 GB10 的 INT8 数值结果；失败就停止，不继续长时间生成。
数值检查通过不代表视频质量通过。INT8 会先跑 4 次预热（其视频不用于质量判断），再跑 49 次正式生成。

## 结果和下载

每次输出在 `/srv/h3/data/benchmark/runs/<时间>-<候选>-<进程号>/`：

- `summary.json`：加载、预热、正式推理的耗时，历史基线相对速度，以及成功/失败状态。
- `turbo-8.mp4`、`turbo-4.mp4` 或 `int8-49.mp4`：需要实际观看和听声音的结果。
- `*.request.json`、`*.media.json`、`logs/sglang-fl2va.log`：参数、媒体检查和详细日志。
- `int8-kernel-gate.log`：仅 INT8，数值兼容检查。

结果目录如果普通用户无法进入，用以下命令导出全部测试结果（不包括大权重和缓存）：

```bash
sudo tar -czf "$HOME/h3-benchmark-results.tar.gz" -C /srv/h3/data/benchmark runs
sudo chown "$(id -u):$(id -g)" "$HOME/h3-benchmark-results.tar.gz"
chmod 600 "$HOME/h3-benchmark-results.tar.gz"
```

用 SFTP/WinSCP 下载自己用户目录下的 `h3-benchmark-results.tar.gz`。再次导出会更新这个结果压缩包，
不会删除原始测试结果。把 `summary.json` 和对应 MP4 发来即可一起判断。

脚本验证 LoRA 确实应用到非零层数、实际 NFE、分辨率、帧数、音频和全片可解码，
并拦截抽样全黑输出；这些检查不保证画面语义、运动连续性或声画同步。
目前是单提示词初筛，不是完整质量评测；胜出的候选再测人物、快速运动、对白。
历史基线使用了不同的启动预热策略，内存驻留也可能不同，所以速度倍数只能作初筛参考。

## 固定来源

- [SGLang 固定版本的 H3 LoRA 说明](https://github.com/sgl-project/sglang/blob/70b5b03e78612c94f86ac98eb4d2d8d19ceda738/docs/cookbook/diffusion/MiniMax/MiniMax-H3.mdx#5-lora-recipes)
- [Larry Turbo 固定版本](https://huggingface.co/larryvrh/MiniMax-H3-Turbo-Lora/tree/43a74557ac3f6539db8e0f2a959d03feb7a81480)
- [Comfy-Org 权重固定版本](https://huggingface.co/Comfy-Org/MiniMax-H3/tree/7e75982b97cd5a41d2dcfa1904ee88d0686d6fd1)

精确文件大小和 SHA256 见 `assets.py`，每次运行前重新验证。
