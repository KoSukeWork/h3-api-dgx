# 双配方 API：Larry + LightX2V Ref2VA

2026-10-21 已在 GB10 正式服务完成 Larry → Ref2VA → Larry 三次串行生成，见
[实机验收记录](PROFILES_DGX_ACCEPTANCE.md)。

非量化 BF16，仍然只有一个持久队列、一个 GPU 推理进程组。每个请求可指定 `profile`；
省略或填 `auto` 时按 task 自动选择。客户端不能传任意 LoRA 路径、alpha、shift 或量化参数。

| profile | task | 步数 | 输入 |
| --- | --- | --- | --- |
| `larry-turbo-v4` | `t2va`、`fl2va` | 默认 8；fast=4、quality=8；自定义 1～49 | 无条件或首/尾帧 |
| `lightx2v-ref2va-v0.1` | `ref2va` | 固定 4；默认、fast 或 steps=4 | 参考图片（最多 9，单图已验收） |

Ref2VA 初版不开放视频/音频参考或 keyframe 混合；`preset: quality`、steps=8/12 等返回 422。
支持网关现有的 480/768 尺度和 4～15 秒时长，但本轮画质验收只有单参考图、480p、约 5 秒，
不宣称所有尺寸/时长/多图组合都已验证，也不等于原作者的 544p 配方复现。
首尾帧要求图像严格作为起止帧；Ref2VA 是内容参考，语义不同。

## 调用

所有 `/v1/*` 均需 `Authorization: Bearer <API key>`。
先 POST `/v1/files` multipart 字段 `file` 上传图片，获得 `file_<32 hex>`，然后提交：

```json
{
  "task": "ref2va",
  "profile": "lightx2v-ref2va-v0.1",
  "prompt": "Use <Picture 1> as the character reference. The character waves. Camera stays fixed.",
  "conditions": [{"type": "image", "role": "reference", "file_id": "file_00000000000000000000000000000000"}],
  "target": {"short_edge": 480, "aspect_ratio": "16:9", "duration_seconds": 5},
  "steps": 4,
  "seed": 42
}
```

文件 ID 是占位符，须替换为上传返回值。POST `/v1/videos` 返回 202；
GET `/v1/videos/{id}` 查询，完成后 GET `/v1/videos/{id}/content` 下载。
提交/查询响应增加实际 `profile`，保留 `sampling`、`status`、`error`、`content_url` 等字段；
`/docs` 的 `JobResponse` 定义了响应结构。旧历史任务没有保存配方时返回 `profile: null`，不猜测。
首尾帧索引仍是 `0`（首帧）和 `-1`（末帧），不是传实际帧数。

GET `/v1/profiles` 列出启用配方、任务和步数约束；GET `/ready` 增加 `loaded_profile` 和 `profiles`。
旧 `/ready.sampling` 的默认步数仍描述 Larry 默认行为；多配方客户端应读取各自的 `profiles`。

同配方重复请求、Larry 切换 4/8/12 步不重载；跨配方必须先结束旧进程组，再校验并加载新底模/LoRA。
缓存按固定配方和权重 SHA 隔离。切换和冷启动会增加等待时间，不是同时驻留，也不承诺热切换。
此版本没有新增定时空闲卸载器；切换、失败清理或停止服务时会卸载模型。

## ComfyUI

更新节点后重启 ComfyUI 并强制刷新页面。新增末尾 `profile` 下拉框，旧控件顺序不变。
通常选 `auto` 即可；`task=ref2va`、图片接 `reference_images`，sampling 选 `server_default` 或 `fast`。
`steps` 仅 custom 时使用，此时 Ref2VA 必须填 4。预览、下载和 `video_path` 接口保持原样。
旧节点没有 profile 字段也能按 task 自动路由，但不可继续向 Ref2VA 发送 8/12 步。

## 在已有 DGX 部署上升级

保留旧安装目录、旧镜像和 `.env`；在新的独立目录解压代码包。不要对非空 service-data 重新运行 prepare。
只在 ARM64 DGX 构建，复用已有 `h3-api:gb10-turbo-v1`，不升级 GPU 依赖。
可将已检查过的旧镜像 ID 打一个独立本地 tag，通过 `H3_PROFILES_BASE_IMAGE` 选择它；
不要把未经验证的任意镜像或已打过参考图补丁的测试镜像作为基础。

将原有 `.env` 私下复制到新目录并保持 0600；地址、端口、密钥保持不变，不要提交 Git。
用已经校验下载过的 LoRA 安装到正式模型目录：

```bash
sudo python3 -m production.install_ref2va --source /srv/h3/ref2va-test/assets/minimax_h3_ref2v_turbo_4step_v0.1_bf16.safetensors
sudo docker compose --env-file .env -f production/compose.yaml -f production/profiles.yaml config --quiet
sudo docker compose --env-file .env -f production/compose.yaml -f production/profiles.yaml build
```

模型来源和首次下载见 [测试说明](../benchmark/ref2va/README.md)。安装脚本验证源/目标完整 SHA，
保留源文件、拒绝覆盖不一致目标。构建核验固定 QKV/LoRA 补丁及参考图缩放补丁，不改变宿主依赖。

确认调用方暂不提交、队列为空后，停止旧服务，备份 SQLite（含 WAL 的一致性备份）、旧 Compose/.env，
并记录旧镜像 ID。新旧 Compose 使用相同 `h3-service` 项目和 `/srv/h3/service-data`，不得同时启动两套。

```bash
# 从旧安装目录停止；先确认没有 queued/loading/running 请求。
sudo docker compose --env-file .env -f production/compose.yaml stop
# 切到新安装目录后：
sudo docker compose --env-file .env -f production/compose.yaml -f production/profiles.yaml up -d --no-build
```

检查 `/health`、鉴权 `/ready`、`/v1/profiles` 和历史任务下载。
验收脚本 `examples/validate_profiles.py` 在 DGX 容器内按 Larry → Ref2VA → Larry 顺序生成三段测试视频。
测试素材和结果只放私有数据目录；脚本不会输出密钥。

回退：先在新目录以两个 `-f` 停服务，再从保留的旧目录以原 Compose `up -d --no-build --force-recreate`。
旧镜像 tag 必须仍指向原 ID。此次不修改数据库表结构，不用旧备份覆盖后续生成结果或数据库。
若回退前已有新 Ref2VA 请求，先处理完或明确取消，不让旧服务把新请求按旧配方执行。

## 升级后的日常管理

本次升级使用新目录 `~/h3-service-profiles`。日常操作始终带上两个 Compose 文件：

```bash
cd ~/h3-service-profiles
sudo docker compose --env-file .env -f production/compose.yaml -f production/profiles.yaml ps
sudo docker compose --env-file .env -f production/compose.yaml -f production/profiles.yaml logs --tail 100 h3-api
# 需要重启时使用 restart；不要在有任务运行时操作。
sudo docker compose --env-file .env -f production/compose.yaml -f production/profiles.yaml restart h3-api
```

不要从旧目录执行 `up`，也不要遗漏 `profiles.yaml` 后执行 `up`，以免退回单配方镜像。
