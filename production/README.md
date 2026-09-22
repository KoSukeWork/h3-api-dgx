# H3 Turbo 正式 API：独立、非量化、可调步数

## 当前方案

- 使用 BF16 主模型 + Larry Turbo v4，动态 LoRA；已有单台 GB10 横评记录，部署到新环境仍需验收。
- 不下载 INT8，不改驱动、不安装新推理依赖、不重新下载大权重。
- 派生镜像 `h3-api:gb10-turbo-v1` 复用已验证的 `h3-bench:gb10-turbo`；QKV 和 LoRA 兼容补丁都保留。
- 默认 8 次；`preset: fast` 为 4 次，`preset: quality` 为 8 次；`steps: 12` 为 12 次。
- 自定义步数目前允许整数 1～49；4、8 次已实测，其余仅开放参数，画质/耗时需实测。
- 6～8 次是作者主要建议范围，多步不保证更好。`quality` 档依旧是 Turbo，不是原始 49 次基线。
- 相同模型常驻，步数按任务调整，不重载。单 GPU 串行排队，不能同时跑多个服务/测试占用 GPU。
- 本轮支持 `t2va`、`fl2va`；拒绝把 FL2VA Turbo LoRA 用到 `ref2va`。已实测的是文生视频。

## 1. 新目录和准备

如有旧安装目录，请保留，不要直接覆盖它。
以下以当前用户的 `~/h3-service` 为新安装目录，把 `h3-api-dgx.tar.gz` 上传进去。
本节是复用现有镜像的快速路径；新机器先按 [从零部署](../docs/FRESH_INSTALL.md) 建立运行环境。

```bash
mkdir -p ~/h3-service
cd ~/h3-service
tar -xzf h3-api-dgx.tar.gz
sudo python3 production/manage.py prepare
```

`prepare` 只会：

1. 检查 ARM64、既有 Turbo 镜像、模型配置，以及旧 H3 容器/GPU 当前没有在运行任务。
2. 校验并复制约 780 MB LoRA 到 `/srv/h3/models/loras/`，保留原始 benchmark 文件。
3. 创建独立空数据目录 `/srv/h3/service-data`，UID/GID 10001，不复制旧队列、缓存或测试视频。
4. 在新安装目录创建随机密钥 `.env`（0600），默认监听 `127.0.0.1:8000`；已有 `.env` 不覆盖。

它不会删除文件，也不会启动 API。已开始使用的非空 `service-data` 会拒绝重新初始化。
旧 `.env`、证书、基础模型、结果都保留。容器内的数据路径仍是 `/srv/h3/data`，
但对应主机上全新的 `/srv/h3/service-data`，不是旧 `/srv/h3/data`。

## 2. 先盘点，后清理

```bash
sudo python3 production/manage.py inventory
```

此命令仅检查目录占用、容器状态、镜像和构建缓存大小，请自行审核后再决定清理目标。
默认检查固定数据目录和当前仓库，不猜测用户名或其他安装位置。
如需额外盘点旧安装目录，可显式指定（`$HOME` 在调用 sudo 前由当前用户的 shell 展开）：

```bash
sudo python3 production/manage.py inventory --legacy-install-dir "$HOME/h3-install"
```

分享盘点输出前，应遮蔽本机路径、容器名称等不打算公开的环境信息。
现在没有任何自动删除命令。应保留：

- `/srv/h3/models` 全部模型与 prepared 配置，包括暂时不用的 Ref2VA 权重。
- `/srv/h3/data/benchmark/runs`、`acceptance`、`outputs` 和其他生成结果。
- 旧安装目录的证书和 `.env`，新服务验收前保留整个旧安装目录。
- 已跑通的 BF16/Turbo Docker 镜像，尤其在正式派生镜像构建成功前。

通常可清的是旧生成缓存、benchmark 缓存、重复安装压缩包，但要先核对实际目录内容。
旧 `/srv/h3/data/sglang` 可能包含视频和上传素材，不应当作纯缓存整目录删除。
不要执行 `docker system prune -a` 或删除 `/srv/h3/data`、`/srv/h3/models` 整体。

## 3. 构建并启动新服务

确认不再运行旧 API/benchmark 后，在新目录执行：

```bash
cd ~/h3-service
sudo docker compose --env-file .env -f production/compose.yaml build
sudo docker compose --env-file .env -f production/compose.yaml up -d
curl -fsS http://127.0.0.1:8000/health
```

不用 `--no-cache`，不重装 torch/SGLang，不重新构建根目录的旧 BF16 镜像。
构建会校验固定源码及兼容补丁；未知源码版本会拒绝。
`/health` 仅代表网关/调度存活，GPU 模型首次收到请求才加载。
首个请求包含加载/首次运行开销；后续切换步数不会重新加载。
这里没有 benchmark 的额外预热视频，启动预热也关闭。

## 4. 首次验收和自定义步数

先跑 4 次，检查画面和声音。容器已持有 API key，不必把密钥写进命令行：

```bash
sudo docker compose --env-file .env -f production/compose.yaml exec h3-api \
  /opt/h3-api/app/.venv/bin/python examples/client.py \
  --url http://127.0.0.1:8000 --preset fast --short-edge 768 \
  --prompt 'A red toy car drives across a white table. Camera: static. Audio: quiet electric motor.' \
  --output /srv/h3/data/acceptance/turbo-fast.mp4
```

主机文件是 `/srv/h3/service-data/acceptance/turbo-fast.mp4`。
下一次把 `--preset fast` 换成 `--preset quality`，或换成 `--steps 12`，输出文件名也相应更改。
`--preset` 和 `--steps` 二选一；两者都不传时，正式 Turbo 服务默认 8 次。
不要把 12 写成 13：外层 API/客户端使用实际推理次数，内部自动 +1 转成 sigma 点数。

正式请求 JSON 示例：

```json
{
  "task": "t2va",
  "prompt": "A red toy car drives across a white table. Camera: static. Audio: quiet electric motor.",
  "target": {"short_edge": 768, "aspect_ratio": "16:9", "duration_seconds": 5},
  "seed": 42,
  "steps": 12
}
```

POST `/v1/videos` 返回 202 和任务 id。GET `/v1/videos/{id}` 查看状态；完成后
GET `/v1/videos/{id}/content` 下载 MP4。所有 `/v1/*` 和 `/ready` 请求需 `Authorization: Bearer <密钥>`。
提交和查询响应中的 `sampling` 返回 `mode`、实际 `steps`、内部 `num_inference_steps`，便于核验。
不要传内部的 `num_inference_steps`、LoRA 路径或量化参数，网关会拒绝。
旧 `quality: lossless` 表示禁用额外近似缓存，不代表 Turbo 输出无质量损失；通常不必显式传它。

参数非法、同时传 preset/steps、Turbo 服务请求 ref2va 都返回 422，不会排队浪费 GPU 时间。
在不启用 Turbo 的旧配置下，默认仍保留原 49 次；不会把旧基线默默改成 Turbo。

## 5. 局域网及 ComfyUI

本机验收成功后，用 `sudo nano .env` 将 `H3_BIND_IP` 改为 DGX 实际的内网 IP，然后执行：

```bash
sudo docker compose --env-file .env -f production/compose.yaml up -d
```

从其他机器访问 `http://<DGX_LAN_IP>:8000/docs`，将占位符替换为实际 IP，Authorize 输入新 `.env` 中的密钥。
密钥不要发到聊天、截图、工作流 JSON 或 Git。当前服务使用 HTTP，仅在可信内网使用；
跨网段/公网访问前增加 HTTPS/访问控制。不要将监听地址直接改为全网开放。

将 `comfyui_client` 复制到 ComfyUI 的 `custom_nodes/h3_api_client/`，在其 Python 环境安装 requirements。
推荐复制节点目录的 `config.example.json` 为 `config.json`，填写服务器根地址和纯 API key，
确保 ComfyUI 服务用户可读（建议权限 600）。安装后重启一次 ComfyUI，不需要修改 systemd 配置。
之后只改本地配置，下一次节点运行会重新读取；缺少文件时也支持原来的环境变量方式。
完整 Ubuntu/systemd 步骤见 [节点安装说明](../comfyui_client/README.md)。
节点新增 `sampling`：`server_default`、`fast`、`quality`、`custom`；仅 custom 时读取 `steps`。
其他开发工具可直接调用同一 HTTP API，无需 ComfyUI。

## 日志与停止

```bash
sudo docker compose --env-file .env -f production/compose.yaml logs --tail 100 h3-api
sudo tail -n 120 /srv/h3/service-data/logs/sglang-fl2va.log
sudo docker compose --env-file .env -f production/compose.yaml stop
```

不要再启动旧 `h3-install` 服务；新服务的 Compose 项目名是 `h3-service`。
模型首次加载会检查 LoRA SHA256 和非零应用层数。需要新一次验收确认整条正式 API 路径，
当前本地检查仅为 CPU 单元测试/Compose 校验，没有在 x86 上进行 GPU 推理。
