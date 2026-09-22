# 新 DGX 从零部署

只在原厂 NVIDIA DGX Spark（GB10 / aarch64）构建、加载、生成。不要在 x86 上模拟推理。
这是新机器的重建流程，已有正常环境请直接使用 `production/README.md` 的派生镜像流程。

## 1. 系统前置条件

通过 NVIDIA 原厂更新机制完成系统/驱动更新并按要求重启，不手动混装 x86 驱动或 `.run` 安装包。
安装 ARM64 Docker Engine、Compose 插件和 NVIDIA Container Toolkit。先检查：

```bash
uname -m
nvidia-smi
nvidia-ctk --version
sudo docker version
sudo docker compose version
```

预期架构 aarch64 / Docker arm64、GPU 名称 GB10。驱动必须支持 CUDA 13。
已验证环境为驱动 580.178.04，版本记录不表示以后必须降级到它。
如 Docker 尚未配置 NVIDIA runtime，按 [NVIDIA Toolkit 官方说明](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
配置；重启 Docker 会中断其他容器，需先安排停机。

## 2. 代码和证书

将本仓库 clone 或安装包解压到当前用户的 `~/h3-service`，也可以选择其他独立目录。
不要复用正在服务的安装目录。以下命令均从仓库根目录运行。

公司 HTTPS 网关使用私有 CA 时，把 IT 确认过指纹的 PEM 公共根/中间 CA 放入 `certs/*.crt`。
每文件一个证书；不复制私钥、PFX/P12 或网站叶证书。DER `.cer` 需要转换，不能只改扩展名。
不要提交实际证书。新 Dockerfile 在 pip、rustup、git 之前导入它们。
拉取 CUDA 基础镜像之前仍需要宿主机/Docker 自身信任网关；HTTPS apt 代理也需另行正确配置。
不关闭 TLS 校验来绕过证书问题。

## 3. 初始化空目录并构建运行环境

```bash
cd ~/h3-service
sudo python3 scripts/bootstrap_host.py
sudo docker compose -f bootstrap/compose.yaml build runtime
sudo docker compose -f bootstrap/compose.yaml build turbo
```

初始化创建 `.env` 随机密钥及 `/srv/h3/models`、`/srv/h3/bootstrap-cache`、`/srv/h3/service-data`。
已有密钥不会覆盖；非空正式数据目录会拒绝重新初始化。脚本不会安装驱动、删除数据或启动服务。
镜像依次为 `h3-api:gb10-bf16` 和 `h3-bench:gb10-turbo`；这两个名字是中间构建产物，不需要旧机器预先提供。
先后顺序不可颠倒，不使用 `docker compose up` 启动 bootstrap 文件。
首次构建会下载 ARM64 推理依赖，可能较慢；不要用 `--no-cache` 重做已完成的步骤。

依赖固定版本来自成功的 GB10 测试记录。若未来包不可用或约束解析失败，应停止并保存日志；
不要自行去掉约束升级全部依赖。优先走已有镜像的备份恢复路径。

## 4. 下载/复用模型，生成本地配置

完整沿用这次验证过的五文件准备流程：约 **189.9 GB 基础权重 + 0.78 GB Turbo LoRA**。
其中 Ref2VA 权重约 66.3 GB，为基线/以后扩展保留；当前 Turbo 正式服务不使用它，
但当前 `h3_api.prepare` 的完整校验流程仍要求它存在。这不是最低磁盘占用方案。
建议空闲至少 300 GiB，构建缓存、模型 spill、输出会继续增长。

已有 `/srv/h3/models` 可直接复用；从备份复制时保持 diffusion_models、text_encoders、vae、loras 层级。
下面的下载脚本会 SHA256 校验已有文件并跳过下载，发现同名文件不匹配则拒绝覆盖。
无 HF token 可公开下载，但可能限速。可在 DGX 终端私下设置 `HF_TOKEN`，不写进 Git 或聊天。

```bash
sudo --preserve-env=HF_TOKEN docker compose -f bootstrap/compose.yaml run --rm download
sudo --preserve-env=HF_TOKEN docker compose -f bootstrap/compose.yaml run --rm prepare
```

`prepare` 再次校验五份权重，只下载固定版本的官方 config/tokenizer 小文件，不下载官方重复大权重。
它在容器路径 `/srv/h3/models` 下生成 `prepared/manifest.json`；此路径必须与正式服务一致。
Windows 预下载脚本 `predownload_h3_bf16.ps1 -IncludeRef2VA` 也可下载基础权重；
它不包含 Turbo LoRA，仍需用上述 download 命令补齐/校验。
准备容器不申请 GPU，CUDA 启动横幅可能显示 Driver 未检测到；它不执行推理，不是 GPU 验收。

## 5. 正式服务

```bash
sudo python3 production/manage.py prepare
sudo docker compose --env-file .env -f production/compose.yaml build
sudo docker compose --env-file .env -f production/compose.yaml up -d
curl -fsS http://127.0.0.1:8000/health
```

生产 LoRA 已放在模型目录时，prepare 直接验证它，不依赖旧 benchmark/assets 路径。
镜像 `h3-api:gb10-turbo-v1` 仅叠加网关代码，不再次安装推理依赖。
接着按 [正式 API 文档](../production/README.md) 跑 fast 出片验收、测试自定义 steps，最后再开放内网访问。
`/health` 成功不代表 GPU 出片通过。新构建流程没有在当前 Windows 主机执行 GPU 测试。
