# 备份与恢复

代码仓库不等于完整运行环境。为了以后直接复用，正式 API 在 DGX 验收成功后，
建议单独保存镜像、模型、配置和所需结果；不要把大文件或密钥提交 Git。

## 保存已验证的正式镜像

在 DGX 将输出路径换成有足够空间的备份目录，以下示例不会自动创建目录：

```bash
sudo docker image save -o /path/to/backup/h3-api-gb10-turbo-v1.tar h3-api:gb10-turbo-v1
sha256sum /path/to/backup/h3-api-gb10-turbo-v1.tar
```

正式镜像包含全部基础层、推理环境、两项兼容补丁和公司 CA（如果构建时添加过），
不包含挂载的模型/数据或运行时 API key。应内部保存，不随意公开镜像。
如还要在换机后用新代码构建派生镜像，也保存 `h3-bench:gb10-turbo`：

```bash
sudo docker image save -o /path/to/backup/h3-turbo-runtime.tar h3-bench:gb10-turbo
```

正式服务停机后再备份 `/srv/h3/service-data`，避免复制到正在变动的 SQLite 队列。
另行保存 `/srv/h3/models`（含 prepared 配置、LoRA）、新安装目录 `.env` 和受信任的公共 CA。
旧生成结果在 `/srv/h3/data` 下，不能当作正式数据目录的副本；如需要也单独保留。
API 密钥备份应加密、限制访问，换机可重新生成。

## 只恢复已验证的服务，不重新构建

1. 新 GB10 按官方流程完成驱动、Docker、NVIDIA Container Toolkit 安装。
2. 获取本仓库代码。用 `docker image load -i ...` 恢复正式镜像并核对其标签/外部 SHA256。
3. 将模型恢复到相同的 `/srv/h3/models`；不要改变 prepared manifest 内的容器路径。
4. 新的空队列用 `sudo python3 scripts/bootstrap_host.py` 创建目录和新密钥；
   若恢复历史正式数据，则手动恢复 `.env` 与 `/srv/h3/service-data`，保留 UID/GID 10001，
   不对已有数据运行初始化脚本。不要递归修改整个 `/srv/h3` 的权限。
5. 启动已恢复的镜像，明确禁止自动构建/拉取：

```bash
sudo docker compose --env-file .env -f production/compose.yaml up -d --no-build --pull never
```

这一恢复路径不需要旧 benchmark 运行目录、中间镜像或重新下载 Python 包。
若只恢复正式镜像，不运行 `production/manage.py prepare`，它是供构建路径使用、会检查中间镜像。
仍需 API/GPU 出片验收，之后才能开启内网访问。

## 清理原则

通过新服务验收后再清理旧资源；先执行 `production/manage.py inventory`。
仅删除已确认闲置的具体缓存目录，保留模型、结果、证书和必要的镜像备份。
不要运行全局 `docker system prune -a`；不要删除 `/srv/h3` 或整个 `/srv/h3/data`。
这个仓库没有自动破坏性清理脚本。
