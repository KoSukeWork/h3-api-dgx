# H3 Remote API：ComfyUI 远程调用节点

将另一台机器上的 ComfyUI 连接到 H3 HTTP API：上传输入 → 提交任务 → 轮询 → 下载 MP4。
H3 推理运行在 DGX 上，客户端不下载 H3 权重、不加载 H3 模型、不安装 SGLang/CUDA 推理依赖。
本节点保留 `H3RemoteVideo` 类型，可更新已有工作流。包含后端 Python 节点和视频预览前端扩展。

## Ubuntu 安装（包括 systemd 启动的 ComfyUI）

1. 将安装包解压到 ComfyUI 的 `custom_nodes`，或将本仓库的 `comfyui_client` 目录复制并改名为
   `h3_api_client`。应直接得到 `custom_nodes/h3_api_client/__init__.py`，不要多套一层目录。
2. 使用 **实际运行 ComfyUI 的 Python 环境** 安装节点目录中的 `requirements.txt`：

```bash
# 先进入 ComfyUI 目录并激活其虚拟环境，或换成该环境 Python 的绝对路径。
python -m pip install -r custom_nodes/h3_api_client/requirements.txt
```

3. 在节点目录把 `config.example.json` 复制为 `config.json`：

```bash
cd custom_nodes/h3_api_client
cp -n config.example.json config.json
chmod 600 config.json
nano config.json
```

内容如下，**把两项占位符都改为实际值**，域名解析到内网 IP 也可以：

```json
{
  "H3_API_URL": "http://<DGX_HOST>:8000",
  "H3_API_KEY": "REPLACE_WITH_YOUR_API_KEY"
}
```

URL 是服务器根地址，不能带 `/docs`、查询参数或用户名/密码。Key 填纯密钥，不加 `Bearer`。
`<DGX_HOST>` 可以换成可访问 DGX 的 IP 或域名；示例占位符不能直接运行。
不要将服务地址和密钥直接写进 Python 代码，也不用把它们作为工作流节点输入。

4. 确保 `config.json` 属于运行 ComfyUI 的用户且该用户有读权限。
   systemd 服务如果使用不同用户，应只修改这一个文件的所属用户，不要递归修改整个 ComfyUI 的权限。
5. 重启 ComfyUI 以加载新节点。若系统服务名是 `comfyui.service`：

```bash
sudo systemctl restart comfyui.service
```

服务名不同请替换；用户级服务使用自己的 `systemctl --user`。**不需要增加 systemd 环境变量配置。**

安装或更新 Python 节点代码后要重启；以后仅修改 `config.json`，下一次节点执行自动重新读取。
配置文件存在时使用其中完整的 URL/Key；不存在时才回退到进程环境变量 `H3_API_URL`/`H3_API_KEY`。
不把文件和环境变量中的半套配置混用；文件损坏或不完整时会报错，不静默回退。

已有旧版时覆盖节点代码即可，务必一起复制新增的 `web/` 目录，保留原来的 `config.json`。
重启 ComfyUI 后在浏览器强制刷新（Ctrl+Shift+R），加载前端扩展。不用修改 DGX 服务或重新填写密钥。

## 在工作流中使用

搜索 `H3 Remote API Video (BF16)`，分类为 `H3 API`。

- **文生视频**：`task=t2va`，填写 prompt；首次可选 seconds=5、short_edge=768、sampling=fast。
- **首/尾帧生视频**：`task=fl2va`，将 ComfyUI 的加载图像节点接到 first_frame 和/或 last_frame，每个输入仅一张图片。
- **多参考**：节点保留 ref2va 兼容原版服务，但当前 Turbo FL2VA 服务会拒绝它；不要在该服务上选择。

| sampling | 实际推理次数 |
| --- | --- |
| server_default | 使用服务默认值，当前 Turbo 默认为 8 |
| fast | 4 |
| quality | 8 |
| custom | 使用 steps，整数 1～49，例如 12 |

非 custom 模式下 steps 不参与请求。外层直接填实际次数，不自行加 1。
这几档共用 DGX 上已加载的模型，不重启服务。4/8 次已有历史实测，其他次数仅提供参数，不保证更高画质。

节点保存视频到 **ComfyUI 所在机器** 的 `output/h3_api/`，输出 STRING 类型的文件路径。
不需要连接任何输出节点：生成完成后，节点内自动出现视频播放器、**下载 MP4** 和 **新窗口播放**。
播放器带进度、音量及浏览器支持的全屏控制，点击播放可听到音频，不自动播放。
下载链接把原始 MP4（含音轨）保存到 **浏览器所在电脑**；播放器不重新编码视频。
如果浏览器不支持视频编码，可下载后用本地播放器打开。
`video_path` 仍是 STRING，兼容后续路径读取节点，不直接输出 IMAGE/AUDIO/VIDEO 张量。
API Key 和 URL 不会出现在节点的输入列表或工作流 JSON 中。

预览与下载走 ComfyUI 自带的 `/view` 输出文件接口，浏览器不直接连接 DGX，不接收 H3 API Key。
文件访问受 ComfyUI 自身的访问控制保护，不要把未鉴权的 ComfyUI 暴露到公网。
预览文件必须保留在 ComfyUI 输出目录。旧版本执行记录没有预览元数据，升级不会自动补出旧记录的播放器。
如果新生成后没有播放器，检查 `web/h3_video.js` 已安装且扩展未被禁用，重启并强制刷新浏览器。

每次排队执行都会向远端提交新任务（不会复用 ComfyUI 上一次缓存结果），按“运行”前确认参数。
生成状态会显示在 ComfyUI 的后端日志中；默认最多等待四小时。
停止本地节点、关闭网页或客户端等待超时，**不会取消已经提交到 DGX 的任务**。
日志中的任务 id 可继续通过 API 查询；提交请求超时也不要盲目重复提交。

## 配置与安全

`config.json` 是明文密钥文件，已被 Git 和打包脚本排除。请保留 600 权限，不要发到聊天、公开仓库，
也不要连同整个 custom_nodes 目录打包分享。安装包仅提供 example，不覆盖已有 config.json。
节点按配置向服务器发送 Bearer key；它只访问明确指定的地址，不跟随 HTTP 重定向。
TLS 证书校验保持开启，不提供关闭验证的选项；如使用网关，自行正确配置证书信任。
HTTP 只适用于可信内网，跨网络访问应使用 HTTPS/VPN 及适当访问控制。

常见错误：401 检查 key 是否匹配；422 检查任务/条件/步数；连接失败检查网络、域名和监听地址；
配置读取失败检查 JSON 和服务用户权限；重定向错误请配置最终 API 地址而不是跳转地址。

## 从源码打包

在仓库根目录运行 `python scripts/package_comfyui.py`，输出 `dist/h3-comfyui-client.zip`。
包内顶层只有 `h3_api_client/`，只包含代码、示例配置、说明和许可，不包含真实配置或视频。
CPU 测试覆盖配置与模拟 HTTP 工作流；`node --experimental-vm-modules --test tests/comfyui_preview.test.mjs`
检查前端扩展的播放器、下载链接和生命周期。首次在真实 ComfyUI/DGX 上仍需验收。
