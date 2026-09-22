# 2026-09-22 部署记录

本记录来自一台 DGX Spark 的实际部署与横评，不代表全部硬件/软件组合的验证结果。
独立仓库不包含上游仓库的 ComfyUI 本地推理安装向导、工作流、视频或模型。
运行和测试不依赖维护者本机路径；模型和数据使用文档约定的容器挂载路径。

## 已实测环境

- DGX Spark GB10，aarch64，GPU capability 12.1，约 121.7 GiB 统一内存。
- 驱动 580.178.04，CUDA 13.0，基础容器 CUDA 13.0.3。
- torch 2.13.0+cu130；完整包版本见 `inference-resolved-20260922.txt`。
- SGLang `70b5b03e78612c94f86ac98eb4d2d8d19ceda738`。
- Comfy-Org 模型 revision `7e75982b97cd5a41d2dcfa1904ee88d0686d6fd1`。
- 官方配置 revision `42ed227ee7df40d41602854ae760620d6eb651fe`。
- Larry Turbo v4 revision `43a74557ac3f6539db8e0f2a959d03feb7a81480`。

## 必须保留的修复

1. `scripts/patch_h3_qkv.py`：固定 Comfy BF16 QKV 排列，避免重复转换导致棕色噪声。
   只有经过固定 SHA 校验的 BF16 权重才由网关开启该分支，不能泛化到任意权重。
2. `benchmark/patch_lora_compat.py`：LoRA 包装层不存在 quant_method 时跳过 MXFP8 输入分支，
   保留普通 BF16 + 动态 LoRA，不通过转发 base_layer 绕过 LoRA。
3. `scripts/check_inference_env.py`：处理本次 cuSPARSELt wheel 平台元数据兼容审计；
   不以删除 NVIDIA 数学库或忽略所有依赖错误代替检查。
4. 公司公共 CA 放入系统信任链，保留 TLS 验证。新从零构建把 CA 导入提前到网络工具之前。

## 同一提示词的单次测速

1344×768、124 帧、24 fps、seed 42，红色玩具车提示词；不含模型加载和预热。

| 模式 | 实际 DiT 次数 | 生成耗时 | 相对历史基线 |
| --- | ---: | ---: | ---: |
| 原 BF16 | 49 | 2103.52 秒 | 1× |
| BF16 + Turbo v4 | 8 | 440.46 秒 | 4.78× |
| BF16 + Turbo v4 | 4 | 248.10 秒 | 8.48× |

Turbo 加载约 44.14 秒、独立预热约 260.79 秒；正式 API 不主动做这条额外预热。
两档共用一次模型加载，LoRA 实际应用到 259 层。原始摘要保存在 `turbo-benchmark-20260922.json`。
历史 BF16 的预热策略不同，不能把单次倍数当作严谨的多样本性能结论。
该轮人工目视评估认为效果可接受，因此未继续 INT8 对照；这不是盲测或标准化质量评分。

正式 API 的新网关代码、从零构建顺序和固定版本约束均只做 CPU/静态检查，
需要在目标 DGX 完成正式服务出片验收，不能把旧横评成功等同于所有新流程已通过 GPU 验收。
