# 来源与许可

本仓库从 DGX H3 API 部署实践中整理，保留原仓库 MIT LICENSE 的版权声明。
原始项目：[dgxspark_comfyui_minimax_h3](https://github.com/luqidaxia/dgxspark_comfyui_minimax_h3)。

运行时使用 SGLang、NVIDIA CUDA 容器、MiniMax H3/Comfy-Org 权重和 Larry Turbo LoRA。
这些第三方软件、容器与模型遵循各自许可证；本仓库的 LICENSE 不重新授予它们的权利。
模型文件不分发在本仓库中。部署前应阅读相应 Hugging Face 仓库与 NVIDIA 镜像许可。

相关上游项目：

- [SGLang](https://github.com/sgl-project/sglang)
- [MiniMax H3 官方模型](https://huggingface.co/MiniMaxAI/MiniMax-H3)
- [Comfy-Org 模型导出](https://huggingface.co/Comfy-Org/MiniMax-H3)
- [Larry Turbo LoRA](https://huggingface.co/larryvrh/MiniMax-H3-Turbo-Lora)
- [NVIDIA Deep Learning Container License](https://developer.nvidia.com/ngc/nvidia-deep-learning-container-license)

本项目与上述组织不存在因使用其代码/模型而自动产生的官方隶属、认证或支持关系。
保留源码许可声明不等同于对模型的商用、再分发或衍生使用作出许可保证。

用于兼容的源代码补丁仅针对固定 SGLang commit，不替换上游许可证或版权声明。
内网 CA、API key、HF token、用户视频和大权重不得提交到代码仓库。
