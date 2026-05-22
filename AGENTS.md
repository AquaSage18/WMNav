# AGENTS.md

## 项目结构
- `scripts/main.py`: 训练/评估入口，读取 `config/*.yaml` 并启动环境。
- `scripts/aggregator.py`: 并行评估时的 Flask + WandB 聚合服务。
- `src/api.py`: 所有视觉语言模型适配层，Qwen/Gemini 的请求格式和返回清洗都在这里。
- `src/WMNav_agent.py`: 主体导航 agent 实现。
- `src/custom_agent.py` / `src/custom_env.py`: 自定义与消融版本。
- `src/WMNav_env.py`: 环境封装与评估逻辑。
- `src/utils.py`: 坐标变换、图像绘制、深度相关工具。
- `config/WMNav.yaml`: 默认实验配置。
- `parallel.sh`: 多 GPU 并行评估脚本模板。
- `readme.md`: 安装、部署和运行说明。
- `logs/`, `data/`, `wandb/`: 运行产物与数据目录。

## 维护约定
- 改模型调用优先看 `src/api.py`，改 agent 行为优先看 `src/WMNav_agent.py`。
- 需要访问境外网站时，先在终端设置代理：
  - `export http_proxy=http://127.0.0.1:7890`
  - `export https_proxy=http://127.0.0.1:7890`
  - 然后用 `curl -I www.google.com` 先确认代理可用。
- 尽量保持改动局部，不回滚用户已有修改。
- 运行前优先用 `rg` 找引用点，再改配置或代码。

## Qwen3.6 参考资料
- vLLM 部署与 API 调用: [Qwen3.6 running guide](https://docs.vllm.ai/projects/recipes/en/latest/Qwen/Qwen3.5.html#running-qwen36)
- 模型页: [Qwen/Qwen3.6-27B](https://huggingface.co/Qwen/Qwen3.6-27B?spm=a2ty_o06.30285417.0.0.7973c921O0YLUU&file=Qwen3.6-27B)
