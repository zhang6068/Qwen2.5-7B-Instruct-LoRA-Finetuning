## 项目概述
- 基座：Qwen2.5-7B-Instruct
- 场景：医疗科普、轻症护理、就诊科室指引（**禁止临床诊断、开具处方**）
- 量化方案：Q4_K_M，原始13GB FP16模型压缩至4GB，6GB显存笔记本可离线运行
- 云端环境：AutoDL Ubuntu容器完成训练、格式转换、量化
- 本地推理：Ollama（底层依赖llama.cpp GGUF标准）

## 仓库目录结构
```
medical-qwen2.5-7b/
├── README.md                     # 项目总文档（本文）
├── data/
│   └── medical_train.jsonl      # ChatML格式医疗微调数据集
├── code/
│   ├── train_lora.py             # LoRA微调完整Python代码
│   ├── test.py                   #  LoRA 模型本地对话测试代码
│   ├── merge_lora_to_full.py     # LoRA合并完整HF模型代码
├── deploy/
│   ├── convert_hf_gguf.py        # HF转GGUF调用代码
│   ├── quantize_model.py         # 调用llama-quantize量化代码
│   └── Modelfile                 # Ollama本地部署配置文件
```

## 一、云端环境依赖安装（AutoDL）
### 1.1 系统依赖
```bash
sudo apt update
sudo apt install -y build-essential cmake unzip
```
### 1.2 Python全量依赖
```bash
# 微调、合并模型依赖
pip install torch transformers peft accelerate datasets bitsandbytes sentencepiece
```
### 1.3 llama.cpp 工具部署（转换&量化必备）
GitHub国内访问超时，推荐手动上传源码zip包解压
```bash
# 上传 master.zip 至 /root/autodl-tmp
unzip master.zip
mv llama.cpp-master llama.cpp
cd llama.cpp
# CMake编译（新版仅支持CMake，关闭CUDA加速，大幅提速）
cmake -B build
cmake --build build --config Release
```
校验编译结果：
```bash
ls llama.cpp/build/bin
# 存在 llama-quantize 即为编译成功
```

## 二、数据集规范（ChatML标准格式，data/medical_train.jsonl）
每行1条对话样本，支持单轮/多轮问答，适配Qwen官方chat模板
```json
{"messages": [
    {"role": "system", "content": "你是专业医疗科普助手，仅提供轻症护理、就医科室指导；不做疾病确诊、不开处方药；出现高烧不退、胸痛、出血等重症，建议线下医院就诊，回答仅作科普参考。"},
    {"role": "user", "content": "成人低烧37.5度怎么处理？"},
    {"role": "assistant", "content": "37.5℃属于低热，多喝温水、减少厚衣物，保证休息；若伴随咽痛、浑身酸痛持续3天以上，可前往内科检查。本内容不能替代执业医师诊断。"}
]}
```

## 三、模块1：LoRA微调 Python代码 `code/train_lora.py`
4bit量化加载基座，仅训练LoRA低秩参数，节省显存
运行微调：
```bash
python code/train_lora.py
```
## 四、模块2： LoRA模型本地对话测试代码
挂载训练完成的LoRA增量权重，实现医疗问答本地推理；
运行测试
```bash
python code/test.py
```

   
## 五、模块3：LoRA合并完整HF模型 Python代码 `code/merge_lora_to_full.py`
LoRA仅为增量权重，必须合并基座得到完整模型，才能转换GGUF格式
运行合并：
```bash
python code/merge_lora_to_full.py
```

## 六、模块4：HF模型转GGUF FP16 `deploy/convert_hf_gguf.py`
调用llama.cpp官方转换脚本，输出未量化原始GGUF文件
运行转换：
```bash
python deploy/convert_hf_gguf.py
```

## 七、模块5：GGUF Q4_K_M量化 Python调用代码 `deploy/quantize_model.py`
调用编译好的 `llama-quantize` 二进制工具，生成低显存轻量化模型
运行量化：
```bash
python deploy/quantize_model.py
```
### 云端打包下载量化模型
```bash
tar -zcvf qwen2.5-7b-medical-q4km.tar.gz qwen2.5-7b-medical-q4_K_M.gguf
```

## 八、本地Ollama离线部署
### 8.1 Modelfile（deploy/Modelfile）
```Modelfile
FROM ./qwen2.5-7b-medical-q4_K_M.gguf

# 医疗助手固定系统提示词
SYSTEM """你是专业医疗科普助手，仅提供日常轻症护理、就诊科室推荐；
禁止判断疾病、开具药物、给出诊疗方案；高烧、胸痛、外伤出血等情况统一引导前往线下医院；
回答通俗易懂，不提及模型、厂商相关信息。"""

```
### 8.2 本地终端部署命令
1. 将压缩包解压，把GGUF文件与Modelfile放入同一文件夹
2. 终端进入目录执行：
```bash
# 创建本地模型
ollama create medical-qwen7b -f Modelfile
# 启动离线对话
ollama run medical-qwen7b
```


## 九、许可证与免责声明
1. 基座Qwen2.5-7B-Instruct遵循通义千问开源协议；llama.cpp工具遵循MIT协议；
2. **重要免责**：本模型仅用于学术医疗科普学习，严禁用于线上问诊、临床诊断、商业医疗服务，任何身体不适请咨询执业医师。

## 十、拓展优化方向
1. 更换Q5_K_M量化档位，提升模型问答精度（模型体积约5GB）；
2. 封装llama.cpp本地API接口，搭建Web对话页面；
3. 扩充细分科室医疗对话数据集，优化专科问答效果。
