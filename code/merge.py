import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE_PATH = "/root/autodl-tmp/qwen_model/qwen/Qwen2___5-7B-Instruct"
LORA_PATH = "/root/autodl-tmp/qwen_medical_lora_output/final_best_lora"
SAVE_MERGED = "./qwen2.5-7b-medical-full"

tokenizer = AutoTokenizer.from_pretrained(BASE_PATH)
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_PATH,
    torch_dtype=torch.float16,
    device_map="auto"
)
# 加载LoRA适配器并合并
model = PeftModel.from_pretrained(base_model, LORA_PATH)
merged_model = model.merge_and_unload()
# 保存完整合并模型
merged_model.save_pretrained(SAVE_MERGED)
tokenizer.save_pretrained(SAVE_MERGED)