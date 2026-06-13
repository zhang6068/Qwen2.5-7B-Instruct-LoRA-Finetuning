import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

MODEL_PATH = "/root/autodl-tmp/qwen_model/qwen/Qwen2___5-7B-Instruct"
LORA_PATH = "/root/autodl-tmp/qwen_medical_lora_output/final_best_lora"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    load_in_4bit=True,
    device_map="auto",
    trust_remote_code=True
)
# 加载微调后的LoRA
model = PeftModel.from_pretrained(base_model, LORA_PATH)

def chat(prompt):
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to("cuda")
    outputs = model.generate(
        **inputs,
        max_new_tokens=512,
        temperature=0.7,
        top_p=0.8
    )
    resp = tokenizer.decode(outputs[0][len(inputs["input_ids"][0]):], skip_special_tokens=True)
    return resp

# 测试多条你业务内的问题
if __name__ == "__main__":
    print(chat("我今年58岁，有高血压3年了，一直在吃硝苯地平控释片，早上空腹吃，血压控制在130/85左右。但最近早上起床时偶尔会觉得头晕，测血压有时候偏高到150/95，是不是药物效果不好了？需要换药吗？"))
    print(chat("医生，我这两天一直打喷嚏、流清鼻涕，还有点喉咙痒，没有发烧，是不是普通感冒？需要吃药吗？"))