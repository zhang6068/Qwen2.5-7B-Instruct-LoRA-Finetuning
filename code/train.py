import os
import torch
import json
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    TrainerCallback
)
from peft import LoraConfig, get_peft_model, TaskType

# ===================== 全局路径配置（AutoDL高速数据盘） =====================
# 本地离线Qwen2.5-7B基座
MODEL_PATH = "/root/autodl-tmp/qwen_model/qwen/Qwen2___5-7B-Instruct"
# 数据集文件
TRAIN_JSONL = "/root/autodl-tmp/train.jsonl"
VAL_JSONL = "/root/autodl-tmp/val.jsonl"
# LoRA权重输出目录
OUTPUT_DIR = "/root/autodl-tmp/qwen_medical_lora_output"
# 模型缓存目录
HF_CACHE = "/root/autodl-tmp/huggingface_cache"
# 上下文最大长度（缩短提速，不丢失核心对话）
MAX_SEQ_LEN = 768

# ===================== 4bit QLoRA量化配置 =====================
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

# ===================== LoRA微调配置（加大dropout防过拟合） =====================
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj"],
    inference_mode=False,
    r=16,
    lora_alpha=32,
    lora_dropout=0.12
)

# ===================== 读取messages格式jsonl =====================
def load_messages_jsonl(file_path):
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            data.append({"messages": item["messages"]})
    return Dataset.from_list(data)

# ===================== 单样本预处理（Qwen标准SFT掩码逻辑） =====================
def process_sample(sample, tokenizer):
    messages = sample["messages"]
    full_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    split_tag = "<|im_start|>assistant\n"
    prompt_part, answer_part = full_text.split(split_tag)
    prompt_full = prompt_part + split_tag

    prompt_enc = tokenizer(prompt_full, add_special_tokens=False)
    full_enc = tokenizer(full_text, add_special_tokens=False)

    input_ids = full_enc["input_ids"]
    attention_mask = full_enc["attention_mask"]
    # system/user部分标签置-100，仅计算assistant损失
    labels = [-100] * len(prompt_enc["input_ids"]) + full_enc["input_ids"][len(prompt_enc["input_ids"]):]

    # 超长截断
    if len(input_ids) > MAX_SEQ_LEN:
        input_ids = input_ids[:MAX_SEQ_LEN]
        attention_mask = attention_mask[:MAX_SEQ_LEN]
        labels = labels[:MAX_SEQ_LEN]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }

# ===================== Epoch结束日志回调，醒目打印本轮loss =====================
class EpochSummaryCallback(TrainerCallback):
    def on_epoch_end(self, args, state, control, logs=None, **kwargs):
        if logs is not None:
            print("\n" + "="*70)
            print(f"✅ EPOCH 完成：{state.epoch:.2f} / {args.num_train_epochs}")
            print(f"📉 本轮训练loss: {logs.get('loss', 0):.5f}")
            print(f"📊 本轮验证eval_loss: {logs.get('eval_loss', 0):.5f}")
            print("="*70 + "\n")

if __name__ == "__main__":
    torch.manual_seed(42)
    os.environ["HF_HOME"] = HF_CACHE

    # 1. 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        cache_dir=HF_CACHE,
        use_fast=False
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # 2. 加载数据集 + 裁剪样本（控制时长，保留7万训练/8千验证）
    train_ds_raw = load_messages_jsonl(TRAIN_JSONL)
    train_ds_raw = train_ds_raw.select(range(70000))  # 裁剪训练集
    val_ds_raw = load_messages_jsonl(VAL_JSONL)
    val_ds_raw = val_ds_raw.select(range(8000))        # 裁剪验证集

    # 批量预处理编码
    def map_func(x):
        return process_sample(x, tokenizer)
    train_ds = train_ds_raw.map(map_func, remove_columns=["messages"])
    val_ds = val_ds_raw.map(map_func, remove_columns=["messages"])
    print(f"训练集 {len(train_ds)} 条，验证集 {len(val_ds)} 条")

    # 3. 4bit加载基座模型，注入LoRA
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        cache_dir=HF_CACHE,
        torch_dtype=torch.bfloat16,
        use_cache=False
    )
    model.gradient_checkpointing_enable()
    model = get_peft_model(model, lora_config)
    model.enable_input_require_grads()
    model.print_trainable_parameters()

    # 4. 训练超参（提速核心配置）
    train_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        learning_rate=1.5e-4,
        num_train_epochs=1,
        bf16=True,
        logging_steps=10,
        evaluation_strategy="steps",
        eval_steps=500,
        save_strategy="steps",
        save_steps=500,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        optim="paged_adamw_8bit",
        report_to="none",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        gradient_checkpointing=True,
        save_total_limit=2,
        dataloader_pin_memory=False
    )

    # 数据填充器
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True)

    # 5. 初始化Trainer，挂载日志回调
    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=data_collator,
        callbacks=[EpochSummaryCallback()]
    )

    print("===== 开始QLoRA微调 =====")
    # 断电续训用法：trainer.train(resume_from_checkpoint=f"{OUTPUT_DIR}/checkpoint-xxxx")
    trainer.train()

    # 保存全程最优LoRA权重
    best_lora_path = os.path.join(OUTPUT_DIR, "final_best_lora")
    trainer.save_model(best_lora_path)
    print(f"训练完成，最优LoRA权重保存在：{best_lora_path}")
