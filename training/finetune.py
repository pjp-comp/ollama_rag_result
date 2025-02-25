import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
from datasets import load_dataset

# Load dataset
dataset = load_dataset("json", data_files="your_data.json")

# Load DeepSeek-R1 model from Ollama
model_name = "deepseek-r1"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")

# Apply LoRA Configuration
lora_config = LoraConfig(
    r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"], lora_dropout=0.1, bias="none"
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# Define Training Arguments
training_args = TrainingArguments(
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    learning_rate=2e-4,
    output_dir="./fine_tuned_deepseek_r1",
    save_strategy="epoch",
    push_to_hub=False
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
)
trainer.train()

# Save fine-tuned model
model.save_pretrained("fine_tuned_deepseek_r1")
tokenizer.save_pretrained("fine_tuned_deepseek_r1")

print("✅ Fine-tuning complete. Saved model in 'fine_tuned_deepseek_r1'.")
