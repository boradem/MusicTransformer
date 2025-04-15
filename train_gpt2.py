import torch
import pickle
from transformers import GPT2LMHeadModel, GPT2Tokenizer, TrainingArguments, Trainer
from datasets import Dataset

# Check if is available
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
print(f"Using device: {device}")

# Load GPT-2 Tokenizer & Model
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Fix: Add a padding token
tokenizer.pad_token = tokenizer.eos_token

model = GPT2LMHeadModel.from_pretrained("gpt2")

# Resize embedding layer to fit our custom token vocabulary
model.resize_token_embeddings(len(tokenizer))


# Move model to Metal GPU
model.to(device)

# Load the tokenized dataset
with open("midi_token_dataset.pkl", "rb") as f:
    tokenized_data = pickle.load(f)

from transformers import DataCollatorForLanguageModeling

# Define maximum sequence length
MAX_LENGTH = 512

# Pad/truncate sequences to ensure uniform input size
def pad_or_truncate(examples):
    return {"input_ids": [seq[:MAX_LENGTH] + [0] * (MAX_LENGTH - len(seq)) for seq in examples["input_ids"]]}

# Convert to Hugging Face Dataset format
dataset = Dataset.from_dict({"input_ids": tokenized_data}).map(pad_or_truncate, batched=True)


data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False  # We don't need masked language modeling for music
)

# Training Configuration
training_args = TrainingArguments(
    output_dir="./gpt2-music",
    evaluation_strategy="no",
    save_strategy="epoch",
    logging_dir="./logs",
    per_device_train_batch_size=2,  # Small batch size for Mac's memory
    per_device_eval_batch_size=2,
    num_train_epochs=5,  # Adjust based on performance
    learning_rate=5e-5,
    weight_decay=0.01,
    save_total_limit=2,
    push_to_hub=False,
    report_to="none",
)

# Define Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=data_collator,  # ✅ Make sure we add this now
)

# Model Training
trainer.train()
