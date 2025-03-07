from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import Dataset
import torch

model_name = "facebook/blenderbot-400M-distill"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)  # BlenderBot is seq2seq, so use AutoModelForSeq2SeqLM

# Loading training data (queries and expected responses)
train_texts = ["Hello", "Track my order", "Goodbye"]
train_labels = ["Hello!", "Your order is being processed.", "Goodbye!"]

# Tokenizing inputs and labels
max_length = 32  # Ensure all sequences are the same length

inputs = tokenizer(train_texts, padding="max_length", truncation=True, max_length=max_length, return_tensors="pt")
labels = tokenizer(train_labels, padding="max_length", truncation=True, max_length=max_length, return_tensors="pt")["input_ids"]

# Replace padding token in labels with -100 to ignore in loss computation
labels[labels == tokenizer.pad_token_id] = -100

# Creating dataset
dataset = Dataset.from_dict({
    "input_ids": inputs["input_ids"],
    "attention_mask": inputs["attention_mask"],  # Include attention mask
    "labels": labels
})

# Training arguments
training_args = TrainingArguments(
    output_dir="./gpt_finetuned",
    evaluation_strategy="no",  # Disable evaluation
    save_strategy="epoch",
    per_device_train_batch_size=2,
    num_train_epochs=3
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset
)

#trainer.train()

# Train the model
trainer.train()

# Explicitly print to confirm saving process
print("Saving the fine-tuned model...")

# Ensure the model and tokenizer are saved
trainer.save_model("fine_tuned_gpt")
tokenizer.save_pretrained("fine_tuned_gpt")

print("Model saved successfully!")

model.save_pretrained("fine_tuned_gpt", save_config=True)
