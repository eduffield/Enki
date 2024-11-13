import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq
from datasets import Dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

csv_file = 'babylonian_english.csv'  # Replace with your CSV file path
data = pd.read_csv(csv_file)
data = data.rename(columns={data.columns[0]: "babylonian", data.columns[1]: "english"})

model_name = "t5-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def preprocess_function(examples):
    inputs = [f"translate Babylonian to English: {ex}" for ex in examples["babylonian"]]
    targets = [ex for ex in examples["english"]]
    model_inputs = tokenizer(inputs, max_length=128, truncation=True)
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=128, truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

dataset = Dataset.from_pandas(data)
tokenized_dataset = dataset.map(preprocess_function, batched=True)

train_test_split = tokenized_dataset.train_test_split(test_size=0.1)
train_dataset = train_test_split["train"]
val_dataset = train_test_split["test"]

model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
checkpoint_path = "./results/checkpoint-11500"

training_args = Seq2SeqTrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=3e-4,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=6,
    predict_with_generate=True,
    fp16=True,
    resume_from_checkpoint=checkpoint_path  # Resuming from checkpoint
)

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

trainer.train(resume_from_checkpoint=checkpoint_path)

def translate_babylonian(text):
    inputs = tokenizer(f"translate Babylonian to English: {text}", return_tensors="pt").to(device)
    outputs = model.generate(**inputs)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Example usage
babylonian_text = "a du10 hu-mu-ne-nag"
english_translation = translate_babylonian(babylonian_text)
print(f"English translation: {english_translation}")