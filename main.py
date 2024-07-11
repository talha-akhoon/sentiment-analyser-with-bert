from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, \
    DataCollatorWithPadding
from huggingface_hub import notebook_login

# Load dataset
dataset = load_dataset('imdb')

# Load tokenizer and model
model_name = 'bert-base-uncased'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)


# Tokenize data
def tokenize_function(examples):
    return tokenizer(examples['text'], padding='max_length', truncation=True)


tokenized_datasets = dataset.map(tokenize_function, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
tokenized_datasets = tokenized_datasets.remove_columns(['text'])
tokenized_datasets = tokenized_datasets.rename_column('label', 'labels')
tokenized_datasets.set_format('torch')

# Training arguments
training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy='epoch',
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=1,
    weight_decay=0.01,
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['test'],
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# Train and evaluate
trainer.train()
results = trainer.evaluate()
# Define the label mapping
label_map = {
    0: 'negative',
    1: 'positive'
}

# Print human-readable evaluation results
mapped_results = {label_map[k]: v for k, v in results.items() if k in label_map}
print(mapped_results)

# Push to Hugging Face Hub
notebook_login()
trainer.push_to_hub()
