import numpy as np
import datasets
import evaluate
import math
import torch
import pandas as pd
from datasets import Dataset
from datasets import load_metric
from transformers import AutoTokenizer, RobertaTokenizer
from transformers import AutoModelForCausalLM, Trainer, TrainerCallback, TrainingArguments, T5ForConditionalGeneration
from transformers import DataCollatorForLanguageModeling
from slither_sol_helpers import get_sol_data

torch.cuda.empty_cache()

block_size = 128
context_length = 32
base_model = '../solidity-generator'

training_args = TrainingArguments('test_trainer', 
        evaluation_strategy="epoch", 
        learning_rate=7e-5, 
        per_device_eval_batch_size=32,
        per_device_train_batch_size=32,
        num_train_epochs=10,
        push_to_hub=False,
        save_total_limit=2,
        load_best_model_at_end=True,
        save_strategy = "epoch",
        prediction_loss_only=True,
        logging_strategy="steps",
        logging_steps=50,
        seed=100)

# raw_sol_data = load_dataset('mwritescode/slither-audited-smart-contracts', 'all-plain-text')

raw_sol_data = Dataset.from_pandas(pd.read_pickle('./slither_processed_contracts_sample100.pkl').sample(10))
#tokenizer = AutoTokenizer.from_pretrained(base_model)
tokenizer = AutoTokenizer.from_pretrained(base_model)
tokenizer.pad_token = tokenizer.eos_token

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False) # shift the input by one to the right to get the labels

def tokenize_function(data):
    # get rid of comments in source files
    t = tokenizer([get_sol_data(sf, True)  for sf in data['sol_file']], 
                    max_length=context_length, 
                    padding="max_length",
                  return_overflowing_tokens=True,
                  return_length=True,
                  truncation=True) #  truncation=True, max_length=16, return_length=True, return_overflowing_tokens=True)
    
    input_batch = []
    for l, ids in zip(t['length'], t['input_ids']):
        if l == context_length:
            input_batch.append(ids)

    return {'input_ids': input_batch}


def group_texts(examples):
    # Concatenate all texts.
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
    # customize this part to your needs.
    if total_length >= block_size:
        total_length = (total_length // block_size) * block_size
    # Split by chunks of block_size.
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result

dataset = raw_sol_data.train_test_split(test_size=0.1)
full_train_dataset = dataset["train"]
full_eval_dataset = dataset["test"]

full_train_dataset = full_train_dataset.map(tokenize_function, batched=True, remove_columns=full_eval_dataset.column_names)#.map(group_texts, batched=True)
full_eval_dataset = full_eval_dataset.map(tokenize_function, batched=True, remove_columns=full_eval_dataset.column_names)#.map(group_texts, batched=True)

model = AutoModelForCausalLM.from_pretrained(base_model)

param_size = 0
for param in model.parameters():
    param_size += param.nelement() * param.element_size()
buffer_size = 0
for buffer in model.buffers():
    buffer_size += buffer.nelement() * buffer.element_size()

size_all_mb = (param_size + buffer_size) / 1024**2
print('model size: {:.3f}MB'.format(size_all_mb))


metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

class PerplexCallback(TrainerCallback):
    "A callback that computes the model perplexity"

    def on_epoch_end(self, args, state, control, **kwargs):
        eval_results = trainer.evaluate()
        print(f"\nModel Perplexity: {math.exp(eval_results['eval_loss']):.2f}")


trainer = Trainer(
    model=model, 
    tokenizer=tokenizer,
    args=training_args, 
    train_dataset=full_train_dataset, 
    eval_dataset=full_eval_dataset, 
    callbacks=[PerplexCallback],
    #compute_metrics=compute_metrics,
    data_collator=data_collator # very important, does the label shifting by 1
)

trainer.train()

tokenizer.save_pretrained('./solidity_gpt2')
trainer.save_model('./solidity_gpt2')
