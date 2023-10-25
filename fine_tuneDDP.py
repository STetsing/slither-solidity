import os, argparse
import numpy as np
import datasets
import evaluate
import math
import torch
import pandas as pd
import torch.distributed as dist
from accelerate import Accelerator

from tqdm import tqdm
from torch.optim import AdamW
from torch.utils.data import DataLoader
from datasets import Dataset
from datasets import load_metric
from transformers import AutoTokenizer, RobertaTokenizer
from transformers import AutoModelForCausalLM, Trainer, TrainerCallback, TrainingArguments, T5ForConditionalGeneration
from transformers import DataCollatorForLanguageModeling, get_scheduler
from slither_sol_helpers import get_sol_data
from torch.nn.parallel import DistributedDataParallel as DDP
torch.cuda.empty_cache()

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=32, help="The training batch size")
parser.add_argument('--learning_rate', type=float, default=1e-5, help="The training learning rate")
parser.add_argument('--weight_decay', type=float, default=1e-2, help="Weight decay param")
parser.add_argument('--num_train_epochs', type=int, default=100, help="Training epochs")

args = parser.parse_args()
accelerator = Accelerator()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
block_size = 128
context_length = 32
base_model = '../solidity-generator'

training_args = TrainingArguments('test_trainer', 
        evaluation_strategy="epoch", 
        learning_rate=args.learning_rate, 
        per_device_eval_batch_size=args.batch_size,
        per_device_train_batch_size=32,
        num_train_epochs=args.num_train_epochs,
        push_to_hub=False,
        save_total_limit=2,
        load_best_model_at_end=True,
        save_strategy = "epoch",
        prediction_loss_only=True,
        logging_strategy="steps",
        logging_steps=50,
        seed=100)

# raw_sol_data = load_dataset('mwritescode/slither-audited-smart-contracts', 'all-plain-text')

raw_sol_data = Dataset.from_pandas(pd.read_pickle('./slither_processed_contracts.pkl').sample(50))
#tokenizer = AutoTokenizer.from_pretrained(base_model)
tokenizer = AutoTokenizer.from_pretrained(base_model)
tokenizer.pad_token = tokenizer.eos_token

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False) # shift the input by one to the right to get the labels

def setup(rank, world_size):
    "Sets up the process group and configuration for PyTorch Distributed Data Parallelism"
    os.environ["MASTER_ADDR"] = 'localhost'
    os.environ["MASTER_PORT"] = "12355"

    # Initialize the process group
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)

def cleanup():
    "Cleans up the distributed environment"
    dist.destroy_process_group()


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

full_train_dataset = full_train_dataset.map(tokenize_function, batched=True, remove_columns=full_eval_dataset.column_names).map(group_texts, batched=True)
full_eval_dataset = full_eval_dataset.map(tokenize_function, batched=True, remove_columns=full_eval_dataset.column_names).map(group_texts, batched=True)

full_train_dataset.set_format('torch')
full_eval_dataset.set_format('torch')
train_loader = DataLoader(full_train_dataset, shuffle=True, collate_fn=data_collator, batch_size=args.batch_size)
eval_loader = DataLoader(full_eval_dataset, shuffle=True, collate_fn=data_collator, batch_size=args.batch_size)

model = AutoModelForCausalLM.from_pretrained(base_model)
optimizer = AdamW(model.parameters(), lr=args.learning_rate)
max_train_steps = args.num_train_epochs * len(train_loader)
metric = load_metric("accuracy")
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=args.num_training_steps
)

# traditional train loop
pbar = tqdm(range(max_train_steps))
for epoch in tqdm(range(args.num_train_epochs)):
    model.train()
    for batch in train_loader:
        batch = {k: v.to(device) for k,v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        pbar.update(1)

    model.eval()
    for batch in eval_loader:
        batch = {k: v.to(device) for k,v in batch.items()}
        outputs = model(**batch)
        preds = outputs.logits.argmax(dim=-1)
        print(batch['input_ids'].shape, batch['labels'].shape)
        metric.add_batch(predictions=preds, references=batch['labels'])
    
    eval_metric = metric.compute()
    print(f"epoch {epoch}: {eval_metric}")


# metric = evaluate.load("accuracy")

# def compute_metrics(eval_pred):
#     logits, labels = eval_pred
#     predictions = np.argmax(logits, axis=-1)
#     return metric.compute(predictions=predictions, references=labels)

# class PerplexCallback(TrainerCallback):
#     "A callback that computes the model perplexity"

#     def on_epoch_end(self, args, state, control, **kwargs):
#         eval_results = trainer.evaluate()
#         print(f"\nModel Perplexity: {math.exp(eval_results['eval_loss']):.2f}")


# trainer = Trainer(
#     model=model, 
#     tokenizer=tokenizer,
#     args=training_args, 
#     train_dataset=full_train_dataset, 
#     eval_dataset=full_eval_dataset, 
#     callbacks=[PerplexCallback],
#     #compute_metrics=compute_metrics,
#     data_collator=data_collator # very important, does the label shifting by 1
# )
# trainer.train()

# tokenizer.save_pretrained('./solidity_gpt2')
# trainer.save_model('./solidity_gpt2')
