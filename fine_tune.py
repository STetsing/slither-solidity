import os 
import numpy as np
import datasets
import evaluate
import math
import torch
import pandas as pd
import yaml
from datasets import Dataset
from datasets import load_metric, load_from_disk, load_dataset
from transformers import AutoTokenizer, RobertaTokenizer
from transformers import AutoModelForCausalLM, Trainer, TrainerCallback, TrainingArguments, T5ForConditionalGeneration
from transformers import DataCollatorForLanguageModeling
from slither_sol_helpers import get_sol_data
from accelerate import Accelerator
from slither_sol_helpers import get_error_or_warning_codes

with open('./config.yml', 'r') as fh:
    config = yaml.safe_load(fh)

#device = 'cuda'
device = Accelerator.device
print('Info: Computing device is:', device)
torch.cuda.empty_cache()

block_size = config["training"]["block_size"]
context_length = block_size
#base_model = 'Pipper/finetuned_sol'
base_model = 'ckandemir/solidity-generator'
dataset_repo = "Pipper/solidity"
accelerator = Accelerator()
process_local = True

training_args = TrainingArguments('test_trainer', 
        evaluation_strategy=config["training"]["eval_strategy"], 
        learning_rate=config["training"]["learning_rate"], 
        per_device_eval_batch_size=config["training"]["batch_size"],
        per_device_train_batch_size=config["training"]["batch_size"],
        num_train_epochs=config["training"]["epochs"],
        push_to_hub=False,
        save_total_limit=2,
        load_best_model_at_end=True,
        save_strategy = "epoch",
        prediction_loss_only=True,
        logging_strategy="steps",
        logging_steps=config["training"]["log_steps"],
        seed=100)

tokenizer = AutoTokenizer.from_pretrained(base_model)
tokenizer.pad_token = tokenizer.eos_token
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False) # shift the input by one to the right to get the labels

def tokenize_function(data):
    # get rid of comments in source files
    return tokenizer([ " ".join(get_sol_data(sf, config["slither"]['rm_comments'])) for sf in data['sol_file']]) #  truncation=True, max_length=16, return_length=True, return_overflowing_tokens=True)
    


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


if process_local:
    # Read and filter processed files
    slither_dataset = pd.read_pickle('./slither_processed_contracts.pkl')
    slither_dataset.reset_index(drop=True, inplace=True)
    slither_dataset["source_dir"]=slither_dataset["source_dir"].apply(lambda x: x.replace("/home/pippertetsing/sourcify_contract_data/", "/Users/pippertetsing/Desktop/work/Remix/solcoder/"))
    slither_dataset["sol_file"]=slither_dataset["sol_file"].apply(lambda x: x.replace("/home/pippertetsing/sourcify_contract_data/", "/Users/pippertetsing/Desktop/work/Remix/solcoder/"))
    slither_dataset["contracts_dirs"]=slither_dataset["contracts_dirs"].apply(lambda x: x.replace("/home/pippertetsing/sourcify_contract_data/", "/Users/pippertetsing/Desktop/work/Remix/solcoder/"))

    slither_processed = slither_dataset[slither_dataset['slither_processed'] == True]

    # eliminate High risk impact files 
    def keep(x, idx_red_list):
        if x == None:
            return True
        else:
            for x_idx in x:
                if x_idx in idx_red_list:
                    return False
        return True

    print('len before keep ', len(slither_processed))
    slither_processed['keep'] = True
    if config["slither"]['rm_high_idx']:
        print('Info: removing high impact warning codes ...')
        high_idx = get_error_or_warning_codes('High')
        slither_processed['keep'] = slither_processed.slither.apply(lambda x: keep(x, high_idx)) # remove high impact slither warnings

    if config["slither"]['rm_med_idx']:
        print('Info: removing medium impact warning codes ...')
        med_idx = get_error_or_warning_codes('Medium')
        slither_processed['keep'] = slither_processed.slither.apply(lambda x: keep(x, med_idx)) # remove high impact slither warnings

    filtered = slither_processed[slither_processed['keep'] == True] 
    print('len after keep', len(filtered))
    dataset = Dataset.from_pandas(filtered)

    if not os.path.exists('./sol_dataset'):
        print('INFO: Length dataset before tokenization:', len(dataset))
        dataset = dataset.map(tokenize_function, batched=True, remove_columns=dataset.column_names, num_proc=os.cpu_count()).map(group_texts, batched=True,  num_proc=os.cpu_count())
        dataset.save_to_disk('./sol_dataset')
        print('INFO: Length dataset after tokenization:', len(dataset))
    else:
        print('Info: loaded preprocessed set from disk!')
        dataset = load_from_disk('./sol_dataset')
        print('INFO: Length dataset loaded', len(dataset))
else:
    print('INFO: Loading dataset from hugginface ...')
    dataset = load_dataset(dataset_repo)
    print('INFO: Loaded dataset from hugginface')

dataset = dataset.train_test_split(test_size=0.1)

train_set = dataset['train']
eval_set = dataset['test']
print('length train set: ', len(train_set))
print('length test set: ', len(eval_set))

model = AutoModelForCausalLM.from_pretrained(base_model)
#model = model.to(dtype=torch.float16)

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
    train_dataset=train_set, 
    eval_dataset=eval_set, 
    callbacks=[PerplexCallback],
    #compute_metrics=compute_metrics,
    data_collator=data_collator # very important, does the label shifting by 1
)

trainer.train()

tokenizer.save_pretrained('./solidity_gpt2')
trainer.save_model('./solidity_gpt2')
