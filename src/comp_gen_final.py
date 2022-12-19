import transformers
import torch
import numpy as np
import nltk
nltk.download('punkt')

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoConfig, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Config
from datasets import load_dataset
from evaluate import load

# add seed lib
from transformers import set_seed
import os 
import pandas as pd

# preprocssing. this encodes command and actions 
def preprocess_function(examples):

    # note that you need to add padding 
    # model_inputs = tokenizer(examples["commands"], max_length=tokenizer.model_max_length, padding="longest", truncation=True, return_tensors='pt')
    # model_inputs = tokenizer(examples["question"], max_length=tokenizer.model_max_length, padding="longest", truncation=True, return_tensors='pt')
    # model_inputs = tokenizer(examples["source"], max_length=tokenizer.model_max_length, padding="longest", truncation=True, return_tensors='pt')
    # model_inputs = tokenizer(examples["user_utterance"], max_length=tokenizer.model_max_length, padding="longest", truncation=True, return_tensors='pt')
    model_inputs = tokenizer(examples["input"], max_length=tokenizer.model_max_length, padding="longest", truncation=True, return_tensors='pt')
    
    # Setup the tokenizer for targets
    # labels = tokenizer(examples["actions"], max_length=tokenizer.model_max_length, padding="longest", truncation=True, return_tensors='pt')
    # labels = tokenizer(examples["query"], max_length=tokenizer.model_max_length, padding="longest", truncation=True, return_tensors='pt')
    # labels = tokenizer(examples["target"], max_length=tokenizer.model_max_length, padding="longest", truncation=True, return_tensors='pt')
    # labels = tokenizer(examples["lispress"], max_length=tokenizer.model_max_length, padding="longest", truncation=True, return_tensors='pt')
    labels = tokenizer(examples["output"], max_length=tokenizer.model_max_length, padding="longest", truncation=True, return_tensors='pt')

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# training function, returns the model (which is trainer)
def model_train():
    # define args for finetuning
    batch_size = 128 # change this to bigger number such as 128 
    args = Seq2SeqTrainingArguments(
        f"{model_name}-pcfgset-1121",
        evaluation_strategy = "epoch",
        learning_rate=5e-5, #5e-5
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        weight_decay=0.01,
        save_total_limit=3,
        num_train_epochs=1, # or 50
        predict_with_generate=True,
        disable_tqdm = False, # False 
        fp16 = True,
        # push_to_hub=True,
        report_to="tensorboard", # after running, use 'tensorboard dev upload --logdir {FILE_PATH}'
    )

    # collator fixed of max length 
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    # assign trainer function
    trainer = Seq2SeqTrainer(
        model,
        args,
        train_dataset=tokenized_datasets["train"],
        # eval_dataset=tokenized_datasets["validation"],
        eval_dataset=tokenized_datasets["dev"],
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    
    # train starts 
    trainer.train()

    return trainer    

# decoding function, this converts list of intergers into sentence 
def decode_data(prediction, label):
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    
    # Replace -100 in the labels as we can't decode them.
    labels = np.where(label_ids != -100, label_ids, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # We convert back into the sentence
    decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
    decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]

    return decoded_preds, decoded_labels

# evaluate function: this calculates exact match score 
def evaluate_exact_match(decoded_preds, decoded_labels):
    
    # load EM
    exact_match_metric = load("exact_match")

    # calculate EM result 
    match = exact_match_metric.compute(predictions=decoded_preds, references=decoded_labels)['exact_match']
    print('########################')
    print('Exact match: ', match)
    print()

    return match

def cogs():
    # this is the path where gen is divided into gen_structural and gen_lexical
    path1 = '/scratch/yp2201/comp_gen/COGS/data/gen_structural.tsv'
    path2 = '/scratch/yp2201/comp_gen/COGS/data/gen_lexical.tsv'
    
    # if you don't have gen sepearted files, conduct as below to generate files 
    if not (os.path.exists(path1) and os.path.exists(path2)):
        # first, divide gen 
        COGS_GEN_PATH = '/scratch/yp2201/comp_gen/COGS/data/gen.tsv'
        gen = pd.read_csv(COGS_GEN_PATH, sep='\t', names=['source', 'target', 'generalization_type'])
        
        # Treat rows that are labeled cp_recursion, pp_recursion, and obj_pp_to_subj_pp as gen-structural and anything else, gen-lexical
        # structural: cp_recursion, pp_recursion, and obj_pp_to_subj_pp
        cond1 = gen['generalization_type'] == 'cp_recursion'
        cond2 = gen['generalization_type'] == 'pp_recursion'
        cond3 = gen['generalization_type'] == 'obj_pp_to_subj_pp'

        # assign gen_structural and gen_lexical 
        gen_structural = gen[cond1 | cond2 | cond3]
        gen_lexical = gen[~(cond1 | cond2 | cond3)]

        # export as seperate tsv files 
        gen_structural.to_csv('/scratch/yp2201/comp_gen/COGS/data/gen_structural.tsv', index=False, header=False, sep='\t')
        gen_lexical.to_csv('/scratch/yp2201/comp_gen/COGS/data/gen_lexical.tsv', index=False, header=False, sep='\t')
          
    # assuming that you have gen seperated data, run below codes 
    COGS_PATH = '/scratch/yp2201/comp_gen/COGS/data'
    
    data_splits = ['train', 'dev', 'test', 'gen_lexical', 'gen_structural']
    data_files = {}
    for data_split in data_splits:
        data_files[data_split] = os.path.join(COGS_PATH, f'{data_split}.tsv')

    raw_datasets = load_dataset('csv', data_files=data_files, column_names=['source', 'target', 'generalization_type'], sep='\t')
    
    return raw_datasets


def pcfg():
    # this is the path where gen is divided into gen_structural and gen_lexical
    path1 = '/scratch/yp2201/comp_gen/am-i-compositional/data/pcfgset/pcfgset/pcfgset_train.tsv'
    path3 = '/scratch/yp2201/comp_gen/am-i-compositional/data/pcfgset/pcfgset/pcfgset_dev.tsv'
    path3 = '/scratch/yp2201/comp_gen/am-i-compositional/data/pcfgset/pcfgset/pcfgset_test.tsv'
    
    # if you don't have gen sepearted files, conduct as below to generate files 
    if not (os.path.exists(path1) and os.path.exists(path2) and os.path.exists(path3)):
        ## train set
        # first, assign path
        train_input_path = '/scratch/yp2201/comp_gen/am-i-compositional/data/pcfgset/pcfgset/train.src'
        train_output_path = '/scratch/yp2201/comp_gen/am-i-compositional/data/pcfgset/pcfgset/train.tgt'
        # import input/output
        train_input = pd.read_csv(train_input_path, names=['input'], sep='/n', engine='python')
        train_output = pd.read_csv(train_output_path, names=['output'], sep='/n', engine='python')
        # concat
        train_df = pd.concat([train_input, train_output], axis=1)
        # export as seperate tsv files 
        train_df.to_csv('/scratch/yp2201/comp_gen/am-i-compositional/data/pcfgset/pcfgset/pcfgset_train.tsv', index=False, header=False, sep='\t')
        
        ## dev set
        # first, assign path
        dev_input_path = '/scratch/yp2201/comp_gen/am-i-compositional/data/pcfgset/pcfgset/dev.src'
        dev_output_path = '/scratch/yp2201/comp_gen/am-i-compositional/data/pcfgset/pcfgset/dev.tgt'
        # import input/output
        dev_input = pd.read_csv(dev_input_path, names=['input'], sep='/n', engine='python')
        dev_output = pd.read_csv(dev_output_path, names=['output'], sep='/n', engine='python')
        # concat
        dev_df = pd.concat([dev_input, dev_output], axis=1)
        # export as seperate tsv files 
        dev_df.to_csv('/scratch/yp2201/comp_gen/am-i-compositional/data/pcfgset/pcfgset/pcfgset_dev.tsv', index=False, header=False, sep='\t')
        
        ## test set
        # first, assign path
        test_input_path = '/scratch/yp2201/comp_gen/am-i-compositional/data/pcfgset/pcfgset/test.src'
        test_output_path = '/scratch/yp2201/comp_gen/am-i-compositional/data/pcfgset/pcfgset/test.tgt'
        # import input/output
        test_input = pd.read_csv(test_input_path, names=['input'], sep='/n', engine='python')
        test_output = pd.read_csv(test_output_path, names=['output'], sep='/n', engine='python')
        # concat
        test_df = pd.concat([test_input, test_output], axis=1)
        # export as seperate tsv files 
        test_df.to_csv('/scratch/yp2201/comp_gen/am-i-compositional/data/pcfgset/pcfgset/pcfgset_test.tsv', index=False, header=False, sep='\t')        

    # assuming that you have gen seperated data, run below codes 
    PCFG_PATH = '/scratch/yp2201/comp_gen/am-i-compositional/data/pcfgset/pcfgset'
    
    data_splits = ['train', 'dev', 'test']
    data_files = {}
    for data_split in data_splits:
        data_files[data_split] = os.path.join(PCFG_PATH, f'pcfgset_{data_split}.tsv')

    raw_datasets = load_dataset('csv', data_files=data_files, column_names=['input', 'output'], sep='\t')
    
    return raw_datasets


if __name__ == "__main__":
           
    ## SEED
    seeds = [0] # change here, you can put a list of seeds or just a seed 
    
    # iterate list of seeds 
    for seed in seeds:
        
        # random seed will be assigned here 
        RANDOM_SEED = seed
        print('random_seed: ', RANDOM_SEED)
        set_seed(RANDOM_SEED)
        
        # dataset 
        # raw_datasets = load_dataset("scan", "simple") # load scan-simple
        # raw_datasets = load_dataset("scan", "addprim_jump")
        # raw_datasets = load_dataset("scan", "addprim_turn_left")
        # raw_datasets = load_dataset("scan", "length")
        # raw_datasets = load_dataset("cfq", "mcd1")
        # raw_datasets = cogs()
        # raw_datasets = load_dataset("iohadrubin/smcalflow") # smcalflow
        raw_datasets = pcfg() # pcfg
        print(raw_datasets)
        
        # assign your model name
        model_name = "t5-small" # t5-small
        # model_name = "t5-base" # t5-base

        # if you need to use pretrained checkpoints, change this code to True
        pretrained = False
        
        # use pretrained tokenizer and model
        tokenizer = T5Tokenizer.from_pretrained(model_name)
        
        # Download configuration from huggingface.co and cache.
        config = T5Config.from_pretrained(model_name)
        
        # this is the case when you are using pretrained checkpoints (including huggingface settings)
        if pretrained:
            # define model based on the config
            model = T5ForConditionalGeneration.from_pretrained(model_name, config=config)
        
        # this applies to most cases that we are testing, cases are dependant to the random seed         
        else: 
            # define the model not from pretrained. This will be fixed by random seed 
            model = T5ForConditionalGeneration(config=config)
            
        # preprocess dataset
        tokenized_datasets = raw_datasets.map(preprocess_function, batched=True)  
        
        # train 
        trainer = model_train()

        # predict test set 
        predictions, label_ids, _ = trainer.predict(tokenized_datasets["test"], max_length=tokenizer.model_max_length) # 512
        # predictions, label_ids, _ = trainer.predict(tokenized_datasets["gen_lexical"], max_length=tokenizer.model_max_length) # 512
        # predictions, label_ids, _ = trainer.predict(tokenized_datasets["gen_structural"], max_length=tokenizer.model_max_length) # 512
        # predictions, label_ids, _ = trainer.predict(tokenized_datasets["validation"], max_length=tokenizer.model_max_length) # 512
        
        # decode results 
        decoded_preds, decoded_labels = decode_data(predictions, label_ids)  

        # evaluate exact match result 
        match = evaluate_exact_match(decoded_preds, decoded_labels)