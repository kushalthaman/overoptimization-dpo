import torch
import csv
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
import os

# Load tokenizer and model for reward calculation
gold_tokenizer = AutoTokenizer.from_pretrained("sileod/deberta-v3-large-tasksource-rlhf-reward-model")
gold_reward = AutoModelForSequenceClassification.from_pretrained("sileod/deberta-v3-large-tasksource-rlhf-reward-model")

# Load dataset
preference_dataset = load_dataset("tatsu-lab/alpaca_farm", "alpaca_human_preference")["preference"]

def create_columns(example):
    chosen = example['output_1'] if example['preference'] == 1 else example['output_2']
    rejected = example['output_2'] if example['preference'] == 1 else example['output_1']
    prompt = example['instruction'] + " " + example['input']
    return {'prompt': prompt, 'chosen': chosen, 'rejected': rejected}

preference_dataset = preference_dataset.map(create_columns)
preference_dataset = preference_dataset.remove_columns(['instruction', 'input', 'output_1', 'output_2', 'preference', 'raw_preference'])
train_test_split = preference_dataset.train_test_split(test_size=0.2)

# Prepare tokenizer and models
model_name = "EleutherAI/pythia-70m-deduped"
dpo_models_path = 'dpo_models'
dpo_tokenizer = AutoTokenizer.from_pretrained(model_name, revision="step3000", cache_dir="./pythia-70m-deduped/step3000")
models = [AutoModelForCausalLM.from_pretrained(os.path.join(dpo_models_path, model_folder)) for model_folder in os.listdir(dpo_models_path) if os.path.isdir(os.path.join(dpo_models_path, model_folder))]

def inference(dpo_model, dpo_tokenizer, prompt):
    inputs = dpo_tokenizer(prompt, return_tensors="pt")
    tokens = dpo_model.generate(**inputs)
    completion = dpo_tokenizer.decode(tokens[0])
    return completion

# Evaluate models
winners = []
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
gold_reward = gold_reward.to(device)

for example in train_test_split["test"]:
    with torch.no_grad():
        completions = [inference(model, dpo_tokenizer, example["prompt"]) for model in models]
        cur_max_reward, model_num = -float('inf'), -1
        for cur_iter, completion in enumerate(completions):
            text = f"{example['prompt']}\n{completion}"
            reward_input = gold_tokenizer(text, return_tensors="pt").to(device)
            reward = gold_reward(**reward_input).logits[0].item()
            if reward > cur_max_reward:
                model_num, cur_max_reward = cur_iter, reward
        winners.append(model_num)

csv_file_path = 'winners.csv'

# Writing the winners list to a CSV file
with open(csv_file_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Model Number'])  # Writing the header
    for winner in winners:
        writer.writerow([winner])  # Writing each winner

print(f"Winners saved to {csv_file_path}")
