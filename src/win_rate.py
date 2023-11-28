import torch 
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments
from datasets import Dataset, load_dataset
import os 
import re 
import gc
import csv

gold_tokenizer = AutoTokenizer.from_pretrained("sileod/deberta-v3-large-tasksource-rlhf-reward-model")
gold_reward = AutoModelForSequenceClassification.from_pretrained("sileod/deberta-v3-large-tasksource-rlhf-reward-model")
preference_dataset = load_dataset("tatsu-lab/alpaca_farm", "alpaca_human_preference")["preference"]
def create_columns(example):
    if example['preference'] == 1:
        chosen = example['output_1']
        rejected = example['output_2']
    else:  # preference == 2
        chosen = example['output_2']
        rejected = example['output_1']

    prompt = example['instruction'] + " " + example['input']
    return {'prompt': prompt, 'chosen': chosen, 'rejected': rejected}

def inference(dpo_model, dpo_tokenizer, prompt):
    inputs = dpo_tokenizer(prompt, return_tensors="pt")
    tokens = dpo_model.generate(**inputs.to('cuda'))
    completion = dpo_tokenizer.decode(tokens[0])
    return completion

def extract_last_number(path):
    numbers = re.findall(r'\d+', path)
    if numbers:
        return int(numbers[-1])
    else:
        return None
# Apply the function to the dataset
preference_dataset = preference_dataset.map(create_columns)

# Select only the new columns
preference_dataset = preference_dataset.remove_columns(['instruction', 'input', 'output_1', 'output_2', 'preference', 'raw_preference'])
train_test_split = preference_dataset.train_test_split(test_size=0.2)
train_dataset = train_test_split["train"]
eval_dataset = train_test_split["test"]
prefix_user = "Human:"
prefix_bot = "\nAssistant:"
model_name = "EleutherAI/pythia-70m-deduped"
dpo_tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    revision="step3000",
    cache_dir="./pythia-70m-deduped/step3000",
)
model_paths = []
winners = []
dpo_models_path = 'dpo_models'
for model_folder in os.listdir(dpo_models_path):
    model_path = os.path.join(dpo_models_path, model_folder)
    if os.path.isdir(model_path):
        model_paths.append({"dpo_iter": extract_last_number(model_path), "path": model_path})
prefix_user = "Human:"
prefix_bot = "\nAssistant:"
winners = []
NUM_EXAMPLES = len(eval_dataset)
winners = [-1] * NUM_EXAMPLES 
highest_rewards = [-100000000] * NUM_EXAMPLES  

for model_info in model_paths:
    with torch.no_grad():
        model = AutoModelForCausalLM.from_pretrained(model_info["path"])
        for idx, example in enumerate(eval_dataset):
            completion = inference(model, dpo_tokenizer, example["prompt"])
            text = prefix_user + example["prompt"] + prefix_bot + completion
            reward_input = gold_tokenizer(text, return_tensors="pt")
            reward = gold_reward(**reward_input).logits[0].item()
            if reward > highest_rewards[idx]:
                highest_rewards[idx] = reward
                winners[idx] = model_info["dpo_iter"]
        del model
        gc.collect()
        torch.cuda.empty_cache()
        
with open('winners.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Example Index', 'Winning Model Number'])
    for index, winner in enumerate(winners):
        writer.writerow([index, winner])

print("Winners saved to winners.csv")
