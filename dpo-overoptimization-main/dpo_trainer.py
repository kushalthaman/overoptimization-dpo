from typing import Dict, Optional
from dataclasses import dataclass, field
from typing import Dict, Optional
from transformers import AutoModelForSequenceClassification, LlamaTokenizer, GPTNeoXForCausalLM
import pandas as pd
import csv
import torch
from datasets import Dataset, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser, TrainingArguments
from transformers import AutoTokenizer
from trl import DPOTrainer

NUM_ITERATIONS = 20 # number of DPO iterations

@dataclass
class ScriptArguments:
    """
    The arguments for the DPO training script. beta in particular is anecodately very influential
    """

    # data parameters
    beta: Optional[float] = field(default=0.1, metadata={"help": "the beta parameter for DPO loss"})

    # training parameters
    model_name_or_path: Optional[str] = field(default="gpt2", metadata={"help": "the model name"})
    learning_rate: Optional[float] = field(default=1e-3, metadata={"help": "optimizer learning rate"})
    per_device_train_batch_size: Optional[int] = field(default=4, metadata={"help": "batch size per device"})
    gradient_accumulation_steps: Optional[int] = field(
        default=1, metadata={"help": "the number of gradient accumulation steps"}
    )
    max_length: Optional[int] = field(default=512, metadata={"help": "max length of each sample"})
    max_prompt_length: Optional[int] = field(default=128, metadata={"help": "max length of each sample's prompt"})
    max_target_length: Optional[int] = field(
        default=128, metadata={"help": "Only used for encoder decoder model. Max target of each sample's prompt"}
    )
    label_pad_token_id: Optional[int] = field(default=-100, metadata={"help": "label for non response tokens"})
    max_steps: Optional[int] = field(default=100, metadata={"help": "max number of training steps"})
    # instrumentation
    sanity_check: Optional[bool] = field(default=True, metadata={"help": "only train on 1000 samples"})
    report_to: Optional[str] = field(
        default='all',
        metadata={
            "help": 'The list of integrations to report the results and logs to. Supported platforms are `"azure_ml"`,'
            '`"comet_ml"`, `"mlflow"`, `"neptune"`, `"tensorboard"`,`"clearml"` and `"wandb"`. '
            'Use `"all"` to report to all integrations installed, `"none"` for no integrations.'
        },
    )
    # debug argument for distributed training
    ignore_bias_buffers: Optional[bool] = field(
        default=False,
        metadata={
            "help": "fix for DDP issues with LM bias/mask buffers - invalid scalar type,`inplace operation. See"
            "https://github.com/huggingface/transformers/issues/22482#issuecomment-1595790992"
        },
    )
    gradient_checkpointing: Optional[bool] = field(
        default=False, metadata={"help": "Whether to use gradient checkpointing or no"}
    )
    gradient_checkpointing_kwargs: Optional[dict] = field(
        default=None,
        metadata={
            "help": "key word arguments to be passed along `torch.utils.checkpoint.checkpoint` method - e.g. `use_reentrant=False`"
        },
    )

def dpo_trainer_wrapper(pretrained_model, pretrained_tokenizer, gold_reward, gold_tokenizer, train_dataset, eval_dataset, model_name):
    """
    Function: dpo_trainer_wrapper
    -----------------------------------------------------------
    This function does DPO on pretrained_model on train_dataset. At certain points of the
    training, the prompts in eval_dataset are fed into current model and the completions are
    evaluated by gold_reward. In addition, the training loss is reported. The models can be hosted
    on HuggingFace directly or the model weights can be saved locally.

    The function returns a matplotlib graph showing training loss in DPO and the rewards
    evaluated. This is used to evaluate overoptimization. The pretrained model & gold_reward
    model can be any HuggingFace autogressive model.

    Parameters:
        - pretrained_model: A HuggingFace AutoModelForCausalLM model.
        - train_dataset:  A HuggingFace dataset with columns including  {"prompts", "choosen", "rejected"}
        - gold_reward: A HuggingFace AutoModelForCausalLM model.
        - eval_dataset: A HuggingFace dataset with columns including {"instructions", "output1", "output2", "preference"}
    Output:
        - Local path of the matplotlib graph.
    TODO: See if you can use model_ref to eliminate need for evaluate_reward.
    """
    parser = HfArgumentParser(ScriptArguments)
    script_args, _ = parser.parse_args_into_dataclasses(return_remaining_strings=True)

    dpo_iter = 1

    # Ensure the tokenizer has a pad token
    if pretrained_tokenizer.pad_token is None:
        pretrained_tokenizer.pad_token = pretrained_tokenizer.eos_token

    reward = []
    model_ref = pretrained_model

    while dpo_iter <= NUM_ITERATIONS:
        training_args = TrainingArguments(
            per_device_train_batch_size=script_args.per_device_train_batch_size,
            max_steps=script_args.max_steps,
            remove_unused_columns=False,
            gradient_accumulation_steps=script_args.gradient_accumulation_steps,
            learning_rate=script_args.learning_rate,
            evaluation_strategy="steps",
            logging_first_step=True,
            logging_steps=10,
            eval_steps=500,
            output_dir=f"./{model_name}_{dpo_iter}",
            optim="rmsprop",
            warmup_steps=150,
            report_to=script_args.report_to,
            gradient_checkpointing=script_args.gradient_checkpointing,
        )

        dpo_trainer = DPOTrainer(
            model=pretrained_model,
            args=training_args,
            beta=script_args.beta,
            train_dataset=train_dataset,
            tokenizer=pretrained_tokenizer
        )

        dpo_trainer.train()
        dpo_trainer.save_model()

        loss_table = pd.DataFrame(dpo_trainer.state.log_history)
        loss_table.to_csv(f'{model_name}_{dpo_iter}.csv', index=False)

        reward.append(evaluate_reward(gold_reward, gold_tokenizer, eval_dataset, pretrained_model, pretrained_tokenizer))

        pretrained_model = AutoModelForCausalLM.from_pretrained(f'./{model_name}_{dpo_iter}')
        model_ref = pretrained_model
        dpo_iter += 1
        # Update the output directory for the next iteration
        training_args.output_dir = f'{model_name}_{dpo_iter}'

    reward_df = pd.DataFrame(reward)
    reward_df.to_csv(f'rewards_{model_name}.csv', index=False)



def train_linear_head(gold_reward, dataset):
    pretrained_model = AutoModelForCausalLM.from_pretrained(gold_reward)
    classifier = torch.nn.Linear(pretrained_model.config.hidden_size, 1)


def evaluate_reward(gold_reward, gold_tokenizer, eval_set, dpo_model, dpo_tokenizer):
    avg_reward, length = 0, 0
    # gold_reward = gold_reward.eval().half().cuda()
    prefix_user = "Human:"
    prefix_bot = "\nAssistant:"
    for example in eval_set:
        with torch.no_grad():
            inputs = dpo_tokenizer(example["prompt"], return_tensors="pt")
            tokens = dpo_model.generate(**inputs.to('cuda'))


            completion = dpo_tokenizer.decode(tokens[0])
            text = prefix_user + example["prompt"] + prefix_bot + completion
            #print(text)
            reward_input = gold_tokenizer(text, return_tensors="pt")
            reward = gold_reward(**reward_input).logits[0].item()

            avg_reward += reward
            length += 1

        """
        inputs = tokenizer.encode(example['instruction'], return_tensors = "pt")
        outputs = gold_reward.generate(input_ids = inputs)
        reward = tokenizer.decode(outputs)
        avg_reward += reward
        """
    avg_reward /= length
    return avg_reward

def main():
    torch.device("cuda")
    """
    model_1b = GPTNeoXForCausalLM.from_pretrained(
        "EleutherAI/pythia-1b-deduped",
        revision="step3000",
        cache_dir="./pythia-1b-deduped/step3000",
    )
    model_410m = GPTNeoXForCausalLM.from_pretrained(
        "EleutherAI/pythia-410m-deduped",
        revision="step3000",
        cache_dir="./pythia-410m-deduped/step3000",
    )
    tokenizer_1b = AutoTokenizer.from_pretrained(
        "EleutherAI/pythia-1b-deduped",
        revision="step3000",
        cache_dir="./pythia-1b-deduped/step3000",
    )
    tokenizer_410m = AutoTokenizer.from_pretrained(
        "EleutherAI/pythia-410m-deduped",
        revision="step3000",
        cache_dir="./pythia-1b-deduped/step3000",
    )
    """

    model_name = "EleutherAI/pythia-70m-deduped"
    model = GPTNeoXForCausalLM.from_pretrained(
        model_name,
        revision="step3000",
        cache_dir="./pythia-70m-deduped/step3000",
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        revision="step3000",
        cache_dir="./pythia-70m-deduped/step3000",
    )

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

    # Apply the function to the dataset
    preference_dataset = preference_dataset.map(create_columns)

    # Select only the new columns
    preference_dataset = preference_dataset.remove_columns(['instruction', 'input', 'output_1', 'output_2', 'preference', 'raw_preference'])
    train_test_split = preference_dataset.train_test_split(test_size=0.2)
    avg_reward, length = 0, 0
    # gold_reward = gold_reward.eval().half().cuda()
    prefix_user = "Human:"
    prefix_bot = "\nAssistant:"
    # The result is a new DatasetDict with 'train' and 'test' splits
    train_dataset = train_test_split["train"]
    eval_dataset = train_test_split["test"]
    """
    for example in eval_dataset:
        with torch.no_grad():
            inputs = tokenizer_410_m(example["prompt"], return_tensors="pt")
            tokens = model_410_m.generate(**inputs)


            completion = tokenizer_410_m.decode(tokens[0])
            text = prefix_user + example["prompt"] + prefix_bot + completion
            #print(text)
            reward_input = gold_tokenizer(text, return_tensors="pt")
            reward = gold_reward(**reward_input).logits[0]
            print(reward)

            avg_reward += reward
    """
    # print(train_dataset["chosen"])
    dpo_trainer_wrapper(model, tokenizer, gold_reward, gold_tokenizer, train_dataset, eval_dataset, model_name.split('/')[1])


main()


