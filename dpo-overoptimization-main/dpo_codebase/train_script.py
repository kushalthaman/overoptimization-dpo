import subprocess

def train_wrapper(mixup: bool):
    """
    Creates a new training process for SFT or DPO. 
    """
    command = [
        "python", "-u", "train.py",
        "model=pythia69",
        "datasets=[hh]",
        "loss=sft",
        "exp_name=anthropic_dpo_pythia69",
        "gradient_accumulation_steps=2",
        "batch_size=64",
        "eval_batch_size=32",
        "trainer=FSDPTrainer",
        "sample_during_eval=false"
    ]

    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    if process.returncode != 0:
        print("Error running the command:")
        print(stderr.decode())
    else:
        print("Command ran successfully. Output:")
        print(stdout.decode())

if __name__ == "__main__":
    train_wrapper(False)
