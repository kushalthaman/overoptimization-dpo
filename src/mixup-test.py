import torch
from transformers import GPTNeoXForCausalLM, AutoTokenizer

# Example to test get_model_input_and_output_embeddings_for_mixup
# Initialize a mock model and tokenizer
model = GPTNeoXForCausalLM.from_pretrained("EleutherAI/pythia-410m-deduped",
        revision="step3000",
        cache_dir="./pythia-70m-deduped/step3000",
        output_hidden_states=True)
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-410m-deduped",
        revision="step3000",
        cache_dir="./pythia-70m-deduped/step3000")

# Create mock input embeddings
input_ids = tokenizer.encode("Hello, world!", return_tensors="pt")
mock_embeddings = model.transformer.wte(input_ids)  # get embeddings from the tokenizer input

# Call your function with mock data
input_activations, output_activations = get_model_input_and_output_embeddings_for_mixup(model, {"input_ids": input_ids})

# Inspect the output
print("Input Activations:", input_activations)
print("Output Activations:", output_activations)