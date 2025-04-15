import torch
import pickle
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load trained model and tokenizer
model_path = "./gpt2-music"
tokenizer = GPT2Tokenizer.from_pretrained(model_path, local_files_only=True)
tokenizer.pad_token = tokenizer.eos_token  # Ensure padding is set
model = GPT2LMHeadModel.from_pretrained(model_path).to("mps")  # Use Apple Metal GPU


# Generate new music tokens
def generate_music(start_tokens, max_length=512):
    input_ids = torch.tensor([prompt_tokens]).to("mps")

    output = model.generate(
        input_ids=input_ids,
        max_length=max_length,
        temperature=1.0,
        top_k=50,
        top_p=0.95,
        num_return_sequences=1,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,  # ✅ Set explicit padding token
        attention_mask=torch.ones(input_ids.shape).to("mps"),  # ✅ Add attention mask
    )


    return output[0].tolist()


# Load tokenized dataset for reference
with open("midi_token_dataset.pkl", "rb") as f:
    tokenized_data = pickle.load(f)

# Select a random starting sequence
prompt_tokens = tokenized_data[0][:50]  # First 50 tokens from a song

# Generate a continuation
generated_tokens = generate_music(prompt_tokens)

# Print generated tokens
print("Generated Music Tokens:", generated_tokens)
