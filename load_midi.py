import os
from miditok import REMI, TokenizerConfig
from symusic import Score


midi_folder = "Your/PATH" # Make sure you paste your own PATH here


config = TokenizerConfig()
tokenizer = REMI(config)


all_tokens = [] # Paste your generated MIDI tokens here


for filename in os.listdir(midi_folder):
    if filename.endswith(".mid") or filename.endswith(".midi"):
        midi_path = os.path.join(midi_folder, filename)
        try:
            with open(midi_path, "rb") as f:
                midi_bytes = f.read()

            # Convert bytes to symusic Score
            score = Score.from_midi(midi_bytes)

            # Encode tokens
            tokens = tokenizer.encode(score)

            all_tokens.append(tokens[0])

            print(f"Tokenized: {filename}")
        except Exception as e:
            print(f"Failed to process {filename}: {e}")

import pickle
with open("tokenized_midi_dataset.pkl", "wb") as f:
    pickle.dump(all_tokens, f)

print("Tokenized all MIDI files in the folder.")
