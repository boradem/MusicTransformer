import os
import pickle
from miditok import REMI, TokenizerConfig
from symusic import Score

# Folder containing MIDI files
midi_folder = "/Users/borandemir/Music/Garageband"


midi_files = [f for f in os.listdir(midi_folder) if f.endswith(".mid")]

if not midi_files:
    print("No MIDI files found in the folder!")
    exit()

print(f"Music Files Found {len(midi_files)} MIDI files. Tokenizing...")


config = TokenizerConfig()
tokenizer = REMI(config)

tokenized_data = []

for midi_file in midi_files:
    midi_path = os.path.join(midi_folder, midi_file)

    try:

        with open(midi_path, "rb") as f:
            midi_bytes = f.read()


        score = Score.from_midi(midi_bytes)

        tokens = tokenizer.encode(score)


        if isinstance(tokens, list):
            for seq in tokens:
                tokenized_data.append(seq.ids)  # Append each sequence separately
        else:
            tokenized_data.append(tokens.ids)  # If it's a single sequence, store directly

        print(f"Tokenized {midi_file} successfully!")

    except Exception as e:
        print(f"Error processing {midi_file}: {e}")

# Save tokenized data to a file
dataset_path = "midi_token_dataset.pkl"
with open(dataset_path, "wb") as f:
    pickle.dump(tokenized_data, f)

print(f"Dataset saved as {dataset_path}!")
