from miditok import REMI, TokenizerConfig


# Loading trained tokenizer
config = TokenizerConfig()
tokenizer = REMI(config)

# Load generated MIDI tokens
generated_tokens = ["There will be your generated MIDI tokens."]

# Convert tokens to a ScoreTick object
score_tick = tokenizer.decode([generated_tokens])

# Make sure `score_tick` is not empty
if not hasattr(score_tick, "notes") or len(score_tick.notes) == 0:
    print("Warning: No notes found in the generated ScoreTick. MIDI may be empty.")

# Convert ScoreTick to Score
try:
    score = score_tick.to_score()  # Properly convert to Score
except AttributeError:
    print("Error: `ScoreTick` does not support `.to_score()`. Ensure your tokenizer is set up correctly.")
    exit(1)

# Convert Score to a PrettyMIDI object
midi = score.to_pretty_midi()  # Convert properly

# Save MIDI file
midi.write("generated_music.mid")  # Save correctly

print("🎶 MIDI file saved as `generated_music.mid`! Play it in any MIDI player.")
