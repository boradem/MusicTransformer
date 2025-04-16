# Music Transformer

This repository contains an implementation of the Music Transformer model for symbolic music generation using PyTorch.
The model is designed to handle long-term musical structure by using relative attention mechanisms.

## Features

- Transformer architecture with relative self-attention
- Input/output in event-based token representation
- Support for loading and saving MIDI files
- Training and inference pipelines
- Evaluation with MIDI playback and token-based metrics

## File Structure
```
.
├── configs/               
├── data/                  
├── models/                
├── preprocess/            
├── checkpoints/           
├── outputs/               
├── train.py               
├── generate.py            
└── requirements.txt
```


## Dependencies

- Python 3.8+
- PyTorch
- NumPy
- pretty_midi
- miditoolkit
- tqdm
- Transformers




