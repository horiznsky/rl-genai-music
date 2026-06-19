import json
import torch
import numpy as np
from typing import List
from music21 import stream, note, meter
from src.actor_critic import ActorCritic

class MusicGenPipeline:
    def __init__(self, model_path: str, config: dict, vocab_path: str):
        self.device = torch.device("cuda" if torch.cuda.is_available() and config["training"]["device"] == "cuda" else "cpu")
        with open(vocab_path, 'r') as f:
            self.vocab = json.load(f)
        self.inv_vocab = {v: k for k, v in self.vocab.items()}
        
        self.model = ActorCritic(config["model"]).to(self.device)
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

    def generate_counter_melody(self, human_pitches: List[int], ticks_per_quarter: int = 4) -> stream.Part:
        seq_len = len(human_pitches)
        beat_seq = np.tile([1, 2, 3, 4], (seq_len // 4) + 1)[:seq_len]
        
        human_tensor = torch.tensor([human_pitches], dtype=torch.long, device=self.device)
        beat_tensor = torch.tensor([beat_seq], dtype=torch.long, device=self.device)
        human_lengths = torch.tensor([seq_len], dtype=torch.long, device=self.device)
        
        with torch.no_grad():
            generated_tokens = self._run_autoregressive_generation(human_tensor, beat_tensor, human_lengths)
            
        decoded_pitches = [self.inv_vocab.get(t.item(), 'rest') for t in generated_tokens[0]]
        return self._sequence_to_stream(decoded_pitches, ticks_per_quarter)

    def _run_autoregressive_generation(self, human, beat, lengths):
        batch_size, max_length = human.size(0), human.size(1)
        generated_seq = torch.zeros(batch_size, max_length, dtype=torch.long, device=self.device)
        input_seq = torch.full((batch_size, 1), fill_value=1, dtype=torch.long, device=self.device)
        input_lengths = torch.full((batch_size,), fill_value=1, dtype=torch.long, device=self.device)

        for t in range(max_length):
            beat_slice = beat[:, :input_seq.size(1)]
            logits = self.model.generator(input_seq, input_lengths, beat_slice)
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).squeeze(1)
            
            generated_seq[:, t] = next_token
            input_seq = torch.cat([input_seq, next_token.unsqueeze(1)], dim=1)
            input_lengths += 1
        return generated_seq

    def _sequence_to_stream(self, pitch_seq: List[str], ticks_per_quarter: int) -> stream.Part:
        part = stream.Part()
        part.append(meter.TimeSignature('4/4'))
        current_pitch, duration = None, 0.0
        step_duration = 1.0 / ticks_per_quarter

        for pitch in pitch_seq:
            if "hold" in pitch or pitch == 'hold':
                duration += step_duration
            else:
                if current_pitch:
                    part.append(note.Rest(quarterLength=duration) if current_pitch == 'rest' 
                                else note.Note(current_pitch, quarterLength=duration))
                current_pitch, duration = pitch, step_duration

        if current_pitch:
            part.append(note.Rest(quarterLength=duration) if current_pitch == 'rest' 
                        else note.Note(current_pitch, quarterLength=duration))
        return part