import json
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

class DuetDataset(Dataset):
    def __init__(self, duet_data, vocab_path, encoding='single'):
        self.duet_data = duet_data
        self.encoding = encoding
        with open(vocab_path, 'r') as f:
            self.pitch_vocab = json.load(f)

    def __len__(self):
        return len(self.duet_data)

    def __getitem__(self, idx):
        entry = self.duet_data[idx]
        human_pitch = [self.pitch_vocab.get(p, 1) for p in entry['human']['pitch']]
        machine_pitch = [self.pitch_vocab.get(p, 1) for p in entry['machine']['pitch']]
        beat = entry['beat']
        length = min(len(human_pitch), len(machine_pitch), len(beat))
        return {
            'human_pitch': torch.tensor(human_pitch[:length], dtype=torch.long),
            'machine_pitch': torch.tensor(machine_pitch[:length], dtype=torch.long),
            'beat': torch.tensor(beat[:length], dtype=torch.float),
        }

    def get_vocab_size(self):
        return len(self.pitch_vocab)

class DuetRangeDataset(DuetDataset):
    def __init__(self, duet_data, vocab_path, window_size=5, encoding='single'):
        super().__init__(duet_data, vocab_path, encoding)
        self.window_size = window_size
        self.half_window = window_size // 2
        self.valid_indices = self._generate_valid_indices()

    def _generate_valid_indices(self):
        indices = []
        for i, entry in enumerate(self.duet_data):
            length = min(len(entry['human']['pitch']), len(entry['machine']['pitch']), len(entry['beat']))
            for t in range(self.half_window, length - self.half_window):
                indices.append((i, t))
        return indices

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        chorale_idx, center_t = self.valid_indices[idx]
        entry = self.duet_data[chorale_idx]
        start = center_t - self.half_window
        end = center_t + self.half_window + 1

        human_pitch = [self.pitch_vocab.get(p, 1) for p in entry['human']['pitch'][start:end]]
        machine_pitch = [self.pitch_vocab.get(p, 1) for p in entry['machine']['pitch'][start:end]]
        beat = entry['beat'][start:end]

        return {
            'human_pitch': torch.tensor(human_pitch, dtype=torch.long),
            'machine_pitch': torch.tensor(machine_pitch, dtype=torch.long),
            'beat': torch.tensor(beat, dtype=torch.float)
        }

def custom_collate_fn(batch):
    PAD_IDX = 0
    human_seqs = [item['human_pitch'] for item in batch]
    machine_seqs = [item['machine_pitch'] for item in batch]
    beat_seqs = [item['beat'] for item in batch]  

    human_lengths = torch.tensor([len(seq) for seq in human_seqs])
    machine_lengths = torch.tensor([len(seq) for seq in machine_seqs])

    human_padded = pad_sequence(human_seqs, batch_first=True, padding_value=PAD_IDX)
    machine_padded = pad_sequence(machine_seqs, batch_first=True, padding_value=PAD_IDX)
    beat_padded = pad_sequence(beat_seqs, batch_first=True, padding_value=PAD_IDX)

    return {
        'human_pitch': human_padded,
        'machine_pitch': machine_padded,
        'human_lengths': human_lengths,
        'machine_lengths': machine_lengths,
        'beat': beat_padded  
    }

def range_collate_fn(batch):
    human_batch = torch.stack([item['human_pitch'] for item in batch])
    machine_batch = torch.stack([item['machine_pitch'] for item in batch])
    beat_batch = torch.stack([item['beat'] for item in batch])
    lengths = torch.tensor([human_batch.shape[1]] * len(batch))

    return {
        'human_pitch': human_batch,
        'machine_pitch': machine_batch,
        'beat': beat_batch,
        'human_lengths': lengths,
        'machine_lengths': lengths
    }