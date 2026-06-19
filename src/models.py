import torch
import torch.nn as nn

class BaseRewardModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim=128, hidden_size=128, dropout_prob=0.3, use_beat=True):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.gru = nn.GRU(embedding_dim, hidden_size, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout_prob)
        self.use_beat = use_beat
        self.hidden_size = hidden_size

        if self.use_beat:
            self.beat_embedding = nn.Embedding(100, embedding_dim)
            self.beat_rnn = nn.GRU(embedding_dim, hidden_size, batch_first=True)

class JointRewardModel(BaseRewardModel):
    def __init__(self, vocab_size, **kwargs):
        super().__init__(vocab_size, **kwargs)
        self.fc = nn.Linear(self.hidden_size * 5, 1)

    def forward(self, human, human_lengths, machine, machine_lengths, beat):
        human_emb = self.embedding(human)
        machine_emb = self.embedding(machine)
        
        human_packed = nn.utils.rnn.pack_padded_sequence(human_emb, human_lengths.cpu(), batch_first=True, enforce_sorted=False)
        machine_packed = nn.utils.rnn.pack_padded_sequence(machine_emb, machine_lengths.cpu(), batch_first=True, enforce_sorted=False)

        _, human_hidden = self.gru(human_packed)
        _, machine_hidden = self.gru(machine_packed)

        human_hidden = torch.cat([human_hidden[-2], human_hidden[-1]], dim=1)
        machine_hidden = torch.cat([machine_hidden[-2], machine_hidden[-1]], dim=1)

        if self.use_beat:
            beat_emb = self.beat_embedding(beat)  
            beat_out, _ = self.beat_rnn(beat_emb)
            beat_feat = self.dropout(beat_out[:, -1, :])
        else:
            beat_feat = torch.zeros(human_hidden.size(0), self.hidden_size, device=human.device)

        combined = torch.cat([self.dropout(human_hidden), self.dropout(machine_hidden), beat_feat], dim=1)
        return self.fc(combined).squeeze(1)

class JointRangeRewardModel(BaseRewardModel):
    def __init__(self, vocab_size, **kwargs):
        super().__init__(vocab_size, **kwargs)
        self.fc = nn.Linear(self.hidden_size * 3, 1)

    def forward(self, human, human_lengths, machine, machine_lengths, beat):
        full_input = torch.cat([human, machine], dim=1)
        full_lengths = human_lengths + machine_lengths
        emb = self.embedding(full_input)
        packed = nn.utils.rnn.pack_padded_sequence(emb, full_lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, hidden = self.gru(packed)
        joint_hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)

        if self.use_beat:
            beat_emb = self.beat_embedding(beat)  
            beat_out, _ = self.beat_rnn(beat_emb)
            beat_feat = self.dropout(beat_out[:, -1, :])
        else:
            beat_feat = torch.zeros(joint_hidden.size(0), self.hidden_size, device=joint_hidden.device)

        combined = torch.cat([self.dropout(joint_hidden), beat_feat], dim=1)
        return self.fc(combined).squeeze(1)

class HorizontalRewardModel(BaseRewardModel):
    def __init__(self, vocab_size, **kwargs):
        super().__init__(vocab_size, **kwargs)
        self.fc = nn.Linear(self.hidden_size * 3, 1)

    def forward(self, human, human_lengths, machine, machine_lengths, beat):
        machine_emb = self.embedding(machine)
        packed = nn.utils.rnn.pack_padded_sequence(machine_emb, machine_lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, machine_hidden = self.gru(packed)
        machine_hidden = torch.cat([machine_hidden[-2], machine_hidden[-1]], dim=1)

        if self.use_beat:
            beat_emb = self.beat_embedding(beat)  
            beat_out, _ = self.beat_rnn(beat_emb)
            beat_feat = self.dropout(beat_out[:, -1, :])
        else:
            beat_feat = torch.zeros(machine_hidden.size(0), self.hidden_size, device=machine.device)

        combined = torch.cat([self.dropout(machine_hidden), beat_feat], dim=1)
        return self.fc(combined).squeeze(1)

class VerticalRewardModel(BaseRewardModel):
    def __init__(self, vocab_size, **kwargs):
        super().__init__(vocab_size, **kwargs)
        self.fc = nn.Linear(self.hidden_size * 3, 1)

    def forward(self, human, human_lengths, machine, machine_lengths, beat):
        human_emb = self.embedding(human)
        packed = nn.utils.rnn.pack_padded_sequence(human_emb, human_lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, human_hidden = self.gru(packed)
        
        # Fixed the scope variable reference typo here:
        human_hidden = torch.cat([human_hidden[-2], human_hidden[-1]], dim=1)

        if self.use_beat:
            beat_emb = self.beat_embedding(beat)  
            beat_out, _ = self.beat_rnn(beat_emb)
            beat_feat = self.dropout(beat_out[:, -1, :])
        else:
            beat_feat = torch.zeros(human_hidden.size(0), self.hidden_size, device=human.device)

        combined = torch.cat([self.dropout(human_hidden), beat_feat], dim=1)
        return self.fc(combined).squeeze(1)