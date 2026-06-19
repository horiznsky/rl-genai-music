import torch
import torch.nn as nn

class GenerationModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim=128, hidden_size=128, dropout_prob=0.3, use_beat=True):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.gru = nn.GRU(embedding_dim, hidden_size, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 3, vocab_size)
        self.dropout = nn.Dropout(dropout_prob)
        self.use_beat = use_beat
        self.hidden_size = hidden_size
        self.hidden_state = None

        if self.use_beat:
            self.beat_embedding = nn.Embedding(100, embedding_dim)
            self.beat_rnn = nn.GRU(embedding_dim, hidden_size, batch_first=True)

    def forward(self, x, lengths, beat):
        emb = self.embedding(x)
        packed = nn.utils.rnn.pack_padded_sequence(emb, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, hidden = self.gru(packed)
        hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)

        if self.use_beat:
            beat_emb = self.beat_embedding(beat)
            beat_out, _ = self.beat_rnn(beat_emb)
            beat_feat = self.dropout(beat_out[:, -1, :])
        else:
            beat_feat = torch.zeros(hidden.size(0), self.hidden_size, device=hidden.device)

        combined = torch.cat([hidden, beat_feat], dim=1)  
        self.hidden_state = combined  
        return self.fc(self.dropout(combined))

    def get_hidden_state(self):
        return self.hidden_state

class ActorCritic(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.generator = GenerationModel(
            vocab_size=config["vocab_size"],
            embedding_dim=config["embedding_dim"],
            hidden_size=config["hidden_size"],
            dropout_prob=config["dropout_prob"],
            use_beat=config["use_beat"]
        )
        self.critic = nn.Linear(config["hidden_size"] * 3, 1)  

    def forward(self, x, lengths, beat):
        logits = self.generator(x, lengths, beat)  
        hidden_state = self.generator.get_hidden_state()  
        value = self.critic(hidden_state)  
        return logits, value