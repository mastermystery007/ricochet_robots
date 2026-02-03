import torch
import torch.nn as nn

BOS_TOKEN = 16
EOS_TOKEN = 17
PAD_TOKEN = 18
ACTION_VOCAB_SIZE = 19


class RicochetSeq2SeqModel(nn.Module):
    def __init__(self, tokenizer, d_model=256, nhead=8, num_layers=6):
        super().__init__()
        self.tokenizer = tokenizer
        self.action_embedding = nn.Embedding(ACTION_VOCAB_SIZE, d_model)

        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        dec_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=num_layers)

        self.output_head = nn.Linear(d_model, ACTION_VOCAB_SIZE)

    def _causal_mask(self, size, device):
        return torch.triu(torch.ones(size, size, device=device), diagonal=1).bool()

    def forward(self, walls, robots, goal, action_in):
        """
        action_in: [B, T] teacher-forced input with BOS/EOS/PAD tokens.
        """
        tokens = self.tokenizer(walls, robots, goal)  # [B,261,d]
        memory = self.encoder(tokens)                 # [B,261,d]

        tgt = self.action_embedding(action_in)        # [B,T,d]
        tgt_mask = self._causal_mask(tgt.size(1), tgt.device)
        decoded = self.decoder(tgt, memory, tgt_mask=tgt_mask)  # [B,T,d]
        return self.output_head(decoded)

    @torch.no_grad()
    def greedy_decode(self, walls, robots, goal, max_len=20):
        B = walls.size(0)
        device = walls.device
        tokens = self.tokenizer(walls, robots, goal)
        memory = self.encoder(tokens)

        seq = torch.full((B, 1), BOS_TOKEN, dtype=torch.long, device=device)
        finished = torch.zeros(B, dtype=torch.bool, device=device)

        for _ in range(max_len):
            tgt = self.action_embedding(seq)
            tgt_mask = self._causal_mask(tgt.size(1), tgt.device)
            decoded = self.decoder(tgt, memory, tgt_mask=tgt_mask)
            logits = self.output_head(decoded[:, -1, :])
            next_token = torch.argmax(logits, dim=1)
            seq = torch.cat([seq, next_token.unsqueeze(1)], dim=1)
            finished = finished | (next_token == EOS_TOKEN)
            if finished.all():
                break
        return seq[:, 1:]
