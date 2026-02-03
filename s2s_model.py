# s2s_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class Seq2SeqActionModel(nn.Module):
    """
    Encoder: same tokenizer(walls, robots, goal) -> TransformerEncoder
    Decoder: autoregressive TransformerDecoder over action tokens (0..15)
    Output: logits over 16 actions at each step
    """
    def __init__(self, tokenizer, d_model=256, nhead=8, num_layers=6, max_seq_len=15, action_vocab=16):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.action_vocab = action_vocab

        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        dec_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=num_layers)

        # action token embeddings (+ position)
        self.action_emb = nn.Embedding(action_vocab + 1, d_model)  # +1 for BOS
        self.pos_emb = nn.Embedding(max_seq_len + 1, d_model)

        self.out = nn.Linear(d_model, action_vocab)

        self.register_buffer("bos_id", torch.tensor(action_vocab, dtype=torch.long))  # BOS = 16

    def _causal_mask(self, T, device):
        # True values are masked in PyTorch's Transformer (float -inf mask also works)
        # We'll use float mask.
        mask = torch.full((T, T), float("-inf"), device=device)
        mask = torch.triu(mask, diagonal=1)
        return mask

    def forward(self, walls, robots, goal, action_inp, attn=None):
        """
        action_inp: [B, T] input tokens including BOS at t=0.
        returns logits [B, T, 16] for predicting next action at each position.
        """
        # encode context
        mem_tokens = self.tokenizer(walls, robots, goal)     # [B,261,d]
        memory = self.encoder(mem_tokens)                    # [B,261,d]

        B, T = action_inp.shape
        device = action_inp.device

        pos = torch.arange(T, device=device).unsqueeze(0).expand(B, -1)
        dec_in = self.action_emb(action_inp) + self.pos_emb(pos)

        tgt_mask = self._causal_mask(T, device=device)       # [T,T]

        # optional key padding mask: True means "ignore"
        tgt_key_padding_mask = None
        if attn is not None:
            # attn is 1 for real tokens, 0 for pad
            tgt_key_padding_mask = (attn == 0)

        dec = self.decoder(
            tgt=dec_in,
            memory=memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask
        )  # [B,T,d]

        logits = self.out(dec)  # [B,T,16]
        return logits

    @torch.no_grad()
    def greedy_decode(self, walls, robots, goal, max_steps=15):
        """
        Returns a Python list of actions length <= max_steps.
        Greedy decoding (argmax).
        """
        device = walls.device
        B = walls.shape[0]
        assert B == 1, "greedy_decode supports batch=1 for simplicity"

        # start with BOS
        tokens = torch.full((1, 1), int(self.bos_id.item()), dtype=torch.long, device=device)

        actions = []
        for t in range(max_steps):
            # attn is all 1s (no padding)
            attn = torch.ones_like(tokens, dtype=torch.long, device=device)
            logits = self.forward(walls, robots, goal, tokens, attn=attn)  # [1,T,16]
            next_logits = logits[:, -1, :]  # [1,16]
            a = int(next_logits.argmax(dim=-1).item())
            actions.append(a)
            # append predicted token
            tokens = torch.cat([tokens, torch.tensor([[a]], device=device, dtype=torch.long)], dim=1)

        return actions
