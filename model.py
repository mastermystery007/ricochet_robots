import torch
import torch.nn as nn

class RicochetModel(nn.Module):
    def __init__(self, tokenizer, d_model=256, nhead=8, num_layers=6, grid_size=16, dist_bins=21):
        super().__init__()
        self.tokenizer = tokenizer
        self.grid_size = grid_size
        self.dist_bins = dist_bins

        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.trunk = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        # Proposer heads
        HW = grid_size * grid_size
        self.proposer_cell = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, HW))
        self.proposer_robot = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, 4))
        self.proposer_cell2 = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, HW))

        # Verifier head (UVFA bins)
        self.verifier_head = nn.Sequential(
            nn.Linear(d_model * 4, d_model), nn.ReLU(),
            nn.Linear(d_model, dist_bins)
        )

    def _encode(self, walls, robots, goal):
        tokens = self.tokenizer(walls, robots, goal)      # [B,261,d]
        ctx = self.trunk(tokens)                          # [B,261,d]
        z = ctx.mean(dim=1)                               # [B,d]
        return z

    def propose(self, walls, robots, goal):
        z = self._encode(walls, robots, goal)
        return {
            "cell_logits": self.proposer_cell(z),
            "robot_logits": self.proposer_robot(z),
            "cell2_logits": self.proposer_cell2(z),
        }

    @torch.no_grad()
    def propose_k(self, walls, robots, goal, k=4, temperature=1.0):
        z = self._encode(walls, robots, goal)
        cell_logits = self.proposer_cell(z) / temperature
        cell_probs = torch.softmax(cell_logits, dim=1)
        samples = torch.multinomial(cell_probs, num_samples=k, replacement=False)
        sample_logprobs = torch.log(torch.gather(cell_probs, 1, samples))

        helper_id = self.proposer_robot(z).argmax(dim=1)
        helper_cell = self.proposer_cell2(z).argmax(dim=1)

        proposals = []
        for idx in range(k):
            proposals.append({
                "mid_cell": samples[:, idx],
                "helper_id": helper_id,
                "helper_cell": helper_cell,
                "logprob": sample_logprobs[:, idx],
            })
        return proposals

    def verify(self, walls, robots_a, robots_b):
        """
        UVFA distance between two full states (A and B) under the same walls.
        We do not condition on the puzzle goal here; use a dummy goal token.
        """
        B = walls.shape[0]
        device = walls.device
        dummy_goal = torch.zeros((B, 3), dtype=torch.long, device=device)

        z_a = self._encode(walls, robots_a, dummy_goal)
        z_b = self._encode(walls, robots_b, dummy_goal)

        h = torch.cat([z_a, z_b, (z_a - z_b).abs(), z_a * z_b], dim=1)
        return self.verifier_head(h)  # [B,21]


class ContrastSeq2SeqModel(nn.Module):
    """
    Seq2seq-style model to predict the sub-goal (midpoint cell) from the same
    inputs as the proposer/verifier model: walls, robots, and goal.
    """
    def __init__(self, tokenizer, d_model=256, nhead=8, num_layers=6, grid_size=16):
        super().__init__()
        self.tokenizer = tokenizer
        self.grid_size = grid_size

        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        dec_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=num_layers)

        self.query = nn.Parameter(torch.randn(1, 1, d_model))
        self.subgoal_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, grid_size * grid_size),
        )

    def forward(self, walls, robots, goal):
        """
        Returns sub-goal (midpoint) logits over 256 cells.
        """
        tokens = self.tokenizer(walls, robots, goal)  # [B,261,d]
        memory = self.encoder(tokens)                 # [B,261,d]

        B = walls.shape[0]
        query = self.query.expand(B, -1, -1)          # [B,1,d]
        decoded = self.decoder(query, memory)         # [B,1,d]
        return self.subgoal_head(decoded.squeeze(1))
