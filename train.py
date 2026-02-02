# train.py
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from environment import RicochetEnv
from tokenizer import RicochetTokenizer
from model import RicochetModel
from data import HindsightDataset

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    env = RicochetEnv(size=16, num_robots=4)

    ds = HindsightDataset(
        env,
        num_episodes=2000,
        rollout_len=15,
        pairs_per_episode=8,
        grid_size=16
    )
    dl = DataLoader(ds, batch_size=32, shuffle=True, num_workers=0)

    tokenizer = RicochetTokenizer(d_model=256, grid_size=16).to(device)
    model = RicochetModel(tokenizer, d_model=256, nhead=8, num_layers=6, grid_size=16, dist_bins=21).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-2)

    # Loss weights:
    # - target midpoint cell: 1.0
    # - helper total: 1.0 split as 0.5 (id) + 0.5 (cell)
    # - verifier: 1.0
    w_target = 1.0
    w_helper_id = 0.5
    w_helper_cell = 0.5
    w_dist = 1.0

    model.train()
    for step, batch in enumerate(dl, start=1):
        walls = batch["walls"].to(device)
        robots = batch["robots"].to(device)
        goal = batch["goal"].to(device)

        final_robots = batch["final_robots"].to(device)
        mid_cell = batch["mid_cell"].to(device)
        helper_id = batch["helper_id"].to(device)
        helper_cell = batch["helper_cell"].to(device)
        dist = batch["dist"].to(device)

        prop = model.propose(walls, robots, goal)
        dist_logits = model.verify(walls, robots, final_robots)

        loss_target = F.cross_entropy(prop["cell_logits"], mid_cell)
        loss_hid = F.cross_entropy(prop["robot_logits"], helper_id)
        loss_hcell = F.cross_entropy(prop["cell2_logits"], helper_cell)
        loss_dist = F.cross_entropy(dist_logits, dist)

        loss = (w_target * loss_target) + (w_helper_id * loss_hid) + (w_helper_cell * loss_hcell) + (w_dist * loss_dist)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step % 50 == 0:
            torch.save(model.state_dict(), "ckpt.pt")
            with torch.no_grad():
                acc_dist = (dist_logits.argmax(dim=1) == dist).float().mean().item()
                acc_target = (prop["cell_logits"].argmax(dim=1) == mid_cell).float().mean().item()
                acc_hid = (prop["robot_logits"].argmax(dim=1) == helper_id).float().mean().item()
                acc_hcell = (prop["cell2_logits"].argmax(dim=1) == helper_cell).float().mean().item()

            print(
                f"step {step:5d} | loss {loss.item():.4f} | "
                f"acc_dist {acc_dist:.3f} | acc_target {acc_target:.3f} | "
                f"acc_helper_id {acc_hid:.3f} | acc_helper_cell {acc_hcell:.3f}"
            )

        if step == 500:
            break

if __name__ == "__main__":
    main()
