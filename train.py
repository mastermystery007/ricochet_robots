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

    model.train()
    for step, batch in enumerate(dl, start=1):
        walls = batch["walls"].to(device)
        robots = batch["robots"].to(device)
        goal = batch["goal"].to(device)

        final_robots = batch["final_robots"].to(device)
        mid_cell = batch["mid_cell"].to(device)
        block_id = batch["block_id"].to(device)
        block_cell = batch["block_cell"].to(device)
        dist = batch["dist"].to(device)

        prop = model.propose(walls, robots, goal)
        dist_logits = model.verify(walls, robots, final_robots)

        loss_cell = F.cross_entropy(prop["cell_logits"], mid_cell)
        loss_block = F.cross_entropy(prop["robot_logits"], block_id) + F.cross_entropy(prop["cell2_logits"], block_cell)
        loss_dist = F.cross_entropy(dist_logits, dist)

        loss = loss_cell + loss_block + loss_dist

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step % 50 == 0:
            with torch.no_grad():
                acc_dist = (dist_logits.argmax(dim=1) == dist).float().mean().item()
                acc_cell = (prop["cell_logits"].argmax(dim=1) == mid_cell).float().mean().item()
                acc_block_r = (prop["robot_logits"].argmax(dim=1) == block_id).float().mean().item()
                acc_block_c = (prop["cell2_logits"].argmax(dim=1) == block_cell).float().mean().item()

            print(
                f"step {step:5d} | loss {loss.item():.4f} | "
                f"acc_dist {acc_dist:.3f} | acc_cell {acc_cell:.3f} | "
                f"acc_block_r {acc_block_r:.3f} | acc_block_c {acc_block_c:.3f}"
            )

        if step == 500:
            break

if __name__ == "__main__":
    main()
