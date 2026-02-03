import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from environment import RicochetEnv
from tokenizer import RicochetTokenizer
from model import RicochetModel
from data import HindsightDataset

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Data: complete subsets
    env = RicochetEnv(size=16, num_robots=4)
    ds = HindsightDataset(
        env,
        num_episodes=2500,
        rollout_len=15,
        pairs_per_episode=None,          # ALL PAIRS
        include_all_one_step=True,       # extra 1-step
        max_pairs_cap=120,               # safety cap per episode (optional; set None if you want full explosion)
        grid_size=16
    )
    dl = DataLoader(ds, batch_size=32, shuffle=True, num_workers=0)

    tokenizer = RicochetTokenizer(d_model=256, grid_size=16).to(device)
    model = RicochetModel(tokenizer, d_model=256, nhead=8, num_layers=6, grid_size=16, dist_bins=21).to(device)

    save_dir = "checkpoints_pv"
    os.makedirs(save_dir, exist_ok=True)

    # ---------- Stage configs ----------
    # Stage 1: target + dist
    stage1_epochs = 3
    stage1_lr = 3e-4

    # Stage 2: helper (and optionally keep small target/dist anchors)
    stage2_epochs = 3
    stage2_lr = 1e-4  # smaller

    # Loss weights (important)
    # Stage 1: only target + dist
    w1_target = 1.0
    w1_dist   = 1.0
    w1_hid    = 0.0
    w1_hcell  = 0.0

    # Stage 2: helper gets full weight; keep small anchor on target/dist to prevent drift
    # Also prevents “helper double importance”: helper is split 0.5/0.5 so total helper weight = 1.0
    w2_target = 0.25
    w2_dist   = 0.25
    w2_hid    = 0.50
    w2_hcell  = 0.50

    def run_stage(epochs, lr, w_target, w_dist, w_hid, w_hcell, tag):
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
        step = 0
        model.train()
        for ep in range(1, epochs + 1):
            for batch in dl:
                step += 1
                walls = batch["walls"].to(device)
                robots = batch["robots"].to(device)
                goal = batch["goal"].to(device)

                final_robots = batch["final_robots"].to(device)
                mid_cell = batch["mid_cell"].to(device)
                helper_id = batch["block_id"].to(device)
                helper_cell = batch["block_cell"].to(device)
                dist = batch["dist"].to(device)

                prop = model.propose(walls, robots, goal)
                dist_logits = model.verify(walls, robots, final_robots)

                loss_target = F.cross_entropy(prop["cell_logits"], mid_cell)
                loss_dist   = F.cross_entropy(dist_logits, dist)
                loss_hid    = F.cross_entropy(prop["robot_logits"], helper_id)
                loss_hcell  = F.cross_entropy(prop["cell2_logits"], helper_cell)

                loss = (w_target * loss_target) + (w_dist * loss_dist) + (w_hid * loss_hid) + (w_hcell * loss_hcell)

                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()

                if step % 200 == 0:
                    with torch.no_grad():
                        acc_dist = (dist_logits.argmax(dim=1) == dist).float().mean().item()
                        acc_t    = (prop["cell_logits"].argmax(dim=1) == mid_cell).float().mean().item()
                        acc_hid  = (prop["robot_logits"].argmax(dim=1) == helper_id).float().mean().item()
                        acc_hc   = (prop["cell2_logits"].argmax(dim=1) == helper_cell).float().mean().item()
                    ckpt = os.path.join(save_dir, f"{tag}_ep{ep}_step{step}.pt")
                    torch.save(model.state_dict(), ckpt)
                    print(
                        f"[{tag}] ep {ep} step {step} | loss {loss.item():.4f} | "
                        f"acc_dist {acc_dist:.3f} acc_target {acc_t:.3f} acc_hid {acc_hid:.3f} acc_hcell {acc_hc:.3f}"
                    )

            # end epoch checkpoint
            ckpt = os.path.join(save_dir, f"{tag}_epoch{ep}.pt")
            torch.save(model.state_dict(), ckpt)
            print(f"[{tag}] saved {ckpt}")

    # ---------- Stage 1 ----------
    run_stage(stage1_epochs, stage1_lr, w1_target, w1_dist, w1_hid, w1_hcell, tag="stage1")

    # ---------- Stage 2 ----------
    run_stage(stage2_epochs, stage2_lr, w2_target, w2_dist, w2_hid, w2_hcell, tag="stage2")

    torch.save(model.state_dict(), os.path.join(save_dir, "pv_final.pt"))
    print("Saved pv_final.pt")

if __name__ == "__main__":
    main()
