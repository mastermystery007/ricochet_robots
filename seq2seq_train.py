# seq2seq_train.py
import os
from datetime import datetime

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from environment import RicochetEnv
from tokenizer import RicochetTokenizer
from seq2seq_model import RicochetSeq2SeqModel, PAD_TOKEN
from seq2seq_data import Seq2SeqDataset, seq2seq_collate


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    env = RicochetEnv(size=16, num_robots=4)

    ds = Seq2SeqDataset(
        env,
        num_episodes=50000,
        rollout_len=25,
        max_pairs_per_episode=1000,
        min_seg_len=1,
        max_seg_len=None,
        grid_size=16,
    )
    if ds.episodes > 0:
        avg_pairs = ds.total_pairs / ds.episodes
    else:
        avg_pairs = 0
    print(f"Dataset size: {len(ds)} examples | avg pairs/episode: {avg_pairs:.1f}")

    dl = DataLoader(ds, batch_size=32, shuffle=True, num_workers=0, collate_fn=seq2seq_collate)

    tokenizer = RicochetTokenizer(d_model=256, grid_size=16).to(device)
    model = RicochetSeq2SeqModel(tokenizer, d_model=256, nhead=8, num_layers=6).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-2)

    max_steps = 200000
    save_every = 5000

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join("runs", run_id)
    os.makedirs(run_dir, exist_ok=True)
    metrics_path = os.path.join(run_dir, "metrics.csv")
    with open(metrics_path, "w", encoding="utf-8") as f:
        f.write("step,loss,token_acc\n")

    model.train()
    for step, batch in enumerate(dl, start=1):
        walls = batch["walls"].to(device)
        robots = batch["robots"].to(device)
        goal = batch["goal"].to(device)
        action_in = batch["action_in"].to(device)
        action_out = batch["action_out"].to(device)

        logits = model(walls, robots, goal, action_in)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), action_out.view(-1), ignore_index=PAD_TOKEN)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step % save_every == 0:
            torch.save(model.state_dict(), os.path.join(run_dir, f"s2s_ckpt_{step}.pt"))
            with torch.no_grad():
                preds = logits.argmax(dim=-1)
                mask = action_out != PAD_TOKEN
                correct = (preds == action_out) & mask
                token_acc = correct.float().sum() / mask.float().sum().clamp_min(1.0)
            print(f"step {step:5d} | loss {loss.item():.4f} | token_acc {token_acc.item():.3f}")
            with open(metrics_path, "a", encoding="utf-8") as f:
                f.write(f"{step},{loss.item():.6f},{token_acc.item():.4f}\n")

        if step >= max_steps:
            break

    torch.save(model.state_dict(), os.path.join(run_dir, "s2s_ckpt_final.pt"))
    print(f"Saved {os.path.join(run_dir, 's2s_ckpt_final.pt')}")


if __name__ == "__main__":
    main()
