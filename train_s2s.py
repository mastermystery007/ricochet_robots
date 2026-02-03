# train_s2s.py
import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from environment import RicochetEnv
from tokenizer import RicochetTokenizer
from data_s2s import Seq2SeqDataset
from s2s_model import Seq2SeqActionModel

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # training knobs
    num_episodes = 4000
    rollout_len = 15
    max_seq_len = 15
    batch_size = 32
    epochs = 5
    lr = 3e-4
    save_dir = "checkpoints_s2s"
    os.makedirs(save_dir, exist_ok=True)

    env = RicochetEnv(size=16, num_robots=4)

    ds = Seq2SeqDataset(
        env,
        num_episodes=num_episodes,
        rollout_len=rollout_len,
        grid_size=16,
        max_pairs_per_episode=None,      # COMPLETE SUBSETS
        include_all_one_step=True,       # extra 1-step supervision
        max_seq_len=max_seq_len
    )
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0)

    tokenizer = RicochetTokenizer(d_model=256, grid_size=16).to(device)
    model = Seq2SeqActionModel(
        tokenizer=tokenizer,
        d_model=256,
        nhead=8,
        num_layers=6,
        max_seq_len=max_seq_len,
        action_vocab=16
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)

    step = 0
    model.train()
    for ep in range(1, epochs + 1):
        for batch in dl:
            step += 1
            walls = batch["walls"].to(device)
            robots = batch["robots"].to(device)
            goal = batch["goal"].to(device)

            # teacher forcing:
            # targets are actions[0:seq_len], but our decoder input includes BOS then previous actions
            # We build action_inp = [BOS, a0, a1, ... a_{L-1}] and predict [a0..a_{L-1}]
            actions = batch["actions"].to(device)          # [B,Lmax] with -100 pad
            attn = batch["attn"].to(device)                # [B,Lmax] 1/0
            B, Lmax = actions.shape

            bos = model.bos_id.to(device).view(1, 1).expand(B, 1)  # [B,1]
            # shift-right input: BOS + actions[:-1]
            prev = actions.clone()
            prev[prev == -100] = 0  # doesn't matter, will be masked by attn
            action_inp = torch.cat([bos, prev[:, :-1]], dim=1)  # [B,Lmax]

            logits = model(walls, robots, goal, action_inp, attn=attn)  # [B,Lmax,16]

            # CE over each timestep; ignore padded labels
            loss = F.cross_entropy(logits.reshape(-1, 16), actions.reshape(-1), ignore_index=-100)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            if step % 200 == 0:
                ckpt_path = os.path.join(save_dir, f"s2s_step{step}.pt")
                torch.save(model.state_dict(), ckpt_path)
                print(f"[ep {ep}] step {step} | loss {loss.item():.4f} | saved {ckpt_path}")

        # epoch checkpoint
        ckpt_path = os.path.join(save_dir, f"s2s_epoch{ep}.pt")
        torch.save(model.state_dict(), ckpt_path)
        print(f"END epoch {ep} | saved {ckpt_path}")

    torch.save(model.state_dict(), os.path.join(save_dir, "s2s_final.pt"))
    print("Saved s2s_final.pt")

if __name__ == "__main__":
    main()
