import argparse
import numpy as np
import torch

from environment import RicochetEnv
from tokenizer import RicochetTokenizer
from model import RicochetModel
from seq2seq_model import RicochetSeq2SeqModel, EOS_TOKEN
from data import HindsightDataset, slide_and_get_stopper


def apply_action(walls, state, action, size=16):
    direction = [1, 2, 4, 8][action % 4]
    robot_id = action // 4
    next_state, moved, _stopper = slide_and_get_stopper(walls, state, robot_id, direction, size=size)
    return next_state, moved


def expected_dist(logits):
    probs = torch.softmax(logits, dim=1)
    bins = torch.arange(probs.size(1), device=probs.device).float()
    return (probs * bins).sum(dim=1)


def cell_to_rc(cell, size=16):
    r = cell // size
    c = cell % size
    return r, c


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pv_ckpt", required=True)
    parser.add_argument("--s2s_ckpt", required=True)
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--rollout_len", type=int, default=25)
    parser.add_argument("--max_pairs_per_episode", type=int, default=200)
    parser.add_argument("--max_len", type=int, default=25)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    env = RicochetEnv(size=16, num_robots=4)
    ds = HindsightDataset(
        env,
        num_episodes=args.episodes,
        rollout_len=args.rollout_len,
        max_pairs_per_episode=args.max_pairs_per_episode,
        min_seg_len=1,
        max_seg_len=None,
        grid_size=16,
    )

    tokenizer = RicochetTokenizer(d_model=256, grid_size=16).to(device)
    pv = RicochetModel(tokenizer, d_model=256, nhead=8, num_layers=6, grid_size=16, dist_bins=21).to(device)
    pv.load_state_dict(torch.load(args.pv_ckpt, map_location=device))
    pv.eval()

    s2s = RicochetSeq2SeqModel(tokenizer, d_model=256, nhead=8, num_layers=6).to(device)
    s2s.load_state_dict(torch.load(args.s2s_ckpt, map_location=device))
    s2s.eval()

    successes = 0
    total = 0
    hallucinations = 0

    for ex in ds:
        walls = ex["walls"].unsqueeze(0).to(device)
        robots_a = ex["robots"].unsqueeze(0).to(device)
        robots_b = ex["final_robots"].unsqueeze(0).to(device)
        goal = ex["goal"].unsqueeze(0).to(device)

        proposals = pv.propose_k(walls, robots_a, goal, k=4)
        scores = []
        for prop in proposals:
            mid_cell = prop["mid_cell"]
            r, c = cell_to_rc(mid_cell, size=16)
            robots_mid = robots_a.clone()
            target_robot = goal[0, 0].item()
            robots_mid[0, target_robot, 0] = r
            robots_mid[0, target_robot, 1] = c

            d1 = expected_dist(pv.verify(walls, robots_a, robots_mid))
            d2 = expected_dist(pv.verify(walls, robots_mid, robots_b))
            scores.append((d1 + d2).item())

        best_idx = int(np.argmin(scores))
        best_mid = proposals[best_idx]["mid_cell"]

        target_robot = int(goal[0, 0].item())
        mid_r, mid_c = cell_to_rc(best_mid.item(), size=16)

        state = ex["robots"].numpy().copy()
        walls_np = ex["walls"].numpy()

        mid_goal = goal.clone()
        mid_goal[0, 1] = mid_r
        mid_goal[0, 2] = mid_c

        with torch.no_grad():
            seq_to_mid = s2s.greedy_decode(walls, robots_a, mid_goal, max_len=args.max_len)[0].cpu().tolist()

        reached_mid = False
        for action in seq_to_mid:
            if action == EOS_TOKEN:
                break
            if action < 0 or action > 15:
                break
            state, moved = apply_action(walls_np, state, action, size=16)
            if not moved:
                break
            if state[target_robot, 0] == mid_r and state[target_robot, 1] == mid_c:
                reached_mid = True
                break

        if not reached_mid:
            hallucinations += 1
            total += 1
            continue

        robots_mid_state = torch.tensor(state, dtype=torch.long).unsqueeze(0).to(device)

        with torch.no_grad():
            seq_to_goal = s2s.greedy_decode(walls, robots_mid_state, goal, max_len=args.max_len)[0].cpu().tolist()

        reached_goal = False
        goal_r = int(goal[0, 1].item())
        goal_c = int(goal[0, 2].item())

        for action in seq_to_goal:
            if action == EOS_TOKEN:
                break
            if action < 0 or action > 15:
                break
            state, moved = apply_action(walls_np, state, action, size=16)
            if not moved:
                break
            if state[target_robot, 0] == goal_r and state[target_robot, 1] == goal_c:
                reached_goal = True
                break

        if reached_goal:
            successes += 1
        else:
            hallucinations += 1
        total += 1

    success_rate = successes / max(total, 1)
    hallucination_rate = hallucinations / max(total, 1)
    print(f"success_rate {success_rate:.4f}")
    print(f"hallucination_rate {hallucination_rate:.4f}")


if __name__ == "__main__":
    main()
