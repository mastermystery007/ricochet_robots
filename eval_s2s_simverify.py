import argparse
import numpy as np
import torch

from environment import RicochetEnv
from tokenizer import RicochetTokenizer
from seq2seq_model import RicochetSeq2SeqModel, EOS_TOKEN
from seq2seq_data import Seq2SeqDataset
from data import slide_and_get_stopper


def apply_action(walls, state, action, size=16):
    direction = [1, 2, 4, 8][action % 4]
    robot_id = action // 4
    next_state, moved, _stopper = slide_and_get_stopper(walls, state, robot_id, direction, size=size)
    return next_state, moved


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--rollout_len", type=int, default=25)
    parser.add_argument("--max_pairs_per_episode", type=int, default=1000)
    parser.add_argument("--max_len", type=int, default=25)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    env = RicochetEnv(size=16, num_robots=4)
    ds = Seq2SeqDataset(
        env,
        num_episodes=args.episodes,
        rollout_len=args.rollout_len,
        max_pairs_per_episode=args.max_pairs_per_episode,
        min_seg_len=1,
        max_seg_len=None,
        grid_size=16,
    )

    tokenizer = RicochetTokenizer(d_model=256, grid_size=16).to(device)
    model = RicochetSeq2SeqModel(tokenizer, d_model=256, nhead=8, num_layers=6).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model.eval()

    successes = 0
    total = 0
    invalid_moves = 0

    for ex in ds:
        walls = ex["walls"].unsqueeze(0).to(device)
        robots = ex["robots"].unsqueeze(0).to(device)
        goal = ex["goal"].unsqueeze(0).to(device)

        with torch.no_grad():
            seq = model.greedy_decode(walls, robots, goal, max_len=args.max_len)[0].cpu().tolist()

        state = ex["robots"].numpy().copy()
        target_robot = int(ex["goal"][0].item())
        goal_r = int(ex["goal"][1].item())
        goal_c = int(ex["goal"][2].item())
        walls_np = ex["walls"].numpy()

        for action in seq:
            if action == EOS_TOKEN:
                break
            if action < 0 or action > 15:
                invalid_moves += 1
                break
            state, moved = apply_action(walls_np, state, action, size=16)
            if not moved:
                invalid_moves += 1
                break

        success = int(state[target_robot, 0] == goal_r and state[target_robot, 1] == goal_c)
        successes += success
        total += 1

    success_rate = successes / max(total, 1)
    invalid_rate = invalid_moves / max(total, 1)
    print(f"success_rate {success_rate:.4f}")
    print(f"hallucination_rate {1 - success_rate:.4f}")
    print(f"invalid_action_rate {invalid_rate:.4f}")


if __name__ == "__main__":
    main()
