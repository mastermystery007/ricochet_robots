# eval_compare.py
import os
import random
import numpy as np
import torch

from environment import RicochetEnv
from tokenizer import RicochetTokenizer
from model import RicochetModel
from s2s_model import Seq2SeqActionModel
from solve_s2s_pv import recursive_solve, goal_satisfied_partial

def run_s2s_direct(env, s2s, walls_np, start_state, goal_flat_np, budget=15, device="cpu", grid_size=16):
    walls_t = torch.tensor(walls_np, dtype=torch.long, device=device).unsqueeze(0)
    robots_t = torch.tensor(start_state, dtype=torch.long, device=device).unsqueeze(0)
    goal_t = torch.tensor(goal_flat_np, dtype=torch.long, device=device).unsqueeze(0)

    acts = s2s.greedy_decode(walls_t, robots_t, goal_t, max_steps=budget)

    env.set_state(start_state.copy())
    for t, a in enumerate(acts, start=1):
        env.step(a)
        st = env.get_state()
        # final goal: target robot at goal cell
        tr = int(goal_flat_np[0])
        goal_cell = int(goal_flat_np[1]) * grid_size + int(goal_flat_np[2])
        if goal_satisfied_partial(st, tr, goal_cell, None, None, size=grid_size):
            return True, acts[:t]
    return False, acts

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    grid_size = 16
    budget = 15
    trials = 200

    # load models
    tokenizer = RicochetTokenizer(d_model=256, grid_size=16).to(device)

    pv = RicochetModel(tokenizer, d_model=256, nhead=8, num_layers=6, grid_size=16, dist_bins=21).to(device)
    pv.load_state_dict(torch.load("checkpoints_pv/pv_final.pt", map_location=device))
    pv.eval()

    s2s = Seq2SeqActionModel(tokenizer, d_model=256, nhead=8, num_layers=6, max_seq_len=15, action_vocab=16).to(device)
    s2s.load_state_dict(torch.load("checkpoints_s2s/s2s_final.pt", map_location=device))
    s2s.eval()

    env = RicochetEnv(size=16, num_robots=4)

    succ_s2s = 0
    succ_pv = 0
    len_s2s = []
    len_pv = []

    for _ in range(trials):
        env.generate_random_board()
        walls = env.walls.copy()
        start_state = env.get_state()

        # pick a random target goal using another random rollout step as "goal"
        # simplest: choose a random desired cell for target robot by simulating a few random moves
        tmp_env = RicochetEnv(size=16, num_robots=4)
        tmp_env.walls = walls.copy()
        tmp_env.set_state(start_state.copy())

        for _t in range(random.randint(1, 10)):
            tmp_env.step(random.randint(0, 15))
        final_state = tmp_env.get_state()

        target_robot = random.randint(0, 3)
        goal_flat = np.array([target_robot, final_state[target_robot, 0], final_state[target_robot, 1]], dtype=np.int64)

        ok_s2s, acts_s2s = run_s2s_direct(env, s2s, walls, start_state, goal_flat, budget=budget, device=device, grid_size=grid_size)
        ok_pv, acts_pv = recursive_solve(env, pv, s2s, walls, start_state, goal_flat, total_budget=budget, k=4, device=device, grid_size=grid_size)

        if ok_s2s:
            succ_s2s += 1
            len_s2s.append(len(acts_s2s))
        if ok_pv:
            succ_pv += 1
            len_pv.append(len(acts_pv))

    print(f"S2S direct success: {succ_s2s}/{trials} = {succ_s2s/trials:.3f}")
    print(f"PV+S2S recursive success: {succ_pv}/{trials} = {succ_pv/trials:.3f}")
    if len_s2s:
        print(f"S2S avg length (successful): {sum(len_s2s)/len(len_s2s):.2f}")
    if len_pv:
        print(f"PV avg length (successful): {sum(len_pv)/len(len_pv):.2f}")

if __name__ == "__main__":
    main()
