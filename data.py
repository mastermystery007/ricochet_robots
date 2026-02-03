import random
import numpy as np
import torch
from torch.utils.data import Dataset

DMAX = 20  # bin 20 means >=20

# direction index: 0=N,1=E,2=S,3=W
DRDC = {
    0: (-1, 0),
    1: (0, +1),
    2: (+1, 0),
    3: (0, -1),
}

def cell_id(r, c, size=16):
    return int(r) * size + int(c)

def action_robot_id(a):
    return a // 4

def action_dir_idx(a):
    return a % 4


class HindsightDataset(Dataset):
    """
    PV training dataset:
      - walls
      - robots_a = state at time i
      - robots_b = state at time j   (verifier supervision)
      - goal = (target_robot, r_goal, c_goal) where (r_goal,c_goal) taken from robots_b[target_robot]
      - mid_cell = target robot cell at midpoint k = floor((i+j)/2)
      - helper/blocker robot id and its cell at FIRST interaction with target (if any)
      - dist bin = min(j-i, 20)

    Key change: supports COMPLETE SUBSETS (all i<j) when pairs_per_episode=None.
    Key change: helper chosen by “most interactions blocking target”.
    """
    def __init__(
        self,
        env,
        num_episodes=1000,
        rollout_len=15,
        pairs_per_episode=None,          # None => ALL PAIRS
        grid_size=16,
        include_all_one_step=True,       # extra 1-step segments
        max_pairs_cap=None               # optional safety cap per episode
    ):
        self.env = env
        self.grid_size = grid_size
        self.data = []
        self._generate_data(
            num_episodes=num_episodes,
            rollout_len=rollout_len,
            pairs_per_episode=pairs_per_episode,
            include_all_one_step=include_all_one_step,
            max_pairs_cap=max_pairs_cap,
        )

    def _random_walk_episode(self, steps=15):
        states = [self.env.get_state()]
        actions = []
        for _ in range(steps):
            for _try in range(20):
                a = random.randint(0, 15)
                moved = self.env.step(a)
                if moved:
                    actions.append(a)
                    states.append(self.env.get_state())
                    break
            else:
                break
        return states, actions

    def _helper_by_interaction(self, states, actions, i, j, target_robot):
        """
        Interaction definition (v1, reliable from (state,action) logs):
        - Consider each step t in [i, j-1]
        - If action at t moves the TARGET robot in direction dir,
          then the target stops either due to wall or due to another robot.
        - We infer "blocked by robot X" if the cell immediately in front of the
          target's final position (in that direction) is occupied by some robot in state_before.
        - Count blockers; choose the most frequent blocker as helper.
        - Helper position label = helper's position at the FIRST blocking event.
        """
        counts = [0, 0, 0, 0]
        first_block_pos = [None, None, None, None]  # store (r,c) when first blocks

        for t in range(i, j):
            if t >= len(actions):
                break
            a = int(actions[t])
            rid = action_robot_id(a)
            if rid != target_robot:
                continue

            dir_idx = action_dir_idx(a)
            dr, dc = DRDC[dir_idx]

            s_before = states[t]
            s_after = states[t + 1]

            # target final position after sliding
            tr, tc = int(s_after[target_robot, 0]), int(s_after[target_robot, 1])
            front = (tr + dr, tc + dc)

            # if front cell is occupied by some other robot in s_before => blocked by that robot
            for r in range(4):
                if r == target_robot:
                    continue
                rr, rc = int(s_before[r, 0]), int(s_before[r, 1])
                if (rr, rc) == front:
                    counts[r] += 1
                    if first_block_pos[r] is None:
                        first_block_pos[r] = (rr, rc)
                    break

        # choose helper with max interaction
        best_r = None
        best_c = 0
        for r in range(4):
            if r == target_robot:
                continue
            if counts[r] > best_c:
                best_c = counts[r]
                best_r = r

        if best_r is None or best_c == 0:
            # fallback: deterministic non-target
            best_r = (target_robot + 1) % 4
            # fallback label position = midpoint position (keeps labels stable)
            s_mid = states[(i + j) // 2]
            hr, hc = int(s_mid[best_r, 0]), int(s_mid[best_r, 1])
            return best_r, (hr, hc)

        # helper cell = helper position at FIRST interaction
        hr, hc = first_block_pos[best_r]
        return best_r, (int(hr), int(hc))

    def _generate_pairs(self, T, pairs_per_episode, include_all_one_step, max_pairs_cap):
        pairs = []

        if pairs_per_episode is None:
            # COMPLETE SUBSETS
            for i in range(0, T - 1):
                for j in range(i + 1, T):
                    pairs.append((i, j))
        else:
            # sampled subsets
            for _ in range(pairs_per_episode):
                i = random.randint(0, T - 2)
                j = random.randint(i + 1, T - 1)
                pairs.append((i, j))

        if include_all_one_step:
            for i in range(0, T - 1):
                pairs.append((i, i + 1))

        # de-dup
        pairs = list(set(pairs))

        # optional cap to avoid explosion (complete subsets can be large if rollout_len grows)
        if max_pairs_cap is not None and len(pairs) > max_pairs_cap:
            pairs = random.sample(pairs, max_pairs_cap)

        return pairs

    def _generate_data(self, num_episodes, rollout_len, pairs_per_episode, include_all_one_step, max_pairs_cap):
        for _ in range(num_episodes):
            self.env.generate_random_board()
            walls = self.env.walls.copy()

            states, actions = self._random_walk_episode(steps=rollout_len)
            T = len(states)
            if T < 4:
                continue

            pairs = self._generate_pairs(T, pairs_per_episode, include_all_one_step, max_pairs_cap)

            for (i, j) in pairs:
                k = (i + j) // 2
                s_i = states[i]
                s_j = states[j]
                s_k = states[k]

                target_robot = random.randint(0, 3)
                goal_flat = np.array([target_robot, s_j[target_robot, 0], s_j[target_robot, 1]], dtype=np.int64)

                mid_cell = cell_id(s_k[target_robot, 0], s_k[target_robot, 1], self.grid_size)

                helper_id, (hr, hc) = self._helper_by_interaction(states, actions, i, j - 1, target_robot)
                helper_cell = cell_id(hr, hc, self.grid_size)

                dist = j - i
                dist_bin = dist if dist < DMAX else DMAX

                self.data.append({
                    "walls": walls,
                    "robots_a": s_i,
                    "robots_b": s_j,
                    "goal": goal_flat,
                    "mid_cell": mid_cell,
                    "helper_id": helper_id,
                    "helper_cell": helper_cell,
                    "dist": dist_bin,
                })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        d = self.data[idx]
        return {
            "walls": torch.tensor(d["walls"], dtype=torch.long),
            "robots": torch.tensor(d["robots_a"], dtype=torch.long),
            "final_robots": torch.tensor(d["robots_b"], dtype=torch.long),
            "goal": torch.tensor(d["goal"], dtype=torch.long),
            "mid_cell": torch.tensor(d["mid_cell"], dtype=torch.long),
            "block_id": torch.tensor(d["helper_id"], dtype=torch.long),
            "block_cell": torch.tensor(d["helper_cell"], dtype=torch.long),
            "dist": torch.tensor(d["dist"], dtype=torch.long),
        }