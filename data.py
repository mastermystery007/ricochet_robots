# data.py
import random
import numpy as np
import torch
from torch.utils.data import Dataset

DMAX = 20  # bin 20 means >=20

# Must match environment.py constants
NORTH, EAST, SOUTH, WEST = 1, 2, 4, 8
DIRS = {NORTH: (-1, 0), EAST: (0, 1), SOUTH: (1, 0), WEST: (0, -1)}
DIR_LIST = [NORTH, EAST, SOUTH, WEST]


def cell_id(r, c, size=16):
    return int(r) * size + int(c)

def action_robot_id(a):
    return a // 4

def action_dir(a):
    return DIR_LIST[a % 4]

def blocked_by_wall(walls, r, c, direction):
    return (int(walls[r, c]) & int(direction)) != 0


def slide_and_get_stopper(walls, state_rc, robot_id, direction, size=16):
    """
    Pure transition (no env mutation) that also returns which robot stopped the slide.
    stopper_id:
      - None if wall/bounds stopped it
      - int robot_id if another robot blocked
    """
    dr, dc = DIRS[direction]
    r, c = int(state_rc[robot_id, 0]), int(state_rc[robot_id, 1])

    occ = {(int(rr), int(cc)) for rr, cc in state_rc}
    occ.remove((r, c))

    stopper = None
    while True:
        # wall blocks leaving current cell
        if blocked_by_wall(walls, r, c, direction):
            stopper = None
            break

        nr, nc = r + dr, c + dc
        if not (0 <= nr < size and 0 <= nc < size):
            stopper = None
            break

        if (nr, nc) in occ:
            # stopped by another robot
            # identify which robot is at (nr,nc)
            for rid in range(state_rc.shape[0]):
                if rid == robot_id:
                    continue
                if int(state_rc[rid, 0]) == nr and int(state_rc[rid, 1]) == nc:
                    stopper = rid
                    break
            break

        # move
        r, c = nr, nc

    next_state = state_rc.copy()
    moved = not (r == int(next_state[robot_id, 0]) and c == int(next_state[robot_id, 1]))
    next_state[robot_id] = (r, c)
    return next_state, moved, stopper


class HindsightDataset(Dataset):
    def __init__(
        self,
        env,
        num_episodes=1000,
        rollout_len=15,
        max_pairs_per_episode=1000,
        min_seg_len=1,
        max_seg_len=None,
        grid_size=16,
    ):
        self.env = env
        self.grid_size = grid_size
        self.data = []
        self.total_pairs = 0
        self.episodes = 0
        self._generate_data(
            num_episodes,
            rollout_len,
            max_pairs_per_episode,
            min_seg_len,
            max_seg_len,
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

    def _choose_helper_by_interaction(self, walls, s_i, actions_segment, target_robot, size=16):
        """
        Helper = robot that most often blocks the target robot during target moves
        in this segment. Also returns helper_cell_label = helper position at first
        such interaction time-step (in trajectory time, before applying that action).

        Returns: (helper_id, helper_cell_id, found_interaction_bool)
        """
        interaction_counts = [0, 0, 0, 0]
        first_pos = {0: None, 1: None, 2: None, 3: None}

        # simulate forward from s_i applying the actions in the segment
        state = s_i.copy()

        for a in actions_segment:
            rid = action_robot_id(a)
            direction = action_dir(a)

            if rid == target_robot:
                # Before applying the move, see who would stop it
                next_state, moved, stopper = slide_and_get_stopper(
                    walls, state, target_robot, direction, size=size
                )
                if stopper is not None and stopper != target_robot:
                    interaction_counts[stopper] += 1
                    if first_pos[stopper] is None:
                        hr, hc = int(state[stopper, 0]), int(state[stopper, 1])
                        first_pos[stopper] = cell_id(hr, hc, size=size)
                # advance state
                state = next_state
            else:
                # apply other robot action without needing stopper info
                next_state, moved, stopper = slide_and_get_stopper(
                    walls, state, rid, direction, size=size
                )
                state = next_state

        # choose best helper
        best_id = None
        best_ct = -1
        for r in range(4):
            if r == target_robot:
                continue
            if interaction_counts[r] > best_ct:
                best_ct = interaction_counts[r]
                best_id = r

        found = (best_ct > 0)
        if not found:
            # fallback: deterministic non-target
            best_id = (target_robot + 1) % 4
            first_helper_cell = None

        else:
            first_helper_cell = first_pos[best_id]
            if first_helper_cell is None:
                hr, hc = int(s_i[best_id, 0]), int(s_i[best_id, 1])
                first_helper_cell = cell_id(hr, hc, size=size)

        return best_id, int(first_helper_cell), found

    def _generate_data(self, num_episodes, rollout_len, max_pairs_per_episode, min_seg_len, max_seg_len):
        for _ in range(num_episodes):
            self.env.generate_random_board()
            walls = self.env.walls.copy()

            states, actions = self._random_walk_episode(steps=rollout_len)
            T = len(states)
            if T < 4:
                continue

            max_len = (T - 1) if max_seg_len is None else min(max_seg_len, T - 1)
            seg_pairs = []
            for i in range(0, T - 1):
                for j in range(i + min_seg_len, T):
                    seg_len = j - i
                    if seg_len < min_seg_len:
                        continue
                    if seg_len > max_len:
                        continue
                    seg_pairs.append((i, j))

            if not seg_pairs:
                continue

            if len(seg_pairs) > max_pairs_per_episode:
                seg_pairs = random.sample(seg_pairs, max_pairs_per_episode)

            for i, j in seg_pairs:
                k = (i + j) // 2

                s_i = states[i]
                s_j = states[j]
                s_k = states[k]

                # Choose target robot and define goal as its final cell at s_j
                target_robot = random.randint(0, 3)
                goal_flat = np.array([target_robot, s_j[target_robot, 0], s_j[target_robot, 1]], dtype=np.int64)

                # Segment actions corresponding to s_i -> s_j
                actions_segment = actions[i:j]

                # Target label: midpoint cell for target robot (unchanged)
                mid_cell = cell_id(s_k[target_robot, 0], s_k[target_robot, 1], self.grid_size)

                # Helper labels: interaction-based + first-interaction cell
                helper_id, helper_cell, found = self._choose_helper_by_interaction(
                    walls, s_i, actions_segment, target_robot, size=self.grid_size
                )
                if not found:
                    hr, hc = int(s_k[helper_id, 0]), int(s_k[helper_id, 1])
                    helper_cell = cell_id(hr, hc, size=self.grid_size)

                # Verifier label: rollout distance bin
                dist = j - i
                dist_bin = dist if dist < DMAX else DMAX

                self.data.append({
                    "walls": walls,
                    "robots_a": s_i,
                    "robots_b": s_j,
                    "goal": goal_flat,
                    "mid_cell": int(mid_cell),
                    "helper_id": int(helper_id),
                    "helper_cell": int(helper_cell),
                    "dist": int(dist_bin),
                })

            self.total_pairs += len(seg_pairs)
            self.episodes += 1

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
            "helper_id": torch.tensor(d["helper_id"], dtype=torch.long),
            "helper_cell": torch.tensor(d["helper_cell"], dtype=torch.long),
            "dist": torch.tensor(d["dist"], dtype=torch.long),
        }
