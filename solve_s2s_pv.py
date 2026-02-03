# solver_recursive_pv_s2s.py
import torch
import torch.nn.functional as F

from environment import RicochetEnv

def cell_to_rc(cell, size=16):
    r = int(cell) // size
    c = int(cell) % size
    return r, c

def goal_satisfied_partial(state_rc, target_robot, target_cell, helper_id=None, helper_cell=None, size=16):
    tr, tc = cell_to_rc(target_cell, size=size)
    if int(state_rc[target_robot, 0]) != tr or int(state_rc[target_robot, 1]) != tc:
        return False
    if helper_id is not None and helper_cell is not None:
        hr, hc = cell_to_rc(helper_cell, size=size)
        if int(state_rc[helper_id, 0]) != hr or int(state_rc[helper_id, 1]) != hc:
            return False
    return True

@torch.no_grad()
def propose_k_candidates(model_pv, walls, robots, goal, k=4):
    """
    Returns a list of candidate subgoals:
      each: dict(target_cell, helper_id, helper_cell)
    Strategy:
      - take topk target cells
      - for each, take argmax helper_id and helper_cell (simple v1)
    """
    prop = model_pv.propose(walls, robots, goal)
    cell_logits = prop["cell_logits"]          # [1,256]
    helper_id_logits = prop["robot_logits"]    # [1,4]
    helper_cell_logits = prop["cell2_logits"]  # [1,256]

    topk = torch.topk(cell_logits, k=min(k, cell_logits.shape[-1]), dim=-1).indices[0].tolist()
    hid = int(helper_id_logits.argmax(dim=-1).item())
    hcell = int(helper_cell_logits.argmax(dim=-1).item())

    cands = []
    for c in topk:
        cands.append({"target_cell": int(c), "helper_id": hid, "helper_cell": hcell})
    return cands

@torch.no_grad()
def rank_candidates_by_verifier(model_pv, walls, robots_a, target_robot, goal_cell, candidates, grid_size=16):
    """
    Uses verifier as a scoring heuristic.
    We need a "robots_b" full state for verifier; we cannot directly create it from partial goal.
    So for ranking we use an approximate proxy:
      - construct a proxy state equal to robots_a but with target robot moved to candidate target_cell
      - and helper robot moved to candidate helper_cell
    This is only a heuristic ranking.
    """
    def proxy_state(base, cand):
        st = base.clone()
        tr, tc = cell_to_rc(cand["target_cell"], size=grid_size)
        st[0, target_robot, 0] = tr
        st[0, target_robot, 1] = tc
        hid = cand["helper_id"]
        hr, hc = cell_to_rc(cand["helper_cell"], size=grid_size)
        st[0, hid, 0] = hr
        st[0, hid, 1] = hc
        return st

    best = None
    best_score = None
    for cand in candidates:
        rb = proxy_state(robots_a.clone(), cand)
        dist_logits = model_pv.verify(walls, robots_a, rb)  # [1,21]
        # lower bin is better
        score = int(dist_logits.argmax(dim=-1).item())
        if best_score is None or score < best_score:
            best_score = score
            best = cand
    return best, best_score

def simulate_actions(env, actions):
    """
    Applies actions in-place to env; returns final state.
    """
    for a in actions:
        env.step(int(a))
    return env.get_state()

@torch.no_grad()
def execute_to_subgoal_s2s(env, s2s_model, walls_t, start_state, goal, subgoal, budget, device="cpu", grid_size=16):
    """
    Uses S2S to generate up to 'budget' actions; simulates them; success if subgoal achieved.
    """
    env.set_state(start_state.copy())

    robots_t = torch.tensor(env.get_state(), dtype=torch.long, device=device).unsqueeze(0)
    goal_t = goal.to(device)

    acts = s2s_model.greedy_decode(walls_t, robots_t, goal_t, max_steps=budget)

    # simulate and check success after each action (early stop)
    env.set_state(start_state.copy())
    for t, a in enumerate(acts, start=1):
        env.step(a)
        st = env.get_state()
        ok = goal_satisfied_partial(
            st,
            target_robot=int(goal_t[0, 0].item()),
            target_cell=subgoal["target_cell"],
            helper_id=subgoal["helper_id"],
            helper_cell=subgoal["helper_cell"],
            size=grid_size
        )
        if ok:
            return True, acts[:t], st

    return False, acts, env.get_state()

def recursive_solve(env, model_pv, model_s2s, walls_np, start_state, goal_flat_np, total_budget=15, k=4, depth=0, max_depth=6, device="cpu", grid_size=16):
    """
    PV proposes subgoal(s), verifier ranks, then S2S executes.
    Recurses until final goal reached or budget exhausted.
    Returns: (success_bool, action_list)
    """
    # goal tensor for tokenizer: [1,3]
    walls_t = torch.tensor(walls_np, dtype=torch.long, device=device).unsqueeze(0)
    goal_t = torch.tensor(goal_flat_np, dtype=torch.long, device=device).unsqueeze(0)

    # base case: check final goal (target robot at goal cell)
    target_robot = int(goal_flat_np[0])
    goal_cell = int(goal_flat_np[1]) * grid_size + int(goal_flat_np[2])
    if goal_satisfied_partial(start_state, target_robot, goal_cell, helper_id=None, helper_cell=None, size=grid_size):
        return True, []

    if total_budget <= 0:
        return False, []

    if depth >= max_depth:
        return False, []

    # propose candidates using current state as "robots" input
    robots_t = torch.tensor(start_state, dtype=torch.long, device=device).unsqueeze(0)
    cands = propose_k_candidates(model_pv, walls_t, robots_t, goal_t, k=k)

    # rank candidates by verifier heuristic
    best_cand, best_score = rank_candidates_by_verifier(
        model_pv, walls_t, robots_t, target_robot, goal_cell, cands, grid_size=grid_size
    )

    # allocate budget roughly half/half
    b1 = max(1, total_budget // 2)
    b2 = total_budget - b1

    # execute to subgoal
    ok1, acts1, mid_state = execute_to_subgoal_s2s(
        env, model_s2s, walls_t, start_state, goal_t, best_cand, budget=b1, device=device, grid_size=grid_size
    )
    if not ok1:
        return False, []

    # now solve from midpoint to final goal with remaining budget
    ok2, acts2 = recursive_solve(
        env, model_pv, model_s2s, walls_np, mid_state, goal_flat_np,
        total_budget=b2, k=k, depth=depth+1, max_depth=max_depth,
        device=device, grid_size=grid_size
    )
    if not ok2:
        return False, []

    return True, acts1 + acts2
