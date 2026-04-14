from __future__ import annotations
import argparse
import random
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import os
import time
from collections import deque

ALL_ACTIONS    = ["L45", "L22", "FW", "R22", "R45"]  
MOTION_ACTIONS = ["FW", "L22", "R22"]                


ROTATION_RECIPES = {
    "wall_no_box":   ["L45", "L45", "L45", "FW"],
    "wall_with_box": ["L45", "L45", "L45", "FW"],
    "escape_follow": ["L45", "L45", "L45", "FW"],
}


class DuelingDRQN(nn.Module):
    def __init__(self, in_dim: int = 18, n_actions: int = len(MOTION_ACTIONS)):
        super().__init__()
        self.feature_layer = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(),
            nn.LayerNorm(256),
        )
        self.lstm = nn.LSTM(input_size=256, hidden_size=256, batch_first=True)
        self.advantage = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, n_actions)
        )
        self.value = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 1)
        )

    def forward(self, x, hidden=None):
        if x.dim() == 2:
            x = x.unsqueeze(0)
        x = self.feature_layer(x)
        output, next_hidden = self.lstm(x, hidden)
        last = output[:, -1, :]
        adv  = self.advantage(last)
        val  = self.value(last)
        q    = val + (adv - adv.mean(dim=1, keepdim=True))
        return q, next_hidden


class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6):
        self.capacity  = capacity
        self.alpha     = alpha
        self.buffer    = []
        self.pos       = 0
        self.priorities = np.zeros((capacity,), dtype=np.float32)

    def push(self, state_seq, action, reward, next_state_seq, done):
        max_prio = self.priorities.max() if self.buffer else 1.0
        entry = (state_seq, action, reward, next_state_seq, done)
        if len(self.buffer) < self.capacity:
            self.buffer.append(entry)
        else:
            self.buffer[self.pos] = entry
        self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size, beta=0.4):
        if len(self.buffer) < batch_size:
            return None
        prios  = self.priorities[: len(self.buffer)]
        probs  = prios ** self.alpha
        probs /= probs.sum()
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[i] for i in indices]
        weights = (len(self.buffer) * probs[indices]) ** (-beta)
        weights /= weights.max()
        return samples, indices, torch.FloatTensor(weights)

    def update_priorities(self, indices, priorities):
        for idx, prio in zip(indices, priorities):
            self.priorities[idx] = prio

def import_obelix(obelix_py: str):
    import importlib.util
    spec = importlib.util.spec_from_file_location("obelix_env", obelix_py)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.OBELIX

def choose_rotation_recipe(ir_on: bool, stuck_on: bool, prev_stuck: bool) -> list[str] | None:
    if stuck_on and not ir_on:
        return list(ROTATION_RECIPES["wall_no_box"])
    if stuck_on and ir_on:
        return list(ROTATION_RECIPES["wall_with_box"])
    if prev_stuck and not stuck_on:
        return list(ROTATION_RECIPES["escape_follow"])
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--obelix_py",     type=str, required=True)
    ap.add_argument("--out",           type=str, default="final_wei.pth")
    ap.add_argument("--episodes",      type=int, default=1500)
    ap.add_argument("--difficulty",    type=int, default=3)
    ap.add_argument("--lr",            type=float, default=1e-4)
    ap.add_argument("--wall_obstacles", action="store_true")
    args = ap.parse_args()

    OBELIX = import_obelix(args.obelix_py)
    env = OBELIX(
        scaling_factor=5, arena_size=500, max_steps=2000,
        difficulty=args.difficulty, wall_obstacles=args.wall_obstacles,
    )

    n_motion = len(MOTION_ACTIONS)
    policy_net = DuelingDRQN(in_dim=18, n_actions=n_motion)
    target_net = DuelingDRQN(in_dim=18, n_actions=n_motion)
    target_net.load_state_dict(policy_net.state_dict())

    optimizer = optim.AdamW(policy_net.parameters(), lr=args.lr)
    memory    = PrioritizedReplayBuffer(30_000)

    epsilon, eps_decay, history_len = 0.9, 0.995, 8
    total_train_start = time.perf_counter()

    for ep in range(1, args.episodes + 1):
        ep_start_time = time.time()
        s = env.reset()
        state_history = deque([s] * history_len, maxlen=history_len)
        train_ret, hidden, is_pushing = 0.0, None, False
        rotation_queue: deque[str] = deque()
        prev_stuck = False

        for t in range(2000):
            ir_on, stuck_on = (s[16] == 1), (s[17] == 1)

            if not rotation_queue:
                recipe = choose_rotation_recipe(ir_on, stuck_on, prev_stuck)
                if recipe: rotation_queue.extend(recipe)

            if rotation_queue:
                action_str = rotation_queue.popleft()
                s2, r, done = env.step(action_str)
                train_ret += r
            else:
                seq_t = torch.FloatTensor(np.array(state_history)).unsqueeze(0)

                if is_pushing and ir_on:
                    a_local = MOTION_ACTIONS.index("FW")
                else:
                    if random.random() < epsilon:
                        a_local = random.randint(0, n_motion - 1)
                    else:
                        policy_net.eval()
                        with torch.no_grad():
                            q_vals, hidden = policy_net(seq_t, hidden)
                            a_local = q_vals.argmax().item()

                action_str = MOTION_ACTIONS[a_local]
                s2, r, done = env.step(action_str)
                shaped_r = r

                ir2, stuck2 = (s2[16] == 1), (s2[17] == 1)
                if ir2 and not stuck2 and action_str == "FW":
                    shaped_r += 40.0
                    is_pushing = True
                if is_pushing and action_str == "FW" and not stuck2:
                    shaped_r += 45.0
                elif not ir2:
                    is_pushing = False

                train_ret += r
                next_history = list(state_history)[1:] + [s2]
                memory.push(np.array(state_history), a_local, shaped_r, np.array(next_history), done)

                if len(memory.buffer) > 128:
                    policy_net.train()
                    data = memory.sample(64)
                    if data:
                        samples, indices, weights = data
                        b_s  = torch.FloatTensor(np.array([samp[0] for samp in samples]))
                        b_a  = torch.LongTensor([samp[1] for samp in samples]).unsqueeze(1)
                        b_r  = torch.FloatTensor([samp[2] for samp in samples])
                        b_s2 = torch.FloatTensor(np.array([samp[3] for samp in samples]))
                        b_d  = torch.FloatTensor([samp[4] for samp in samples])

                        curr_q, _ = policy_net(b_s)
                        curr_q = curr_q.gather(1, b_a).squeeze()
                        with torch.no_grad():
                            next_a = policy_net(b_s2)[0].argmax(1).unsqueeze(1)
                            next_q = target_net(b_s2)[0].gather(1, next_a).squeeze()
                            tgt_q  = b_r + (1 - b_d) * 0.99 * next_q

                        loss = (weights * nn.functional.mse_loss(curr_q, tgt_q, reduction='none')).mean()
                        optimizer.zero_grad()
                        loss.backward()
                        nn.utils.clip_grad_norm_(policy_net.parameters(), 1.0)
                        optimizer.step()
                        memory.update_priorities(indices, torch.abs(curr_q - tgt_q).detach().numpy() + 1e-5)

            prev_stuck = stuck_on
            state_history.append(s2)
            s = s2
            if done: break

        ep_duration = time.time() - ep_start_time
        total_elapsed = time.perf_counter() - total_train_start
        epsilon = max(0.05, epsilon * eps_decay)
        
        if ep % 5 == 0: 
            target_net.load_state_dict(policy_net.state_dict())
        
        torch.save(policy_net.state_dict(), args.out)
        
        print(f"Ep {ep:4d} | Ret: {train_ret:8.1f} | Eps: {epsilon:.3f} | "
              f"Ep Time: {ep_duration:5.2f}s | Total: {total_elapsed/60:6.2f}m | "
              f"Saved: {os.path.basename(args.out)}")

    print(f"\nTraining Complete. Final weights saved to: {args.out}")

if __name__ == "__main__":
    main()