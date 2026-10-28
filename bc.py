"""Behaviour cloning: a small MLP (NumPy only, written out by hand) that maps observations to actions, trained on
successful teleoperated demonstrations, then run closed-loop in the simulator on object poses it never saw.
"""
import numpy as np
from sim import HandEnv
from teleop import sample_pose
from human import demo, DT, T_TOTAL
from hand import OBJ_HALF_HEIGHT


class MLP:
    """obs -> action. Two hidden layers (ReLU), inputs and outputs standardised, Adam, mini-batches."""

    def __init__(self, n_in, n_out, hidden=256, seed=0):
        r = np.random.default_rng(seed)
        sizes = [n_in, hidden, hidden, n_out]
        self.W = [r.normal(0, np.sqrt(2 / a), (a, b)) for a, b in zip(sizes[:-1], sizes[1:])]
        self.b = [np.zeros(b) for b in sizes[1:]]

    def forward(self, x, keep=False):
        hs = [x]
        for k, (W, b) in enumerate(zip(self.W, self.b)):
            x = x @ W + b
            if k < len(self.W) - 1:
                x = np.maximum(x, 0)
            hs.append(x)
        return (x, hs) if keep else x

    def fit(self, X, Y, epochs=60, batch=256, lr=1e-3, wd=1e-5, seed=0, val_frac=0.1):
        self.xm, self.xs = X.mean(0), X.std(0) + 1e-6
        self.ym, self.ys = Y.mean(0), Y.std(0) + 1e-6
        Xn, Yn = (X - self.xm) / self.xs, (Y - self.ym) / self.ys
        r = np.random.default_rng(seed)
        idx = r.permutation(len(X))
        nv = int(len(X) * val_frac)
        va, tr = idx[:nv], idx[nv:]
        params = self.W + self.b
        m = [np.zeros_like(p) for p in params]
        v = [np.zeros_like(p) for p in params]
        t, hist = 0, []
        for ep in range(epochs):
            lr_ep = lr * 0.5 * (1 + np.cos(np.pi * ep / epochs))            # cosine decay
            perm = r.permutation(tr)
            for s in range(0, len(perm), batch):
                bi = perm[s:s + batch]
                out, hs = self.forward(Xn[bi], keep=True)
                g = 2 * (out - Yn[bi]) / len(bi) / Yn.shape[1]
                gW, gb = [None] * 3, [None] * 3
                for k in (2, 1, 0):
                    gW[k] = hs[k].T @ g + wd * self.W[k]
                    gb[k] = g.sum(0)
                    if k > 0:
                        g = (g @ self.W[k].T) * (hs[k] > 0)
                t += 1
                for i, (p, gr) in enumerate(zip(params, gW + gb)):
                    m[i] = 0.9 * m[i] + 0.1 * gr
                    v[i] = 0.999 * v[i] + 0.001 * gr * gr
                    p -= lr_ep * (m[i] / (1 - 0.9 ** t)) / (np.sqrt(v[i] / (1 - 0.999 ** t)) + 1e-8)
            hist.append((self.loss(Xn[tr[:5000]], Yn[tr[:5000]]), self.loss(Xn[va], Yn[va])))
        return hist

    def loss(self, Xn, Yn):
        return float(np.mean((self.forward(Xn) - Yn) ** 2))

    def act(self, x):
        return self.forward((x - self.xm) / self.xs) * self.ys + self.ym


def dataset(episodes):
    X = np.concatenate([e["obs"] for e in episodes])
    Y = np.concatenate([e["act"] for e in episodes])
    return X, Y


def rollout(policy, env, rng, obj_noise=0.0, tactile=True):
    """Closed-loop run of a policy on a fresh random object pose. obj_noise = std (m) of a constant error in the
    tracked object position, redrawn each episode (a biased pose estimate), plus 1 mm frame-to-frame jitter."""
    xy, yaw = sample_pose(rng)
    start = demo(env.obj, xy, yaw, rng, OBJ_HALF_HEIGHT[env.obj], kp_noise=0, wrist_noise=0)["palm"][0]
    env.reset(xy, yaw, start)
    n = int(round(T_TOTAL / DT)) + 1
    bias = rng.normal(0, obj_noise, 3) if obj_noise > 0 else np.zeros(3)
    touched = False
    for i in range(n):
        noise = bias + (rng.normal(0, 0.001, 3) if obj_noise > 0 else 0)
        a = policy.act(env.observation(i / (n - 1), obj_noise=noise, tactile=tactile))
        palm_t, q = env.action_to_targets(a)
        env.step(palm_t, q)
        touched |= env.contacts_with_hand() > 0
    from sim import failure_reason
    return env.lifted(), ("ok" if env.lifted() else failure_reason(env, touched))


def evaluate(policy, n_per_obj, seed, obj_noise=0.0, tactile=True, objects=("cube", "cylinder", "sphere")):
    rng = np.random.default_rng(seed)
    res = {}
    for obj in objects:
        env = HandEnv(obj)
        res[obj] = [rollout(policy, env, rng, obj_noise, tactile) for _ in range(n_per_obj)]
    return res
