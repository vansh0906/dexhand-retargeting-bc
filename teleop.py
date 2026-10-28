"""Teleoperation: play a tracked human demonstration through a retargeter into the simulated robot hand, and
record (observation, action) pairs for imitation learning.

Noise injection (DART-style): with inject > 0 the executed palm and finger targets get random offsets that persist
for a few steps, but the recorded label is still the demonstration's target. The data then shows the policy how
to get back on track after small mistakes, which plain behaviour cloning never sees.
"""
import time
import numpy as np
from sim import HandEnv, failure_reason
from human import demo, T_TOTAL
from hand import OBJ_HALF_HEIGHT
from retarget import Retargeter, joint_angles_from_keypoints


def sample_pose(rng):
    """Random object placement on the table: +/- 6 cm, yaw +/- 0.4 rad."""
    return rng.uniform(-0.06, 0.06, 2), rng.uniform(-0.4, 0.4)


def teleop_episode(env, rng, method="optimize", retargeter=None, inject=0.0, record=True, tactile=True, ema=None):
    """One demonstration on a random object pose. method = "optimize" (fingertip retargeting) or "copy" (joint
    angles copied). ema = optional low-pass filter factor on the finger commands (0..1, smaller = smoother).
    Returns success, failure reason, tip errors, retargeting time per frame, finger commands and recorded data."""
    xy, yaw = sample_pose(rng)
    dm = demo(env.obj, xy, yaw, rng, OBJ_HALF_HEIGHT[env.obj])
    rt = retargeter or Retargeter()
    rt.q[:] = 0
    env.reset(xy, yaw, dm["palm"][0])
    n = len(dm["t"])
    obs, act, errs, solve_t = [], [], [], []
    touched = False
    off_palm, off_f = np.zeros(4), np.zeros(16)
    qs, q_f = [], None
    for i in range(n):
        t0 = time.perf_counter()
        if method == "optimize":
            q, _ = rt.solve(dm["kp"][i])
        else:
            q = joint_angles_from_keypoints(dm["kp"][i])
        if ema is not None:
            q_f = q if q_f is None else ema * q + (1 - ema) * q_f
            q = q_f
        solve_t.append(time.perf_counter() - t0)
        qs.append(q)
        errs.append(rt.tip_error(q, dm["kp"][i]))
        palm_t = dm["palm"][i]
        if record:
            obs.append(env.observation(i / (n - 1), tactile=tactile))
            act.append(env.targets_to_action(palm_t, q))
        if inject > 0 and i % 5 == 0:               # new disturbance every 0.25 s
            off_palm = rng.normal(0, inject, 4) * np.array([1, 1, 0.5, 2.0])
            off_f = rng.normal(0, inject * 10, 16)
        env.step(palm_t + off_palm, q + off_f)
        touched |= env.contacts_with_hand() > 0
    ok = bool(env.lifted())
    return {"ok": ok, "why": "ok" if ok else failure_reason(env, touched), "tip_err": np.array(errs),
            "solve_ms": 1000 * np.array(solve_t), "obs": np.array(obs), "act": np.array(act),
            "q": np.array(qs), "phase": dm["phase"], "obj": env.obj}


def collect(n_per_obj, seed, inject=0.0, objects=("cube", "cylinder", "sphere"), tactile=True):
    """Teleoperate n_per_obj demonstrations per object; keep the successful ones as training data."""
    rng = np.random.default_rng(seed)
    rt = Retargeter()
    data, stats = [], []
    for obj in objects:
        env = HandEnv(obj)
        for _ in range(n_per_obj):
            ep = teleop_episode(env, rng, retargeter=rt, inject=inject, tactile=tactile)
            stats.append(ep["ok"])
            if ep["ok"]:
                data.append(ep)
    return data, float(np.mean(stats))
