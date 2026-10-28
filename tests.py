"""Unit and integration checks. Run: python tests.py  (about 10 seconds)"""
import os
os.environ.setdefault("MUJOCO_GL", "disable")
import functools
import numpy as np
import mujoco

import human
import teleop
from hand import model_xml, OBJECTS
from human import keypoints, POSTURES, H_DIGITS, H_LINKS
from retarget import Retargeter, joint_angles_from_keypoints, KP_INDEX
from sim import HandEnv, FINGER_JOINTS
from bc import MLP

passed = 0


def check(name, cond, info=""):
    global passed
    if not cond:
        raise AssertionError(f"{name} FAILED {info}")
    passed += 1
    print(f"ok  {name} {info}")


def test_models():
    for obj in OBJECTS:
        m = mujoco.MjModel.from_xml_string(model_xml(obj))
        check(f"model {obj}: 4 wrist + 16 finger joints + free object", m.nu == 20 and m.nv == 26, f"nu={m.nu} nv={m.nv}")
    check("4 fingertip touch sensors", m.nsensor == 4)


def test_human_hand():
    q = {d: np.array(v) for d, v in POSTURES["cube"].items()}
    kp = keypoints(q)
    check("21 keypoints", kp.shape == (21, 3))
    for d in H_DIGITS:
        p = kp[KP_INDEX[d]:KP_INDEX[d] + 4]
        lens = np.linalg.norm(np.diff(p, axis=0), axis=1)
        assert np.allclose(lens, H_LINKS[d]), d
    check("keypoint bone lengths equal the human link lengths", True)
    est = joint_angles_from_keypoints(kp)
    truth = np.concatenate([q["thumb"], q["index"], q["middle"], 0.5 * (q["ring"] + q["pinky"])])
    check("joint-copy baseline recovers the joint angles from clean keypoints", np.allclose(est, truth, atol=1e-6),
          f"max err {np.abs(est - truth).max():.1e} rad")


def test_jacobian():
    rt = Retargeter()
    q = np.random.default_rng(0).uniform(rt.lo, rt.hi)
    p0 = rt.fk(q)
    J = np.zeros((3, rt.m.nv))
    mujoco.mj_jacSite(rt.m, rt.d, J, None, rt.sites[1])
    Jn = np.zeros((3, 16))
    for k in range(16):
        dq = np.zeros(16)
        dq[k] = 1e-6
        Jn[:, k] = (rt.fk(q + dq)[1] - p0[1]) / 1e-6
    rt.fk(q)
    # the palm site does not move with finger joints, so fingertip Jacobian = d(tip - palm)/dq
    check("MuJoCo fingertip Jacobian matches finite differences", np.allclose(J[:, rt.fdof], Jn, atol=1e-5),
          f"max diff {np.abs(J[:, rt.fdof] - Jn).max():.1e}")


def test_retarget():
    rt = Retargeter(w_pinch=0.0, aperture_gain=None, smooth=0.0, iters=60)
    q_open = {d: np.array([0.0, 0.3, 0.2, 0.1]) for d in H_DIGITS}
    q, err = rt.solve(keypoints(q_open), warm=False)
    check("fingertip retargeting reaches open-hand targets", err < 0.006, f"rms tip error {1000 * err:.1f} mm")
    check("retargeted angles respect the joint limits", np.all(q >= rt.lo - 1e-9) and np.all(q <= rt.hi + 1e-9))


def test_action_roundtrip():
    env = HandEnv("cube")
    env.reset([0.01, -0.02], 0.2, [0.0, 0.0, 0.2, 0.3])
    palm_t, fq = np.array([0.02, 0.01, 0.18, 0.35]), np.linspace(0, 0.3, 16)
    a = env.targets_to_action(palm_t, fq)
    p2, q2 = env.action_to_targets(a)
    check("action <-> target conversion is exact", np.allclose(p2, palm_t) and np.allclose(q2, fq))
    check("observation is 30-D (26 without touch)", env.observation(0.0).shape == (30,)
          and env.observation(0.0, tactile=False).shape == (26,))
    check("no touch force with the hand in the air", np.all(env.tactile() == 0))
    check("16 finger joints", len(FINGER_JOINTS) == 16)


def test_mlp():
    r = np.random.default_rng(0)
    X = r.normal(size=(3000, 6))
    Y = np.c_[np.sin(X[:, 0]) + X[:, 1] * X[:, 2], X[:, 3] - 0.5 * X[:, 4] ** 2]
    mlp = MLP(6, 2, hidden=64)
    hist = mlp.fit(X, Y, epochs=60)
    Yn = (Y - Y.mean(0)) / Y.std(0)
    A = np.c_[X, np.ones(len(X))]
    lin = np.mean((A @ np.linalg.lstsq(A, Yn, rcond=None)[0] - Yn) ** 2)
    check("NumPy MLP learns a nonlinear function", hist[-1][1] < 0.25 * lin and hist[-1][0] < hist[0][0] / 5,
          f"validation loss {hist[-1][1]:.3f} vs. linear fit {lin:.3f} (normalised)")


def test_episode():
    clean = functools.partial(human.demo, kp_noise=0.0, wrist_noise=0.0)
    orig = teleop.demo
    teleop.demo = clean
    try:
        env = HandEnv("cube")
        ep = teleop.teleop_episode(env, np.random.default_rng(3), "optimize")
        check("teleoperated cube grasp lifts the object", ep["ok"], ep["why"])
        check("fingertip touch sensors register the grasp", np.sum(env.tactile() > 0.1) >= 2,
              f"forces {np.round(env.tactile(), 1)} N")
        check("recorded one (obs, action) pair per step", len(ep["obs"]) == len(ep["act"]) == len(ep["q"]))
    finally:
        teleop.demo = orig


if __name__ == "__main__":
    for t in (test_models, test_human_hand, test_jacobian, test_retarget, test_action_roundtrip, test_mlp,
              test_episode):
        t()
    print(f"\nall {passed} checks passed")
