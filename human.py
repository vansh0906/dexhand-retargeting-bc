"""Synthetic human hand demonstrations: a 21-keypoint human hand doing top-down grasps.

The human hand is a kinematic model with adult proportions (MANO-style topology: wrist + 4 joints per digit,
5 digits). It has five fingers and a longer, differently placed thumb, so its kinematics do not match the
4-finger robot hand. That mismatch is what retargeting has to solve.

A demonstration is a wrist trajectory (approach, descend, close, squeeze, lift, hold) plus finger joint angles from a
per-object grasp posture, with person-to-person variation. Keypoints get 4 mm of noise and the wrist gets
3 mm, roughly what a camera-based hand tracker delivers.
"""
import numpy as np

# human digit link lengths (m), MCP positions in the palm frame (palm centre at origin, fingers along +x)
H_LINKS = {"thumb": (0.044, 0.034, 0.029), "index": (0.047, 0.027, 0.021), "middle": (0.051, 0.031, 0.022),
           "ring": (0.048, 0.030, 0.022), "pinky": (0.038, 0.022, 0.019)}
H_BASES = {"thumb": (-0.030, 0.022), "index": (0.040, 0.030), "middle": (0.042, 0.009),
           "ring": (0.040, -0.011), "pinky": (0.036, -0.030)}
H_DIGITS = ["thumb", "index", "middle", "ring", "pinky"]

# grasp postures per object: (abduction, flex1, flex2, flex3) per digit, human hand
POSTURES = {
    "cube": {"thumb": (0.0, 1.30, 0.75, 0.45), "index": (0.05, 1.35, 0.80, 0.45), "middle": (0.0, 1.35, 0.80, 0.45),
             "ring": (-0.05, 1.30, 0.80, 0.45), "pinky": (-0.1, 1.2, 0.75, 0.4)},
    "cylinder": {"thumb": (0.0, 1.30, 0.80, 0.45), "index": (0.05, 1.35, 0.85, 0.45), "middle": (0.0, 1.35, 0.85, 0.45),
                 "ring": (-0.05, 1.30, 0.85, 0.45), "pinky": (-0.1, 1.2, 0.8, 0.4)},
    "sphere": {"thumb": (0.0, 1.25, 0.80, 0.50), "index": (0.12, 1.30, 0.85, 0.50), "middle": (0.0, 1.35, 0.85, 0.50),
               "ring": (-0.12, 1.30, 0.85, 0.50), "pinky": (-0.2, 1.2, 0.8, 0.45)},
}

DT = 0.05                      # demonstrations sampled at 20 Hz
PHASES = (("approach", 1.0), ("descend", 1.0), ("close", 1.0), ("squeeze", 0.5), ("lift", 1.0), ("hold", 0.8))
T_TOTAL = sum(d for _, d in PHASES)


def digit_points(name, q):
    """Positions of MCP, PIP, DIP, tip for one digit in the palm frame (palm facing down)."""
    abd, f1, f2, f3 = q
    bx, by = H_BASES[name]
    s = -1.0 if name == "thumb" else 1.0
    pts = [np.array([bx, by, 0.0])]
    ang = 0.0
    p = pts[0].copy()
    for L, f in zip(H_LINKS[name], (f1, f2, f3)):
        ang += f
        d = np.array([s * np.cos(ang) * np.cos(abd), np.cos(ang) * np.sin(abd), -np.sin(ang)])
        p = p + L * d
        pts.append(p.copy())
    return np.array(pts)


def keypoints(q_all):
    """21 keypoints: wrist, then 4 points per digit (thumb, index, middle, ring, pinky), in the palm frame."""
    wrist = np.array([[-0.075, 0.0, 0.0]])
    return np.vstack([wrist] + [digit_points(d, q_all[d]) for d in H_DIGITS])


def tips(q_all):
    return {d: digit_points(d, q_all[d])[-1] for d in H_DIGITS}


def demo(obj, obj_xy, obj_yaw, rng, obj_half_h, grasp_height=0.065, lift=0.12, kp_noise=0.004, wrist_noise=0.003):
    """One human demonstration. Returns time, palm-centre trajectory (x, y, z, yaw), per-frame keypoints, phase."""
    post = POSTURES[obj]
    person = {d: np.array(post[d]) * (1 + rng.normal(0, 0.05, 4)) + np.r_[rng.normal(0, 0.03), 0, 0, 0]
              for d in H_DIGITS}
    z_grasp = obj_half_h + grasp_height          # palm centre height: fingertips close around the middle
    z_high = z_grasp + 0.12
    yaw = obj_yaw + rng.normal(0, 0.05)
    start_off = np.array([-0.04, 0.0]) + rng.uniform(-0.02, 0.02, 2)     # where the hand starts, behind the object
    t = np.arange(0, T_TOTAL + 1e-9, DT)
    W, K, PH = [], [], []
    for tt in t:
        acc, name, frac = 0.0, "hold", 1.0
        for nm, dur in PHASES:
            if tt <= acc + dur:
                name, frac = nm, (tt - acc) / dur
                break
            acc += dur
        sm = 0.5 - 0.5 * np.cos(np.pi * np.clip(frac, 0, 1))
        xy = np.array(obj_xy, float)
        if name == "approach":
            z, c = z_high + 0.05 * (1 - sm), 0.0
            xy = xy + (1 - sm) * start_off
        elif name == "descend":
            z, c = z_high + (z_grasp - z_high) * sm, 0.15 * sm
        elif name == "close":
            z, c = z_grasp, 0.15 + 0.85 * sm
        elif name == "squeeze":
            z, c = z_grasp, 1.0
        elif name == "lift":
            z, c = z_grasp + lift * sm, 1.0
        else:
            z, c = z_grasp + lift, 1.0
        q = {d: np.r_[person[d][0] * min(1, c * 2), person[d][1:] * c] for d in H_DIGITS}
        W.append(np.r_[xy, z, yaw] + np.r_[rng.normal(0, wrist_noise, 3), rng.normal(0, 0.01)])
        K.append(keypoints(q) + rng.normal(0, kp_noise, (21, 3)))
        PH.append(name)
    return {"t": t, "palm": np.array(W), "kp": np.array(K), "phase": PH, "obj": obj}
