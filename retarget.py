"""Human-to-robot hand retargeting.

Two methods:
  * joint copy (baseline): estimate the human joint angles from the tracked keypoints and send the same
    angles to the robot (ring + pinky averaged onto the robot's ring finger). Fast, but the robot's links,
    finger bases and thumb placement differ, so the fingertips land in the wrong places.
  * fingertip optimization: find robot joint angles whose fingertips match the human fingertips, each finger
    scaled to the robot's length and measured from its own base, plus thumb-to-finger "pinch" vectors
    (DexPilot / AnyTeleop style):
      - pinch terms switch on as the human thumb and a finger close in (below 12 cm apart);
      - below 9 cm their vector is shrunk towards an aperture gain of 0.55, so the robot closes tighter than
        the human and keeps pressing once it touches the object (the idea of DexPilot's pinch projection);
      - a temporal-smoothness term penalises change from the previous frame, filtering tracker jitter.
    Solved per frame by damped least squares with MuJoCo fingertip Jacobians, warm-started from the previous
    frame, with joint limits handled by an active set.
"""
import os
os.environ.setdefault("MUJOCO_GL", "disable")      # physics only, no rendering window needed
import numpy as np
import mujoco
from hand import model_xml, FINGERS, LINKS, BASES
from human import H_LINKS, H_BASES, H_DIGITS
from sim import FINGER_JOINTS

KP_INDEX = {d: 1 + 4 * k for k, d in enumerate(H_DIGITS)}       # MCP keypoint of each digit; tip = +3


def human_tips_and_bases(kp):
    tips = {d: kp[KP_INDEX[d] + 3] for d in H_DIGITS}
    bases = {d: kp[KP_INDEX[d]] for d in H_DIGITS}
    tips["ring"] = 0.5 * (tips["ring"] + tips["pinky"])
    bases["ring"] = 0.5 * (bases["ring"] + bases["pinky"])
    return tips, bases


def joint_angles_from_keypoints(kp):
    """Estimate (abd, flex1, flex2, flex3) per digit from 4 keypoints, palm frame."""
    out = {}
    for d in H_DIGITS:
        p = kp[KP_INDEX[d]:KP_INDEX[d] + 4]
        s = -1.0 if d == "thumb" else 1.0
        segs = np.diff(p, axis=0)
        horiz = np.hypot(segs[0, 0], segs[0, 1])
        abd = np.arctan2(segs[0, 1], s * segs[0, 0]) if horiz > 1e-6 else 0.0
        pitch = [np.arctan2(-sg[2], np.hypot(sg[0], sg[1]) * (1 if s * sg[0] >= 0 else -1)) for sg in segs]
        f1 = pitch[0]
        f2 = pitch[1] - pitch[0]
        f3 = pitch[2] - pitch[1]
        out[d] = np.array([abd, f1, f2, f3])
    ring = 0.5 * (out["ring"] + out["pinky"])
    return np.concatenate([out["thumb"], out["index"], out["middle"], ring])


class Retargeter:
    def __init__(self, w_pinch=2.0, damping=1e-6, smooth=2e-3, iters=15, aperture_gain=0.55):
        self.m = mujoco.MjModel.from_xml_string(model_xml("cube"))
        self.d = mujoco.MjData(self.m)
        self.fq = np.array([self.m.joint(j).qposadr[0] for j in FINGER_JOINTS])
        self.fdof = np.array([self.m.joint(j).dofadr[0] for j in FINGER_JOINTS])
        jid = [self.m.joint(j).id for j in FINGER_JOINTS]
        self.lo, self.hi = self.m.jnt_range[jid, 0], self.m.jnt_range[jid, 1]
        self.sites = [self.m.site(f"{f}_tip").id for f in FINGERS]
        self.palm = self.m.site("palm_center").id
        self.scale = {f: sum(LINKS[f]) / (sum(H_LINKS[f]) if f != "ring" else
                                          0.5 * (sum(H_LINKS["ring"]) + sum(H_LINKS["pinky"]))) for f in FINGERS}
        self.w_pinch, self.damping, self.smooth, self.iters = w_pinch, damping, smooth, iters
        self.aperture_gain = aperture_gain        # None = scale pinch vectors like the fingers
        self.q = np.zeros(len(FINGER_JOINTS))

    def fk(self, q):
        self.d.qpos[:] = 0
        self.d.qpos[self.fq] = q
        mujoco.mj_kinematics(self.m, self.d)
        mujoco.mj_comPos(self.m, self.d)
        palm = self.d.site_xpos[self.palm]
        return np.array([self.d.site_xpos[s] - palm for s in self.sites])

    def targets(self, kp):
        tips, bases = human_tips_and_bases(kp)
        tgt = []
        for f in FINGERS:
            rb = np.array([BASES[f][0], BASES[f][1], 0.012])        # robot finger base relative to palm centre
            tgt.append(rb + self.scale[f] * (tips[f] - bases[f]))
        tgt = np.array(tgt)
        s_mean = np.mean(list(self.scale.values()))
        pinch, wts = [], []
        for f in FINGERS[1:]:
            v = tips[f] - tips["thumb"]
            dist = np.linalg.norm(v)
            # DexPilot-style switching: thumb-finger vectors only matter when the human is pinching; with an open
            # hand they would fight the per-finger targets (the two hands have different palm proportions)
            wts.append(np.clip((0.12 - dist) / 0.06, 0.0, 1.0))
            g = s_mean
            if self.aperture_gain is not None:
                # closer than 9 cm: blend from the plain scale to the aperture gain (reached at 5 cm and below)
                w = np.clip((0.09 - dist) / 0.04, 0.0, 1.0)
                g = (1 - w) * s_mean + w * self.aperture_gain
            pinch.append(g * v)
        self._pinch_w = np.array(wts)
        return tgt, np.array(pinch)

    def solve(self, kp, warm=True):
        tgt, pinch = self.targets(kp)
        q = self.q.copy() if warm else np.zeros_like(self.q)
        q_prev = q.copy()
        for _ in range(self.iters):
            p = self.fk(q)
            J = []
            for s in self.sites:
                jp = np.zeros((3, self.m.nv))
                mujoco.mj_jacSite(self.m, self.d, jp, None, s)
                J.append(jp[:, self.fdof])
            J = np.array(J)                                           # (4, 3, 16)
            r = [(p - tgt).ravel()]
            A = [J.reshape(-1, len(q))]
            for k in range(1, 4):
                wp = np.sqrt(self.w_pinch * self._pinch_w[k - 1])
                r.append(wp * ((p[k] - p[0]) - pinch[k - 1]))
                A.append(wp * (J[k] - J[0]))
            ws = np.sqrt(self.smooth)
            r.append(ws * (q - q_prev))
            A.append(ws * np.eye(len(q)))
            r, A = np.concatenate(r), np.vstack(A)
            H, g = A.T @ A + self.damping * np.eye(len(q)), A.T @ r
            # joint limits by an active set: a joint sitting on a limit whose step would push it further out is
            # frozen and the step is solved again for the others (clipping alone makes fingers curl the wrong way)
            free = np.ones(len(q), bool)
            for _ in range(4):
                dq = np.zeros(len(q))
                dq[free] = -np.linalg.solve(H[np.ix_(free, free)], g[free])
                push = ((q <= self.lo + 1e-9) & (dq < 0)) | ((q >= self.hi - 1e-9) & (dq > 0))
                if not np.any(push & free):
                    break
                free &= ~push
            dq *= min(1.0, 0.3 / max(np.abs(dq).max(), 1e-12))          # limit each step to 0.3 rad
            q = np.clip(q + dq, self.lo, self.hi)
            if np.abs(dq).max() < 1e-5:
                break
        self.q = q
        return q, float(np.sqrt(np.mean(np.sum((self.fk(q) - tgt) ** 2, 1))))

    def tip_error(self, q, kp):
        tgt, _ = self.targets(kp)
        return float(np.sqrt(np.mean(np.sum((self.fk(q) - tgt) ** 2, 1))))
