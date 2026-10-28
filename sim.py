"""MuJoCo environment for the hand: reset an object, step position targets at 20 Hz, report success and failures.

Also defines what a learned policy sees and outputs:
  observation: object position and yaw in the hand frame (as a pose tracker would report it), palm height,
               16 finger joint angles, object type, episode progress, and optionally 4 fingertip touch forces
  action:      palm motion for the next step in the hand frame (dx, dy, dz, dyaw) + 16 finger joint targets
"""
import os
os.environ.setdefault("MUJOCO_GL", "disable")      # physics only, no rendering window needed
import numpy as np
import mujoco
from hand import model_xml, FINGERS, WRIST, OBJ_HALF_HEIGHT, PALM_HALF_THICK

ROOT_Z = 0.30
FINGER_JOINTS = [f"{f}_{j}" for f in FINGERS for j in ("abd", "j1", "j2", "j3")]
CTRL_DT = 0.05
LIFT_OK = 0.08                  # success: object raised at least 8 cm and still held at the end
OBJ_TYPES = ["cube", "cylinder", "sphere"]


class HandEnv:
    def __init__(self, obj="cube"):
        self.obj = obj
        self.m = mujoco.MjModel.from_xml_string(model_xml(obj))
        self.d = mujoco.MjData(self.m)
        self.wq = np.array([self.m.joint(j).qposadr[0] for j in WRIST])
        self.fq = np.array([self.m.joint(j).qposadr[0] for j in FINGER_JOINTS])
        self.wact = np.array([self.m.actuator(f"a_{j}").id for j in WRIST])
        self.fact = np.array([self.m.actuator(f"a_{j}").id for j in FINGER_JOINTS])
        ids = [self.m.joint(j).id for j in FINGER_JOINTS]
        self.lo, self.hi = self.m.jnt_range[ids, 0], self.m.jnt_range[ids, 1]
        self.obj_q = self.m.joint("obj_free").qposadr[0]
        self.palm_site = self.m.site("palm_center").id
        self.obj_body = self.m.body("object").id
        self.g_obj = self.m.geom("obj").id
        self.g_table = self.m.geom("table").id
        self.touch_adr = np.array([self.m.sensor(f"touch_{f}").adr[0] for f in FINGERS])
        self.n_sub = int(round(CTRL_DT / self.m.opt.timestep))

    def reset(self, xy, yaw, palm0):
        """Object on the table at (x, y, yaw); hand open with its palm centre at palm0 = (x, y, z, yaw)."""
        mujoco.mj_resetData(self.m, self.d)
        h = OBJ_HALF_HEIGHT[self.obj]
        self.d.qpos[self.obj_q:self.obj_q + 3] = [xy[0], xy[1], h + 0.0005]
        self.d.qpos[self.obj_q + 3:self.obj_q + 7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
        w = self.wrist_from_palm(palm0)
        self.d.qpos[self.wq] = w
        self.d.ctrl[self.wact] = w
        self.d.ctrl[self.fact] = 0.0
        mujoco.mj_forward(self.m, self.d)
        self.z0 = self.d.xpos[self.obj_body][2]
        self.palm_target = np.array(palm0, float)

    @staticmethod
    def wrist_from_palm(p):
        """Wrist joint targets (x, y, z, yaw) that put the palm centre at p = (x, y, z, yaw)."""
        return np.array([p[0], p[1], p[2] + PALM_HALF_THICK - ROOT_Z, p[3]])

    def step(self, palm_target, finger_target):
        self.palm_target = np.array(palm_target, float)
        self.d.ctrl[self.wact] = self.wrist_from_palm(palm_target)
        self.d.ctrl[self.fact] = np.clip(finger_target, self.lo, self.hi)
        for _ in range(self.n_sub):
            mujoco.mj_step(self.m, self.d)

    # ---- state -------------------------------------------------------------------------------------------
    def palm_pose(self):
        p = self.d.site_xpos[self.palm_site]
        return np.r_[p, self.d.qpos[self.wq[3]]]

    def obj_pose(self):
        q = self.d.qpos[self.obj_q:self.obj_q + 7]
        yaw = np.arctan2(2 * (q[3] * q[6] + q[4] * q[5]), 1 - 2 * (q[5] ** 2 + q[6] ** 2))
        return q[:3].copy(), yaw

    def finger_q(self):
        return self.d.qpos[self.fq].copy()

    def tactile(self):
        """Fingertip normal forces (N) from the four touch sensors."""
        return self.d.sensordata[self.touch_adr].copy()

    def lifted(self):
        return self.d.xpos[self.obj_body][2] - self.z0 >= LIFT_OK

    def contacts_with_hand(self):
        n = 0
        for k in range(self.d.ncon):
            c = self.d.contact[k]
            if self.g_obj in (c.geom1, c.geom2):
                other = c.geom2 if c.geom1 == self.g_obj else c.geom1
                n += other != self.g_table
        return n

    # ---- policy interface --------------------------------------------------------------------------------
    def observation(self, progress, obj_noise=None, tactile=True):
        """Policy input. obj_noise = (dx, dy, dz) error added to the tracked object position (perception noise)."""
        palm = self.palm_pose()
        p, oyaw = self.obj_pose()
        if obj_noise is not None:
            p = p + obj_noise
        c, s = np.cos(palm[3]), np.sin(palm[3])
        rel = p - palm[:3]
        rel_xy = [c * rel[0] + s * rel[1], -s * rel[0] + c * rel[1]]
        dy = (oyaw - palm[3] + np.pi / 4) % (np.pi / 2) - np.pi / 4          # 90-degree symmetry of the objects
        onehot = np.eye(3)[OBJ_TYPES.index(self.obj)]
        x = np.r_[rel_xy, rel[2], np.sin(4 * dy), np.cos(4 * dy), palm[2], self.finger_q(), onehot, progress]
        if tactile:
            x = np.r_[x, np.log1p(self.tactile())]
        return x

    def action_to_targets(self, a):
        """Policy action -> palm target (x, y, z, yaw) and finger targets. Palm motion is relative, hand frame."""
        palm = self.palm_target
        c, s = np.cos(palm[3]), np.sin(palm[3])
        dx, dy = c * a[0] - s * a[1], s * a[0] + c * a[1]
        return palm + np.r_[dx, dy, a[2], a[3]], np.clip(a[4:], self.lo, self.hi)

    def targets_to_action(self, palm_target, finger_target):
        """Inverse of action_to_targets: the label a demonstration provides at this step."""
        palm = self.palm_target
        c, s = np.cos(palm[3]), np.sin(palm[3])
        d = np.asarray(palm_target) - palm
        return np.r_[c * d[0] + s * d[1], -s * d[0] + c * d[1], d[2], d[3], finger_target]


def failure_reason(env, touched):
    """After a failed episode: why? (for the failure breakdown)."""
    if not touched:
        return "missed"
    if env.contacts_with_hand() == 0:
        return "dropped"
    return "not lifted"
