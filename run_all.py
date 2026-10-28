"""Run every experiment, print the results and the requirements table, save the plots.

  python run_all.py            full run (about 10 minutes on 2 cores)
  python run_all.py --quick    small version for CI (about 2 minutes)
"""
import os
os.environ.setdefault("MUJOCO_GL", "disable")
import sys
import time
import numpy as np
from multiprocessing import Pool

from sim import HandEnv, OBJ_TYPES
from retarget import Retargeter
from teleop import teleop_episode, collect
from bc import MLP, dataset, evaluate

QUICK = "--quick" in sys.argv
N_TELEOP = 15 if QUICK else 50          # demonstrations per object for the retargeting comparison
N_COLLECT = 25 if QUICK else 60         # teleoperated demonstrations per object recorded for learning
N_EVAL = 10 if QUICK else 30            # closed-loop test episodes per object (new object poses)
CURVE = [5, 20] if QUICK else [5, 10, 20, 40]
NOISE_MM = [0, 10] if QUICK else [0, 5, 10]
INJECT = 0.004                          # noise injection while recording (m; fingers 10x in rad)

METHODS = [
    ("Joint-angle copy", dict(method="copy")),
    ("Joint copy + low-pass filter", dict(method="copy", ema=0.3)),
    ("Fingertip optimization only", dict(rt=dict(w_pinch=0.0, aperture_gain=None, smooth=2e-5))),
    ("+ thumb-finger pinch vectors", dict(rt=dict(aperture_gain=None, smooth=2e-5))),
    ("+ aperture gain 0.55", dict(rt=dict(smooth=2e-5))),
    ("+ temporal smoothing (full)", dict(rt=dict())),
]


def teleop_job(args):
    name, cfg, obj, n, seed = args
    env = HandEnv(obj)
    rng = np.random.default_rng(seed)
    rt = Retargeter(**cfg.get("rt", {}))
    out = []
    for _ in range(n):
        ep = teleop_episode(env, rng, cfg.get("method", "optimize"), rt, record=False, ema=cfg.get("ema"))
        grasp = [i for i, p in enumerate(ep["phase"]) if p in ("squeeze", "lift", "hold")]
        chatter = float(np.sqrt(np.mean(np.diff(ep["q"][grasp], axis=0) ** 2)))
        opn = [i for i, p in enumerate(ep["phase"]) if p in ("approach", "descend")]
        out.append((ep["ok"], ep["why"], float(ep["tip_err"].mean()), ep["solve_ms"], chatter,
                    float(ep["tip_err"][opn].mean())))
    return name, obj, out


def collect_job(args):
    inject, seed = args
    return inject, collect(N_COLLECT, seed=seed, inject=inject)


def bc_job(args):
    tag, X, Y, noise_list, seed, tactile = args
    pol = MLP(X.shape[1], Y.shape[1], seed=seed)
    hist = pol.fit(X, Y, epochs=60, seed=seed)
    res = {nm: evaluate(pol, N_EVAL, seed=999, obj_noise=nm / 1000, tactile=tactile) for nm in noise_list}
    return tag, res, hist[-1]


def rate(res):
    v = [r[0] for rs in res.values() for r in rs]
    return 100.0 * np.mean(v)


def take(data, k):
    """First k successful demonstrations of each object."""
    out = []
    for obj in OBJ_TYPES:
        out += [e for e in data if e["obj"] == obj][:k]
    return out


def main():
    t_start = time.time()
    pool = Pool(min(4, os.cpu_count() or 1))
    R = {}

    # ---------------------------------------------------------------------------------------------------------
    print(f"[1  Retargeting a tracked human hand onto the robot hand: {N_TELEOP} demonstrations per object, "
          f"{N_TELEOP * 3} per method]")
    print("  human: 5 fingers, 21 tracked keypoints with 4 mm jitter, wrist 3 mm; robot: 4 fingers, 16 joints, "
          "different link lengths and finger placement")
    jobs = [(nm, cfg, obj, N_TELEOP, 11 + k) for nm, cfg in METHODS for k, obj in enumerate(OBJ_TYPES)]
    tele = {}
    for nm, obj, out in pool.map(teleop_job, jobs):
        tele.setdefault(nm, {})[obj] = out
    print(f"  {'method':32s} {'success':>8s} {'cube':>5s} {'cyl.':>5s} {'sphere':>6s} {'tip err':>8s} "
          f"{'open':>7s} {'chatter':>8s} {'solve p95':>9s}  failures")
    for nm, _ in METHODS:
        allr = [r for o in OBJ_TYPES for r in tele[nm][o]]
        succ = 100 * np.mean([r[0] for r in allr])
        per = [100 * np.mean([r[0] for r in tele[nm][o]]) for o in OBJ_TYPES]
        tip = 1000 * np.mean([r[2] for r in allr])
        tip_open = 1000 * np.mean([r[5] for r in allr])
        chat = np.mean([r[4] for r in allr])
        p95 = np.percentile(np.concatenate([r[3] for r in allr]), 95)
        why = [r[1] for r in allr if not r[0]]
        fails = ", ".join(f"{w} {why.count(w)}" for w in ("missed", "dropped", "not lifted") if why.count(w))
        print(f"  {nm:32s} {succ:7.0f}% {per[0]:4.0f}% {per[1]:4.0f}% {per[2]:5.0f}% {tip:6.1f}mm "
              f"{tip_open:5.1f}mm {chat:7.3f} {p95:7.2f}ms  {fails or '-'}")
        R[nm] = dict(succ=succ, tip=tip, tip_open=tip_open, p95=p95, chatter=chat)
    print("  success = object lifted 8 cm and still held after 0.8 s; tip err = mean distance of robot fingertips "
          "from the scaled human fingertips\n  (whole demo / open hand during approach; while grasping the pinch "
          "terms pull the tips together on purpose);\n  chatter = RMS change of the finger commands per step (rad) while "
          "holding; solve p95 = retargeting time per frame, 95th percentile")

    # ---------------------------------------------------------------------------------------------------------
    print(f"\n[2  Recording demonstrations for imitation learning: {N_COLLECT} teleoperated attempts per object, "
          "full retargeter]")
    coll = dict(pool.map(collect_job, [(0.0, 100), (INJECT, 100)]))
    for inj, (data, tr) in coll.items():
        X, Y = dataset(data)
        lab = "noise injection 4 mm" if inj else "plain"
        print(f"  {lab:22s} teleop success {100 * tr:5.1f}%  kept {len(data):3d} successful demos, "
              f"{len(X)} (observation, action) pairs; obs {X.shape[1]}-D, action {Y.shape[1]}-D")
    R["collect_rate"] = 100 * coll[0.0][1]

    # ---------------------------------------------------------------------------------------------------------
    print(f"\n[3  Behaviour cloning, closed loop on new object poses ({N_EVAL} per object): "
          "learning curve]")
    jobs = []
    for k in CURVE:
        for inj in (0.0, INJECT):
            X, Y = dataset(take(coll[inj][0], k))
            jobs.append(((k, inj), X, Y, [0], 0, True))
    curve = {tag: (res, h) for tag, res, h in pool.map(bc_job, jobs)}
    print(f"  {'demos per object':>16s} {'plain BC':>9s} {'with noise injection':>21s}")
    for k in CURVE:
        print(f"  {k:16d} {rate(curve[(k, 0.0)][0][0]):8.0f}% {rate(curve[(k, INJECT)][0][0]):20.0f}%")
    R["curve"] = {k: (rate(curve[(k, 0.0)][0][0]), rate(curve[(k, INJECT)][0][0])) for k in CURVE}
    kmax = CURVE[-1]

    # ---------------------------------------------------------------------------------------------------------
    print(f"\n[4  Robustness to object-pose errors (constant offset per episode, std shown, + 1 mm jitter), "
          f"{kmax} demos per object]")
    jobs = []
    for tag, inj, tac in (("plain BC", 0.0, True), ("noise injection", INJECT, True),
                          ("noise injection, no touch", INJECT, False)):
        X, Y = dataset(take(coll[inj][0], kmax))
        if not tac:
            X = X[:, :-4]
        jobs.append((tag, X, Y, NOISE_MM, 1, tac))
    rob = {tag: res for tag, res, _ in pool.map(bc_job, jobs)}
    print(f"  {'policy':28s}" + "".join(f"{f'{nm} mm':>9s}" for nm in NOISE_MM))
    for tag, res in rob.items():
        print(f"  {tag:28s}" + "".join(f"{rate(res[nm]):8.0f}%" for nm in NOISE_MM))
    R["rob"] = {tag: {nm: rate(res[nm]) for nm in NOISE_MM} for tag, res in rob.items()}
    worst = rob["noise injection"][NOISE_MM[-1]]
    why = [r[1] for rs in worst.values() for r in rs if not r[0]]
    print(f"  failures of the noise-injection policy at {NOISE_MM[-1]} mm: "
          + (", ".join(f"{w} {why.count(w)}" for w in ("missed", "dropped", "not lifted") if why.count(w)) or "none"))
    pool.close()

    # ---------------------------------------------------------------------------------------------------------
    full, copy = R["+ temporal smoothing (full)"], R["Joint-angle copy"]
    nosmooth = R["+ aperture gain 0.55"]
    bc_best = R["curve"][kmax][1]
    req = [
        ("R1", "Teleop success, full retargeter", f"{full['succ']:.0f} %", ">= 90 %", full["succ"] >= 90),
        ("R2", "Fingertip retargeting vs. joint copy", f"+{full['succ'] - copy['succ']:.0f} pts", ">= +50 pts",
         full["succ"] - copy["succ"] >= 50),
        ("R3", "Fingertip retargeting vs. filtered joint copy",
         f"+{full['succ'] - R['Joint copy + low-pass filter']['succ']:.0f} pts", ">= +10 pts",
         full["succ"] - R["Joint copy + low-pass filter"]["succ"] >= 10),
        ("R4", "Fingertip error, open hand (full retargeter)", f"{full['tip_open']:.1f} mm", "<= 12 mm",
         full["tip_open"] <= 12),
        ("R5", "Smoothing cuts finger-command chatter", f"-{100 * (1 - full['chatter'] / nosmooth['chatter']):.0f} %",
         ">= 50 %", full["chatter"] <= 0.5 * nosmooth["chatter"]),
        ("R6", "Retargeting time per frame (p95)", f"{full['p95']:.1f} ms", "<= 5 ms", full["p95"] <= 5),
        ("R7", f"BC success on new poses ({kmax} demos/object)", f"{bc_best:.0f} %", ">= 90 %", bc_best >= 90),
        ("R8", "BC vs. teleop success", f"{bc_best - full['succ']:+.0f} pts", ">= -5 pts",
         bc_best >= full["succ"] - 5),
        ("R9", f"BC with {NOISE_MM[-1]} mm object-pose error (best policy)",
         f"{max(v[NOISE_MM[-1]] for v in R['rob'].values()):.0f} %", ">= 70 %",
         max(v[NOISE_MM[-1]] for v in R["rob"].values()) >= 70),
    ]
    print(f"\n{'ID':4s} {'Requirement':48s} {'Result':>9s} {'Target':>11s}  Status")
    for r in req:
        print(f"{r[0]:4s} {r[1]:48s} {r[2]:>9s} {r[3]:>11s}  {'PASS' if r[4] else 'FAIL'}")
    npass = sum(r[4] for r in req)
    print(f"{npass}/{len(req)} requirements pass")

    plots(R, tele)
    print(f"\nsaved results.png, hands.png  (total {time.time() - t_start:.0f} s)")
    return 0 if npass == len(req) else 1


def plots(R, tele):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    blue, gray, orange = "#2a6fb0", "#9aa3ad", "#d9822b"
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))
    names = [nm for nm, _ in METHODS]
    vals = [R[nm]["succ"] for nm in names]
    ax[0].barh(range(len(names))[::-1], vals, color=[gray, gray, blue, blue, blue, blue])
    ax[0].set_yticks(range(len(names))[::-1], names, fontsize=9)
    for i, v in enumerate(vals):
        ax[0].text(v + 1, len(names) - 1 - i, f"{v:.0f}%", va="center", fontsize=9)
    ax[0].set_xlim(0, 112)
    ax[0].set_xlabel("teleoperated grasp success (%)")
    ax[0].set_title("Retargeting method")
    ks = sorted(R["curve"])
    ax[1].plot(ks, [R["curve"][k][0] for k in ks], "o-", color=gray, label="plain BC")
    ax[1].plot(ks, [R["curve"][k][1] for k in ks], "o-", color=blue, label="BC + noise injection")
    ax[1].axhline(R["+ temporal smoothing (full)"]["succ"], ls="--", color=orange, label="teleop (full retargeter)")
    ax[1].set_xlabel("demonstrations per object")
    ax[1].set_ylabel("success on new object poses (%)")
    ax[1].set_ylim(0, 105)
    ax[1].set_title("Learning curve")
    ax[1].legend(fontsize=8, loc="lower right")
    for (tag, d), c in zip(R["rob"].items(), [gray, blue, orange]):
        nm = sorted(d)
        ax[2].plot(nm, [d[n] for n in nm], "o-", color=c, label=tag)
    ax[2].set_xlabel("object-pose error std (mm)")
    ax[2].set_ylabel("success (%)")
    ax[2].set_ylim(0, 105)
    ax[2].set_title("Perception error")
    ax[2].legend(fontsize=8, loc="lower left")
    for a in ax:
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig("results.png", dpi=130)
    plt.close(fig)
    hands_figure()


def hands_figure():
    """Human keypoints and the retargeted robot hand at the grasp, side and top views."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import mujoco
    from human import keypoints, POSTURES, H_DIGITS
    from hand import FINGERS
    kp = keypoints({d: np.array(v) for d, v in POSTURES["cylinder"].items()})
    rt = Retargeter()
    for _ in range(30):
        q, _ = rt.solve(kp)
    m, d = rt.m, rt.d
    rt.fk(q)
    palm = d.site_xpos[rt.palm]
    chains = []
    for f in FINGERS:
        pts = [d.xpos[m.body(f"{f}_{s}").id] - palm for s in ("base", "l1", "l2", "l3")]
        pts.append(d.site_xpos[m.site(f"{f}_tip").id] - palm)
        chains.append(np.array(pts))
    fig, ax = plt.subplots(1, 2, figsize=(10, 4.2))
    for a, (i, j, lab) in zip(ax, [(0, 2, "side view (x, z)"), (0, 1, "top view (x, y)")]):
        for k, dname in enumerate(H_DIGITS):
            c = np.vstack([kp[0], kp[1 + 4 * k:5 + 4 * k]]) + np.array([0, 0, 0.012])   # MCPs level with the robot finger bases
            a.plot(c[:, i], c[:, j], "o-", color="#d9822b", ms=3, lw=1.5, label="human (21 keypoints)" if k == 0 else None)
        for k, c in enumerate(chains):
            a.plot(c[:, i], c[:, j], "s-", color="#2a6fb0", ms=4, lw=3, alpha=0.8, label="robot (retargeted)" if k == 0 else None)
        a.set_aspect("equal")
        a.set_title(lab)
        a.grid(alpha=0.3)
        a.set_xlabel("m")
    ax[0].legend(fontsize=8)
    fig.suptitle("Cylinder grasp: human hand pose and the robot joint angles found by fingertip retargeting")
    fig.tight_layout()
    fig.savefig("hands.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    sys.exit(main())
