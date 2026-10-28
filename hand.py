"""MuJoCo model: a 16-DOF four-finger robot hand on a 4-DOF wrist (x, y, z, yaw), a table and one object.

Hand layout (palm frame, palm facing down): three fingers (index, middle, ring) on the front edge pointing +x,
an opposed thumb on the back edge pointing -x. Every digit has an abduction joint and three flexion joints,
like LEAP/Allegro-style research hands, but with simpler capsule links so the model needs no mesh files.
Each distal link carries a touch sensor (fingertip normal force), the simulated stand-in for a tactile pad.
"""
import numpy as np

FINGERS = ["thumb", "index", "middle", "ring"]
LINKS = {"thumb": (0.040, 0.032, 0.026), "index": (0.045, 0.030, 0.024),
         "middle": (0.048, 0.032, 0.025), "ring": (0.044, 0.030, 0.024)}
BASES = {"thumb": (-0.045, 0.0), "index": (0.045, 0.026), "middle": (0.045, 0.0), "ring": (0.045, -0.026)}
FLEX_RANGE = [(0.0, 1.7), (0.0, 1.8), (0.0, 1.5)]
ABD_RANGE = (-0.35, 0.35)
WRIST = ["wx", "wy", "wz", "wyaw"]
OBJECTS = {
    "cube": '<geom name="obj" type="box" size="0.024 0.024 0.024" mass="0.08" rgba="0.85 0.45 0.2 1"/>',
    "cylinder": '<geom name="obj" type="cylinder" size="0.022 0.04" mass="0.08" rgba="0.2 0.55 0.85 1"/>',
    "sphere": '<geom name="obj" type="sphere" size="0.028" mass="0.06" rgba="0.3 0.7 0.35 1"/>',
}
OBJ_HALF_HEIGHT = {"cube": 0.024, "cylinder": 0.04, "sphere": 0.028}
PALM_HALF_THICK = 0.012


def _digit(name):
    l1, l2, l3 = LINKS[name]
    bx, by = BASES[name]
    s = -1.0 if name == "thumb" else 1.0                 # thumb points -x
    ax = "0 -1 0" if name == "thumb" else "0 1 0"         # flexion curls the digit down (-z)
    r = 0.0085
    out = f'<body name="{name}_base" pos="{bx} {by} 0">'
    out += f'<joint name="{name}_abd" axis="0 0 1" range="{ABD_RANGE[0]} {ABD_RANGE[1]}"/>'
    out += f'<geom type="sphere" size="{r}" rgba="0.6 0.6 0.62 1"/>'
    out += f'<body name="{name}_l1">'
    out += f'<joint name="{name}_j1" axis="{ax}" range="{FLEX_RANGE[0][0]} {FLEX_RANGE[0][1]}"/>'
    out += f'<geom type="capsule" fromto="0 0 0 {s * l1} 0 0" size="{r}" rgba="0.75 0.75 0.78 1"/>'
    out += f'<body name="{name}_l2" pos="{s * l1} 0 0">'
    out += f'<joint name="{name}_j2" axis="{ax}" range="{FLEX_RANGE[1][0]} {FLEX_RANGE[1][1]}"/>'
    out += f'<geom type="capsule" fromto="0 0 0 {s * l2} 0 0" size="{r}" rgba="0.75 0.75 0.78 1"/>'
    out += f'<body name="{name}_l3" pos="{s * l2} 0 0">'
    out += f'<joint name="{name}_j3" axis="{ax}" range="{FLEX_RANGE[2][0]} {FLEX_RANGE[2][1]}"/>'
    out += f'<geom type="capsule" fromto="0 0 0 {s * l3} 0 0" size="{r * 0.95}" rgba="0.3 0.3 0.32 1"/>'
    out += f'<site name="{name}_tip" pos="{s * (l3 + r)} 0 0" size="0.004"/>'
    # tactile pad: a touch sensor sums the normal force of every contact inside this volume (distal link)
    out += f'<site name="{name}_pad" type="capsule" fromto="0 0 0 {s * l3} 0 0" size="{r * 1.25}" rgba="0 0 0 0"/>'
    out += '</body></body></body></body>'
    return out


def model_xml(obj="cube"):
    digits = "".join(_digit(f) for f in FINGERS)
    acts = "".join(f'<position name="a_{j}" joint="{j}" kp="4000" kv="150" ctrlrange="-1 1" ctrllimited="false"/>'
                   if j in ("wx", "wy", "wz") else "" for j in WRIST)
    acts += '<position name="a_wyaw" joint="wyaw" kp="60" kv="6"/>'
    for f in FINGERS:
        for j in ("abd", "j1", "j2", "j3"):
            acts += f'<position name="a_{f}_{j}" joint="{f}_{j}" kp="6" kv="0.15" forcerange="-2 2"/>'
    return f"""
<mujoco model="dexhand_{obj}">
  <compiler angle="radian"/>
  <option timestep="0.002" cone="elliptic" impratio="10" integrator="implicitfast" noslip_iterations="10"/>
  <default>
    <geom condim="4" friction="2.0 0.02 0.002" solref="0.004 1" solimp="0.95 0.99 0.001"/>
    <joint damping="0.03" armature="0.0005" limited="true"/>
  </default>
  <worldbody>
    <light pos="0 0 1" dir="0 0 -1"/>
    <geom name="table" type="box" size="0.4 0.4 0.02" pos="0 0 -0.02" rgba="0.9 0.9 0.88 1"/>
    <body name="hand_root" pos="0 0 0.30" gravcomp="1">
      <joint name="wx" type="slide" axis="1 0 0" range="-0.3 0.3" damping="5"/>
      <joint name="wy" type="slide" axis="0 1 0" range="-0.3 0.3" damping="5"/>
      <joint name="wz" type="slide" axis="0 0 1" range="-0.3 0.2" damping="5"/>
      <joint name="wyaw" type="hinge" axis="0 0 1" range="-3.2 3.2" damping="0.5"/>
      <inertial pos="0 0 0" mass="0.05" diaginertia="1e-5 1e-5 1e-5"/>
      <body name="palm" gravcomp="1">
        <geom name="palm" type="box" size="0.048 0.042 {PALM_HALF_THICK}" mass="0.4" rgba="0.55 0.57 0.6 1"/>
        <site name="palm_center" pos="0 0 -{PALM_HALF_THICK}" size="0.005"/>
        {digits}
      </body>
    </body>
    <body name="object" pos="0 0 0.05">
      <freejoint name="obj_free"/>
      {OBJECTS[obj]}
    </body>
  </worldbody>
  <contact>{"".join(f'<exclude body1="palm" body2="{f}_l1"/>' for f in FINGERS)}</contact>
  <actuator>{acts}</actuator>
  <sensor>{"".join(f'<touch name="touch_{f}" site="{f}_pad"/>' for f in FINGERS)}</sensor>
</mujoco>"""


if __name__ == "__main__":
    import sys
    for o in OBJECTS:
        open(f"/tmp/dexhand_{o}.xml" if len(sys.argv) < 2 else f"{sys.argv[1]}/dexhand_{o}.xml", "w").write(model_xml(o))
