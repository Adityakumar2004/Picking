import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=" keyboard teleop")
parser.add_argument("--num_envs", type=int, default=2, help="Number of environments to spawn.")
# add argparse arguments
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()
# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app
import omni.ui as ui
from pxr import Usd, UsdPhysics
import os


ROBOT_ASSET_DIR = os.path.join(os.path.dirname(__file__), "usd")
usd_path=os.path.join(ROBOT_ASSET_DIR, "Robots/Kinova/gen3n7.usd")

stage = Usd.Stage.Open(usd_path)
joint_prim = stage.GetPrimAtPath("/World/Robot/Joint1")
armature_attr = joint_prim.GetAttribute("physics:armature")

if armature_attr:
    armature_value = armature_attr.Get()
    print("Armature for Joint1:", armature_value)
