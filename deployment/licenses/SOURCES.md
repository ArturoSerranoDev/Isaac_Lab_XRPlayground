# Robot description sources

The authored deployment catalog points to these immutable revisions. A verified
entry means its intended source can be redistributed under the recorded notices;
it does not mean that a normalized Unity prefab or calibrated policy bundle is
present.

| Robot | Source revision | Material used | Notice |
|---|---|---|---|
| UR10e | `UniversalRobots/Universal_Robots_ROS2_Description@ae333289875f9ba5a9ea6649a54036efb5ccabee` (`4.3.1`) | UR10e Xacro/configuration and meshes; UR10e is not one of the models covered by the repository's separate Graphical Documentation terms | `Universal_Robots_ROS2_Description-ae333289.LICENSE.txt` |
| Robotiq 2F-85 | `ros-industrial-attic/robotiq@45196f6558fe8ba9d89bc8a105396c68c3e7e892` | `robotiq_2f_85_gripper_visualization` Xacro and meshes | `Robotiq-45196f65.LICENSE.txt` |
| Spot | `RAI-Opensource/spot_description@156d1802bfb117f219dbfce7597d283d5fabc968` | Spot Xacro, meshes, and model metadata; physics truth must still come from the Isaac articulation export | `RAI-Spot-Description-156d1802.LICENSE.txt` |

Immutable source URLs:

- https://github.com/UniversalRobots/Universal_Robots_ROS2_Description/tree/ae333289875f9ba5a9ea6649a54036efb5ccabee
- https://github.com/ros-industrial-attic/robotiq/tree/45196f6558fe8ba9d89bc8a105396c68c3e7e892/robotiq_2f_85_gripper_visualization
- https://github.com/RAI-Opensource/spot_description/tree/156d1802bfb117f219dbfce7597d283d5fabc968

`Isaac_XRPlayground/scripts/deployment/prepare_unity_robot_sources.py` verifies
the full UR and Robotiq commit IDs before expanding Xacro. It copies only meshes
referenced by the generated `ur10e_robotiq_2f85.urdf` into Unity and records the
URDF, live robot-definition, and per-mesh SHA-256 values in
`source.manifest.json`.

The AgiBot A2D entry remains unverified because the checked-in USD does not
contain or reference redistribution terms. It must not be normalized or bundled
for Unity until its exact source license is recorded.
