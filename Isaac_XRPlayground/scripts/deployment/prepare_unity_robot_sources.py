"""Prepare pinned, deterministic Unity conversion sources for UR10e + Robotiq.

This command does not train or load a policy.  It verifies the two source
repositories recorded in ``deployment/stations.json``, expands their Xacros,
then rebuilds the exact link/joint topology exported from the live Isaac
articulation.  Unity's pinned URDF Importer can import the resulting file; the
runtime RobotDefinition remains authoritative for physics values.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import shutil
import subprocess
from pathlib import Path
from typing import Iterable
from xml.dom import minidom


UR_REVISION = "ae333289875f9ba5a9ea6649a54036efb5ccabee"
ROBOTIQ_REVISION = "45196f6558fe8ba9d89bc8a105396c68c3e7e892"
ROBOT_ID = "ur10e_robotiq_2f85"
ARM_LINKS = (
    "base_link",
    "shoulder_link",
    "upper_arm_link",
    "forearm_link",
    "wrist_1_link",
    "wrist_2_link",
    "wrist_3_link",
)
GRIPPER_LINKS = (
    "base_link_0",
    "left_outer_knuckle",
    "right_outer_knuckle",
    "left_outer_finger",
    "right_outer_finger",
    "left_inner_finger",
    "right_inner_finger",
    "left_inner_knuckle",
    "right_inner_knuckle",
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _git_revision(path: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _verify_revision(path: Path, expected: str, label: str) -> None:
    if not path.is_dir():
        raise FileNotFoundError(f"{label} source directory does not exist: {path}")
    actual = _git_revision(path)
    if actual != expected:
        raise RuntimeError(f"{label} revision {actual} != pinned {expected}")


def _expand_xacro(path: Path, packages: dict[str, Path], **mappings: str) -> minidom.Document:
    try:
        import xacro
        import xacro.substitution_args
    except ImportError as exc:
        raise RuntimeError(
            "xacro is required; run this script with the Isaac Lab environment Python"
        ) from exc

    previous = xacro.substitution_args._eval_find

    def find_package(package: str) -> str:
        try:
            return str(packages[package])
        except KeyError as exc:
            raise RuntimeError(f"Xacro requested unpinned package '{package}'") from exc

    xacro.substitution_args._eval_find = find_package
    try:
        return xacro.process_file(str(path), mappings=mappings)
    finally:
        xacro.substitution_args._eval_find = previous


def _named_elements(document: minidom.Document, tag: str) -> dict[str, minidom.Element]:
    return {
        element.getAttribute("name"): element
        for element in document.getElementsByTagName(tag)
    }


def _direct_children(element: minidom.Element, tag: str) -> list[minidom.Element]:
    return [
        child
        for child in element.childNodes
        if child.nodeType == child.ELEMENT_NODE and child.tagName == tag
    ]


def _remove_direct_children(element: minidom.Element, tag: str) -> None:
    for child in _direct_children(element, tag):
        element.removeChild(child)


def _float_text(values: Iterable[float]) -> str:
    return " ".join(f"{float(value):.10g}" for value in values)


def _quat_multiply(left: list[float], right: list[float]) -> list[float]:
    lx, ly, lz, lw = left
    rx, ry, rz, rw = right
    return [
        lw * rx + lx * rw + ly * rz - lz * ry,
        lw * ry - lx * rz + ly * rw + lz * rx,
        lw * rz + lx * ry - ly * rx + lz * rw,
        lw * rw - lx * rx - ly * ry - lz * rz,
    ]


def _quat_inverse(value: list[float]) -> list[float]:
    x, y, z, w = value
    norm = x * x + y * y + z * z + w * w
    return [-x / norm, -y / norm, -z / norm, w / norm]


def _rotate(rotation: list[float], vector: list[float]) -> list[float]:
    pure = [vector[0], vector[1], vector[2], 0.0]
    result = _quat_multiply(_quat_multiply(rotation, pure), _quat_inverse(rotation))
    return result[:3]


def _unity_position_to_source(value: list[float]) -> list[float]:
    return [value[0], value[2], value[1]]


def _unity_quat_to_source(value: list[float]) -> list[float]:
    return [value[0], value[2], value[1], -value[3]]


def _quat_to_rpy(value: list[float]) -> list[float]:
    x, y, z, w = value
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)
    sinp = max(-1.0, min(1.0, 2.0 * (w * y - z * x)))
    pitch = math.asin(sinp)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return [roll, pitch, yaw]


def _joint_origin(item: dict) -> tuple[list[float], list[float]]:
    parent_position = [float(value) for value in item["parent_anchor_position"]]
    parent_rotation = [float(value) for value in item["parent_anchor_rotation"]]
    child_position = [float(value) for value in item["anchor_position"]]
    child_rotation = [float(value) for value in item["anchor_rotation"]]
    local_rotation = _quat_multiply(parent_rotation, _quat_inverse(child_rotation))
    rotated_anchor = _rotate(local_rotation, child_position)
    local_position = [
        parent_position[index] - rotated_anchor[index] for index in range(3)
    ]
    source_position = _unity_position_to_source(local_position)
    source_rotation = _unity_quat_to_source(local_rotation)
    return source_position, _quat_to_rpy(source_rotation)


def _origin(document: minidom.Document, xyz: Iterable[float], rpy: Iterable[float]) -> minidom.Element:
    origin = document.createElement("origin")
    origin.setAttribute("xyz", _float_text(xyz))
    origin.setAttribute("rpy", _float_text(rpy))
    return origin


def _replace_inertial(document: minidom.Document, link: minidom.Element, item: dict) -> None:
    _remove_direct_children(link, "inertial")
    inertial = document.createElement("inertial")
    source_com = _unity_position_to_source([float(value) for value in item["center_of_mass"]])
    source_rotation = _unity_quat_to_source(
        [float(value) for value in item["inertia_tensor_rotation"]]
    )
    inertial.appendChild(_origin(document, source_com, _quat_to_rpy(source_rotation)))
    mass = document.createElement("mass")
    mass.setAttribute("value", _float_text([item["mass"]]))
    inertial.appendChild(mass)
    tensor = [float(value) for value in item["inertia_tensor"]]
    inertia = document.createElement("inertia")
    inertia.setAttribute("ixx", _float_text([tensor[0]]))
    inertia.setAttribute("ixy", "0")
    inertia.setAttribute("ixz", "0")
    inertia.setAttribute("iyy", _float_text([tensor[1]]))
    inertia.setAttribute("iyz", "0")
    inertia.setAttribute("izz", _float_text([tensor[2]]))
    inertial.appendChild(inertia)
    link.insertBefore(inertial, link.firstChild)


def _rename_link_references(element: minidom.Element, old: str, new: str) -> None:
    links = [element] if element.tagName == "link" else element.getElementsByTagName("link")
    for link in links:
        if link.getAttribute("name") == old:
            link.setAttribute("name", new)
    for tag in ("parent", "child"):
        for node in element.getElementsByTagName(tag):
            if node.getAttribute("link") == old:
                node.setAttribute("link", new)


def _merge_arm_base(arm: minidom.Document) -> None:
    links = _named_elements(arm, "link")
    base = links["base_link"]
    inertia_link = links["base_link_inertia"]
    for tag in ("visual", "collision", "inertial"):
        for child in _direct_children(inertia_link, tag):
            base.appendChild(child.cloneNode(deep=True))


def _merge_gripper_pads(gripper: minidom.Document) -> None:
    links = _named_elements(gripper, "link")
    joints = _named_elements(gripper, "joint")
    for side in ("left", "right"):
        finger = links[f"{side}_inner_finger"]
        pad = links[f"{side}_inner_finger_pad"]
        joint = joints[f"{side}_inner_finger_pad_joint"]
        joint_origin = _direct_children(joint, "origin")[0]
        for tag in ("visual", "collision"):
            for child in _direct_children(pad, tag):
                copied = child.cloneNode(deep=True)
                _remove_direct_children(copied, "origin")
                copied.insertBefore(joint_origin.cloneNode(deep=True), copied.firstChild)
                finger.appendChild(copied)


def _rewrite_mesh_paths(document: minidom.Document) -> None:
    replacements = {
        "package://ur_description/meshes/": "Meshes/UniversalRobots/",
        "package://robotiq_2f_85_gripper_visualization/meshes/": "Meshes/Robotiq/",
    }
    for mesh in document.getElementsByTagName("mesh"):
        filename = mesh.getAttribute("filename")
        for prefix, replacement in replacements.items():
            if filename.startswith(prefix):
                mesh.setAttribute("filename", replacement + filename[len(prefix) :])
                break
        else:
            raise RuntimeError(f"URDF contains an unpinned mesh reference: {filename}")


def _create_joint(document: minidom.Document, item: dict, fixed: bool) -> minidom.Element:
    joint = document.createElement("joint")
    joint.setAttribute("name", item["name"])
    if fixed:
        joint.setAttribute("type", "fixed")
    elif item["joint_type"] == "PrismaticJoint":
        joint.setAttribute("type", "prismatic")
    elif item["limit_mode"] == "continuous":
        joint.setAttribute("type", "continuous")
    else:
        joint.setAttribute("type", "revolute")
    parent = document.createElement("parent")
    parent.setAttribute("link", item["parent_link"])
    joint.appendChild(parent)
    child = document.createElement("child")
    child.setAttribute("link", item["child_link"])
    joint.appendChild(child)
    xyz, rpy = _joint_origin(item)
    joint.appendChild(_origin(document, xyz, rpy))
    if not fixed:
        axis = document.createElement("axis")
        axis.setAttribute(
            "xyz",
            _float_text(
                _unity_position_to_source([float(value) for value in item["axis"]])
            ),
        )
        joint.appendChild(axis)
        limit = document.createElement("limit")
        limit.setAttribute("lower", _float_text([item["lower_limit"]]))
        limit.setAttribute("upper", _float_text([item["upper_limit"]]))
        limit.setAttribute("effort", _float_text([item["force_limit"]]))
        limit.setAttribute("velocity", _float_text([item["max_velocity"]]))
        joint.appendChild(limit)
    return joint


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_if_changed(path: Path, data: bytes) -> None:
    if path.is_file() and path.read_bytes() == data:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(data)
    temporary.replace(path)


def _copy_referenced_meshes(
    document: minidom.Document,
    ur_source: Path,
    gripper_package: Path,
    destination: Path,
) -> dict[str, str]:
    mappings = {
        "Meshes/UniversalRobots/": ur_source / "meshes",
        "Meshes/Robotiq/": gripper_package / "meshes",
    }
    referenced: dict[Path, Path] = {}
    for mesh in document.getElementsByTagName("mesh"):
        filename = mesh.getAttribute("filename").replace("\\", "/")
        for prefix, source_root in mappings.items():
            if filename.startswith(prefix):
                relative = Path(filename[len(prefix) :])
                source = (source_root / relative).resolve()
                target = (destination / prefix.removeprefix("Meshes/") / relative).resolve()
                if not source.is_file():
                    raise FileNotFoundError(f"Pinned URDF mesh is absent: {source}")
                if destination.resolve() not in target.parents:
                    raise RuntimeError(f"Refusing mesh output outside {destination}: {target}")
                referenced[target] = source
                break
        else:
            raise RuntimeError(f"URDF contains an unmapped prepared mesh: {filename}")

    wanted_files = set(referenced)
    wanted_directories = {destination.resolve()}
    for target in wanted_files:
        wanted_directories.update(
            parent for parent in target.parents if destination.resolve() in (parent, *parent.parents)
        )
    if destination.exists():
        for existing in sorted(destination.rglob("*"), reverse=True):
            resolved = existing.resolve()
            if existing.is_file():
                retained_meta = existing.name.endswith(".meta") and Path(str(resolved)[:-5]) in (
                    wanted_files | wanted_directories
                )
                if resolved not in wanted_files and not retained_meta:
                    existing.unlink()
            elif existing.is_dir():
                try:
                    existing.rmdir()
                except OSError:
                    pass
    hashes: dict[str, str] = {}
    for target, source in sorted(referenced.items(), key=lambda item: str(item[0])):
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.is_file() or _sha256(target) != _sha256(source):
            shutil.copy2(source, target)
        hashes[target.relative_to(destination.parent).as_posix()] = _sha256(target)
    return hashes


def prepare(ur_source: Path, robotiq_source: Path, unity_project: Path) -> Path:
    _verify_revision(ur_source, UR_REVISION, "Universal Robots")
    _verify_revision(robotiq_source, ROBOTIQ_REVISION, "Robotiq")
    definition_path = (
        unity_project
        / "Assets/_Project/Features/Deployment/RobotDefinitions"
        / ROBOT_ID
        / "robot.definition.json"
    )
    definition = json.loads(definition_path.read_text(encoding="utf-8"))
    if definition.get("robot_id") != ROBOT_ID or definition.get("schema_version") != 6:
        raise RuntimeError("A schema-6 live Isaac UR10e/Robotiq definition is required")

    gripper_package = robotiq_source / "robotiq_2f_85_gripper_visualization"
    packages = {
        "ur_description": ur_source,
        "robotiq_2f_85_gripper_visualization": gripper_package,
    }
    arm = _expand_xacro(
        ur_source / "urdf/ur.urdf.xacro", packages, ur_type="ur10e", name="ur10e"
    )
    gripper = _expand_xacro(
        gripper_package / "urdf/robotiq_arg2f_85_model.xacro", packages
    )
    _merge_arm_base(arm)
    _merge_gripper_pads(gripper)
    gripper_root = gripper.documentElement
    _rename_link_references(gripper_root, "robotiq_arg2f_base_link", "base_link_0")

    output = minidom.Document()
    robot = output.createElement("robot")
    robot.setAttribute("name", ROBOT_ID)
    output.appendChild(robot)
    arm_links = _named_elements(arm, "link")
    gripper_links = _named_elements(gripper, "link")
    source_links = {
        **{name: arm_links[name] for name in ARM_LINKS},
        **{name: gripper_links[name] for name in GRIPPER_LINKS},
    }
    definition_links = {item["name"]: item for item in definition["links"]}
    expected = set(ARM_LINKS + GRIPPER_LINKS)
    if set(definition_links) != expected:
        raise RuntimeError("Live Isaac link order differs from the pinned Unity source map")
    for item in definition["links"]:
        link = output.importNode(source_links[item["name"]], deep=True)
        _replace_inertial(output, link, item)
        robot.appendChild(link)
    for item in definition["joints"]:
        robot.appendChild(_create_joint(output, item, fixed=False))
    for item in definition["fixed_joints"]:
        robot.appendChild(_create_joint(output, item, fixed=True))
    _rewrite_mesh_paths(output)

    destination = unity_project / "Assets/_Project/Features/Robots/UR10e/URDF"
    urdf_path = destination / f"{ROBOT_ID}.urdf"
    xml = output.toprettyxml(indent="  ", encoding="utf-8")
    _write_if_changed(urdf_path, xml)

    mesh_destination = destination / "Meshes"
    mesh_hashes = _copy_referenced_meshes(
        output, ur_source, gripper_package, mesh_destination
    )
    manifest = {
        "schema_version": 1,
        "robot_id": ROBOT_ID,
        "robot_definition_sha256": definition["definition_sha256"],
        "ur_source_revision": UR_REVISION,
        "robotiq_source_revision": ROBOTIQ_REVISION,
        "urdf_sha256": _sha256(urdf_path),
        "mesh_sha256": mesh_hashes,
        "generated_from_live_isaac_definition": True,
        "policy_checkpoint_used": False,
    }
    _write_if_changed(
        destination / "source.manifest.json",
        (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )
    return urdf_path


def main() -> int:
    root = _repo_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ur-source",
        type=Path,
        default=Path("C:/tmp/xr-robot-sources/ur_description"),
    )
    parser.add_argument(
        "--robotiq-source",
        type=Path,
        default=Path("C:/tmp/xr-robot-sources/robotiq"),
    )
    parser.add_argument(
        "--unity-project",
        type=Path,
        default=root / "Unity_XRPlayground",
    )
    args = parser.parse_args()
    output = prepare(
        args.ur_source.resolve(), args.robotiq_source.resolve(), args.unity_project.resolve()
    )
    print(f"Prepared {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
