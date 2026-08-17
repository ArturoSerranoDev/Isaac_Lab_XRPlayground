from __future__ import annotations

import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace

import pytest

from XRPlayground.bridge.frame_math import (
    isaac_pos_to_unity,
    isaac_quat_xyzw_to_unity,
    unity_pos_to_isaac,
    unity_quat_xyzw_to_isaac,
)
from XRPlayground.bridge.protocol import SequenceGate, encode_message, make_envelope, try_decode_buffer
from XRPlayground.cli import _bridge_invocation, _forwarded, build_parser
from XRPlayground.deployment.catalog import CatalogError, load_catalog
from XRPlayground.deployment.calibration import (
    ParameterBound,
    bounded_coordinate_search,
    compare_open_loop_traces,
    derive_randomization_ranges,
    write_open_loop_action_sequence,
)
from XRPlayground.deployment.contract import PolicyContract, PolicyContractError
from XRPlayground.deployment.descriptors import direct_descriptors, flat_dimension, manager_descriptors
from XRPlayground.deployment.evaluation import EvaluationError, build_evaluation_report
from XRPlayground.deployment.exporter import (
    candidate_bundle_path,
    reference_bundle_path,
    unity_robot_definition_path,
    validate_normalization_embedding,
)
from XRPlayground.deployment.generate import generate
from XRPlayground.deployment.handoff import build_handoff_report, write_handoff_report
from XRPlayground.deployment.robot_definition import (
    RobotDefinitionDocument,
    _actuator_joint_properties,
    _body_name_aliases,
    _definition_hash,
    _row,
    definition_from_environment,
)
from XRPlayground.deployment.trace import GoldenTraceError, GoldenTraceWriter, load_golden_trace
from XRPlayground.deployment.validation import _reject_reference_marker


def _cfg(**kwargs):
    return SimpleNamespace(**kwargs)


def test_prepared_ur10e_robotiq_source_matches_live_definition():
    root = Path(__file__).resolve().parents[2]
    source = root / "Unity_XRPlayground/Assets/_Project/Features/Robots/UR10e/URDF"
    urdf_path = source / "ur10e_robotiq_2f85.urdf"
    manifest = json.loads((source / "source.manifest.json").read_text(encoding="utf-8"))
    definition = json.loads(
        (
            root
            / "Unity_XRPlayground/Assets/_Project/Features/Deployment/RobotDefinitions"
            / "ur10e_robotiq_2f85/robot.definition.json"
        ).read_text(encoding="utf-8")
    )
    robot = ET.parse(urdf_path).getroot()
    assert robot.attrib["name"] == "ur10e_robotiq_2f85"
    assert [item.attrib["name"] for item in robot.findall("link")] == [
        item["name"] for item in definition["links"]
    ]
    assert [item.attrib["name"] for item in robot.findall("joint")] == [
        item["name"] for item in definition["joints"] + definition["fixed_joints"]
    ]
    assert manifest["robot_definition_sha256"] == definition["definition_sha256"]
    assert manifest["urdf_sha256"] == hashlib.sha256(urdf_path.read_bytes()).hexdigest()
    assert manifest["mesh_sha256"]
    for relative, expected_hash in manifest["mesh_sha256"].items():
        mesh_path = source / relative
        assert mesh_path.is_file()
        assert hashlib.sha256(mesh_path.read_bytes()).hexdigest() == expected_hash
    assert manifest["policy_checkpoint_used"] is False


@pytest.mark.parametrize("case", ("missing", "empty", "duplicate"))
def test_catalog_rejects_invalid_robot_license_paths(tmp_path, case):
    root = Path(__file__).resolve().parents[2]
    value = json.loads((root / "deployment/stations.json").read_text(encoding="utf-8"))
    robot = next(item for item in value["robot_assets"] if item["robot_id"] == "spot")
    if case == "missing":
        robot.pop("license_paths")
    elif case == "empty":
        robot["license_paths"] = [""]
    else:
        robot["license_paths"] = [robot["license_paths"][0], robot["license_paths"][0]]
    path = tmp_path / "stations.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(CatalogError, match="license_paths"):
        load_catalog(path)


@pytest.mark.parametrize(
    ("policy_id", "cfg", "observation_dim", "action_dim"),
    [
        (
            "ball_catch.throw",
            _cfg(
                arm_joint_names=[f"a{i}" for i in range(7)],
                gripper_joint_names=[f"g{i}" for i in range(6)],
                dof_velocity_scale=0.1,
                action_scale=5.0,
            ),
            30,
            8,
        ),
        (
            "conveyor_color.sort",
            _cfg(
                arm_joint_names=[f"a{i}" for i in range(6)],
                gripper_joint_names=["g"],
                dof_velocity_scale=0.1,
                action_scale=4.0,
                num_colors=3,
                num_object_slots=4,
                gripper_open=0.0,
                gripper_close=0.8,
            ),
            66,
            7,
        ),
        (
            "pick_place_table.place",
            _cfg(
                arm_joint_names=[f"a{i}" for i in range(7)],
                gripper_joint_names=["g", "s1", "s2"],
                gripper_driver_name="g",
                dof_velocity_scale=0.1,
                action_scale=5.0,
            ),
            30,
            8,
        ),
        (
            "balance_bot.two_ball",
            _cfg(
                joint_names=["roll_joint", "pitch_joint"],
                dof_velocity_scale=0.25,
                action_scale=1.5,
            ),
            20,
            2,
        ),
    ],
)
def test_direct_descriptor_dimensions(policy_id, cfg, observation_dim, action_dim):
    descriptors = direct_descriptors(policy_id, cfg)
    assert flat_dimension(descriptors["observations"]) == observation_dim
    assert flat_dimension(descriptors["actions"]) == action_dim


def test_spot_manager_descriptors_keep_order_joint_offsets_and_scales():
    joints = [f"joint_{index}" for index in range(12)]
    observations = [
        {"name": "base_lin_vel", "shape": [3], "dtype": "torch.float32"},
        {"name": "base_ang_vel", "shape": [3], "dtype": "torch.float32"},
        {"name": "projected_gravity", "shape": [3], "dtype": "torch.float32"},
        {"name": "velocity_commands", "shape": [3], "dtype": "torch.float32"},
        {"name": "joint_pos_rel", "shape": [12], "dtype": "torch.float32", "joint_names": joints},
        {"name": "joint_vel_rel", "shape": [12], "dtype": "torch.float32", "joint_names": joints},
        {"name": "last_action", "shape": [12], "dtype": "torch.float32"},
    ]
    actions = [{
        "name": "joint_position_action",
        "shape": [12],
        "dtype": "torch.float32",
        "full_path": "isaaclab.envs.mdp.actions.joint_actions.JointPositionAction",
        "action_type": "JointAction",
        "joint_names": joints,
        "scale": [0.2] * 12,
        "offset": [0.1 * index for index in range(12)],
    }]
    unwrapped = SimpleNamespace(
        get_IO_descriptors=lambda: {"observations": {"policy": observations}, "actions": actions}
    )
    descriptors = manager_descriptors(SimpleNamespace(unwrapped=unwrapped), "spot.locomotion")
    assert [term["name"] for term in descriptors["observations"]] == [
        term["name"] for term in observations
    ]
    assert flat_dimension(descriptors["observations"]) == 48
    action = descriptors["actions"][0]
    assert action["names"] == joints
    assert action["scale"] == pytest.approx(0.2)
    assert action["offset_values"] == pytest.approx([0.1 * index for index in range(12)])
    assert action["target_type"] == "joint_position"


def test_spot_follow_pose_descriptor_is_four_values():
    observations = [
        {"name": "base_lin_vel", "shape": [3], "dtype": "float32"},
        {"name": "projected_gravity", "shape": [3], "dtype": "float32"},
        {"name": "pose_command", "shape": [4], "dtype": "float32"},
    ]
    actions = [{"name": "pre_trained_policy_action", "shape": [3], "dtype": "float32"}]
    unwrapped = SimpleNamespace(
        get_IO_descriptors=lambda: {"observations": {"policy": observations}, "actions": actions}
    )
    descriptors = manager_descriptors(SimpleNamespace(unwrapped=unwrapped), "spot.follow")
    assert descriptors["observations"][-1]["names"] == [
        "position_x", "position_y", "position_z", "heading"
    ]
    assert flat_dimension(descriptors["observations"]) == 10
    assert descriptors["actions"][0]["names"] == ["linear_x", "linear_y", "yaw_rate"]


def test_spot_follow_action_shape_uses_live_manager_dimension_when_descriptor_omits_it():
    observations = [
        {"name": "base_lin_vel", "shape": [3], "dtype": "float32"},
        {"name": "projected_gravity", "shape": [3], "dtype": "float32"},
        {"name": "pose_command", "shape": [4], "dtype": "float32"},
    ]
    actions = [{"name": "pre_trained_policy_action", "shape": None, "dtype": None}]
    unwrapped = SimpleNamespace(
        get_IO_descriptors=lambda: {"observations": {"policy": observations}, "actions": actions},
        action_manager=SimpleNamespace(action_term_dim=[3]),
    )
    descriptors = manager_descriptors(SimpleNamespace(unwrapped=unwrapped), "spot.follow")
    assert descriptors["actions"][0]["shape"] == [3]
    assert flat_dimension(descriptors["actions"]) == 3


def test_catalog_and_generated_artifacts_are_consistent():
    catalog = load_catalog()
    assert len(catalog.stations) == 6
    assert len(catalog.policies) == 6
    assert len(catalog.robot_assets) == 5
    assert catalog.robot_asset("kinova_jaco2_n7s300").redistribution_verified is True
    assert catalog.robot_asset("ur10e_robotiq_2f85").redistribution_verified is True
    assert len(catalog.robot_asset("ur10e_robotiq_2f85").license_paths) == 2
    assert catalog.robot_asset("spot").redistribution_verified is True
    assert catalog.robot_asset("agibot_a2d").redistribution_verified is False
    assert catalog.policy("spot.follow").observation_dim == 10
    assert catalog.policy("spot.follow").policy_hz == 5
    assert generate(check=True) == []


def test_reference_exports_are_outside_the_promotion_candidate_tree():
    candidate = candidate_bundle_path("spot.locomotion")
    reference = reference_bundle_path("spot.locomotion")
    assert candidate != reference
    assert "Candidates" in candidate.parts
    assert "References" in reference.parts
    assert candidate.parent.parent == reference.parent.parent
    definition = unity_robot_definition_path("spot")
    assert "Bundles" not in definition.parts
    assert definition.name == "robot.definition.json"


def test_cli_accepts_checkpoint_free_definition_export():
    args = build_parser().parse_args(
        ["export", "--policy", "spot.locomotion", "--definition-only"]
    )
    assert args.definition_only is True
    assert args.checkpoint is None


def test_cli_marks_local_spot_dependency_bypass_explicitly():
    args = build_parser().parse_args(
        [
            "train",
            "--policy",
            "spot.follow",
            "--allow-unvalidated-dependency",
            "--",
            "--headless",
        ]
    )
    assert args.allow_unvalidated_dependency is True
    assert args.args == ["--", "--headless"]


def test_reference_marker_is_rejected_even_if_copied_into_a_candidate(tmp_path):
    (tmp_path / "REFERENCE_ONLY.json").write_text("{}", encoding="utf-8")
    with pytest.raises(PolicyContractError, match="never be validated as candidates"):
        _reject_reference_marker(tmp_path)


def test_spot_bridge_invocation_selects_matching_play_task():
    script, arguments = _bridge_invocation("spot_follow")
    assert script.endswith("run_xr_bridge_spot.py")
    assert "--station-id=spot_follow" in arguments
    assert "--task=Template-Xrplayground-Spot-Follow-Play-v0" in arguments
    assert not any("Spot-Loco" in item for item in arguments)
    assert _forwarded(["--", "--checkpoint=model.pt"]) == ["--checkpoint=model.pt"]
    _, ball_arguments = _bridge_invocation("ball_catch")
    assert "--task=Template-Xrplayground-Ball-Catch-Throw-B-v0" in ball_arguments


def test_bridge_v2_framing_and_sequence_gate():
    first = make_envelope(
        "robot_state", {"joints": [], "links": []}, station_id="ball_catch", sequence=4, sim_time_s=1.25
    )
    wire = encode_message(first)
    partial = bytearray(wire[:7])
    decoded, partial = try_decode_buffer(partial)
    assert decoded is None
    partial.extend(wire[7:])
    decoded, partial = try_decode_buffer(partial)
    assert decoded == first
    assert partial == bytearray()
    gate = SequenceGate()
    assert gate.accept(first)
    assert not gate.accept(first)
    older = {**first, "sequence": 3}
    assert not gate.accept(older)


def test_bridge_rejects_v1_envelope():
    with pytest.raises(ValueError, match="missing"):
        encode_message({"topic": "/xr/robot_state", "data": {}})


def test_frame_conversion_round_trip():
    position = [1.5, -2.0, 0.25]
    quaternion = [0.1, -0.2, 0.3, 0.9]
    assert unity_pos_to_isaac(isaac_pos_to_unity(position)) == position
    assert unity_quat_xyzw_to_isaac(isaac_quat_xyzw_to_unity(quaternion)) == quaternion


def test_procedural_balance_robot_definition_round_trip(tmp_path):
    document = definition_from_environment(
        SimpleNamespace(unwrapped=SimpleNamespace()), "balance_tray_2dof"
    )
    output = tmp_path / "robot.definition.json"
    document.write_atomic(output)
    loaded = RobotDefinitionDocument.load(output)
    assert loaded.robot_id == "balance_tray_2dof"
    assert loaded.schema_version == 6
    assert len(loaded.definition_sha256) == 64
    assert [joint["name"] for joint in loaded.joints] == ["roll_joint", "pitch_joint"]
    assert loaded.fixed_joints == []
    assert loaded.auxiliary_joints == []
    assert "auxiliary_joints" not in loaded.to_dict()
    assert all(link["mass"] > 0 for link in loaded.links)
    assert loaded.links[-1]["collision_shape_count"] == 1


def test_robot_definition_rejects_tampered_physics_payload(tmp_path):
    document = definition_from_environment(
        SimpleNamespace(unwrapped=SimpleNamespace()), "balance_tray_2dof"
    )
    payload = document.to_dict()
    payload["links"][-1]["mass"] = 99.0
    output = tmp_path / "robot.definition.json"
    output.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(Exception, match="SHA-256"):
        RobotDefinitionDocument.load(output)


def test_robot_definition_rejects_invalid_collision_descriptors():
    root = Path(__file__).resolve().parents[2]
    payload = json.loads(
        (
            root
            / "Unity_XRPlayground/Assets/_Project/Features/Deployment/RobotDefinitions"
            / "spot/robot.definition.json"
        ).read_text(encoding="utf-8")
    )
    foot = next(item for item in payload["links"] if item["name"] == "fl_foot")
    foot["collision_shapes"][0]["radius"] = -1.0
    with pytest.raises(Exception, match="invalid radius"):
        RobotDefinitionDocument(**payload).validate()


def test_robot_definition_hashes_and_validates_auxiliary_loop_constraints():
    document = definition_from_environment(
        SimpleNamespace(unwrapped=SimpleNamespace()), "balance_tray_2dof"
    )
    payload = document.to_dict()
    auxiliary = {
        "name": "tray_four_bar_closure",
        "source_prim_path": "/World/envs/env_0/Robot/tray_four_bar_closure",
        "source_joint_type": "PhysicsRevoluteJoint",
        "joint_type": "RevoluteJoint",
        "topology_role": "loop_closure",
        "parent_link": "tray_root",
        "child_link": "tray",
        "axis": [1.0, 0.0, 0.0],
        "parent_anchor_position": [0.0, 0.0, 0.0],
        "parent_anchor_rotation": [0.0, 0.0, 0.0, 1.0],
        "anchor_position": [0.0, 0.0, 0.0],
        "anchor_rotation": [0.0, 0.0, 0.0, 1.0],
    }
    payload["auxiliary_joints"] = [auxiliary]
    payload["definition_sha256"] = _definition_hash(
        payload["robot_id"],
        payload["source_asset"],
        payload["links"],
        payload["joints"],
        payload["fixed_joints"],
        payload["auxiliary_joints"],
    )
    RobotDefinitionDocument(**payload).validate()
    payload["auxiliary_joints"][0]["topology_role"] = "tree_connector"
    with pytest.raises(Exception, match="mislabeled"):
        RobotDefinitionDocument(**payload).validate()


def test_robot_definition_reads_warp_style_numpy_arrays():
    class WarpLike:
        @staticmethod
        def numpy():
            import numpy as np

            return np.asarray([[1.0, 2.0]], dtype=np.float32)

    assert _row(WarpLike(), 0) == [1.0, 2.0]


def test_physx_merged_root_name_has_a_stable_usd_alias():
    assert _body_name_aliases("base_link_0") == ("base_link_0", "base_link")
    assert _body_name_aliases("link_name") == ("link_name",)


def test_robot_definition_prefers_resolved_actuator_gains_and_torque_curve():
    class RemotizedPDActuator:
        joint_names = ["knee"]
        stiffness = [[60.0]]
        damping = [[1.5]]
        effort_limit = [[float("inf")]]
        velocity_limit = [[float("inf")]]
        is_implicit_model = False
        cfg = _cfg(
            min_delay=0,
            max_delay=4,
            joint_parameter_lookup=[[-1.0, 1.0, 25.0], [0.0, 1.0, 45.0]],
        )

    robot = _cfg(actuators={"spot_knee": RemotizedPDActuator()})
    properties = _actuator_joint_properties(
        robot,
        ["knee"],
        0,
        [0.0],
        [0.0],
        [1.0e9],
        [20.0],
    )[0]
    assert properties["actuator_model"] == "RemotizedPDActuator"
    assert properties["stiffness"] == 60.0
    assert properties["damping"] == 1.5
    assert properties["force_limit"] == 45.0
    assert properties["max_velocity"] == 20.0
    assert properties["actuator_nominal_delay_steps"] == 2
    assert properties["effort_limit_curve"][-1] == {
        "position": 0.0,
        "max_effort": 45.0,
    }


def test_bounded_calibration_and_residual_randomization():
    result = bounded_coordinate_search(
        [ParameterBound("damping", 0.0, 10.0, 1.0, 2.0)],
        lambda values: (values["damping"] - 5.0) ** 2,
    )
    assert result.parameters["damping"] == pytest.approx(5.0, abs=0.01)
    assert result.normalized_error < 1.0e-4
    assert derive_randomization_ranges({"mass": 10.0}, {"mass": 0.1})["mass"] == pytest.approx(
        (8.8, 11.2)
    )


def test_open_loop_comparison_requires_identical_actions(tmp_path):
    def sample(position, action=0.2, loop_closures=None):
        body = {
            "position": [position, 0.0, 0.0],
            "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
            "linear_velocity": [0.0, 0.0, 0.0],
            "angular_velocity": [0.0, 0.0, 0.0],
        }
        return {
            "schema_version": 1,
            "policy_id": "test.policy",
            "station_id": "test",
            "reset_seed": 7,
            "sim_time_s": position,
            "reset_state": {},
            "observations": [0.0],
            "raw_actions": [action],
            "processed_actions": [action],
            "joint_state": {
                "names": ["joint"],
                "position_units": ["radian"],
                "velocity_units": ["radian_per_second"],
                "position": [position],
                "velocity": [0.0],
            },
            "root_state": body,
            "object_state": [],
            "contacts": [],
            "assist": [],
            "loop_closures": loop_closures or [],
            "task_score": 0.0,
        }

    isaac = tmp_path / "isaac.jsonl"
    unity = tmp_path / "unity.jsonl"
    with GoldenTraceWriter(isaac) as writer:
        writer.append(sample(0.0))
        writer.append(sample(0.1))
    with GoldenTraceWriter(unity) as writer:
        writer.append(sample(0.0))
        writer.append(sample(0.11))
    comparison = compare_open_loop_traces(isaac, unity)
    assert comparison.sample_count == 2
    assert 0.0 < comparison.normalized_error < 0.1

    loop = {
        "constraint_id": "four_bar",
        "anchor_error_m": 0.001,
        "axis_error_deg": 0.2,
        "position_unit": "radian",
        "velocity_unit": "radian_per_second",
        "position": 0.0,
        "velocity": 0.0,
        "proxy_mass_kg": 0.1,
        "healthy": True,
    }
    with GoldenTraceWriter(unity) as writer:
        writer.append(sample(0.0, loop_closures=[loop]))
        writer.append(sample(0.1, loop_closures=[loop]))
    comparison = compare_open_loop_traces(isaac, unity)
    assert comparison.components["unity_loop_anchor_error"] == pytest.approx(0.2)
    assert comparison.components["unity_loop_axis_error"] == pytest.approx(0.1)

    loop["healthy"] = False
    with GoldenTraceWriter(unity) as writer:
        writer.append(sample(0.0, loop_closures=[loop]))
        writer.append(sample(0.1, loop_closures=[loop]))
    with pytest.raises(ValueError, match="loop closure 'four_bar' is unhealthy"):
        compare_open_loop_traces(isaac, unity)

    with GoldenTraceWriter(unity) as writer:
        writer.append(sample(0.0, action=0.3))
        writer.append(sample(0.11, action=0.3))
    with pytest.raises(ValueError, match="processed actions are not identical"):
        compare_open_loop_traces(isaac, unity)


def test_open_loop_action_sequence_uses_catalog_dimensions(tmp_path):
    output = write_open_loop_action_sequence(
        "spot.follow", tmp_path / "actions.json", seed=17, frame_count=100, amplitude=0.1
    )
    value = json.loads(output.read_text(encoding="utf-8"))
    assert value["station_id"] == "spot_follow"
    assert value["policy_id"] == "spot.follow"
    assert value["seed"] == 17
    assert len(value["frames"]) == 100
    assert all(len(item["processed_action"]) == 3 for item in value["frames"])
    assert value["frames"][0]["processed_action"] == pytest.approx([0.0, 0.0, 0.0])


def test_handoff_report_is_dependency_ordered_and_never_claims_training(tmp_path):
    report = build_handoff_report()
    assert report["schema_version"] == 1
    assert len(report["policies"]) == 6
    assert report["policy_order"].index("spot.locomotion") < report["policy_order"].index(
        "spot.follow"
    )
    assert report["policies_retrained_by_this_command"] is False
    assert report["complete_six_policy_gate"] is False
    assert report["legacy_removal_allowed"] is False
    entries = {item["policy_id"]: item for item in report["policies"]}
    assert len(entries["ball_catch.throw"]["commands"]["train_phases"]) == 3
    assert "spot.locomotion" in entries["spot.follow"]["depends_on_policy_ids"]
    assert any("dependency 'spot.locomotion'" in item for item in entries["spot.follow"]["training_blockers"])
    assert any("provenance is unverified" in item for item in entries["pick_place_table.place"]["training_blockers"])

    output = write_handoff_report(tmp_path / "handoff.json")
    written = json.loads(output.read_text(encoding="utf-8"))
    assert written["catalog_sha256"] == report["catalog_sha256"]
    assert written["policy_order"] == report["policy_order"]


def test_golden_trace_is_atomic_and_rejects_non_finite(tmp_path):
    output = tmp_path / "golden_trace.jsonl"
    sample = {name: {} for name in ("reset_state", "joint_state", "root_state")}
    sample.update(
        schema_version=1, station_id="test", policy_id="test.policy", reset_seed=7,
        sim_time_s=0.0, observations=[0.0], raw_actions=[0.2],
        processed_actions=[0.2], object_state=[], contacts=[], assist=[],
        loop_closures=[], task_score=0.5,
    )
    with GoldenTraceWriter(output) as writer:
        writer.append(sample)
    assert load_golden_trace(output) == [sample]
    with pytest.raises(GoldenTraceError, match="non-finite"):
        with GoldenTraceWriter(tmp_path / "bad.jsonl") as writer:
            writer.append({**sample, "task_score": float("nan")})
    with pytest.raises(GoldenTraceError, match="schema_version"):
        with GoldenTraceWriter(tmp_path / "wrong-schema.jsonl") as writer:
            writer.append({**sample, "schema_version": 2})


def _valid_contract() -> PolicyContract:
    sha = "a" * 64
    return PolicyContract(
        policy_id="test.policy",
        source_task_id="Test-v0",
        station_id="test",
        robot_id="robot",
        adapter_id="adapter",
        checkpoint="model.pt",
        checkpoint_sha256=sha,
        config_sha256=sha,
        model_sha256=sha,
        onnx={
            "sha256": sha,
            "opset": 15,
            "input_name": "obs",
            "input_shape": [1, 2],
            "output_name": "actions",
            "output_shape": [1, 1],
            "inputs": [{"name": "obs", "shape": [1, 2]}],
            "outputs": [{"name": "actions", "shape": [1, 1]}],
        },
        observations=[{"name": "obs", "shape": [2], "dtype": "float32"}],
        actions=[{
            "name": "action",
            "shape": [1],
            "dtype": "float32",
            "clip": [-1, 1],
            "target_type": "joint_position",
            "integration": "absolute",
        }],
        timing={"source_sim_dt": 0.01, "deployment_sim_dt": 0.01, "policy_dt": 0.02},
        frame={},
        physics_profile_id="physics",
        assist_profile_id="none",
        evaluation={
            "seeded_scenarios": 100,
            "task_thresholds": [
                {"metric": "score", "comparison": "min", "value": 0.8}
            ],
        },
    )


def test_contract_rejects_model_shape_and_opset_mismatch():
    contract = _valid_contract()
    contract.validate()
    contract.onnx["input_shape"] = [1, 3]
    with pytest.raises(PolicyContractError, match="input shape"):
        contract.validate()
    contract.onnx["input_shape"] = [1, 2]
    contract.onnx["opset"] = 18
    with pytest.raises(PolicyContractError, match="opset 15"):
        contract.validate()


def test_contract_rejects_malformed_dependency_hashes():
    contract = _valid_contract()
    contract.dependencies = [{
        "policy_id": "spot.locomotion",
        "model_sha256": "bad",
        "torchscript_sha256": "b" * 64,
    }]
    with pytest.raises(PolicyContractError, match="dependency spot.locomotion.model_sha256"):
        contract.validate()


def test_contract_accepts_paired_static_recurrent_state_and_rejects_mismatch():
    contract = _valid_contract()
    state_input = {"name": "hidden_in", "shape": [1, 2, 8]}
    state_output = {"name": "hidden_out", "shape": [1, 2, 8]}
    contract.onnx["inputs"].append(state_input)
    contract.onnx["outputs"].append(state_output)
    contract.recurrent_state = [state_input, state_output]
    contract.validate()
    contract.onnx["outputs"][1] = {"name": "hidden_out", "shape": [1, 2, 7]}
    with pytest.raises(PolicyContractError, match="different input/output shapes"):
        contract.validate()


def test_export_requires_stateful_normalizer_when_actor_enables_it():
    class Identity:
        pass

    class EmpiricalNormalizer:
        pass

    class ExportModule:
        def __init__(self, include_normalizer):
            self.include_normalizer = include_normalizer

        def named_modules(self):
            modules = [("", self)]
            if self.include_normalizer:
                modules.append(("actor_normalizer", EmpiricalNormalizer()))
            return modules

        def state_dict(self):
            return {"actor_normalizer.running_mean": [0.0]} if self.include_normalizer else {}

    enabled = SimpleNamespace(actor=SimpleNamespace(obs_normalization=True))
    disabled = SimpleNamespace(actor=SimpleNamespace(obs_normalization=False))
    assert validate_normalization_embedding(ExportModule(True), enabled) is True
    assert validate_normalization_embedding(ExportModule(False), disabled) is True
    with pytest.raises(PolicyContractError, match="no stateful normalizer"):
        validate_normalization_embedding(ExportModule(False), enabled)
    with pytest.raises(PolicyContractError, match="explicitly declare"):
        validate_normalization_embedding(ExportModule(True), SimpleNamespace())


def test_contract_rejects_external_observation_normalization():
    contract = _valid_contract()
    contract.normalization_embedded = False
    with pytest.raises(PolicyContractError, match="normalization embedded"):
        contract.validate()


def test_evaluation_report_requires_and_pairs_exact_seed_set(tmp_path):
    def scenario(seed, score):
        return {
            "schema_version": 1,
            "policy_id": "ball_catch.throw",
            "seed": seed,
            "normalized_task_score": score,
            "task_metrics": {
                "catch_rate": score,
                "retained_grasp_rate": score,
                "pre_contact_assist_events": 0,
            },
            "has_nan": False,
            "invalid_actions": False,
            "missing_joints": False,
            "joint_limit_violations": False,
        }

    isaac = tmp_path / "isaac.jsonl"
    unity = tmp_path / "unity.jsonl"
    isaac.write_text("\n".join(json.dumps(scenario(seed, 0.9)) for seed in range(100)), encoding="utf-8")
    unity.write_text("\n".join(json.dumps(scenario(seed, 0.8)) for seed in range(100)), encoding="utf-8")
    parity = tmp_path / "parity.json"
    parity.write_text(json.dumps({
        "schema_version": 1,
        "python_unity_abs_errors": [1.0e-5],
        "observation_abs_errors": [2.0e-5],
        "mirror_position_errors_m": [2.0e-4],
        "mirror_rotation_errors_deg": [0.02],
        "frame_times_ms": [10.0] * 95 + [12.0] * 5,
    }), encoding="utf-8")
    output = tmp_path / "evaluation.json"
    build_evaluation_report("ball_catch.throw", isaac, unity, parity, output=output)
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["seeded_scenarios"] == 100
    assert report["task_gate_passed"] is True
    assert report["p95_frame_time_ms"] == pytest.approx(10.0)

    mismatched = tmp_path / "mismatched.jsonl"
    mismatched.write_text(
        "\n".join(json.dumps(scenario(seed + 1, 0.8)) for seed in range(100)),
        encoding="utf-8",
    )
    with pytest.raises(EvaluationError, match="seeds differ"):
        build_evaluation_report("ball_catch.throw", isaac, mismatched, parity, output=output)
