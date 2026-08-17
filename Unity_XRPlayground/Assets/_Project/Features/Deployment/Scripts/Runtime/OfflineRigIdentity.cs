using System;
using UnityEngine;

namespace XRPlayground.Deployment
{
    [DisallowMultipleComponent]
    public sealed class OfflineRigIdentity : MonoBehaviour
    {
        public string robotId;
        public string definitionSha256;
        public bool projectAuthoredProcedural;
        public RobotDefinition robotDefinition;
        public ArticulationRobotDriver articulationDriver;

        public bool ValidateForStation(
            StationCatalog catalog,
            StationCatalogEntry station,
            bool bindArticulation,
            out string error)
        {
            if (catalog == null || station == null)
                return Fail("offline rig validation requires a catalog station", out error);
            if (string.IsNullOrWhiteSpace(robotId) || robotId != station.robot_id)
                return Fail(
                    $"offline rig robot '{robotId}' does not match station robot '{station.robot_id}'",
                    out error);
            StationCatalogRobotAsset asset = catalog.FindRobotAsset(robotId);
            if (asset == null)
                return Fail($"robot '{robotId}' is absent from the catalog provenance table", out error);
            if (!asset.redistribution_verified)
                return Fail(
                    $"robot '{robotId}' redistribution provenance has not been verified",
                    out error);

            if (projectAuthoredProcedural)
            {
                if (robotDefinition != null || articulationDriver != null)
                    return Fail("procedural offline rig must not reference an articulation definition", out error);
                if (!string.Equals(asset.redistribution_license, "project-authored",
                        StringComparison.OrdinalIgnoreCase))
                    return Fail("only catalog project-authored robots may use procedural rigs", out error);
                error = null;
                return true;
            }

            if (robotDefinition == null || articulationDriver == null)
                return Fail("normalized offline rig requires a robot definition and articulation driver", out error);
            if (robotDefinition.robotId != robotId || articulationDriver.definition != robotDefinition)
                return Fail("offline rig definition identity is inconsistent", out error);
            if (!IsSha256(definitionSha256) ||
                definitionSha256 != robotDefinition.definitionSha256)
                return Fail("offline rig physics hash does not match its robot definition", out error);
            if (!articulationDriver.TryValidateHierarchy(out error))
                return false;
            ArticulationLoopClosureBinding loopBinding =
                GetComponent<ArticulationLoopClosureBinding>();
            bool requiresLoopBinding = false;
            if (robotDefinition.auxiliaryJoints != null)
                foreach (RobotAuxiliaryJointDefinition joint in robotDefinition.auxiliaryJoints)
                    requiresLoopBinding |= joint != null && joint.IsLoopClosure;
            if (requiresLoopBinding && loopBinding == null)
                return Fail(
                    "robot definition requires auxiliary loop-closure bindings",
                    out error);
            if (loopBinding != null &&
                !(Application.isPlaying
                    ? loopBinding.TryValidate(out error)
                    : loopBinding.TryValidateMetadata(out error)))
                return false;
            if (bindArticulation && !articulationDriver.TryBind(out error))
                return false;
            error = null;
            return true;
        }

        static bool IsSha256(string value)
        {
            if (string.IsNullOrWhiteSpace(value) || value.Length != 64)
                return false;
            foreach (char character in value)
                if (!Uri.IsHexDigit(character))
                    return false;
            return true;
        }

        static bool Fail(string message, out string error)
        {
            error = message;
            return false;
        }
    }
}
