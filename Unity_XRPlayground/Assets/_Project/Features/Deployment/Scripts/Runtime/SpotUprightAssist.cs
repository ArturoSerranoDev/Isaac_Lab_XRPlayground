using System;
using UnityEngine;

namespace XRPlayground.Deployment
{
    [DisallowMultipleComponent]
    public sealed class SpotUprightAssist : MonoBehaviour
    {
        public AssistProfile profile;
        public ArticulationBody rootBody;
        public ContactSensor[] feet;
        public bool unassistedDiagnostic;
        public event Action<AssistTelemetry> Telemetry;

        void FixedUpdate()
        {
            if (unassistedDiagnostic || profile == null || !profile.enabledInProduction ||
                profile.kind != AssistKind.SpotUprightDamping || rootBody == null)
                return;
            int contacts = 0;
            if (feet != null)
                foreach (ContactSensor foot in feet)
                    if (foot != null && foot.IsTouching)
                        contacts++;
            float tilt = Vector3.Angle(rootBody.transform.up, Vector3.up);
            if (contacts < 2 || tilt >= profile.maximumUprightTiltDegrees)
                return;
            // Contact-gated angular PD only: restore the up axis while damping
            // rotation. This never changes root position and disengages before
            // it could conceal a genuine fall.
            Vector3 correctionAxis = Vector3.Cross(rootBody.transform.up, Vector3.up);
            Vector3 torque = correctionAxis * (profile.uprightAngularDamping * 4f) -
                             rootBody.angularVelocity * profile.uprightAngularDamping;
            rootBody.AddTorque(torque, ForceMode.Acceleration);
            Telemetry?.Invoke(new AssistTelemetry(profile.profileId, true, 0f, torque.magnitude, false));
        }
    }
}
