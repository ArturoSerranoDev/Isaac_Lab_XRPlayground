using UnityEngine;

namespace XRPlayground.Deployment
{
    public enum AssistKind
    {
        None,
        ContactGrip,
        SpotUprightDamping,
    }

    [CreateAssetMenu(menuName = "XRPlayground/Deployment/Assist Profile")]
    public sealed class AssistProfile : ScriptableObject
    {
        public string profileId;
        public AssistKind kind;
        [Min(0)] public int requiredDistinctContacts;
        [Min(0)] public float breakForce = 150f;
        [Min(0)] public float breakTorque = 50f;
        [Min(0)] public float uprightAngularDamping = 3f;
        [Range(0f, 90f)] public float maximumUprightTiltDegrees = 20f;
        public bool enabledInProduction = true;
    }

    public readonly struct AssistTelemetry
    {
        public readonly string ProfileId;
        public readonly bool Active;
        public readonly float Force;
        public readonly float Torque;
        public readonly bool Broke;

        public AssistTelemetry(string profileId, bool active, float force, float torque, bool broke)
        {
            ProfileId = profileId;
            Active = active;
            Force = force;
            Torque = torque;
            Broke = broke;
        }
    }
}
