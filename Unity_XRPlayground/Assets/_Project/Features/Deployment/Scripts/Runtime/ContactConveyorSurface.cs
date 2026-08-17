using UnityEngine;

namespace XRPlayground.Deployment
{
    [DisallowMultipleComponent]
    [RequireComponent(typeof(Collider))]
    public sealed class ContactConveyorSurface : MonoBehaviour
    {
        public Transform environmentAnchor;
        public Vector3 surfaceVelocityIsaac = new(0f, 0.22f, 0f);
        public float maximumAccelerationMps2 = 4f;

        Collider _surface;

        void Awake() => _surface = GetComponent<Collider>();

        void OnCollisionStay(Collision collision)
        {
            Rigidbody body = collision.rigidbody;
            if (body == null || body.isKinematic || _surface == null)
                return;
            Vector3 up = environmentAnchor != null ? environmentAnchor.up : Vector3.up;
            if (Vector3.Dot(body.worldCenterOfMass - _surface.bounds.center, up) <= 0f)
                return;
            Vector3 localVelocity = DeploymentFrameConverter.IsaacToUnity(surfaceVelocityIsaac);
            Vector3 targetVelocity = environmentAnchor != null
                ? environmentAnchor.TransformDirection(localVelocity)
                : localVelocity;
            float speed = targetVelocity.magnitude;
            if (speed <= 1e-6f)
                return;
            Vector3 direction = targetVelocity / speed;
            float currentSpeed = Vector3.Dot(body.linearVelocity, direction);
            float maximumDelta = Mathf.Max(0f, maximumAccelerationMps2) * Time.fixedDeltaTime;
            float velocityDelta = Mathf.Clamp(speed - currentSpeed, -maximumDelta, maximumDelta);
            body.AddForce(direction * velocityDelta, ForceMode.VelocityChange);
        }
    }
}
