using System;
using UnityEngine;

namespace XRPlayground.Deployment
{
    [Serializable]
    public sealed class PhysicalObjectSlot
    {
        public string id;
        public Rigidbody body;
        public int color;
        public bool active;
        public bool grasped;
        public bool useConfiguredResetPose;
        public Vector3 resetPositionIsaac;
        public Vector3 resetEulerDegreesIsaac;
        public Vector3 resetJitterIsaac;

        [NonSerialized] bool _cached;
        [NonSerialized] Vector3 _cachedWorldPosition;
        [NonSerialized] Quaternion _cachedWorldRotation;

        public void Reset(Transform environmentAnchor, System.Random random)
        {
            if (body == null)
                return;
            body.gameObject.SetActive(active);
            if (!_cached)
            {
                _cachedWorldPosition = body.position;
                _cachedWorldRotation = body.rotation;
                _cached = true;
            }
            if (useConfiguredResetPose)
            {
                Vector3 jitter = new(
                    Sample(random, -resetJitterIsaac.x, resetJitterIsaac.x),
                    Sample(random, -resetJitterIsaac.y, resetJitterIsaac.y),
                    Sample(random, -resetJitterIsaac.z, resetJitterIsaac.z));
                Vector3 localPosition = DeploymentFrameConverter.IsaacToUnity(
                    resetPositionIsaac + jitter);
                Quaternion localRotation = DeploymentFrameConverter.IsaacToUnity(
                    Quaternion.Euler(resetEulerDegreesIsaac));
                body.position = environmentAnchor != null
                    ? environmentAnchor.TransformPoint(localPosition)
                    : localPosition;
                body.rotation = environmentAnchor != null
                    ? environmentAnchor.rotation * localRotation
                    : localRotation;
            }
            else
            {
                body.position = _cachedWorldPosition;
                body.rotation = _cachedWorldRotation;
            }
            body.linearVelocity = Vector3.zero;
            body.angularVelocity = Vector3.zero;
            grasped = false;
        }

        static float Sample(System.Random random, float minimum, float maximum) =>
            random == null ? 0.5f * (minimum + maximum) :
            minimum + (maximum - minimum) * (float)random.NextDouble();
    }
}
