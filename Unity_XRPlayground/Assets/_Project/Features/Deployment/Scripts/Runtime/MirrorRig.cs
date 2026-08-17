using System;
using System.Collections.Generic;
using UnityEngine;

namespace XRPlayground.Deployment
{
    [Serializable]
    public sealed class NamedTransformBinding
    {
        public string id;
        public Transform target;
    }

    [DisallowMultipleComponent]
    public sealed class MirrorRig : MonoBehaviour
    {
        public BridgeV2Client bridge;
        public Transform environmentAnchor;
        public GameObject[] visualRoots;
        public NamedTransformBinding[] links;
        public NamedTransformBinding[] objects;

        sealed class Sample
        {
            public Vector3 previousPosition;
            public Quaternion previousRotation;
            public Vector3 currentPosition;
            public Quaternion currentRotation;
            public float previousArrival;
            public float currentArrival;
            public bool initialized;
            public bool active = true;
        }

        readonly Dictionary<string, Transform> _linkTargets = new();
        readonly Dictionary<string, Transform> _objectTargets = new();
        readonly Dictionary<string, Sample> _linkSamples = new();
        readonly Dictionary<string, Sample> _objectSamples = new();

        void Awake()
        {
            BuildMap(links, _linkTargets);
            BuildMap(objects, _objectTargets);
        }

        void OnEnable()
        {
            SetVisuals(true);
            if (bridge == null)
                return;
            bridge.RobotStateReceived += OnRobotState;
            bridge.ObjectsStateReceived += OnObjectsState;
        }

        void OnDisable()
        {
            SetVisuals(false);
            if (bridge == null)
                return;
            bridge.RobotStateReceived -= OnRobotState;
            bridge.ObjectsStateReceived -= OnObjectsState;
        }

        void SetVisuals(bool value)
        {
            if (visualRoots == null)
                return;
            foreach (GameObject root in visualRoots)
                if (root != null && root != gameObject)
                    root.SetActive(value);
        }

        void Update()
        {
            if (bridge == null || bridge.IsStale)
                return; // authoritative visuals freeze; they are never extrapolated.
            Render(_linkTargets, _linkSamples);
            Render(_objectTargets, _objectSamples);
        }

        void OnRobotState(RobotStatePayload payload, double _)
        {
            if (payload.links == null)
                return;
            foreach (NamedPoseState pose in payload.links)
                Record(_linkSamples, pose.name, pose.position, pose.orientation_xyzw, true);
        }

        void OnObjectsState(ObjectsStatePayload payload, double _)
        {
            if (payload.objects == null)
                return;
            foreach (SceneObjectState item in payload.objects)
                Record(_objectSamples, item.id, item.position, item.orientation_xyzw, item.active);
        }

        void Record(
            Dictionary<string, Sample> samples,
            string id,
            float[] position,
            float[] rotation,
            bool active)
        {
            if (string.IsNullOrEmpty(id) || position == null || position.Length < 3 ||
                rotation == null || rotation.Length < 4)
                return;
            if (!samples.TryGetValue(id, out Sample sample))
            {
                sample = new Sample();
                samples[id] = sample;
            }
            Vector3 p = DeploymentFrameConverter.IsaacToUnity(DeploymentFrameConverter.Vector3From(position));
            Quaternion q = DeploymentFrameConverter.IsaacToUnity(DeploymentFrameConverter.QuaternionFrom(rotation));
            float arrival = Time.realtimeSinceStartup;
            if (!sample.initialized)
            {
                sample.previousPosition = sample.currentPosition = p;
                sample.previousRotation = sample.currentRotation = q;
                sample.previousArrival = sample.currentArrival = arrival;
                sample.initialized = true;
            }
            else
            {
                sample.previousPosition = sample.currentPosition;
                sample.previousRotation = sample.currentRotation;
                sample.previousArrival = sample.currentArrival;
                sample.currentPosition = p;
                sample.currentRotation = q;
                sample.currentArrival = arrival;
            }
            sample.active = active;
        }

        void Render(Dictionary<string, Transform> targets, Dictionary<string, Sample> samples)
        {
            foreach (var pair in targets)
            {
                Transform target = pair.Value;
                if (target == null || !samples.TryGetValue(pair.Key, out Sample sample) || !sample.initialized)
                    continue;
                target.gameObject.SetActive(sample.active);
                if (!sample.active)
                    continue;
                float duration = Mathf.Max(0.0001f, sample.currentArrival - sample.previousArrival);
                float alpha = Mathf.Clamp01((Time.realtimeSinceStartup - sample.currentArrival + duration) / duration);
                Vector3 localPosition = Vector3.Lerp(sample.previousPosition, sample.currentPosition, alpha);
                Quaternion localRotation = Quaternion.Slerp(sample.previousRotation, sample.currentRotation, alpha);
                if (environmentAnchor != null)
                    target.SetPositionAndRotation(
                        environmentAnchor.TransformPoint(localPosition), environmentAnchor.rotation * localRotation);
                else
                    target.SetPositionAndRotation(localPosition, localRotation);
            }
        }

        static void BuildMap(NamedTransformBinding[] bindings, Dictionary<string, Transform> destination)
        {
            destination.Clear();
            if (bindings == null)
                return;
            foreach (NamedTransformBinding binding in bindings)
                if (binding != null && !string.IsNullOrEmpty(binding.id) && binding.target != null)
                    destination[binding.id] = binding.target;
        }
    }
}
