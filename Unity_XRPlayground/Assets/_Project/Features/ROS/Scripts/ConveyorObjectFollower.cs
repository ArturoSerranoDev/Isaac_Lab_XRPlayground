using System.Collections.Generic;
using UnityEngine;

namespace XRPlayground.ROS
{
    /// <summary>
    /// Mirrors Isaac conveyor objects onto a fixed pool of Unity cube visuals.
    /// </summary>
    public sealed class ConveyorObjectFollower : MonoBehaviour
    {
        public RosTcpClient client;
        public Transform envAnchor;
        public Transform[] objectSlots;
        public bool followingEnabled = true;

        static readonly Color[] Colors =
        {
            new Color(0.90f, 0.15f, 0.12f),
            new Color(0.15f, 0.75f, 0.25f),
            new Color(0.15f, 0.35f, 0.90f),
        };

        void OnEnable()
        {
            if (client != null)
                client.MessageReceived += OnMessage;
        }

        void OnDisable()
        {
            if (client != null)
                client.MessageReceived -= OnMessage;
        }

        void OnMessage(string json)
        {
            if (!followingEnabled)
                return;
            if (!RosJson.TryParseTopic(json, out var topic) || topic != RosTopics.ConveyorObjectsState)
                return;
            if (!RosJson.TryParseConveyorObjects(json, out var state) || state == null)
                return;
            Apply(state);
        }

        public void Apply(ConveyorObjectsStateData state)
        {
            if (state.objects == null || objectSlots == null)
                return;
            Transform anchor = envAnchor != null ? envAnchor : transform;

            for (int i = 0; i < objectSlots.Length; i++)
            {
                var slot = objectSlots[i];
                if (slot == null)
                    continue;

                ConveyorObjectData obj = null;
                if (i < state.objects.Length)
                    obj = state.objects[i];

                bool active = obj != null && obj.active;
                slot.gameObject.SetActive(active);
                if (!active)
                    continue;

                var p = XrFrameConverter.IsaacPosToUnity(XrFrameConverter.FromArray3(obj.position));
                var q = XrFrameConverter.IsaacQuatToUnity(XrFrameConverter.FromXyzw(obj.orientation_xyzw));
                slot.SetPositionAndRotation(anchor.TransformPoint(p), anchor.rotation * q);

                int c = Mathf.Clamp(obj.color, 0, Colors.Length - 1);
                var rend = slot.GetComponentInChildren<Renderer>();
                if (rend != null)
                {
                    // Unique material instance so colors don't share
                    if (rend.sharedMaterial == null || !rend.material.name.Contains("ConveyorCube"))
                    {
                        var mat = new Material(Shader.Find("Universal Render Pipeline/Lit") ?? Shader.Find("Standard"));
                        mat.name = "ConveyorCubeMat";
                        rend.material = mat;
                    }
                    rend.material.color = Colors[c];
                }
            }
        }
    }
}
