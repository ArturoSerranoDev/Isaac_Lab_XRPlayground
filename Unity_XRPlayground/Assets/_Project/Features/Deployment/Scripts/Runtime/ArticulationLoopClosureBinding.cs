using System;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.SceneManagement;

namespace XRPlayground.Deployment
{
    [Serializable]
    public sealed class ArticulationLoopClosureEntry
    {
        public string constraintName;
        public ArticulationBody parentBody;
        public ArticulationBody childBody;
        public Vector3 parentAnchorPosition;
        public Quaternion parentAnchorRotation = Quaternion.identity;
        public Vector3 childAnchorPosition;
        public Quaternion childAnchorRotation = Quaternion.identity;
        public Vector3 axis = Vector3.right;
        [Min(1e-5f)] public float proxyMass = 0.1f;
    }

    /// <summary>
    /// Stores prefab-safe loop metadata and spawns the required Rigidbody proxies
    /// as separate scene roots before the articulation driver binds.
    /// </summary>
    [DefaultExecutionOrder(-200)]
    [DisallowMultipleComponent]
    public sealed class ArticulationLoopClosureBinding : MonoBehaviour
    {
        public ArticulationLoopClosureEntry[] entries =
            Array.Empty<ArticulationLoopClosureEntry>();

        readonly Dictionary<string, ArticulationLoopClosureProxy> _proxies = new();

        public IReadOnlyDictionary<string, ArticulationLoopClosureProxy> Proxies => _proxies;

        void Awake()
        {
            if (!TryInitialize(out string error))
                Debug.LogError(error, this);
        }

        void OnDestroy() => DestroyProxies();

        public bool Contains(string constraintName)
        {
            if (entries == null)
                return false;
            foreach (ArticulationLoopClosureEntry entry in entries)
                if (entry != null && entry.constraintName == constraintName)
                    return true;
            return false;
        }

        public bool TryGetProxy(
            string constraintName, out ArticulationLoopClosureProxy proxy) =>
            _proxies.TryGetValue(constraintName, out proxy) && proxy != null;

        public bool TryInitialize(out string error)
        {
            if (_proxies.Count > 0)
                return TryValidate(out error);
            if (!TryValidateMetadata(out error))
                return false;
            try
            {
                foreach (ArticulationLoopClosureEntry entry in entries)
                {
                    var proxyObject = new GameObject(
                        $"{name}__Loop__{entry.constraintName}");
                    Scene ownerScene = gameObject.scene;
                    if (ownerScene.IsValid() && proxyObject.scene != ownerScene)
                        SceneManager.MoveGameObjectToScene(proxyObject, ownerScene);
                    ArticulationLoopClosureProxy proxy =
                        proxyObject.AddComponent<ArticulationLoopClosureProxy>();
                    proxy.proxyMass = entry.proxyMass;
                    proxy.Configure(
                        entry.constraintName,
                        entry.parentBody,
                        entry.childBody,
                        entry.parentAnchorPosition,
                        entry.parentAnchorRotation,
                        entry.childAnchorPosition,
                        entry.childAnchorRotation,
                        entry.axis);
                    _proxies.Add(entry.constraintName, proxy);
                }
            }
            catch (Exception exception)
            {
                DestroyProxies();
                return Fail(
                    $"Could not initialize articulation loop closures: {exception.Message}",
                    out error);
            }
            return TryValidate(out error);
        }

        public bool TryValidateMetadata(out string error)
        {
            if (entries == null)
                return Fail("Loop-closure binding has no entries array", out error);
            var names = new HashSet<string>();
            foreach (ArticulationLoopClosureEntry entry in entries)
                if (entry == null || string.IsNullOrWhiteSpace(entry.constraintName) ||
                    !names.Add(entry.constraintName) || entry.parentBody == null ||
                    entry.childBody == null || entry.parentBody == entry.childBody ||
                    entry.axis.sqrMagnitude < 0.999f || entry.axis.sqrMagnitude > 1.001f ||
                    !float.IsFinite(entry.proxyMass) || entry.proxyMass <= 0f)
                    return Fail(
                        "Loop-closure binding contains an invalid or duplicate entry",
                        out error);
            error = null;
            return true;
        }

        public bool TryValidate(out string error)
        {
            if (!TryValidateMetadata(out error))
                return false;
            if (_proxies.Count != entries.Length)
                return Fail("Not all loop-closure proxies are active", out error);
            foreach (ArticulationLoopClosureEntry entry in entries)
            {
                if (entry == null)
                    return Fail(
                        "Loop-closure binding contains a null entry",
                        out error);
                if (!_proxies.TryGetValue(
                        entry.constraintName, out ArticulationLoopClosureProxy proxy) ||
                    proxy == null)
                    return Fail(
                        $"Loop closure '{entry.constraintName}' has no active proxy",
                        out error);
                if (!proxy.TryValidate(out string proxyError))
                    return Fail(
                        $"Loop closure '{entry.constraintName}' is unhealthy: {proxyError}",
                        out error);
            }
            error = null;
            return true;
        }

        void DestroyProxies()
        {
            foreach (ArticulationLoopClosureProxy proxy in _proxies.Values)
                if (proxy != null)
                    Destroy(proxy.gameObject);
            _proxies.Clear();
        }

        static bool Fail(string message, out string error)
        {
            error = message;
            return false;
        }
    }
}
