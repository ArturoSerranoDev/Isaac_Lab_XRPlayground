using System;
using System.Collections.Generic;
using System.IO;
using Unity.InferenceEngine;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;

namespace XRPlayground.Deployment.Editor
{
    /// <summary>Serializes only the atomically promoted release into loaded station shells.</summary>
    public sealed class ReadyPolicyBundleBinder : AssetPostprocessor
    {
        const string ReadyRoot =
            "Assets/_Project/Features/Deployment/Bundles/Ready";
        const string CatalogPath = ReadyRoot + "/catalog.json";
        const string CandidateRoot =
            "Assets/_Project/Features/Deployment/Bundles/Candidates";

        [Serializable]
        sealed class ReadyCatalog
        {
            public int schema_version;
            public string active_release;
            public bool complete_station_gate;
            public ReadyPolicy[] policies;
        }

        [Serializable]
        sealed class ReadyPolicy
        {
            public string policy_id;
            public string station_id;
            public string bundle;
        }

        static void OnPostprocessAllAssets(
            string[] importedAssets,
            string[] deletedAssets,
            string[] movedAssets,
            string[] movedFromAssetPaths)
        {
            foreach (string path in importedAssets)
            {
                if (string.Equals(path, CatalogPath, StringComparison.OrdinalIgnoreCase))
                {
                    EditorApplication.delayCall += BindPromotedReleaseAfterImport;
                    break;
                }
                if (path.StartsWith(
                        CandidateRoot + "/spot.follow/",
                        StringComparison.OrdinalIgnoreCase))
                {
                    EditorApplication.delayCall += TryBindLocalSpotAfterImport;
                    break;
                }
            }
        }

        static void TryBindLocalSpotAfterImport()
        {
            try
            {
                BindLocalSpotFollow(false);
            }
            catch (Exception exception)
            {
                Debug.LogWarning(
                    $"Local Spot candidates are not bindable yet: {exception.Message}");
            }
        }

        static void BindPromotedReleaseAfterImport()
        {
            try
            {
                BindPromotedRelease(false);
            }
            catch (Exception exception)
            {
                Debug.LogError($"Could not bind promoted deployment release: {exception.Message}");
            }
        }

        [MenuItem("XRPlayground/Deployment/Bind Promoted Policy Release")]
        public static void BindPromotedReleaseFromMenu() => BindPromotedRelease(true);

        [MenuItem("XRPlayground/Deployment/Bind Local Spot Follow Candidates")]
        public static void BindLocalSpotFollowFromMenu() => BindLocalSpotFollow(true);

        /// <summary>
        /// Bind only the two Spot development candidates. This is intentionally
        /// separate from promotion and leaves allowUnpromotedCandidate enabled.
        /// </summary>
        static void BindLocalSpotFollow(bool requireEveryRuntime)
        {
            var required = new Dictionary<string, string>
            {
                { "spot.locomotion", "spot_loco" },
                { "spot.follow", "spot_follow" },
            };
            var models = new Dictionary<string, ModelAsset>();
            var contractAssets = new Dictionary<string, TextAsset>();
            foreach (KeyValuePair<string, string> item in required)
            {
                string bundlePath = CandidateRoot + "/" + item.Key;
                ModelAsset model = AssetDatabase.LoadAssetAtPath<ModelAsset>(
                    bundlePath + "/policy.onnx");
                TextAsset contractAsset = AssetDatabase.LoadAssetAtPath<TextAsset>(
                    bundlePath + "/policy.contract.json");
                string parseError = null;
                PolicyContract contract = null;
                bool parsed = contractAsset != null &&
                    PolicyContract.TryParse(contractAsset, out contract, out parseError);
                if (model == null || !parsed || contract.policy_id != item.Key ||
                    contract.station_id != item.Value)
                    throw new InvalidDataException(
                        $"Local bundle '{item.Key}' is missing or invalid: {parseError}");
                if ((item.Key == "spot.locomotion" &&
                     (contract.ObservationDimension != 48 || contract.ActionDimension != 12 ||
                      contract.timing.deployment_physics_hz != 250 ||
                      contract.timing.policy_hz != 50)) ||
                    (item.Key == "spot.follow" &&
                     (contract.ObservationDimension != 10 || contract.ActionDimension != 3 ||
                      contract.timing.deployment_physics_hz != 250 ||
                      contract.timing.policy_hz != 5)))
                    throw new InvalidDataException(
                        $"Local bundle '{item.Key}' does not use the corrected Spot contract");

                models.Add(item.Key, model);
                contractAssets.Add(item.Key, contractAsset);
            }

            PolicyRuntime[] runtimes = UnityEngine.Object.FindObjectsByType<PolicyRuntime>(
                FindObjectsInactive.Include);
            var boundPolicies = new HashSet<string>();
            int boundRuntimes = 0;
            foreach (PolicyRuntime runtime in runtimes)
            {
                if (runtime == null ||
                    !required.TryGetValue(runtime.expectedPolicyId, out string stationId))
                    continue;
                Undo.RecordObject(runtime, "Bind local Spot policy candidate");
                runtime.modelAsset = models[runtime.expectedPolicyId];
                runtime.policyContractAsset = contractAssets[runtime.expectedPolicyId];
                runtime.expectedStationId = stationId;
                runtime.allowUnpromotedCandidate = true;
                EditorUtility.SetDirty(runtime);
                if (runtime.gameObject.scene.IsValid())
                    EditorSceneManager.MarkSceneDirty(runtime.gameObject.scene);
                boundPolicies.Add(runtime.expectedPolicyId);
                boundRuntimes++;
            }
            if (requireEveryRuntime && boundPolicies.Count != required.Count)
                throw new InvalidOperationException(
                    $"Only {boundPolicies.Count}/{required.Count} local Spot policies were bound");
            if (boundPolicies.Count == required.Count)
            {
                AssetDatabase.SaveAssets();
                EditorSceneManager.SaveOpenScenes();
                Debug.Log(
                    $"Bound {boundRuntimes} corrected local Spot locomotion and Follow runtimes. " +
                    "These remain development bundles and are not promoted.");
            }
        }

        static void BindPromotedRelease(bool requireEveryRuntime)
        {
            if (!File.Exists(CatalogPath))
                throw new FileNotFoundException(
                    "No promoted release catalog exists. Run validate --promote after all gates pass.",
                    CatalogPath);
            ReadyCatalog catalog = JsonUtility.FromJson<ReadyCatalog>(File.ReadAllText(CatalogPath));
            if (catalog == null || catalog.schema_version != 1 ||
                !catalog.complete_station_gate || string.IsNullOrWhiteSpace(catalog.active_release) ||
                catalog.policies == null || catalog.policies.Length != 6)
                throw new InvalidDataException("Ready catalog is not a complete six-policy release");

            var policies = new Dictionary<string, ReadyPolicy>();
            foreach (ReadyPolicy policy in catalog.policies)
            {
                if (policy == null || string.IsNullOrWhiteSpace(policy.policy_id) ||
                    string.IsNullOrWhiteSpace(policy.station_id) ||
                    !IsReleaseBundle(policy.bundle, catalog.active_release) ||
                    !policies.TryAdd(policy.policy_id, policy))
                    throw new InvalidDataException("Ready catalog contains an invalid policy entry");
            }

            PolicyRuntime[] runtimes = UnityEngine.Object.FindObjectsByType<PolicyRuntime>(
                FindObjectsInactive.Include);
            int bound = 0;
            foreach (PolicyRuntime runtime in runtimes)
            {
                if (runtime == null || string.IsNullOrWhiteSpace(runtime.expectedPolicyId))
                    continue;
                if (!policies.TryGetValue(runtime.expectedPolicyId, out ReadyPolicy policy))
                {
                    if (requireEveryRuntime)
                        throw new InvalidDataException(
                            $"No promoted bundle exists for '{runtime.expectedPolicyId}'");
                    continue;
                }
                string bundlePath = ReadyRoot + "/" + policy.bundle;
                ModelAsset model = AssetDatabase.LoadAssetAtPath<ModelAsset>(
                    bundlePath + "/policy.onnx");
                TextAsset contractAsset = AssetDatabase.LoadAssetAtPath<TextAsset>(
                    bundlePath + "/policy.contract.json");
                string parseError = null;
                PolicyContract contract = null;
                bool parsed = contractAsset != null &&
                    PolicyContract.TryParse(contractAsset, out contract, out parseError);
                if (model == null || !parsed ||
                    contract.policy_id != policy.policy_id ||
                    contract.station_id != policy.station_id ||
                    contract.evaluation == null || !contract.evaluation.ready ||
                    contract.evaluation.release_id != catalog.active_release)
                    throw new InvalidDataException(
                        $"Promoted bundle '{policy.policy_id}' is missing or invalid: {parseError}");

                Undo.RecordObject(runtime, "Bind promoted deployment policy");
                runtime.modelAsset = model;
                runtime.policyContractAsset = contractAsset;
                runtime.expectedStationId = policy.station_id;
                runtime.allowUnpromotedCandidate = false;
                EditorUtility.SetDirty(runtime);
                if (runtime.gameObject.scene.IsValid())
                    EditorSceneManager.MarkSceneDirty(runtime.gameObject.scene);
                bound++;
            }
            if (requireEveryRuntime && bound < 6)
                throw new InvalidOperationException(
                    $"Only {bound} policy runtimes were bound; build/configure all station shells first");
            AssetDatabase.SaveAssets();
            EditorSceneManager.SaveOpenScenes();
            Debug.Log(
                $"Bound {bound} Unity policy runtimes to promoted release " +
                $"'{catalog.active_release}'.");
        }

        public static bool IsReleaseBundle(string bundle, string releaseId)
        {
            if (string.IsNullOrWhiteSpace(bundle) || string.IsNullOrWhiteSpace(releaseId) ||
                bundle.Contains("..") || bundle.Contains('\\') || Path.IsPathRooted(bundle))
                return false;
            string prefix = $"releases/{releaseId}/policies/";
            return bundle.StartsWith(prefix, StringComparison.Ordinal) &&
                bundle.Length > prefix.Length;
        }
    }
}
