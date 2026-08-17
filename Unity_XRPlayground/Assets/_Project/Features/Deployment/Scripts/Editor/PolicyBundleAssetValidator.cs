using System;
using System.IO;
using System.Security.Cryptography;
using UnityEditor;
using UnityEngine;

namespace XRPlayground.Deployment.Editor
{
    public sealed class PolicyBundleAssetValidator : AssetPostprocessor
    {
        [Serializable]
        sealed class OnnxSignature { public string sha256; }

        [Serializable]
        sealed class ContractHeader
        {
            public int schema_version;
            public string model_sha256;
            public OnnxSignature onnx;
        }

        static void OnPostprocessAllAssets(
            string[] importedAssets,
            string[] deletedAssets,
            string[] movedAssets,
            string[] movedFromAssetPaths)
        {
            foreach (string path in importedAssets)
            {
                if (!path.EndsWith("policy.onnx", StringComparison.OrdinalIgnoreCase) &&
                    !path.EndsWith("policy.contract.json", StringComparison.OrdinalIgnoreCase))
                    continue;
                ValidateDirectory(Path.GetDirectoryName(path)?.Replace('\\', '/'));
            }
        }

        static void ValidateDirectory(string directory)
        {
            if (string.IsNullOrEmpty(directory))
                return;
            string modelPath = directory + "/policy.onnx";
            string contractPath = directory + "/policy.contract.json";
            if (!File.Exists(modelPath) || !File.Exists(contractPath))
                return;
            ContractHeader contract = JsonUtility.FromJson<ContractHeader>(File.ReadAllText(contractPath));
            string actual = Sha256(modelPath);
            if (contract == null || contract.schema_version != 2 || contract.model_sha256 != actual ||
                contract.onnx == null || contract.onnx.sha256 != actual)
                Debug.LogError($"Deployment bundle hash validation failed: {directory}");
        }

        static string Sha256(string path)
        {
            using SHA256 algorithm = SHA256.Create();
            using FileStream stream = File.OpenRead(path);
            return BitConverter.ToString(algorithm.ComputeHash(stream)).Replace("-", "").ToLowerInvariant();
        }
    }
}
