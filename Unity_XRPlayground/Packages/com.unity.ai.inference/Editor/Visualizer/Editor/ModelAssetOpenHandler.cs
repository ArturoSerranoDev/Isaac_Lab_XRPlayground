using System;
using UnityEditor;
using UnityEditor.Callbacks;
using UnityEngine;

namespace Unity.InferenceEngine.Editor.Visualizer.Editor
{
    static class ModelAssetOpenHandler
    {
        // Unity 6.5+ treats InstanceIDToObject(int) as a compile error (EntityId migration).
#if UNITY_6000_3_OR_NEWER
        [OnOpenAsset(9999)]
        public static bool OnOpenAssetCallback(EntityId entityId, int line)
        {
            var obj = EditorUtility.EntityIdToObject(entityId);
#else
        [OnOpenAsset(9999)]
        public static bool OnOpenAssetCallback(int instanceID, int line)
        {
            var obj = EditorUtility.InstanceIDToObject(instanceID);
#endif
            if (obj is ModelAsset modelAsset)
            {
                ModelVisualizerWindow.VisualizeModel(modelAsset);
                return true;
            }

            return false;
        }
    }
}
