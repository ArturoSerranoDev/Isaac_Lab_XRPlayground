using System;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;

namespace XRPlayground.Deployment.Editor
{
    public static class ProceduralBalanceBotBuilder
    {
        const string RobotFolder = "Assets/_Project/Features/Robots/BalanceBot";
        const string PrefabFolder = RobotFolder + "/Prefabs";
        const string MaterialFolder = RobotFolder + "/Materials";
        const string PrefabPath = PrefabFolder + "/BalanceBot_Offline.prefab";
        const string TrayMaterialPath = MaterialFolder + "/BalanceBot_Physics.physicMaterial";
        const string BallMaterialPath = MaterialFolder + "/BalanceBot_Ball_Physics.physicMaterial";

        [MenuItem("XRPlayground/Deployment/Build Procedural Balance Bot Offline Rig")]
        public static void Build()
        {
            EnsureFolder(PrefabFolder);
            EnsureFolder(MaterialFolder);
            PhysicsMaterial trayMaterial = LoadOrCreateMaterial(
                TrayMaterialPath, "BalanceBot_Physics", 0.95f, 0.85f);
            PhysicsMaterial ballMaterial = LoadOrCreateMaterial(
                BallMaterialPath, "BalanceBot_Ball_Physics", 0.7f, 0.6f);

            GameObject authored = CreateRig(trayMaterial, ballMaterial);
            GameObject prefab;
            try
            {
                prefab = PrefabUtility.SaveAsPrefabAsset(authored, PrefabPath);
            }
            finally
            {
                UnityEngine.Object.DestroyImmediate(authored);
            }
            if (prefab == null)
                throw new InvalidOperationException("Could not save the procedural Balance Bot prefab");

            GameObject shell = GameObject.Find("Deployment Runtime/balance_bot") ??
                throw new InvalidOperationException(
                    "Build the V2 runtime shells before installing the Balance Bot rig");
            Transform existing = shell.transform.Find("OfflineRig_BalanceBot");
            if (existing != null)
                UnityEngine.Object.DestroyImmediate(existing.gameObject);
            GameObject instance = (GameObject)PrefabUtility.InstantiatePrefab(prefab, shell.transform);
            instance.name = "OfflineRig_BalanceBot";
            GameObject stationAnchor = GameObject.Find(
                "Environment/Lab Zones/01_Manipulation/Robot Stations/Station_D_BalanceBot");
            if (stationAnchor != null)
                instance.transform.SetPositionAndRotation(
                    stationAnchor.transform.position, stationAnchor.transform.rotation);
            instance.SetActive(false);

            StationRuntime runtime = shell.GetComponent<StationRuntime>() ??
                throw new InvalidOperationException("Balance Bot shell has no StationRuntime");
            BalanceBotStationAdapter adapter = instance.GetComponent<BalanceBotStationAdapter>();
            runtime.offlineRig = instance;
            runtime.adapterBehaviour = adapter;
            EditorUtility.SetDirty(runtime);
            EditorSceneManager.MarkSceneDirty(EditorSceneManager.GetActiveScene());
            EditorSceneManager.SaveOpenScenes();
            AssetDatabase.SaveAssets();
            Debug.Log(
                "Built the committed procedural Balance Bot offline rig. It remains fail-closed " +
                "until a validated bundle is assigned and tray_120hz is calibrated.");
        }

        static GameObject CreateRig(PhysicsMaterial trayMaterial, PhysicsMaterial ballMaterial)
        {
            var root = new GameObject("OfflineRig_BalanceBot");
            OfflineRigIdentity identity = root.AddComponent<OfflineRigIdentity>();
            identity.robotId = "balance_tray_2dof";
            identity.projectAuthoredProcedural = true;
            var tray = GameObject.CreatePrimitive(PrimitiveType.Cube);
            tray.name = "Tray";
            tray.transform.SetParent(root.transform, false);
            tray.transform.localPosition = DeploymentFrameConverter.IsaacToUnity(
                new Vector3(0f, 0f, 0.75f));
            tray.transform.localScale = new Vector3(0.60f, 0.02f, 0.60f);
            tray.GetComponent<Collider>().material = trayMaterial;
            Rigidbody trayBody = tray.AddComponent<Rigidbody>();
            trayBody.mass = 3f;
            trayBody.useGravity = false;
            trayBody.isKinematic = true;
            trayBody.interpolation = RigidbodyInterpolation.Interpolate;
            trayBody.collisionDetectionMode = CollisionDetectionMode.ContinuousSpeculative;

            var stand = GameObject.CreatePrimitive(PrimitiveType.Cylinder);
            stand.name = "StandVisual";
            stand.transform.SetParent(root.transform, false);
            stand.transform.localPosition = new Vector3(0f, 0.37f, 0f);
            stand.transform.localScale = new Vector3(0.08f, 0.37f, 0.08f);
            UnityEngine.Object.DestroyImmediate(stand.GetComponent<Collider>());

            Rigidbody[] balls = new Rigidbody[2];
            for (int i = 0; i < balls.Length; i++)
            {
                GameObject ball = GameObject.CreatePrimitive(PrimitiveType.Sphere);
                ball.name = $"Ball_{i}";
                ball.transform.SetParent(root.transform, false);
                Vector3 positionIsaac = new(
                    i == 0 ? 0f : 0.09f,
                    i == 0 ? 0f : 0.06f,
                    0.875f);
                ball.transform.localPosition = DeploymentFrameConverter.IsaacToUnity(positionIsaac);
                ball.transform.localScale = Vector3.one * 0.07f;
                ball.GetComponent<Collider>().material = ballMaterial;
                balls[i] = ball.AddComponent<Rigidbody>();
                balls[i].mass = 0.057f;
                balls[i].interpolation = RigidbodyInterpolation.Interpolate;
                balls[i].collisionDetectionMode = CollisionDetectionMode.ContinuousDynamic;
                balls[i].maxAngularVelocity = 50f;
            }

            BalanceBotStationAdapter adapter = root.AddComponent<BalanceBotStationAdapter>();
            adapter.environmentAnchor = root.transform;
            adapter.tray = tray.transform;
            adapter.trayBody = trayBody;
            adapter.balls = balls;
            adapter.ballActive = new[] { true, true };
            adapter.trayCenterIsaac = new Vector3(0f, 0f, 0.75f);
            adapter.trayThicknessM = 0.02f;
            adapter.ballRadiusM = 0.035f;
            adapter.spawnHeightAboveTrayM = 0.08f;
            adapter.spawnXYHalfM = 0.12f;
            adapter.maximumTiltRad = 0.35f;
            adapter.trayHalfExtentM = 0.27f;
            adapter.dropHeightBelowTrayM = 0.4f;
            return root;
        }

        static PhysicsMaterial LoadOrCreateMaterial(
            string path, string name, float staticFriction, float dynamicFriction)
        {
            PhysicsMaterial material = AssetDatabase.LoadAssetAtPath<PhysicsMaterial>(path);
            if (material == null)
            {
                material = new PhysicsMaterial(name);
                AssetDatabase.CreateAsset(material, path);
            }
            material.name = name;
            material.staticFriction = staticFriction;
            material.dynamicFriction = dynamicFriction;
            material.bounciness = 0f;
            material.frictionCombine = PhysicsMaterialCombine.Average;
            material.bounceCombine = PhysicsMaterialCombine.Minimum;
            EditorUtility.SetDirty(material);
            return material;
        }

        static void EnsureFolder(string folder)
        {
            string[] parts = folder.Split('/');
            string current = parts[0];
            for (int i = 1; i < parts.Length; i++)
            {
                string next = current + "/" + parts[i];
                if (!AssetDatabase.IsValidFolder(next))
                    AssetDatabase.CreateFolder(current, parts[i]);
                current = next;
            }
        }
    }
}
