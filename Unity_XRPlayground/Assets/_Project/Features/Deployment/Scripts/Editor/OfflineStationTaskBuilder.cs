using System;
using System.Collections.Generic;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;

namespace XRPlayground.Deployment.Editor
{
    public static class OfflineStationTaskBuilder
    {
        const string PhysicsFolder = "Assets/_Project/Features/Deployment/PhysicsMaterials";

        [MenuItem("XRPlayground/Deployment/Configure Selected Offline Station Rig")]
        public static void ConfigureSelected()
        {
            Configure(Selection.activeGameObject);
        }

        public static void Configure(GameObject rig)
        {
            OfflineRigIdentity identity = rig != null ? rig.GetComponent<OfflineRigIdentity>() : null;
            StationRuntime runtime = rig != null ? rig.GetComponentInParent<StationRuntime>() : null;
            if (identity == null || runtime == null || runtime.offlineRig != rig)
                throw new InvalidOperationException(
                    "Select the installed OfflineRig_* root assigned to a StationRuntime");
            if (!StationCatalog.TryLoad(out StationCatalog catalog, out string error))
                throw new InvalidOperationException(error);
            StationCatalogEntry station = catalog.Find(runtime.stationId) ??
                throw new InvalidOperationException($"Unknown station '{runtime.stationId}'");
            if (!identity.ValidateForStation(catalog, station, false, out error))
                throw new InvalidOperationException(error);
            if (identity.articulationDriver == null)
                throw new InvalidOperationException("Station task authoring requires an articulation driver");

            Dictionary<string, ArticulationBody> links = BuildLinkMap(identity.articulationDriver);
            ArticulationBody rootBody = RequireLink(links, station.offline_bindings.root_link);
            Transform endEffector = string.IsNullOrWhiteSpace(station.offline_bindings.end_effector_link)
                ? null
                : RequireLink(links, station.offline_bindings.end_effector_link).transform;
            ContactSensor[] contacts = EnsureContactSensors(links, station.offline_bindings.contact_links);

            switch (station.adapter_id)
            {
                case "ball_catch":
                    ConfigureBall(rig, runtime, identity.articulationDriver, endEffector, contacts);
                    break;
                case "conveyor_color":
                    ConfigureConveyor(rig, runtime, identity.articulationDriver, endEffector, contacts);
                    break;
                case "pick_place_table":
                    ConfigurePickPlace(rig, runtime, identity.articulationDriver, endEffector, contacts);
                    break;
                case "spot_loco":
                    ConfigureSpot(rig, runtime, identity.articulationDriver, rootBody, contacts, false);
                    break;
                case "spot_follow":
                    ConfigureSpot(rig, runtime, identity.articulationDriver, rootBody, contacts, true);
                    break;
                default:
                    throw new InvalidOperationException(
                        $"Station adapter '{station.adapter_id}' is not an articulated offline station");
            }
            rig.SetActive(false);
            EditorUtility.SetDirty(runtime);
            EditorSceneManager.MarkSceneDirty(EditorSceneManager.GetActiveScene());
            EditorSceneManager.SaveOpenScenes();
            AssetDatabase.SaveAssets();
            Debug.Log(
                $"Configured physical task bindings for '{station.station_id}'. OfflinePolicy remains " +
                "fail-closed until its promoted bundle and calibrated physics profile are available.");
        }

        static void ConfigureBall(
            GameObject rig,
            StationRuntime runtime,
            ArticulationRobotDriver driver,
            Transform endEffector,
            ContactSensor[] contacts)
        {
            if (endEffector == null || contacts.Length != 3)
                throw new InvalidOperationException("Ball Catch requires one end effector and three fingertips");
            Transform task = EnsureChild(rig.transform, "TaskPhysics");
            PhysicsMaterial material = LoadOrCreatePhysicsMaterial(
                "BallCatch_Ball", 3.5f, 3.0f, 0f);
            GameObject ballObject = EnsurePrimitive(task, "Ball", PrimitiveType.Sphere);
            SetIsaacTransform(ballObject.transform, new Vector3(0.55f, 0f, 0.95f),
                new Vector3(0.0825f, 0.0825f, 0.0825f));
            ConfigureCollider(ballObject, material);
            Rigidbody ball = ConfigureBody(ballObject, 0.055f, CollisionDetectionMode.ContinuousDynamic);
            ContactGripAssist assist = EnsureGripAssist(endEffector.gameObject, contacts);

            BallCatchStationAdapter adapter = GetOrAdd<BallCatchStationAdapter>(rig);
            adapter.robot = driver;
            adapter.environmentAnchor = rig.transform;
            adapter.endEffector = endEffector;
            adapter.fingertips = Array.ConvertAll(contacts, sensor => sensor.transform);
            adapter.ball = ball;
            adapter.gripAssist = assist;
            adapter.resetBallPositionIsaac = new Vector3(0.55f, 0f, 0.95f);
            adapter.ballGravityScale = 0.55f;
            runtime.adapterBehaviour = adapter;
        }

        static void ConfigureConveyor(
            GameObject rig,
            StationRuntime runtime,
            ArticulationRobotDriver driver,
            Transform endEffector,
            ContactSensor[] contacts)
        {
            if (endEffector == null || contacts.Length != 2)
                throw new InvalidOperationException("Conveyor requires one end effector and two opposing contacts");
            Transform task = EnsureChild(rig.transform, "TaskPhysics");
            PhysicsMaterial surfaceMaterial = LoadOrCreatePhysicsMaterial(
                "Conveyor_Surface", 1.2f, 1.0f, 0f);
            PhysicsMaterial objectMaterial = LoadOrCreatePhysicsMaterial(
                "Conveyor_Object", 1.2f, 1.0f, 0f);

            GameObject belt = EnsurePrimitive(task, "Belt", PrimitiveType.Cube);
            SetIsaacTransform(belt.transform, new Vector3(0.55f, 0f, 0.40f),
                new Vector3(0.30f, 1.40f, 0.04f));
            ConfigureCollider(belt, surfaceMaterial);
            ContactConveyorSurface surface = GetOrAdd<ContactConveyorSurface>(belt);
            surface.environmentAnchor = rig.transform;
            surface.surfaceVelocityIsaac = new Vector3(0f, 0.22f, 0f);

            GameObject target = EnsurePrimitive(task, "TargetBin", PrimitiveType.Cube);
            SetIsaacTransform(target.transform, new Vector3(0.20f, 0.78f, 0.405f),
                new Vector3(0.36f, 0.36f, 0.03f));
            ConfigureCollider(target, surfaceMaterial);
            GameObject reject = EnsurePrimitive(task, "RejectBin", PrimitiveType.Cube);
            SetIsaacTransform(reject.transform, new Vector3(0.55f, 0.78f, 0.405f),
                new Vector3(0.34f, 0.28f, 0.03f));
            ConfigureCollider(reject, surfaceMaterial);

            var slots = new PhysicalObjectSlot[4];
            for (int index = 0; index < slots.Length; index++)
            {
                GameObject cube = EnsurePrimitive(task, $"Object_{index}", PrimitiveType.Cube);
                SetIsaacTransform(cube.transform, new Vector3(0.55f, -0.50f, 0.445f),
                    new Vector3(0.05f, 0.05f, 0.05f));
                ConfigureCollider(cube, objectMaterial);
                Rigidbody body = ConfigureBody(cube, 0.05f, CollisionDetectionMode.ContinuousDynamic);
                slots[index] = new PhysicalObjectSlot
                {
                    id = $"object_{index}",
                    body = body,
                    color = index % 3,
                    active = index == 0,
                    useConfiguredResetPose = true,
                    resetPositionIsaac = new Vector3(0.55f, -0.50f, 0.445f),
                    resetJitterIsaac = new Vector3(0.11f, 0.10f, 0f),
                };
            }
            ContactGripAssist assist = EnsureGripAssist(endEffector.gameObject, contacts);
            ConveyorColorStationAdapter adapter = GetOrAdd<ConveyorColorStationAdapter>(rig);
            adapter.robot = driver;
            adapter.environmentAnchor = rig.transform;
            adapter.endEffector = endEffector;
            adapter.targetBin = target.transform;
            adapter.rejectBin = reject.transform;
            adapter.objectSlots = slots;
            adapter.gripAssist = assist;
            adapter.binAcceptanceRadiusM = 0.18f;
            adapter.rejectAcceptanceRadiusM = 0.16f;
            adapter.acceptanceHeightIsaac = new Vector2(0.38f, 0.48f);
            adapter.spawnIntervalSeconds = 2.5f;
            runtime.adapterBehaviour = adapter;
        }

        static void ConfigurePickPlace(
            GameObject rig,
            StationRuntime runtime,
            ArticulationRobotDriver driver,
            Transform endEffector,
            ContactSensor[] contacts)
        {
            if (endEffector == null || contacts.Length != 2)
                throw new InvalidOperationException("Pick and Place requires one end effector and two pad contacts");
            Transform task = EnsureChild(rig.transform, "TaskPhysics");
            PhysicsMaterial surfaceMaterial = LoadOrCreatePhysicsMaterial(
                "PickPlace_Surface", 1.2f, 1.0f, 0f);
            PhysicsMaterial pieceMaterial = LoadOrCreatePhysicsMaterial(
                "PickPlace_Piece", 1.2f, 1.0f, 0f);
            GameObject table = EnsurePrimitive(task, "Table", PrimitiveType.Cube);
            SetIsaacTransform(table.transform, new Vector3(0.45f, 0f, 0.38f),
                new Vector3(0.75f, 0.60f, 0.04f));
            ConfigureCollider(table, surfaceMaterial);
            GameObject bucket = EnsurePrimitive(task, "Bucket", PrimitiveType.Cube);
            SetIsaacTransform(bucket.transform, new Vector3(0.22f, 0f, 0.46f),
                new Vector3(0.16f, 0.16f, 0.12f));
            ConfigureCollider(bucket, surfaceMaterial);
            Transform bucketTarget = EnsureChild(task, "BucketTarget");
            SetIsaacTransform(bucketTarget, new Vector3(0.22f, 0f, 0.54f), Vector3.one);

            var pieces = new PhysicalObjectSlot[4];
            for (int index = 0; index < pieces.Length; index++)
            {
                GameObject cube = EnsurePrimitive(task, $"Piece_{index}", PrimitiveType.Cube);
                SetIsaacTransform(cube.transform, new Vector3(0.55f, 0f, 0.42f),
                    new Vector3(0.04f, 0.04f, 0.04f));
                ConfigureCollider(cube, pieceMaterial);
                Rigidbody body = ConfigureBody(cube, 0.04f, CollisionDetectionMode.ContinuousDynamic);
                pieces[index] = new PhysicalObjectSlot
                {
                    id = $"piece_{index}",
                    body = body,
                    color = index,
                    active = index == 0,
                    useConfiguredResetPose = true,
                    resetPositionIsaac = new Vector3(0.55f, 0f, 0.42f),
                    resetJitterIsaac = new Vector3(0.15f, 0.22f, 0f),
                };
            }
            ContactGripAssist assist = EnsureGripAssist(endEffector.gameObject, contacts);
            PickPlaceStationAdapter adapter = GetOrAdd<PickPlaceStationAdapter>(rig);
            adapter.robot = driver;
            adapter.environmentAnchor = rig.transform;
            adapter.endEffector = endEffector;
            adapter.bucketTarget = bucketTarget;
            adapter.pieces = pieces;
            adapter.gripAssist = assist;
            adapter.liftHeightIsaac = 0.52f;
            adapter.placementRadiusM = 0.11f;
            runtime.adapterBehaviour = adapter;
        }

        static void ConfigureSpot(
            GameObject rig,
            StationRuntime runtime,
            ArticulationRobotDriver driver,
            ArticulationBody rootBody,
            ContactSensor[] feet,
            bool follow)
        {
            if (feet.Length != 4)
                throw new InvalidOperationException("Spot requires four named foot contacts");
            SpotLocomotionStationAdapter locomotion = GetOrAdd<SpotLocomotionStationAdapter>(rig);
            locomotion.robot = driver;
            locomotion.rootBody = rootBody;
            locomotion.environmentAnchor = rig.transform;
            locomotion.footContactSensors = feet;
            SpotUprightAssist upright = GetOrAdd<SpotUprightAssist>(rootBody.gameObject);
            upright.rootBody = rootBody;
            upright.feet = feet;
            if (!follow)
            {
                runtime.adapterBehaviour = locomotion;
                return;
            }

            Transform dependencyNode = EnsureChild(rig.transform, "LocomotionPolicyDependency");
            PolicyRuntime dependency = GetOrAdd<PolicyRuntime>(dependencyNode.gameObject);
            dependency.expectedPolicyId = "spot.locomotion";
            dependency.expectedStationId = "spot_loco";
            SpotFollowStationAdapter adapter = GetOrAdd<SpotFollowStationAdapter>(rig);
            adapter.locomotion = locomotion;
            adapter.locomotionPolicy = dependency;
            adapter.rootBody = rootBody;
            adapter.environmentAnchor = rig.transform;
            GameObject xrOrigin = GameObject.Find("XR Origin");
            adapter.target = Camera.main != null
                ? Camera.main.transform
                : xrOrigin != null ? xrOrigin.transform : null;
            runtime.adapterBehaviour = adapter;
        }

        static Dictionary<string, ArticulationBody> BuildLinkMap(ArticulationRobotDriver driver)
        {
            if (!driver.TryValidateHierarchy(out string error))
                throw new InvalidOperationException(error);
            var result = new Dictionary<string, ArticulationBody>();
            foreach (ArticulationBody body in driver.articulationRoot.GetComponentsInChildren<ArticulationBody>(true))
                result.Add(body.name, body);
            return result;
        }

        static ArticulationBody RequireLink(
            IReadOnlyDictionary<string, ArticulationBody> links, string name) =>
            links.TryGetValue(name, out ArticulationBody body)
                ? body
                : throw new InvalidOperationException($"Offline articulation is missing semantic link '{name}'");

        static ContactSensor[] EnsureContactSensors(
            IReadOnlyDictionary<string, ArticulationBody> links, string[] names)
        {
            var result = new ContactSensor[names.Length];
            for (int index = 0; index < names.Length; index++)
                result[index] = GetOrAdd<ContactSensor>(RequireLink(links, names[index]).gameObject);
            return result;
        }

        static ContactGripAssist EnsureGripAssist(GameObject target, ContactSensor[] contacts)
        {
            ContactGripAssist assist = GetOrAdd<ContactGripAssist>(target);
            assist.contactSensors = contacts;
            assist.gripperArticulation = target.GetComponent<ArticulationBody>();
            assist.gripperBody = null;
            return assist;
        }

        static Transform EnsureChild(Transform parent, string name)
        {
            Transform child = parent.Find(name);
            if (child != null)
                return child;
            var created = new GameObject(name);
            created.transform.SetParent(parent, false);
            return created.transform;
        }

        static GameObject EnsurePrimitive(Transform parent, string name, PrimitiveType type)
        {
            Transform existing = parent.Find(name);
            if (existing != null)
                return existing.gameObject;
            GameObject created = GameObject.CreatePrimitive(type);
            created.name = name;
            created.transform.SetParent(parent, false);
            return created;
        }

        static void SetIsaacTransform(Transform target, Vector3 position, Vector3 size)
        {
            target.localPosition = DeploymentFrameConverter.IsaacToUnity(position);
            target.localRotation = Quaternion.identity;
            target.localScale = new Vector3(size.x, size.z, size.y);
        }

        static void ConfigureCollider(GameObject target, PhysicsMaterial material)
        {
            Collider collider = target.GetComponent<Collider>() ??
                throw new InvalidOperationException($"Task object '{target.name}' has no collider");
            collider.sharedMaterial = material;
            collider.contactOffset = 0.005f;
        }

        static Rigidbody ConfigureBody(
            GameObject target, float mass, CollisionDetectionMode collisionMode)
        {
            Rigidbody body = GetOrAdd<Rigidbody>(target);
            body.mass = mass;
            body.useGravity = true;
            body.isKinematic = false;
            body.interpolation = RigidbodyInterpolation.Interpolate;
            body.collisionDetectionMode = collisionMode;
            body.maxDepenetrationVelocity = 5f;
            return body;
        }

        static PhysicsMaterial LoadOrCreatePhysicsMaterial(
            string name, float staticFriction, float dynamicFriction, float restitution)
        {
            EnsureFolder(PhysicsFolder);
            string path = $"{PhysicsFolder}/{name}.physicMaterial";
            PhysicsMaterial material = AssetDatabase.LoadAssetAtPath<PhysicsMaterial>(path);
            if (material == null)
            {
                material = new PhysicsMaterial(name);
                AssetDatabase.CreateAsset(material, path);
            }
            material.name = name;
            material.staticFriction = staticFriction;
            material.dynamicFriction = dynamicFriction;
            material.bounciness = restitution;
            material.frictionCombine = PhysicsMaterialCombine.Multiply;
            material.bounceCombine = PhysicsMaterialCombine.Multiply;
            EditorUtility.SetDirty(material);
            return material;
        }

        static T GetOrAdd<T>(GameObject target) where T : Component
        {
            // Unity's missing-component sentinel is non-null to the CLR, so the
            // null-coalescing operator can return an unusable Component wrapper.
            T existing = target.GetComponent<T>();
            return existing != null ? existing : target.AddComponent<T>();
        }

        static void EnsureFolder(string folder)
        {
            string[] parts = folder.Split('/');
            string current = parts[0];
            for (int index = 1; index < parts.Length; index++)
            {
                string next = current + "/" + parts[index];
                if (!AssetDatabase.IsValidFolder(next))
                    AssetDatabase.CreateFolder(current, parts[index]);
                current = next;
            }
        }
    }
}
