using NUnit.Framework;
using UnityEditor;
using UnityEngine;
using XRPlayground.Deployment.Editor;

namespace XRPlayground.Deployment.Tests
{
    public sealed class PolicyContractTests
    {
        const string Sha = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";

        [Test]
        public void ContractClipsOnlyTermsThatDeclareClip()
        {
            var contract = new PolicyContract
            {
                actions = new[]
                {
                    new PolicyTerm { name = "bounded", shape = new[] { 2 }, clip = new[] { -1f, 1f } },
                    new PolicyTerm { name = "unbounded", shape = new[] { 1 } },
                },
            };
            float[] values = { -2f, 3f, 9f };
            contract.ClipActionsInPlace(values);
            Assert.That(values, Is.EqualTo(new[] { -1f, 1f, 9f }));
        }

        [Test]
        public void ContractRejectsWrongTensorShape()
        {
            PolicyContract contract = ValidContract();
            contract.onnx.input_shape = new[] { 1, 3 };
            Assert.That(contract.Validate(out string error), Is.False);
            StringAssert.Contains("input shape", error);
        }

        [Test]
        public void ContractRejectsExternalObservationNormalization()
        {
            PolicyContract contract = ValidContract();
            contract.normalization_embedded = false;
            Assert.That(contract.Validate(out string error), Is.False);
            Assert.That(error, Does.Contain("normalization embedded"));
        }

        [Test]
        public void ContractAcceptsPairedStaticRecurrentStateAndRejectsMismatch()
        {
            PolicyContract contract = ValidContract();
            var stateInput = new PolicyTensorInfo
            {
                name = "hidden_in", shape = new[] { 1, 2, 8 },
            };
            var stateOutput = new PolicyTensorInfo
            {
                name = "hidden_out", shape = new[] { 1, 2, 8 },
            };
            contract.onnx.inputs = new[] { contract.onnx.inputs[0], stateInput };
            contract.onnx.outputs = new[] { contract.onnx.outputs[0], stateOutput };
            contract.recurrent_state = new[] { stateInput, stateOutput };
            Assert.That(contract.Validate(out string validError), Is.True, validError);

            stateOutput.shape = new[] { 1, 2, 7 };
            Assert.That(contract.Validate(out string mismatchError), Is.False);
            Assert.That(mismatchError, Does.Contain("different shapes"));
        }

        [Test]
        public void FrameConversionRoundTrips()
        {
            Vector3 source = new(1.2f, -0.7f, 2.4f);
            Assert.That(DeploymentFrameConverter.UnityToIsaac(
                DeploymentFrameConverter.IsaacToUnity(source)), Is.EqualTo(source));
            Quaternion rotation = Quaternion.Euler(10f, 20f, 30f);
            Quaternion roundTrip = DeploymentFrameConverter.UnityToIsaac(
                DeploymentFrameConverter.IsaacToUnity(rotation));
            Assert.That(Quaternion.Angle(rotation, roundTrip), Is.LessThan(1e-4f));

            Vector3 angular = new(0.3f, -0.5f, 0.8f);
            Assert.That(DeploymentFrameConverter.IsaacAngularToUnity(angular),
                Is.EqualTo(new Vector3(-0.3f, -0.8f, 0.5f)));
            Assert.That(DeploymentFrameConverter.UnityAngularToIsaac(
                DeploymentFrameConverter.IsaacAngularToUnity(angular)), Is.EqualTo(angular));
        }

        [Test]
        public void RobotDefinitionAxesMustBeFiniteAndNormalized()
        {
            Assert.That(RobotDefinitionValidation.IsFiniteUnitAxis(Vector3.right), Is.True);
            Assert.That(RobotDefinitionValidation.IsFiniteUnitAxis(new Vector3(2f, 0f, 0f)),
                Is.False);
            Assert.That(RobotDefinitionValidation.IsFiniteUnitAxis(
                new Vector3(float.NaN, 0f, 0f)), Is.False);
            Assert.That(RobotDefinitionValidation.IsFiniteUnitAxis(
                new Vector3(float.PositiveInfinity, 0f, 0f)), Is.False);
        }

        [Test]
        public void GeneratedStationCatalogLoadsAllStations()
        {
            Assert.That(StationCatalog.TryLoad(out StationCatalog catalog, out string error),
                Is.True, error);
            Assert.That(catalog.stations, Has.Length.EqualTo(6));
            Assert.That(catalog.robot_assets, Has.Length.EqualTo(5));
            Assert.That(catalog.FindRobotAsset("kinova_jaco2_n7s300").redistribution_verified,
                Is.True);
            Assert.That(catalog.FindRobotAsset("ur10e_robotiq_2f85").license_paths,
                Has.Length.EqualTo(2));
            Assert.That(catalog.FindRobotAsset("spot").redistribution_verified, Is.True);
            Assert.That(catalog.FindRobotAsset("agibot_a2d").redistribution_verified, Is.False);
            Assert.That(catalog.Find("spot_follow").policies[0].observation_dim, Is.EqualTo(10));
            Assert.That(catalog.Find("spot_follow").policies[0].policy_hz, Is.EqualTo(5));
            Assert.That(catalog.Find("ball_catch").offline_bindings.contact_links,
                Is.EqualTo(new[]
                {
                    "j2n7s300_link_finger_tip_1",
                    "j2n7s300_link_finger_tip_2",
                    "j2n7s300_link_finger_tip_3",
                }));
            Assert.That(catalog.Find("conveyor_color").offline_bindings.robot_position_isaac,
                Is.EqualTo(new[] { -0.22f, -0.10f, 0f }));
        }

        [Test]
        public void CatalogRejectsMissingEmptyAndDuplicateRobotLicensePaths()
        {
            Assert.That(StationCatalog.TryLoad(out StationCatalog catalog, out string loadError),
                Is.True, loadError);
            StationCatalogRobotAsset spot = catalog.FindRobotAsset("spot");
            string[] original = spot.license_paths;
            try
            {
                spot.license_paths = null;
                Assert.That(catalog.Validate(out _), Is.False);
                spot.license_paths = new[] { "" };
                Assert.That(catalog.Validate(out _), Is.False);
                spot.license_paths = new[] { original[0], original[0] };
                Assert.That(catalog.Validate(out _), Is.False);
            }
            finally
            {
                spot.license_paths = original;
            }
        }

        [Test]
        public void ProceduralRigRequiresMatchingVerifiedProjectAuthoredRobot()
        {
            Assert.That(StationCatalog.TryLoad(out StationCatalog catalog, out string loadError),
                Is.True, loadError);
            var root = new GameObject("procedural-rig");
            try
            {
                OfflineRigIdentity identity = root.AddComponent<OfflineRigIdentity>();
                identity.robotId = "balance_tray_2dof";
                identity.projectAuthoredProcedural = true;
                Assert.That(identity.ValidateForStation(
                    catalog, catalog.Find("balance_bot"), false, out string validError),
                    Is.True, validError);
                Assert.That(identity.ValidateForStation(
                    catalog, catalog.Find("ball_catch"), false, out string mismatchError),
                    Is.False);
                StringAssert.Contains("does not match", mismatchError);
            }
            finally
            {
                Object.DestroyImmediate(root);
            }
        }

        [Test]
        public void NormalizedRigRejectsUnverifiedRobotProvenance()
        {
            Assert.That(StationCatalog.TryLoad(out StationCatalog catalog, out string loadError),
                Is.True, loadError);
            var root = new GameObject("agibot-rig");
            try
            {
                OfflineRigIdentity identity = root.AddComponent<OfflineRigIdentity>();
                identity.robotId = "agibot_a2d";
                Assert.That(identity.ValidateForStation(
                    catalog, catalog.Find("pick_place_table"), false, out string error), Is.False);
                StringAssert.Contains("provenance", error);
            }
            finally
            {
                Object.DestroyImmediate(root);
            }
        }

        [Test]
        public void ConveyorRejectMetricRequiresPhysicalRejectZone()
        {
            var root = new GameObject("conveyor-test");
            var target = new GameObject("target");
            var reject = new GameObject("reject");
            var item = new GameObject("item");
            try
            {
                target.transform.SetParent(root.transform, false);
                reject.transform.SetParent(root.transform, false);
                item.transform.SetParent(root.transform, false);
                target.transform.localPosition = DeploymentFrameConverter.IsaacToUnity(
                    new Vector3(0.20f, 0.78f, 0.405f));
                reject.transform.localPosition = DeploymentFrameConverter.IsaacToUnity(
                    new Vector3(0.55f, 0.78f, 0.405f));
                Rigidbody body = item.AddComponent<Rigidbody>();
                var adapter = root.AddComponent<ConveyorColorStationAdapter>();
                adapter.environmentAnchor = root.transform;
                adapter.targetBin = target.transform;
                adapter.rejectBin = reject.transform;
                adapter.targetColor = 0;
                adapter.objectSlots = new[]
                {
                    new PhysicalObjectSlot { id = "object_0", body = body, color = 1, active = true },
                };
                body.position = DeploymentFrameConverter.IsaacToUnity(
                    new Vector3(0.55f, 0.0f, 0.445f));
                StationEvaluationSummary missed = adapter.CompleteEvaluationScenario();
                Assert.That(missed.task_metrics.reject_rate, Is.Zero);
                body.position = DeploymentFrameConverter.IsaacToUnity(
                    new Vector3(0.55f, 0.78f, 0.445f));
                StationEvaluationSummary rejected = adapter.CompleteEvaluationScenario();
                Assert.That(rejected.task_metrics.reject_rate, Is.EqualTo(1f));
                Assert.That(rejected.task_metrics.wrong_bin_rate, Is.Zero);
            }
            finally
            {
                Object.DestroyImmediate(root);
            }
        }

        [Test]
        public void ArticulationBindingPreservesPrismaticUnitsAndJointAxis()
        {
            var root = new GameObject("root_link");
            var child = new GameObject("slide_link");
            var revolute = new GameObject("revolute_link");
            RobotDefinition definition = ScriptableObject.CreateInstance<RobotDefinition>();
            try
            {
                child.transform.SetParent(root.transform, false);
                revolute.transform.SetParent(child.transform, false);
                ArticulationBody rootBody = root.AddComponent<ArticulationBody>();
                ArticulationBody childBody = child.AddComponent<ArticulationBody>();
                ArticulationBody revoluteBody = revolute.AddComponent<ArticulationBody>();
                definition.robotId = "slider";
                definition.links = new[]
                {
                    new RobotLinkDefinition
                    {
                        name = "root_link", mass = 1f, inertiaTensor = Vector3.one,
                        collisionShapeCount = 0,
                    },
                    new RobotLinkDefinition
                    {
                        name = "slide_link", mass = 1f, inertiaTensor = Vector3.one,
                        collisionShapeCount = 0,
                    },
                    new RobotLinkDefinition
                    {
                        name = "revolute_link", mass = 1f, inertiaTensor = Vector3.one,
                        collisionShapeCount = 0,
                    },
                };
                definition.joints = new[]
                {
                    new RobotJointDefinition
                    {
                        name = "slide_joint",
                        parentLink = "root_link",
                        childLink = "slide_link",
                        jointType = ArticulationJointType.PrismaticJoint,
                        positionUnit = "meter",
                        axis = Vector3.up,
                        lowerLimit = -0.2f,
                        upperLimit = 0.3f,
                        defaultPosition = 0.1f,
                        maxVelocity = 0.75f,
                    },
                    new RobotJointDefinition
                    {
                        name = "revolute_joint", parentLink = "slide_link",
                        childLink = "revolute_link",
                        jointType = ArticulationJointType.RevoluteJoint,
                        positionUnit = "radian", limitMode = "limited", axis = Vector3.right,
                        lowerLimit = -1f, upperLimit = 1f, maxVelocity = 2.5f,
                    },
                };
                ArticulationRobotDriver driver = root.AddComponent<ArticulationRobotDriver>();
                driver.definition = definition;
                driver.articulationRoot = rootBody;
                Assert.That(driver.TryBind(out string error), Is.True, error);
                Assert.That(childBody.xDrive.lowerLimit, Is.EqualTo(-0.2f).Within(1e-5f));
                Assert.That(childBody.xDrive.upperLimit, Is.EqualTo(0.3f).Within(1e-5f));
                Assert.That(childBody.xDrive.target, Is.EqualTo(0.1f).Within(1e-5f));
                Assert.That(childBody.maxJointVelocity, Is.EqualTo(0.75f).Within(1e-5f));
                Assert.That(revoluteBody.maxJointVelocity, Is.EqualTo(2.5f).Within(1e-5f));
                Assert.That(Vector3.Angle(
                    childBody.anchorRotation * Vector3.right, Vector3.up), Is.LessThan(1e-3f));
            }
            finally
            {
                Object.DestroyImmediate(root);
                Object.DestroyImmediate(definition);
            }
        }

        [Test]
        public void ArticulationBindingRejectsUnrepresentablePositiveRestOffset()
        {
            var root = new GameObject("root_link");
            var child = new GameObject("slide_link");
            RobotDefinition definition = ScriptableObject.CreateInstance<RobotDefinition>();
            try
            {
                child.transform.SetParent(root.transform, false);
                ArticulationBody rootBody = root.AddComponent<ArticulationBody>();
                child.AddComponent<ArticulationBody>();
                child.AddComponent<BoxCollider>();
                definition.robotId = "rest-offset-test";
                definition.links = new[]
                {
                    new RobotLinkDefinition
                    {
                        name = "root_link", mass = 1f, inertiaTensor = Vector3.one,
                        collisionShapeCount = 0,
                    },
                    new RobotLinkDefinition
                    {
                        name = "slide_link", mass = 1f, inertiaTensor = Vector3.one,
                        collisionShapeCount = 1, collisionEnabled = true,
                        staticFriction = 0.5f, dynamicFriction = 0.5f, restitution = 0f,
                        contactOffset = 0.02f, restOffset = 0.001f,
                    },
                };
                definition.joints = new[]
                {
                    new RobotJointDefinition
                    {
                        name = "slide_joint", parentLink = "root_link", childLink = "slide_link",
                        jointType = ArticulationJointType.PrismaticJoint,
                        positionUnit = "meter", limitMode = "limited", axis = Vector3.right,
                        lowerLimit = -0.1f, upperLimit = 0.1f, maxVelocity = 1f,
                    },
                };
                ArticulationRobotDriver driver = root.AddComponent<ArticulationRobotDriver>();
                driver.definition = definition;
                driver.articulationRoot = rootBody;
                Assert.That(driver.TryBind(out string error), Is.False);
                Assert.That(error, Does.Contain("rest offset"));
                Assert.That(error, Does.Contain("cannot be represented"));
            }
            finally
            {
                Object.DestroyImmediate(root);
                Object.DestroyImmediate(definition);
            }
        }

        [Test]
        public void RemotizedActuatorCurveClampsAndInterpolatesTorque()
        {
            var joint = new RobotJointDefinition
            {
                forceLimit = 100f,
                effortLimitCurve = new[]
                {
                    new RobotJointEffortLimitSample { position = -2f, maxEffort = 30f },
                    new RobotJointEffortLimitSample { position = -1f, maxEffort = 50f },
                },
            };
            Assert.That(joint.EffortLimitAt(-3f), Is.EqualTo(30f));
            Assert.That(joint.EffortLimitAt(-1.5f), Is.EqualTo(40f).Within(1e-5f));
            Assert.That(joint.EffortLimitAt(0f), Is.EqualTo(50f));
        }

        [Test]
        public void ReadyBundleBinderAcceptsOnlyTheActiveReleaseTree()
        {
            const string release = "abc123";
            Assert.That(ReadyPolicyBundleBinder.IsReleaseBundle(
                "releases/abc123/policies/spot.locomotion", release), Is.True);
            Assert.That(ReadyPolicyBundleBinder.IsReleaseBundle(
                "References/spot.locomotion", release), Is.False);
            Assert.That(ReadyPolicyBundleBinder.IsReleaseBundle(
                "releases/abc123/../References/spot.locomotion", release), Is.False);
            Assert.That(ReadyPolicyBundleBinder.IsReleaseBundle(
                "releases/old/policies/spot.locomotion", release), Is.False);
        }

        [Test]
        public void LiveSpotReferenceImportsResolvedIsaacActuators()
        {
            const string jsonPath =
                "Assets/_Project/Features/Deployment/Bundles/References/spot.locomotion/robot.definition.json";
            const string assetPath =
                "Assets/_Project/Features/Deployment/Bundles/References/spot.locomotion/robot.definition.asset";
            AssetDatabase.ImportAsset(jsonPath, ImportAssetOptions.ForceUpdate);
            RobotDefinition definition = AssetDatabase.LoadAssetAtPath<RobotDefinition>(assetPath);
            Assert.That(definition, Is.Not.Null);
            Assert.That(definition.robotId, Is.EqualTo("spot"));
            Assert.That(definition.links, Has.Length.EqualTo(17));
            Assert.That(definition.joints, Has.Length.EqualTo(12));
            RobotJointDefinition hip = definition.FindJoint("fl_hx");
            Assert.That(hip.actuatorModel, Is.EqualTo("DelayedPDActuator"));
            Assert.That(hip.stiffness, Is.EqualTo(60f));
            Assert.That(hip.damping, Is.EqualTo(1.5f));
            Assert.That(hip.forceLimit, Is.EqualTo(45f));
            Assert.That(hip.actuatorNominalDelaySteps, Is.EqualTo(2));
            RobotJointDefinition knee = definition.FindJoint("fl_kn");
            Assert.That(knee.actuatorModel, Is.EqualTo("RemotizedPDActuator"));
            Assert.That(knee.effortLimitCurve.Length, Is.GreaterThan(10));
            Assert.That(knee.EffortLimitAt(knee.defaultPosition), Is.GreaterThan(0f));
        }

        [Test]
        public void AllPolicyIndependentRobotDefinitionsImportAsSchemaSixAssets()
        {
            Assert.That(RobotDefinitionImporter.ImportAllDefinitions(), Is.GreaterThanOrEqualTo(6));
            var expected = new[]
            {
                ("spot", 17, 12),
                ("kinova_jaco2_n7s300", 15, 13),
                ("ur10e_robotiq_2f85", 16, 12),
                ("agibot_a2d", 46, 34),
                ("balance_tray_2dof", 3, 2),
            };
            foreach (var item in expected)
            {
                string path =
                    $"Assets/_Project/Features/Deployment/RobotDefinitions/{item.Item1}/robot.definition.asset";
                RobotDefinition definition = AssetDatabase.LoadAssetAtPath<RobotDefinition>(path);
                Assert.That(definition, Is.Not.Null, path);
                Assert.That(definition.robotId, Is.EqualTo(item.Item1));
                Assert.That(definition.links, Has.Length.EqualTo(item.Item2));
                Assert.That(definition.joints, Has.Length.EqualTo(item.Item3));
                Assert.That(definition.auxiliaryJoints, Is.Not.Null);
                Assert.That(definition.definitionSha256, Has.Length.EqualTo(64));
            }
            RobotDefinition jaco = AssetDatabase.LoadAssetAtPath<RobotDefinition>(
                "Assets/_Project/Features/Deployment/RobotDefinitions/kinova_jaco2_n7s300/robot.definition.asset");
            Assert.That(jaco.FindJoint("j2n7s300_joint_1").HasLimits, Is.False);
            RobotDefinition agibot = AssetDatabase.LoadAssetAtPath<RobotDefinition>(
                "Assets/_Project/Features/Deployment/RobotDefinitions/agibot_a2d/robot.definition.asset");
            Assert.That(agibot.auxiliaryJoints, Has.Length.EqualTo(4));
            Assert.That(agibot.auxiliaryJoints,
                Has.All.Matches<RobotAuxiliaryJointDefinition>(item =>
                    item.IsLoopClosure && item.jointType == "RevoluteJoint" &&
                    item.sourceJointType == "PhysicsRevoluteJoint"));
        }

        [Test]
        public void AgibotTopologySelectsOneTreeAndFourPassiveLoopProxies()
        {
            Assert.That(
                UsdArticulationPrefabBuilder.CanBuild("agibot_a2d", out string blocker),
                Is.False);
            Assert.That(blocker, Does.Contain("redistribution provenance is unverified"));
            Assert.That(blocker, Does.Not.Contain("loop adapter"));

            UsdArticulationPrefabBuilder.TopologySummary topology =
                UsdArticulationPrefabBuilder.AnalyzeTopology("agibot_a2d");
            Assert.That(topology.rootLink, Is.EqualTo("base_link"));
            Assert.That(topology.externalLoopJointNames, Is.EquivalentTo(new[]
            {
                "left_Right_RevoluteJoint",
                "left_Left_RevoluteJoint",
                "right_Right_RevoluteJoint",
                "right_Left_RevoluteJoint",
            }));
            Assert.That(topology.auxiliaryTreeJointNames, Is.EquivalentTo(new[]
            {
                "left_Right_2_Joint",
                "left_Left_2_Joint",
                "right_Right_2_Joint",
                "right_Left_2_Joint",
            }));
        }

        [Test]
        public void LiveCollisionDescriptorsAndNormalizedPrefabsStayDefinitionBound()
        {
            RobotDefinition spot = AssetDatabase.LoadAssetAtPath<RobotDefinition>(
                "Assets/_Project/Features/Deployment/RobotDefinitions/spot/robot.definition.asset");
            RobotLinkDefinition foot = System.Array.Find(
                spot.links, item => item.name == "fl_foot");
            Assert.That(foot.collisionShapes, Has.Length.EqualTo(1));
            Assert.That(foot.collisionShapes[0].shapeType, Is.EqualTo("sphere"));
            Assert.That(foot.collisionShapes[0].radius, Is.GreaterThan(0f));

            RobotDefinition ur = AssetDatabase.LoadAssetAtPath<RobotDefinition>(
                "Assets/_Project/Features/Deployment/RobotDefinitions/ur10e_robotiq_2f85/robot.definition.asset");
            RobotLinkDefinition innerFinger = System.Array.Find(
                ur.links, item => item.name == "left_inner_finger");
            Assert.That(innerFinger.collisionShapes, Has.Length.EqualTo(2));

            var prefabs = new[]
            {
                "Assets/_Project/Features/Robots/KinovaJaco2/Prefabs/KinovaJaco2_Normalized.prefab",
                "Assets/_Project/Features/Robots/UR10e/Prefabs/UR10eRobotiq2F85_Normalized.prefab",
                "Assets/_Project/Features/Robots/Spot/Prefabs/Spot_Normalized.prefab",
            };
            foreach (string path in prefabs)
            {
                GameObject prefab = AssetDatabase.LoadAssetAtPath<GameObject>(path);
                Assert.That(prefab, Is.Not.Null, path);
                OfflineRigIdentity identity = prefab.GetComponent<OfflineRigIdentity>();
                ArticulationRobotDriver driver = prefab.GetComponent<ArticulationRobotDriver>();
                Assert.That(identity, Is.Not.Null, path);
                Assert.That(driver, Is.Not.Null, path);
                Assert.That(identity.definitionSha256,
                    Is.EqualTo(identity.robotDefinition.definitionSha256), path);
                Assert.That(driver.TryValidateHierarchy(out string error), Is.True,
                    $"{path}: {error}");
            }
        }

        [Test]
        public void GoldenTraceSampleRejectsDimensionAndNonFiniteData()
        {
            GoldenTraceSample sample = ValidTraceSample();
            Assert.That(sample.Validate(2, 1, out string validError), Is.True, validError);
            sample.observations = new[] { 0f };
            Assert.That(sample.Validate(2, 1, out string dimensionError), Is.False);
            StringAssert.Contains("dimensions", dimensionError);
            sample = ValidTraceSample();
            sample.root_state.linear_velocity[1] = float.NaN;
            Assert.That(sample.Validate(2, 1, out string finiteError), Is.False);
            StringAssert.Contains("non-finite", finiteError);
            sample = ValidTraceSample();
            sample.loop_closures = new[]
            {
                new LoopClosureTraceTelemetry
                {
                    constraint_id = "passive_joint", position_unit = "radian",
                    velocity_unit = "radian_per_second", proxy_mass_kg = 0.1f,
                    healthy = true,
                },
            };
            Assert.That(sample.Validate(2, 1, out string loopError), Is.True, loopError);
            sample.loop_closures[0].velocity_unit = null;
            Assert.That(sample.Validate(2, 1, out string unitError), Is.False);
            StringAssert.Contains("loop-closure", unitError);
        }

        [Test]
        public void ScenarioRecordRejectsNonFiniteTaskEvidence()
        {
            var record = new StationScenarioRecord
            {
                policy_id = "ball_catch.throw",
                normalized_task_score = 0.8f,
                task_metrics = new StationTaskMetrics { catch_rate = 1f },
            };
            Assert.That(record.Validate(out string validError), Is.True, validError);
            record.task_metrics.retained_grasp_rate = float.PositiveInfinity;
            Assert.That(record.Validate(out string invalidError), Is.False);
            StringAssert.Contains("non-finite", invalidError);
        }

        static GoldenTraceSample ValidTraceSample()
        {
            BodyStateTelemetry body = new()
            {
                position = new float[3],
                orientation_xyzw = new[] { 0f, 0f, 0f, 1f },
                linear_velocity = new float[3],
                angular_velocity = new float[3],
            };
            return new GoldenTraceSample
            {
                station_id = "test",
                policy_id = "test.policy",
                reset_state = new StationTelemetrySnapshot(),
                observations = new float[2],
                raw_actions = new float[1],
                processed_actions = new float[1],
                joint_state = new JointStateTelemetry(),
                root_state = body,
                object_state = System.Array.Empty<ObjectStateTelemetry>(),
                contacts = System.Array.Empty<ContactTraceTelemetry>(),
                assist = System.Array.Empty<AssistTraceTelemetry>(),
            };
        }

        static PolicyContract ValidContract()
        {
            return new PolicyContract
            {
                schema_version = 2,
                policy_id = "test.policy",
                source_task_id = "Test-v0",
                station_id = "test",
                robot_id = "robot",
                adapter_id = "adapter",
                checkpoint_sha256 = Sha,
                config_sha256 = Sha,
                model_sha256 = Sha,
                normalization_embedded = true,
                onnx = new PolicyOnnxContract
                {
                    sha256 = Sha,
                    opset = 15,
                    input_name = "obs",
                    output_name = "actions",
                    input_shape = new[] { 1, 2 },
                    output_shape = new[] { 1, 1 },
                    inputs = new[]
                    {
                        new PolicyTensorInfo { name = "obs", shape = new[] { 1, 2 } },
                    },
                    outputs = new[]
                    {
                        new PolicyTensorInfo { name = "actions", shape = new[] { 1, 1 } },
                    },
                },
                observations = new[] { new PolicyTerm { name = "obs", shape = new[] { 2 } } },
                actions = new[]
                {
                    new PolicyTerm
                    {
                        name = "action", shape = new[] { 1 }, target_type = "joint_position"
                    }
                },
                timing = new PolicyTiming
                {
                    source_physics_hz = 120,
                    deployment_physics_hz = 120,
                    policy_hz = 60,
                    source_sim_dt = 1f / 120f,
                    deployment_sim_dt = 1f / 120f,
                    policy_dt = 1f / 60f,
                    deployment_decimation = 2,
                },
                recurrent_state = System.Array.Empty<PolicyTensorInfo>(),
                evaluation = new PolicyEvaluation
                {
                    seeded_scenarios = 100,
                    task_thresholds = new[]
                    {
                        new PolicyMetricThreshold
                        {
                            metric = "score", comparison = "min", value = 0.8f,
                        },
                    },
                },
            };
        }
    }
}
