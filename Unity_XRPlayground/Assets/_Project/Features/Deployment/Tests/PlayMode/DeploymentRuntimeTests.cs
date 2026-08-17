using System.Collections;
using System.Reflection;
using NUnit.Framework;
using UnityEngine;
using UnityEngine.TestTools;

namespace XRPlayground.Deployment.Tests
{
    public sealed class DeploymentRuntimeTests
    {
        [UnityTest]
        public IEnumerator BridgeRejectsOutOfOrderStateAndBecomesStale()
        {
            var owner = new GameObject("BridgeTest");
            BridgeV2Client bridge = owner.AddComponent<BridgeV2Client>();
            bridge.stationId = "ball_catch";
            bridge.staleAfterSeconds = 0.01f;
            bridge.disconnectAfterSeconds = 1f;
            int received = 0;
            bridge.RobotStateReceived += (_, _) => received++;
            MethodInfo receive = typeof(BridgeV2Client).GetMethod(
                "OnJson", BindingFlags.Instance | BindingFlags.NonPublic);
            Assert.That(receive, Is.Not.Null);
            receive.Invoke(bridge, new object[]
            {
                "{\"schema_version\":2,\"station_id\":\"ball_catch\",\"sequence\":2," +
                "\"sim_time_s\":1.0,\"frame_id\":\"isaac_env\",\"message_type\":\"robot_state\"," +
                "\"payload\":{\"joints\":[],\"links\":[]}}"
            });
            receive.Invoke(bridge, new object[]
            {
                "{\"schema_version\":2,\"station_id\":\"ball_catch\",\"sequence\":1," +
                "\"sim_time_s\":0.5,\"frame_id\":\"isaac_env\",\"message_type\":\"robot_state\"," +
                "\"payload\":{\"joints\":[],\"links\":[]}}"
            });
            Assert.That(received, Is.EqualTo(1));
            Assert.That(bridge.IsStale, Is.False);
            yield return new WaitForSecondsRealtime(0.03f);
            Assert.That(bridge.IsStale, Is.True);
            Object.Destroy(owner);
        }

        [UnityTest]
        public IEnumerator ModeSwitchingKeepsOfflineFailClosedWithoutBundleAndAdapter()
        {
            var owner = new GameObject("StationTest");
            StationRuntime runtime = owner.AddComponent<StationRuntime>();
            runtime.stationId = "ball_catch";
            var mirrorObject = new GameObject("Mirror");
            mirrorObject.transform.SetParent(owner.transform);
            runtime.mirrorRig = mirrorObject.AddComponent<MirrorRig>();
            runtime.offlineRig = new GameObject("Offline");
            runtime.offlineRig.transform.SetParent(owner.transform);
            Assert.That(runtime.SetMode(StationMode.Mirror, out string mirrorError),
                Is.True, mirrorError);
            Assert.That(runtime.mirrorRig.gameObject.activeSelf, Is.True);
            Assert.That(runtime.offlineRig.activeSelf, Is.False);
            Assert.That(runtime.SetMode(StationMode.OfflinePolicy, out string offlineError), Is.False);
            Assert.That(offlineError, Does.Contain("adapter"));
            Assert.That(runtime.Mode, Is.EqualTo(StationMode.Mirror));
            yield return null;
            Object.Destroy(owner);
        }

        [UnityTest]
        public IEnumerator GripAssistRequiresAllContactsAndReleasesOnOpening()
        {
            var owner = new GameObject("GripAssistTest");
            ContactGripAssist assist = owner.AddComponent<ContactGripAssist>();
            AssistProfile profile = ScriptableObject.CreateInstance<AssistProfile>();
            profile.profileId = "three_fingertip_grip";
            profile.kind = AssistKind.ContactGrip;
            profile.requiredDistinctContacts = 3;
            profile.breakForce = 150f;
            profile.breakTorque = 50f;
            profile.enabledInProduction = true;
            assist.profile = profile;
            assist.contactSensors = new ContactSensor[3];
            for (int i = 0; i < assist.contactSensors.Length; i++)
            {
                var finger = new GameObject($"Finger{i}");
                finger.transform.SetParent(owner.transform);
                finger.transform.position = new Vector3(0.55f, (i - 1) * 0.2f, 0f);
                Rigidbody body = finger.AddComponent<Rigidbody>();
                body.isKinematic = true;
                finger.AddComponent<BoxCollider>().size = Vector3.one * 0.15f;
                assist.contactSensors[i] = finger.AddComponent<ContactSensor>();
                if (i == 0)
                    assist.gripperBody = body;
            }
            var candidate = new GameObject("Candidate");
            Rigidbody candidateBody = candidate.AddComponent<Rigidbody>();
            candidateBody.useGravity = false;
            candidate.AddComponent<BoxCollider>().size = Vector3.one;
            candidate.transform.position = Vector3.right * 10f;

            assist.SetClosingCommand(true);
            yield return new WaitForFixedUpdate();
            Assert.That(candidate.GetComponent<FixedJoint>(), Is.Null,
                "assist must exert no pre-contact attraction");
            Assert.That(candidate.transform.position.x, Is.EqualTo(10f).Within(0.01f));

            candidateBody.position = Vector3.zero;
            candidateBody.linearVelocity = Vector3.zero;
            Physics.SyncTransforms();
            yield return new WaitForFixedUpdate();
            yield return new WaitForFixedUpdate();
            yield return new WaitForFixedUpdate();
            yield return new WaitForFixedUpdate();
            Assert.That(candidate.GetComponent<FixedJoint>(), Is.Not.Null,
                "all three fingertip contacts should permit the bounded constraint");

            assist.SetClosingCommand(false);
            yield return null;
            Assert.That(candidate.GetComponent<FixedJoint>(), Is.Null,
                "opening command must release immediately");
            Object.Destroy(candidate);
            Object.Destroy(owner);
            Object.Destroy(profile);
        }

        [UnityTest]
        public IEnumerator BalanceAdapterCapturesContractOrderedPhysicalState()
        {
            var owner = new GameObject("BalanceTelemetryTest");
            BalanceBotStationAdapter adapter = owner.AddComponent<BalanceBotStationAdapter>();
            adapter.environmentAnchor = owner.transform;
            adapter.tray = new GameObject("Tray").transform;
            adapter.tray.SetParent(owner.transform);
            adapter.balls = new Rigidbody[2];
            for (int i = 0; i < adapter.balls.Length; i++)
            {
                var ball = GameObject.CreatePrimitive(PrimitiveType.Sphere);
                ball.name = $"Ball{i}";
                ball.transform.SetParent(owner.transform);
                adapter.balls[i] = ball.AddComponent<Rigidbody>();
                adapter.balls[i].useGravity = false;
            }

            adapter.ResetStation(42);
            yield return new WaitForFixedUpdate();
            StationTelemetrySnapshot telemetry = adapter.CaptureTelemetry();

            Assert.That(telemetry.joint_state.names,
                Is.EqualTo(new[] { "roll_joint", "pitch_joint" }));
            Assert.That(telemetry.joint_state.position.Length, Is.EqualTo(2));
            Assert.That(telemetry.joint_state.position_units,
                Is.EqualTo(new[] { "radian", "radian" }));
            Assert.That(telemetry.root_state.position.Length, Is.EqualTo(3));
            Assert.That(telemetry.root_state.orientation_xyzw.Length, Is.EqualTo(4));
            Assert.That(telemetry.object_state.Length, Is.EqualTo(2));
            Assert.That(telemetry.object_state[0].id, Is.EqualTo("ball_0"));
            Assert.That(telemetry.object_state[1].id, Is.EqualTo("ball_1"));
            Assert.That(telemetry.object_state[0].active, Is.True);

            Object.Destroy(owner);
        }

        [UnityTest]
        public IEnumerator ContactSensorDetectsStaticGroundForSpotFeet()
        {
            var ground = GameObject.CreatePrimitive(PrimitiveType.Cube);
            ground.name = "StaticGround";
            ground.transform.position = new Vector3(0f, -0.05f, 0f);
            ground.transform.localScale = new Vector3(2f, 0.1f, 2f);

            var foot = GameObject.CreatePrimitive(PrimitiveType.Cube);
            foot.name = "SpotFootSensor";
            foot.transform.position = new Vector3(0f, 0.25f, 0f);
            foot.transform.localScale = Vector3.one * 0.1f;
            Rigidbody body = foot.AddComponent<Rigidbody>();
            body.constraints = RigidbodyConstraints.FreezeRotation;
            ContactSensor sensor = foot.AddComponent<ContactSensor>();

            for (int i = 0; i < 20 && !sensor.IsTouching; i++)
                yield return new WaitForFixedUpdate();

            Assert.That(sensor.IsTouching, Is.True,
                "foot contact must include colliders without a Rigidbody");
            Object.Destroy(foot);
            Object.Destroy(ground);
        }

        [UnityTest]
        public IEnumerator ContactSensorOnArticulationChildReceivesCollisionCallbacks()
        {
            var ground = GameObject.CreatePrimitive(PrimitiveType.Cube);
            ground.name = "ArticulationContactGround";
            ground.transform.position = new Vector3(0f, -0.05f, 0f);
            ground.transform.localScale = new Vector3(2f, 0.1f, 2f);

            var rootObject = new GameObject("contact_root");
            rootObject.transform.position = new Vector3(0f, 0.5f, 0f);
            rootObject.AddComponent<ArticulationBody>();
            var footObject = new GameObject("contact_foot");
            footObject.transform.SetParent(rootObject.transform, false);
            footObject.transform.localPosition = new Vector3(0f, -0.2f, 0f);
            ArticulationBody footBody = footObject.AddComponent<ArticulationBody>();
            footBody.jointType = ArticulationJointType.FixedJoint;
            BoxCollider footCollider = footObject.AddComponent<BoxCollider>();
            footCollider.size = Vector3.one * 0.1f;
            ContactSensor sensor = footObject.AddComponent<ContactSensor>();

            for (int i = 0; i < 60 && !sensor.IsTouching; i++)
                yield return new WaitForFixedUpdate();

            Assert.That(sensor.IsTouching, Is.True,
                "contact sensors on imported articulation links must receive collision callbacks");
            Object.Destroy(rootObject);
            Object.Destroy(ground);
        }

        [UnityTest]
        public IEnumerator RigidbodyProxyClosesAnArticulationCycleWithoutAnchorDrift()
        {
            var wrapper = new GameObject("LoopClosureTest");
            var rootObject = new GameObject("root_link");
            rootObject.transform.SetParent(wrapper.transform, false);
            ArticulationBody root = rootObject.AddComponent<ArticulationBody>();
            root.immovable = true;

            var childObject = new GameObject("child_link");
            childObject.transform.SetParent(rootObject.transform, false);
            childObject.transform.localPosition = Vector3.right;
            ArticulationBody child = childObject.AddComponent<ArticulationBody>();
            child.jointType = ArticulationJointType.RevoluteJoint;
            child.twistLock = ArticulationDofLock.FreeMotion;
            child.anchorPosition = Vector3.left;
            child.parentAnchorPosition = Vector3.zero;
            Quaternion zAxis = Quaternion.FromToRotation(Vector3.right, Vector3.forward);
            child.anchorRotation = zAxis;
            child.parentAnchorRotation = zAxis;
            childObject.AddComponent<BoxCollider>().size = new Vector3(2f, 0.1f, 0.1f);

            var proxyObject = new GameObject("secondary_revolute_proxy");
            ArticulationLoopClosureProxy proxy =
                proxyObject.AddComponent<ArticulationLoopClosureProxy>();
            proxy.proxyMass = 0.1f;
            proxy.maximumHealthyAnchorErrorMeters = 0.002f;
            proxy.Configure(
                "secondary_revolute",
                root,
                child,
                Vector3.up,
                Quaternion.identity,
                new Vector3(-1f, 1f, 0f),
                Quaternion.identity,
                Vector3.forward);
            Assert.That(proxy.TryValidate(out string initialError), Is.True, initialError);

            child.AddForce(Vector3.up * 20f, ForceMode.Force);
            for (int i = 0; i < 30; i++)
                yield return new WaitForFixedUpdate();

            Assert.That(proxy.TryValidate(out string settledError), Is.True, settledError);
            Assert.That(proxy.AnchorErrorMeters, Is.LessThan(0.002f));
            Assert.That(float.IsFinite(child.transform.position.x), Is.True);
            Assert.That(float.IsFinite(child.transform.position.y), Is.True);
            Object.Destroy(proxyObject);
            Object.Destroy(wrapper);
        }

        [UnityTest]
        public IEnumerator LoopBindingSpawnsBeforeDriverAndRoutesExternalJointTelemetry()
        {
            var wrapper = new GameObject("LoopBindingTest");
            wrapper.SetActive(false);
            var rootObject = new GameObject("root_link");
            rootObject.transform.SetParent(wrapper.transform, false);
            ArticulationBody root = rootObject.AddComponent<ArticulationBody>();
            root.immovable = true;
            var childObject = new GameObject("child_link");
            childObject.transform.SetParent(rootObject.transform, false);
            childObject.transform.localPosition = Vector3.right;
            ArticulationBody child = childObject.AddComponent<ArticulationBody>();

            RobotDefinition definition = ScriptableObject.CreateInstance<RobotDefinition>();
            definition.robotId = "closed_chain_fixture";
            definition.links = new[]
            {
                new RobotLinkDefinition
                {
                    name = "root_link", mass = 1f, inertiaTensor = Vector3.one,
                    collisionShapeCount = 0,
                },
                new RobotLinkDefinition
                {
                    name = "child_link", mass = 1f, inertiaTensor = Vector3.one,
                    collisionShapeCount = 0,
                },
            };
            definition.joints = new[]
            {
                RevoluteFixtureJoint(
                    "primary_revolute", Vector3.zero, Vector3.left),
                RevoluteFixtureJoint(
                    "secondary_revolute", Vector3.up, new Vector3(-1f, 1f, 0f)),
            };

            ArticulationLoopClosureBinding binding =
                wrapper.AddComponent<ArticulationLoopClosureBinding>();
            binding.entries = new[]
            {
                new ArticulationLoopClosureEntry
                {
                    constraintName = "secondary_revolute",
                    parentBody = root,
                    childBody = child,
                    parentAnchorPosition = Vector3.up,
                    childAnchorPosition = new Vector3(-1f, 1f, 0f),
                    axis = Vector3.forward,
                    proxyMass = 0.1f,
                },
            };
            ArticulationRobotDriver driver = wrapper.AddComponent<ArticulationRobotDriver>();
            driver.definition = definition;
            driver.articulationRoot = root;
            wrapper.SetActive(true);
            yield return new WaitForFixedUpdate();

            Assert.That(binding.Proxies, Has.Count.EqualTo(1));
            ArticulationLoopClosureProxy proxy =
                binding.Proxies["secondary_revolute"];
            Assert.That(proxy.transform.parent, Is.Null,
                "Rigidbody proxies must remain outside the articulation hierarchy");
            Assert.That(driver.IsBound, Is.True);
            Assert.That(driver.TryValidateHealth(out string initialHealthError),
                Is.True, initialHealthError);
            Assert.That(driver.TryRead(
                "secondary_revolute", out float position, out float velocity), Is.True);
            Assert.That(float.IsFinite(position), Is.True);
            Assert.That(float.IsFinite(velocity), Is.True);

            child.AddForce(Vector3.up * 20f, ForceMode.Force);
            for (int i = 0; i < 30; i++)
                yield return new WaitForFixedUpdate();
            Assert.That(binding.TryValidate(out string error), Is.True, error);
            Assert.That(driver.TryValidateHealth(out string healthError),
                Is.True, healthError);

            Object.Destroy(wrapper);
            Object.Destroy(definition);
        }

        static RobotJointDefinition RevoluteFixtureJoint(
            string name, Vector3 parentAnchor, Vector3 childAnchor) =>
            new RobotJointDefinition
            {
                name = name,
                parentLink = "root_link",
                childLink = "child_link",
                jointType = ArticulationJointType.RevoluteJoint,
                positionUnit = "radian",
                limitMode = "continuous",
                axis = Vector3.forward,
                parentAnchorPosition = parentAnchor,
                anchorPosition = childAnchor,
                lowerLimit = -Mathf.PI,
                upperLimit = Mathf.PI,
                defaultPosition = 0f,
                stiffness = 0f,
                damping = 0f,
                forceLimit = 100f,
                maxVelocity = 10f,
                actuatorModel = "PassiveFixture",
                actuatorDelayMinSteps = 0,
                actuatorDelayMaxSteps = 0,
                actuatorNominalDelaySteps = 0,
            };
    }
}
