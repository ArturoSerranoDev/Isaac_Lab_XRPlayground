#if UNITY_EDITOR
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.InputSystem;
using UnityEngine.XR.Hands;
using UnityEngine.XR.Interaction.Toolkit.Attachment;
using UnityEngine.XR.Interaction.Toolkit.Inputs;
using UnityEngine.XR.Interaction.Toolkit.Interactables;
using UnityEngine.XR.Interaction.Toolkit.Interactors;
using UnityEngine.XR.Interaction.Toolkit.Interactors.Casters;
using XRPlayground.VR;

namespace XRPlayground.VR.Editor
{
    /// <summary>
    /// One-shot scene wiring for XRIT + XR Hands (no prefabs).
    /// Menu: XRPlayground / Setup Hand Tracking Scene
    /// </summary>
    public static class XRHandSceneSetup
    {
        const string ActionsPath = "Assets/_Project/Features/VR/Input/XRI Default Input Actions.inputactions";

        [MenuItem("XRPlayground/Setup Hand Tracking Scene")]
        public static void Setup()
        {
            Undo.IncrementCurrentGroup();
            Undo.SetCurrentGroupName("XR Hand Tracking Setup");

            var actionAsset = AssetDatabase.LoadAssetAtPath<InputActionAsset>(ActionsPath);
            if (actionAsset == null)
            {
                Debug.LogError($"Missing Input Actions at {ActionsPath}");
                return;
            }

            var subassets = AssetDatabase.LoadAllAssetsAtPath(ActionsPath);

            var env = FindOrCreateRoot("Environment");
            var xrRoot = FindOrCreateRoot("XR");
            var interactables = FindOrCreateRoot("Interactables");

            ReparentIfExists("Directional Light", env.transform);
            ReparentIfExists("Global Volume", env.transform);
            ReparentIfExists("Floor", env.transform);

            var mgr = GameObject.Find("XR Interaction Manager");
            if (mgr != null)
                Undo.SetTransformParent(mgr.transform, xrRoot.transform, "Parent Interaction Manager");

            var origin = GameObject.Find("XR Origin (VR)") ?? GameObject.Find("XR Origin");
            if (origin == null)
            {
                Debug.LogError("Create GameObject/XR/XR Origin (VR) first.");
                return;
            }

            origin.name = "XR Origin";
            Undo.SetTransformParent(origin.transform, xrRoot.transform, "Parent XR Origin");

            var eventSys = GameObject.Find("EventSystem") ?? GameObject.Find("XR Event System");
            if (eventSys != null)
            {
                eventSys.name = "XR Event System";
                Undo.SetTransformParent(eventSys.transform, xrRoot.transform, "Parent Event System");
            }

            // Remove orphan Main Camera (XR Origin already has one)
            var cameras = Object.FindObjectsByType<Camera>();
            foreach (var cam in cameras)
            {
                if (cam.gameObject.CompareTag("MainCamera") &&
                    (cam.transform.parent == null || cam.transform.parent.name != "Camera Offset"))
                {
                    Undo.DestroyObjectImmediate(cam.gameObject);
                }
            }

            // Remove leftover root Near-Far Interactor from earlier menu create
            var leftover = GameObject.Find("Near-Far Interactor");
            if (leftover != null && leftover.transform.parent == null)
                Undo.DestroyObjectImmediate(leftover);

            EnsureFloor(env.transform);

            var iam = origin.GetComponent<InputActionManager>();
            if (iam != null)
            {
                var soIam = new SerializedObject(iam);
                var list = soIam.FindProperty("m_ActionAssets");
                list.arraySize = 1;
                list.GetArrayElementAtIndex(0).objectReferenceValue = actionAsset;
                soIam.ApplyModifiedProperties();
            }

            var camOffset = origin.transform.Find("Camera Offset");
            if (camOffset == null)
            {
                Debug.LogError("XR Origin is missing Camera Offset.");
                return;
            }

            CreateHand(
                camOffset,
                "Left Hand",
                Handedness.Left,
                FindRef(subassets, "XRI Left Interaction/Select"),
                FindRef(subassets, "XRI Left Interaction/Select Value"),
                FindRef(subassets, "XRI Left Interaction/UI Press"));

            CreateHand(
                camOffset,
                "Right Hand",
                Handedness.Right,
                FindRef(subassets, "XRI Right Interaction/Select"),
                FindRef(subassets, "XRI Right Interaction/Select Value"),
                FindRef(subassets, "XRI Right Interaction/UI Press"));

            EnsureGrabCube(interactables.transform);

            EditorSceneManager.MarkSceneDirty(EditorSceneManager.GetActiveScene());
            Debug.Log("XRPlayground: Hand tracking scene setup complete. Save the scene.");
        }

        static GameObject FindOrCreateRoot(string name)
        {
            var existing = GameObject.Find(name);
            if (existing != null && existing.transform.parent == null)
                return existing;

            var go = new GameObject(name);
            Undo.RegisterCreatedObjectUndo(go, $"Create {name}");
            return go;
        }

        static void ReparentIfExists(string name, Transform parent)
        {
            var go = GameObject.Find(name);
            if (go == null || go.transform.parent == parent)
                return;
            Undo.SetTransformParent(go.transform, parent, $"Parent {name}");
        }

        static void EnsureFloor(Transform env)
        {
            var floor = env.Find("Floor");
            if (floor != null)
                return;

            var go = GameObject.CreatePrimitive(PrimitiveType.Plane);
            go.name = "Floor";
            go.transform.SetParent(env, false);
            go.transform.localScale = new Vector3(1.5f, 1f, 1.5f);
            Undo.RegisterCreatedObjectUndo(go, "Create Floor");
        }

        static InputActionReference FindRef(Object[] subassets, string name)
        {
            foreach (var s in subassets)
            {
                if (s is InputActionReference reference && reference.name == name)
                    return reference;
            }

            return null;
        }

        static void WireButton(SerializedProperty buttonProp, InputActionReference performed, InputActionReference value)
        {
            if (buttonProp == null)
                return;

            buttonProp.FindPropertyRelative("m_InputSourceMode").enumValueIndex = 2; // InputActionReference
            buttonProp.FindPropertyRelative("m_InputActionReferencePerformed").objectReferenceValue = performed;
            if (value != null)
                buttonProp.FindPropertyRelative("m_InputActionReferenceValue").objectReferenceValue = value;
        }

        static void CreateHand(
            Transform parent,
            string handName,
            Handedness handedness,
            InputActionReference select,
            InputActionReference selectValue,
            InputActionReference uiPress)
        {
            var existing = parent.Find(handName);
            if (existing != null)
                Undo.DestroyObjectImmediate(existing.gameObject);

            var hand = new GameObject(handName);
            Undo.RegisterCreatedObjectUndo(hand, $"Create {handName}");
            hand.transform.SetParent(parent, false);

            var events = hand.AddComponent<XRHandTrackingEvents>();
            var soEvents = new SerializedObject(events);
            soEvents.FindProperty("m_Handedness").enumValueIndex = (int)handedness;
            soEvents.ApplyModifiedProperties();

            hand.AddComponent<XRHandRootPoseDriver>();
            hand.AddComponent<XRHandJointSphereVisual>();

            // Remove legacy single-palm visual if present from older setups
            var legacyPalm = hand.transform.Find("Palm Visual");
            if (legacyPalm != null)
                Undo.DestroyObjectImmediate(legacyPalm.gameObject);

            var interactorGo = new GameObject("Near-Far Interactor");
            Undo.RegisterCreatedObjectUndo(interactorGo, "Create Near-Far Interactor");
            interactorGo.transform.SetParent(hand.transform, false);

            var nf = interactorGo.AddComponent<NearFarInteractor>();
            interactorGo.AddComponent<InteractionAttachController>();
            interactorGo.AddComponent<SphereInteractionCaster>();
            interactorGo.AddComponent<CurveInteractionCaster>();

            var so = new SerializedObject(nf);
            so.FindProperty("m_Handedness").enumValueIndex = handedness == Handedness.Left ? 1 : 2;
            WireButton(so.FindProperty("m_SelectInput"), select, selectValue);
            WireButton(so.FindProperty("m_UIPressInput"), uiPress, null);
            so.ApplyModifiedProperties();
        }

        static void EnsureGrabCube(Transform parent)
        {
            var existing = parent.Find("Grab Cube");
            if (existing != null)
                return;

            var cube = GameObject.CreatePrimitive(PrimitiveType.Cube);
            cube.name = "Grab Cube";
            cube.transform.SetParent(parent, false);
            cube.transform.position = new Vector3(0f, 1.1f, 0.6f);
            cube.transform.localScale = Vector3.one * 0.12f;
            cube.AddComponent<Rigidbody>();
            cube.AddComponent<XRGrabInteractable>();
            Undo.RegisterCreatedObjectUndo(cube, "Create Grab Cube");
        }
    }
}
#endif
