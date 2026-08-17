using System;
using System.Collections.Generic;
using UnityEngine;

namespace XRPlayground.Deployment
{
    [DisallowMultipleComponent]
    public sealed class ContactGripAssist : MonoBehaviour
    {
        public AssistProfile profile;
        public ContactSensor[] contactSensors;
        public Rigidbody gripperBody;
        public ArticulationBody gripperArticulation;
        public bool unassistedDiagnostic;

        FixedJoint _constraint;
        Rigidbody _held;
        bool _closing;
        bool _brokenWhileClosing;
        float _lastForce;
        float _lastTorque;
        int _activationCount;
        int _preContactActivationCount;

        public event Action<AssistTelemetry> Telemetry;
        public Rigidbody HeldBody => _held;
        public bool ConstraintActive => _constraint != null;
        public int ActivationCount => _activationCount;
        public int PreContactActivationCount => _preContactActivationCount;

        public void SetClosingCommand(bool closing)
        {
            _closing = closing;
            if (!closing)
            {
                _brokenWhileClosing = false;
                Release(false);
            }
        }

        void FixedUpdate()
        {
            if (_constraint != null)
            {
                _lastForce = _constraint.currentForce.magnitude;
                _lastTorque = _constraint.currentTorque.magnitude;
                Telemetry?.Invoke(new AssistTelemetry(
                    profile?.profileId, true, _lastForce, _lastTorque, false));
            }
            if (_held != null && _constraint == null)
            {
                Telemetry?.Invoke(new AssistTelemetry(
                    profile?.profileId, false, _lastForce, _lastTorque, true));
                _held = null;
                _brokenWhileClosing = true;
            }
            if (unassistedDiagnostic || _brokenWhileClosing || profile == null || !profile.enabledInProduction ||
                profile.kind != AssistKind.ContactGrip || !_closing || _constraint != null)
                return;
            Rigidbody candidate = CommonContact();
            if (candidate == null)
                return; // no pre-contact attraction is ever applied.
            if (!HasRequiredContact(candidate))
                _preContactActivationCount++;
            _activationCount++;
            _held = candidate;
            _constraint = candidate.gameObject.AddComponent<FixedJoint>();
            if (gripperArticulation != null)
                _constraint.connectedArticulationBody = gripperArticulation;
            else
                _constraint.connectedBody = gripperBody;
            _constraint.breakForce = profile.breakForce;
            _constraint.breakTorque = profile.breakTorque;
            _constraint.enableCollision = true;
            _lastForce = 0f;
            _lastTorque = 0f;
            Telemetry?.Invoke(new AssistTelemetry(profile.profileId, true, 0f, 0f, false));
        }

        Rigidbody CommonContact()
        {
            if (contactSensors == null || contactSensors.Length < profile.requiredDistinctContacts)
                return null;
            var counts = new Dictionary<Rigidbody, int>();
            foreach (ContactSensor sensor in contactSensors)
            {
                if (sensor == null)
                    continue;
                foreach (Rigidbody body in sensor.Contacts)
                    counts[body] = counts.TryGetValue(body, out int value) ? value + 1 : 1;
            }
            foreach (var pair in counts)
                if (pair.Value >= profile.requiredDistinctContacts)
                    return pair.Key;
            return null;
        }

        public bool HasRequiredContact(Rigidbody candidate)
        {
            if (candidate == null || profile == null || contactSensors == null)
                return false;
            int count = 0;
            foreach (ContactSensor sensor in contactSensors)
                if (sensor != null && sensor.Contains(candidate))
                    count++;
            return count >= profile.requiredDistinctContacts;
        }

        public void ResetTelemetryCounters()
        {
            _activationCount = 0;
            _preContactActivationCount = 0;
            _lastForce = 0f;
            _lastTorque = 0f;
        }

        public void Release(bool broke)
        {
            if (_constraint != null)
                Destroy(_constraint);
            if (_held != null && profile != null)
                Telemetry?.Invoke(new AssistTelemetry(
                    profile.profileId, false, _lastForce, _lastTorque, broke));
            _constraint = null;
            _held = null;
            _lastForce = 0f;
            _lastTorque = 0f;
        }
    }
}
