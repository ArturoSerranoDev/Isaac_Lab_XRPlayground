using System.Collections.Generic;
using UnityEngine;

namespace XRPlayground.Deployment
{
    [DisallowMultipleComponent]
    public sealed class ContactSensor : MonoBehaviour
    {
        readonly Dictionary<Rigidbody, int> _contacts = new();
        readonly HashSet<Collider> _colliders = new();
        public IReadOnlyCollection<Rigidbody> Contacts => _contacts.Keys;
        public IReadOnlyCollection<Collider> Colliders => _colliders;
        public bool IsTouching => _colliders.Count > 0;
        public bool Contains(Rigidbody body) => body != null && _contacts.ContainsKey(body);

        void OnCollisionEnter(Collision collision)
        {
            if (collision.collider != null)
                _colliders.Add(collision.collider);
            if (collision.rigidbody != null)
                _contacts[collision.rigidbody] =
                    _contacts.TryGetValue(collision.rigidbody, out int count) ? count + 1 : 1;
        }

        void OnCollisionExit(Collision collision)
        {
            if (collision.collider != null)
                _colliders.Remove(collision.collider);
            if (collision.rigidbody != null)
            {
                if (_contacts.TryGetValue(collision.rigidbody, out int count) && count > 1)
                    _contacts[collision.rigidbody] = count - 1;
                else
                    _contacts.Remove(collision.rigidbody);
            }
        }

        void OnDisable()
        {
            _contacts.Clear();
            _colliders.Clear();
        }
    }
}
