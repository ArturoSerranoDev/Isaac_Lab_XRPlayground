using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.IO;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using UnityEngine;

namespace XRPlayground.Deployment
{
    [DisallowMultipleComponent]
    public sealed class LengthPrefixedJsonClient : MonoBehaviour
    {
        public string host = "127.0.0.1";
        public int port = 9090;
        public bool autoConnect = true;
        public float reconnectSeconds = 2f;
        public int maxMessageBytes = 1 << 20;

        readonly ConcurrentQueue<string> _incoming = new();
        readonly ConcurrentQueue<string> _outgoing = new();
        CancellationTokenSource _cancellation;
        Task _task;
        volatile bool _connected;

        public bool IsConnected => _connected;
        public event Action<string> MessageReceived;

        void OnEnable()
        {
            if (autoConnect)
                Connect();
        }

        void OnDisable() => Disconnect();

        void Update()
        {
            while (_incoming.TryDequeue(out string message))
                MessageReceived?.Invoke(message);
        }

        public void Connect()
        {
            if (_task != null && !_task.IsCompleted)
                return;
            _cancellation = new CancellationTokenSource();
            _task = Task.Run(() => Run(_cancellation.Token));
        }

        public void Disconnect()
        {
            _cancellation?.Cancel();
            _cancellation = null;
            _connected = false;
        }

        public void PublishJson(string json)
        {
            if (!string.IsNullOrWhiteSpace(json))
                _outgoing.Enqueue(json);
        }

        async Task Run(CancellationToken token)
        {
            while (!token.IsCancellationRequested)
            {
                try
                {
                    using var client = new TcpClient();
                    await client.ConnectAsync(host, port);
                    _connected = true;
                    using NetworkStream stream = client.GetStream();
                    var buffer = new byte[65536];
                    var assembled = new List<byte>(4096);
                    while (!token.IsCancellationRequested && client.Connected)
                    {
                        while (_outgoing.TryDequeue(out string message))
                        {
                            byte[] payload = Encoding.UTF8.GetBytes(message);
                            if (payload.Length <= 0 || payload.Length > maxMessageBytes)
                                throw new InvalidDataException($"Invalid outgoing frame size {payload.Length}");
                            byte[] header = BitConverter.GetBytes(payload.Length);
                            if (!BitConverter.IsLittleEndian)
                                Array.Reverse(header);
                            await stream.WriteAsync(header, 0, header.Length, token);
                            await stream.WriteAsync(payload, 0, payload.Length, token);
                        }
                        if (!stream.DataAvailable)
                        {
                            await Task.Delay(5, token);
                            continue;
                        }
                        int count = await stream.ReadAsync(buffer, 0, buffer.Length, token);
                        if (count <= 0)
                            break;
                        for (int i = 0; i < count; i++)
                            assembled.Add(buffer[i]);
                        while (TryPop(assembled, maxMessageBytes, out string json))
                            _incoming.Enqueue(json);
                    }
                }
                catch (OperationCanceledException)
                {
                    break;
                }
                catch (Exception exception)
                {
                    Debug.LogWarning($"[BridgeV2] {exception.Message}");
                }
                _connected = false;
                try { await Task.Delay(TimeSpan.FromSeconds(reconnectSeconds), token); }
                catch (OperationCanceledException) { break; }
            }
            _connected = false;
        }

        static bool TryPop(List<byte> buffer, int maximum, out string json)
        {
            json = null;
            if (buffer.Count < 4)
                return false;
            int length = BitConverter.ToInt32(buffer.GetRange(0, 4).ToArray(), 0);
            if (length <= 0 || length > maximum)
            {
                buffer.Clear();
                throw new InvalidDataException($"Invalid incoming frame size {length}");
            }
            if (buffer.Count < 4 + length)
                return false;
            json = Encoding.UTF8.GetString(buffer.GetRange(4, length).ToArray());
            buffer.RemoveRange(0, 4 + length);
            return true;
        }
    }
}
