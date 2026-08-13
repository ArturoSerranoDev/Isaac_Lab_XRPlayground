using System;
using System.Collections.Concurrent;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using UnityEngine;

namespace XRPlayground.ROS
{
    /// <summary>
    /// Length-prefixed JSON TCP client (matches Isaac RosTcpServer).
    /// </summary>
    public sealed class RosTcpClient : MonoBehaviour
    {
        public string host = "127.0.0.1";
        public int port = 9090;
        public bool autoConnect = true;
        public float reconnectSeconds = 2f;

        public event Action<string> MessageReceived;

        readonly ConcurrentQueue<string> _inbox = new ConcurrentQueue<string>();
        readonly ConcurrentQueue<string> _outbox = new ConcurrentQueue<string>();
        CancellationTokenSource _cts;
        Task _ioTask;
        volatile bool _connected;

        public bool IsConnected => _connected;

        void OnEnable()
        {
            if (autoConnect)
                Connect();
        }

        void OnDisable() => Disconnect();

        void Update()
        {
            while (_inbox.TryDequeue(out var json))
                MessageReceived?.Invoke(json);
        }

        public void Connect()
        {
            if (_ioTask != null && !_ioTask.IsCompleted)
                return;
            _cts = new CancellationTokenSource();
            _ioTask = Task.Run(() => IoLoop(_cts.Token));
        }

        public void Disconnect()
        {
            try { _cts?.Cancel(); } catch { /* ignore */ }
            _cts = null;
            _connected = false;
        }

        public void PublishJson(string json)
        {
            if (!string.IsNullOrEmpty(json))
                _outbox.Enqueue(json);
        }

        async Task IoLoop(CancellationToken token)
        {
            while (!token.IsCancellationRequested)
            {
                try
                {
                    using var client = new TcpClient();
                    await client.ConnectAsync(host, port);
                    _connected = true;
                    Debug.Log($"[RosTcpClient] Connected to {host}:{port}");
                    using var stream = client.GetStream();
                    var readBuf = new byte[1 << 16];
                    var assemble = new System.Collections.Generic.List<byte>(4096);

                    while (!token.IsCancellationRequested && client.Connected)
                    {
                        // flush outbox
                        while (_outbox.TryDequeue(out var msg))
                        {
                            var payload = Encoding.UTF8.GetBytes(msg);
                            var header = BitConverter.GetBytes(payload.Length);
                            if (!BitConverter.IsLittleEndian)
                                Array.Reverse(header);
                            await stream.WriteAsync(header, 0, 4, token);
                            await stream.WriteAsync(payload, 0, payload.Length, token);
                        }

                        if (stream.DataAvailable)
                        {
                            int n = await stream.ReadAsync(readBuf, 0, readBuf.Length, token);
                            if (n <= 0)
                                break;
                            for (int i = 0; i < n; i++)
                                assemble.Add(readBuf[i]);
                            while (TryPopMessage(assemble, out var json))
                                _inbox.Enqueue(json);
                        }
                        else
                        {
                            await Task.Delay(5, token);
                        }
                    }
                }
                catch (OperationCanceledException)
                {
                    break;
                }
                catch (Exception ex)
                {
                    Debug.LogWarning($"[RosTcpClient] {ex.Message}");
                }

                _connected = false;
                try { await Task.Delay(TimeSpan.FromSeconds(reconnectSeconds), token); }
                catch (OperationCanceledException) { break; }
            }

            _connected = false;
        }

        static bool TryPopMessage(System.Collections.Generic.List<byte> buf, out string json)
        {
            json = null;
            if (buf.Count < 4)
                return false;
            int len = BitConverter.ToInt32(buf.GetRange(0, 4).ToArray(), 0);
            if (buf.Count < 4 + len)
                return false;
            var payload = buf.GetRange(4, len).ToArray();
            buf.RemoveRange(0, 4 + len);
            json = Encoding.UTF8.GetString(payload);
            return true;
        }
    }
}
