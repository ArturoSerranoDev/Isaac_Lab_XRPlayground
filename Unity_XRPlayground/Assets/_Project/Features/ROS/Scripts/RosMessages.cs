using System;
using UnityEngine;

namespace XRPlayground.ROS
{
    [Serializable]
    public class BallStateData
    {
        public float[] position;
        public float[] orientation_xyzw;
        public float[] linear_velocity;
        public float[] angular_velocity;
        public bool grasped;
        public bool throw_event;
    }

    [Serializable]
    public class PoseData
    {
        public float[] position;
        public float[] orientation_xyzw;
    }

    [Serializable]
    public class LinkPoseData
    {
        public string name;
        public float[] position;
        public float[] orientation_xyzw;
    }

    [Serializable]
    public class RobotStateData
    {
        public string[] joint_names;
        public float[] joint_positions;
        public PoseData ee;
        public LinkPoseData[] links;
    }

    [Serializable]
    public class HeartbeatData
    {
        public string role;
        public float sim_time;
        public float unity_time;
    }

    [Serializable]
    public class SessionCommandData
    {
        public string mode;
    }

    [Serializable]
    public class SessionStatusData
    {
        public string mode;
        public string phase;
        public bool policy_loaded;
        public int clients;
    }

    public static class RosTopics
    {
        public const string BallState = "/xr/ball_state";
        public const string RobotState = "/xr/robot_state";
        public const string Heartbeat = "/xr/heartbeat";
        public const string SessionCommand = "/xr/session_command";
        public const string SessionStatus = "/xr/session_status";

        public const string ModeMirror = "mirror";
        public const string ModeAwaitThrow = "await_throw";
    }

    public static class RosJson
    {
        [Serializable]
        class RobotEnvelope
        {
            public string topic;
            public double stamp_s;
            public string frame_id;
            public RobotStateData data;
        }

        [Serializable]
        class BallEnvelope
        {
            public string topic;
            public double stamp_s;
            public string frame_id;
            public BallStateData data;
        }

        [Serializable]
        class HeartbeatEnvelope
        {
            public string topic;
            public double stamp_s;
            public string frame_id;
            public HeartbeatData data;
        }

        [Serializable]
        class SessionCommandEnvelope
        {
            public string topic;
            public double stamp_s;
            public string frame_id;
            public SessionCommandData data;
        }

        [Serializable]
        class SessionStatusEnvelope
        {
            public string topic;
            public double stamp_s;
            public string frame_id;
            public SessionStatusData data;
        }

        public static string SerializeBall(string topic, BallStateData data, string frameId = "unity")
        {
            var env = new BallEnvelope
            {
                topic = topic,
                stamp_s = Time.realtimeSinceStartupAsDouble,
                frame_id = frameId,
                data = data
            };
            return JsonUtility.ToJson(env);
        }

        public static string SerializeHeartbeat(string role)
        {
            var env = new HeartbeatEnvelope
            {
                topic = RosTopics.Heartbeat,
                stamp_s = Time.realtimeSinceStartupAsDouble,
                frame_id = "unity",
                data = new HeartbeatData { role = role, unity_time = Time.time }
            };
            return JsonUtility.ToJson(env);
        }

        public static string SerializeSessionCommand(string mode)
        {
            var env = new SessionCommandEnvelope
            {
                topic = RosTopics.SessionCommand,
                stamp_s = Time.realtimeSinceStartupAsDouble,
                frame_id = "unity",
                data = new SessionCommandData { mode = mode }
            };
            return JsonUtility.ToJson(env);
        }

        public static bool TryParseRobotState(string json, out RobotStateData data, out string topic)
        {
            data = null;
            topic = null;
            try
            {
                var env = JsonUtility.FromJson<RobotEnvelope>(json);
                if (env == null || env.data == null)
                    return false;
                topic = env.topic;
                data = env.data;
                return topic == RosTopics.RobotState;
            }
            catch
            {
                return false;
            }
        }

        public static bool TryParseSessionStatus(string json, out SessionStatusData data)
        {
            data = null;
            try
            {
                var env = JsonUtility.FromJson<SessionStatusEnvelope>(json);
                if (env == null || env.data == null || env.topic != RosTopics.SessionStatus)
                    return false;
                data = env.data;
                return true;
            }
            catch
            {
                return false;
            }
        }

        public static bool TryParseTopic(string json, out string topic)
        {
            topic = null;
            const string key = "\"topic\":\"";
            int i = json.IndexOf(key, StringComparison.Ordinal);
            if (i < 0)
                return false;
            i += key.Length;
            int j = json.IndexOf('"', i);
            if (j < 0)
                return false;
            topic = json.Substring(i, j - i);
            return true;
        }
    }
}
