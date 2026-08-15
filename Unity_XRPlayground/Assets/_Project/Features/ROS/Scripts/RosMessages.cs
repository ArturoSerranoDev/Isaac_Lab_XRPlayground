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
        /// <summary>"unity" or "isaac" — follower only applies isaac-sourced packets.</summary>
        public string source;
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
        public int target_color;
        public string target_color_name;
    }

    [Serializable]
    public class DemoRecordData
    {
        public float[] obs;
        public float[] action;
        public string[] joint_names;
        public float[] joint_positions;
        public float gripper;
        public bool grasped;
        public int frame;
    }

    [Serializable]
    public class ConveyorObjectData
    {
        public int id;
        public bool active;
        public int color;
        public string color_name;
        public bool grasped;
        public float[] position;
        public float[] orientation_xyzw;
        public float[] linear_velocity;
        public float[] angular_velocity;
    }

    [Serializable]
    public class ConveyorObjectsStateData
    {
        public ConveyorObjectData[] objects;
        public int target_color;
        public string source;
    }

    [Serializable]
    public class ConveyorSpawnData
    {
        public int color;
        public float[] position;
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
        public string task;
    }

    public static class RosTopics
    {
        public const string BallState = "/xr/ball_state";
        public const string RobotState = "/xr/robot_state";
        public const string Heartbeat = "/xr/heartbeat";
        public const string SessionCommand = "/xr/session_command";
        public const string SessionStatus = "/xr/session_status";

        public const string ConveyorRobotState = "/xr/conveyor/robot_state";
        public const string ConveyorObjectsState = "/xr/conveyor/objects_state";
        public const string ConveyorSpawn = "/xr/conveyor/spawn";

        public const string PickPlaceRobotState = "/xr/pick_place/robot_state";
        public const string PickPlaceObjectsState = "/xr/pick_place/objects_state";
        public const string PickPlaceSpawn = "/xr/pick_place/spawn";
        public const string PickPlaceDemoRecord = "/xr/pick_place/demo_record";

        public const string ModeMirror = "mirror";
        public const string ModeAwaitThrow = "await_throw";
        public const string ModeAwaitSpawn = "await_spawn";
        public const string ModeRecordDemo = "record_demo";
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

        [Serializable]
        class ConveyorSpawnEnvelope
        {
            public string topic;
            public double stamp_s;
            public string frame_id;
            public ConveyorSpawnData data;
        }

        [Serializable]
        class DemoRecordEnvelope
        {
            public string topic;
            public double stamp_s;
            public string frame_id;
            public DemoRecordData data;
        }

        [Serializable]
        class ConveyorObjectsEnvelope
        {
            public string topic;
            public double stamp_s;
            public string frame_id;
            public ConveyorObjectsStateData data;
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

        public static string SerializeConveyorSpawn(int color, float[] positionIsaac = null)
        {
            var env = new ConveyorSpawnEnvelope
            {
                topic = RosTopics.ConveyorSpawn,
                stamp_s = Time.realtimeSinceStartupAsDouble,
                frame_id = "unity",
                data = new ConveyorSpawnData { color = color, position = positionIsaac }
            };
            return JsonUtility.ToJson(env);
        }

        public static string SerializeDemoRecord(DemoRecordData data, string frameId = "unity")
        {
            var env = new DemoRecordEnvelope
            {
                topic = RosTopics.PickPlaceDemoRecord,
                stamp_s = Time.realtimeSinceStartupAsDouble,
                frame_id = frameId,
                data = data
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
                return topic == RosTopics.RobotState
                    || topic == RosTopics.ConveyorRobotState
                    || topic == RosTopics.PickPlaceRobotState;
            }
            catch
            {
                return false;
            }
        }

        public static bool TryParseBallState(string json, out BallStateData data)
        {
            data = null;
            try
            {
                var env = JsonUtility.FromJson<BallEnvelope>(json);
                if (env == null || env.data == null || env.topic != RosTopics.BallState)
                    return false;
                data = env.data;
                return true;
            }
            catch
            {
                return false;
            }
        }

        public static bool TryParseConveyorObjects(string json, out ConveyorObjectsStateData data)
        {
            data = null;
            try
            {
                var env = JsonUtility.FromJson<ConveyorObjectsEnvelope>(json);
                if (env == null || env.data == null || env.topic != RosTopics.ConveyorObjectsState)
                    return false;
                data = env.data;
                return true;
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
