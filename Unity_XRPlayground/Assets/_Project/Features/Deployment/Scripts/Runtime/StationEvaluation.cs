using System;

namespace XRPlayground.Deployment
{
    [Serializable]
    public sealed class StationTaskMetrics
    {
        public float catch_rate;
        public float retained_grasp_rate;
        public float pre_contact_assist_events;
        public float correct_sort_rate;
        public float wrong_bin_rate;
        public float reject_rate;
        public float contact_grasp_rate;
        public float lift_rate;
        public float release_rate;
        public float stable_placement_rate;
        public float full_episode_hold_rate;
        public float drop_rate;
        public float ten_second_stand_rate;
        public float command_tracking_score;
        public float foot_contact_valid_rate;
        public float fall_rate;
        public float follow_observation_dim;
        public float high_level_policy_hz;
        public float low_level_policy_hz;
        public float target_distance_error_m;
        public float heading_error_rad;

        public bool IsFinite()
        {
            float[] values =
            {
                catch_rate, retained_grasp_rate, pre_contact_assist_events,
                correct_sort_rate, wrong_bin_rate, reject_rate,
                contact_grasp_rate, lift_rate, release_rate, stable_placement_rate,
                full_episode_hold_rate, drop_rate, ten_second_stand_rate,
                command_tracking_score, foot_contact_valid_rate, fall_rate,
                follow_observation_dim, high_level_policy_hz, low_level_policy_hz,
                target_distance_error_m, heading_error_rad,
            };
            foreach (float value in values)
                if (float.IsNaN(value) || float.IsInfinity(value))
                    return false;
            return true;
        }
    }

    [Serializable]
    public sealed class StationEvaluationSummary
    {
        public float normalized_task_score;
        public StationTaskMetrics task_metrics = new();
    }

    [Serializable]
    public sealed class StationScenarioRecord
    {
        public int schema_version = 1;
        public string policy_id;
        public int seed;
        public float normalized_task_score;
        public StationTaskMetrics task_metrics = new();
        public bool has_nan;
        public bool invalid_actions;
        public bool missing_joints;
        public bool joint_limit_violations;

        public bool Validate(out string error)
        {
            if (schema_version != 1 || string.IsNullOrWhiteSpace(policy_id))
            {
                error = "scenario schema and policy_id are required";
                return false;
            }
            if (float.IsNaN(normalized_task_score) || float.IsInfinity(normalized_task_score) ||
                normalized_task_score < 0f || normalized_task_score > 1f ||
                task_metrics == null || !task_metrics.IsFinite())
            {
                error = "scenario metrics contain non-finite values";
                return false;
            }
            error = null;
            return true;
        }
    }
}
