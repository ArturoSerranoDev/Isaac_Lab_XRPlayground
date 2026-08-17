namespace XRPlayground.Deployment
{
    public interface IStationAdapter
    {
        string AdapterId { get; }
        int ObservationDimension { get; }
        int ActionDimension { get; }
        void BindContract(PolicyContract contract);
        void ResetStation(int seed);
        void BuildObservation(float[] destination);
        void ApplyAction(float[] action, float policyDeltaTime);
        bool TryHandleCommand(string command, string payloadJson, out string error);
        StationTelemetrySnapshot CaptureTelemetry();
        float TaskScore { get; }
        bool IsHealthy { get; }
        bool HasJointLimitViolation { get; }
        void BeginEvaluationScenario(int seed);
        void SampleEvaluationScenario(float policyDeltaTime);
        StationEvaluationSummary CompleteEvaluationScenario();
    }
}
