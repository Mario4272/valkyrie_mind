# Self-Monitoring System with Diagnostics Interface

from datetime import datetime
from enum import Enum

class MetricType(Enum):
    PERFORMANCE = "performance"
    RESOURCE = "resource"
    ERROR_RATE = "error_rate"

class ResourceType(Enum):
    CPU = "cpu"
    MEMORY = "memory"
    DISK = "disk"

class SystemMetric:
    def __init__(self, metric_type: MetricType, value: float, timestamp: datetime):
        self.metric_type = metric_type
        self.value = value
        self.timestamp = timestamp

class SystemDiagnosticsInterface:
    def __init__(self):
        self.health_status = "Healthy"
        self.error_logs = []

    def update_health_status(self, status: str):
        self.health_status = status

    def log_error(self, error: str):
        self.error_logs.append(error)

    def get_diagnostics(self):
        return {"health_status": self.health_status, "error_logs": self.error_logs}

class SelfMonitoringSystem:
    def __init__(self, diagnostics_interface: SystemDiagnosticsInterface):
        self.metrics = []
        self.diagnostics_interface = diagnostics_interface

    def track_metric(self, metric: SystemMetric):
        self.metrics.append(metric)
        self.evaluate_system_health()

    def evaluate_system_health(self):
        performance_metrics = [m.value for m in self.metrics if m.metric_type == MetricType.PERFORMANCE]
        if any(val < 50 for val in performance_metrics):  # Example threshold
            self.diagnostics_interface.update_health_status("Degraded")
        else:
            self.diagnostics_interface.update_health_status("Healthy")

    def get_health_status(self):
        return self.diagnostics_interface.get_diagnostics()

class PerformanceOptimizer:
    def suggest_optimizations(self):
        return "Optimize resource allocation and reduce latency."

class AdaptationManager:
    def adapt_system(self, health_status: str):
        if health_status == "Degraded":
            return "Switching to failover mode."
        return "System is stable."
