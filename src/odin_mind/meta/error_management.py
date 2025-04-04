
# Error Management System with Diagnostics Interface

from datetime import datetime
from enum import Enum

class ErrorSeverity(Enum):
    CRITICAL = "critical"
    MAJOR = "major"
    MINOR = "minor"

class ErrorCategory(Enum):
    SYSTEM = "system"
    NETWORK = "network"
    APPLICATION = "application"

class ErrorData:
    def __init__(self, error_id: str, severity: ErrorSeverity, category: ErrorCategory, message: str, timestamp: datetime):
        self.error_id = error_id
        self.severity = severity
        self.category = category
        self.message = message
        self.timestamp = timestamp

class ErrorManagementSystem:
    def __init__(self, diagnostics_interface: 'SystemDiagnosticsInterface'):
        self.errors = []
        self.diagnostics_interface = diagnostics_interface

    def log_error(self, error: ErrorData):
        self.errors.append(error)
        if error.severity == ErrorSeverity.CRITICAL:
            self.trigger_alert(error.message)

    def trigger_alert(self, message: str):
        self.diagnostics_interface.log_error(f"Critical alert: {message}")
