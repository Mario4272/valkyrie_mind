from enum import Enum, auto
from dataclasses import dataclass
from typing import Dict, List, Set, Tuple, Optional, Any, Callable
from abc import ABC, abstractmethod
import asyncio
import hashlib
import hmac
import secrets
import uuid
from datetime import datetime, timedelta
import numpy as np

class SecurityLevel(Enum):
    LOW = auto()
    MEDIUM = auto()
    HIGH = auto()
    CRITICAL = auto()

class AccessType(Enum):
    READ = auto()
    WRITE = auto()
    EXECUTE = auto()
    ADMIN = auto()

class SecurityEvent(Enum):
    AUTH_SUCCESS = auto()
    AUTH_FAILURE = auto()
    PERMISSION_DENIED = auto()
    ANOMALY_DETECTED = auto()
    RESOURCE_ACCESS = auto()
    CONFIGURATION_CHANGE = auto()
    SYSTEM_ERROR = auto()

@dataclass
class SecurityConfig:
    min_password_length: int = 12
    max_auth_attempts: int = 3
    session_timeout_minutes: int = 30
    require_2fa: bool = True
    encryption_algorithm: str = "AES-256"
    log_level: str = "INFO"
    allowed_ips: Set[str] = None
    rate_limits: Dict[str, int] = None

@dataclass
class SecurityCredentials:
    id: str
    access_level: SecurityLevel
    permissions: Set[AccessType]
    api_key: str
    created_at: datetime
    expires_at: datetime
    last_used: datetime
    is_active: bool = True

class SecurityContext:
    def __init__(self, level: SecurityLevel, permissions: Set[AccessType]):
        self.level = level
        self.permissions = permissions
        self.session_id = str(uuid.uuid4())
        self.created_at = datetime.now()
        self.last_accessed = datetime.now()
        self.access_count = 0
        self.anomaly_score = 0.0

class SecurityMonitor:
    def __init__(self):
        self.events: List[Tuple[datetime, SecurityEvent, Dict[str, Any]]] = []
        self.anomaly_threshold = 0.8
        self.alert_callbacks: List[Callable] = []

    async def log_event(self, event: SecurityEvent, details: Dict[str, Any]) -> None:
        """Log a security event and check for anomalies."""
        timestamp = datetime.now()
        self.events.append((timestamp, event, details))
        
        # Check for anomalies
        if await self._check_anomaly(event, details):
            await self._trigger_alert({
                'timestamp': timestamp,
                'event': event,
                'details': details,
                'anomaly_score': self.anomaly_threshold
            })

    async def _check_anomaly(self, event: SecurityEvent, details: Dict[str, Any]) -> bool:
        """Check if the event represents an anomaly."""
        # Implement anomaly detection logic here
        recent_events = [e for e in self.events[-100:] if e[1] == event]
        
        if event == SecurityEvent.AUTH_FAILURE:
            # Check for brute force attempts
            recent_failures = len([e for e in recent_events 
                                 if e[2].get('ip') == details.get('ip')])
            return recent_failures >= 3
            
        elif event == SecurityEvent.PERMISSION_DENIED:
            # Check for potential privilege escalation attempts
            return len(recent_events) >= 5
            
        return False

    async def _trigger_alert(self, alert_data: Dict[str, Any]) -> None:
        """Trigger security alerts."""
        for callback in self.alert_callbacks:
            try:
                await callback(alert_data)
            except Exception as e:
                print(f"Error in alert callback: {str(e)}")

class SecurityManager:
    def __init__(self, config: SecurityConfig):
        self.config = config
        self.monitor = SecurityMonitor()
        self.credentials: Dict[str, SecurityCredentials] = {}
        self.active_sessions: Dict[str, SecurityContext] = {}
        self._encryption_key = secrets.token_bytes(32)
        
    async def authenticate(self, credentials_id: str, api_key: str) -> Optional[SecurityContext]:
        """Authenticate a request and create a security context."""
        try:
            if credentials_id not in self.credentials:
                await self.monitor.log_event(
                    SecurityEvent.AUTH_FAILURE,
                    {'credentials_id': credentials_id, 'reason': 'invalid_id'}
                )
                return None
                
            stored_creds = self.credentials[credentials_id]
            
            if not stored_creds.is_active:
                await self.monitor.log_event(
                    SecurityEvent.AUTH_FAILURE,
                    {'credentials_id': credentials_id, 'reason': 'inactive'}
                )
                return None
                
            if not self._verify_api_key(api_key, stored_creds.api_key):
                await self.monitor.log_event(
                    SecurityEvent.AUTH_FAILURE,
                    {'credentials_id': credentials_id, 'reason': 'invalid_key'}
                )
                return None
                
            # Create new security context
            context = SecurityContext(stored_creds.access_level, stored_creds.permissions)
            self.active_sessions[context.session_id] = context
            
            await self.monitor.log_event(
                SecurityEvent.AUTH_SUCCESS,
                {'credentials_id': credentials_id, 'session_id': context.session_id}
            )
            
            return context
            
        except Exception as e:
            await self.monitor.log_event(
                SecurityEvent.SYSTEM_ERROR,
                {'error': str(e), 'credentials_id': credentials_id}
            )
            raise

    def _verify_api_key(self, provided_key: str, stored_key: str) -> bool:
        """Verify an API key using constant-time comparison."""
        return hmac.compare_digest(provided_key.encode(), stored_key.encode())

    async def authorize(self, context: SecurityContext, required_level: SecurityLevel,
                       required_permissions: Set[AccessType]) -> bool:
        """Check if a security context has required level and permissions."""
        try:
            if not self._validate_session(context):
                return False
                
            has_level = context.level.value >= required_level.value
            has_permissions = required_permissions.issubset(context.permissions)
            
            if not (has_level and has_permissions):
                await self.monitor.log_event(
                    SecurityEvent.PERMISSION_DENIED,
                    {
                        'session_id': context.session_id,
                        'required_level': required_level.name,
                        'required_permissions': [p.name for p in required_permissions]
                    }
                )
                return False
                
            context.last_accessed = datetime.now()
            context.access_count += 1
            
            await self.monitor.log_event(
                SecurityEvent.RESOURCE_ACCESS,
                {'session_id': context.session_id, 'access_count': context.access_count}
            )
            
            return True
            
        except Exception as e:
            await self.monitor.log_event(
                SecurityEvent.SYSTEM_ERROR,
                {'error': str(e), 'session_id': context.session_id}
            )
            raise

    def _validate_session(self, context: SecurityContext) -> bool:
        """Validate that a session is active and not expired."""
        if context.session_id not in self.active_sessions:
            return False
            
        session_age = datetime.now() - context.created_at
        if session_age > timedelta(minutes=self.config.session_timeout_minutes):
            self.active_sessions.pop(context.session_id)
            return False
            
        return True

    async def create_credentials(self, level: SecurityLevel, 
                               permissions: Set[AccessType]) -> SecurityCredentials:
        """Create new security credentials."""
        credentials = SecurityCredentials(
            id=str(uuid.uuid4()),
            access_level=level,
            permissions=permissions,
            api_key=secrets.token_urlsafe(32),
            created_at=datetime.now(),
            expires_at=datetime.now() + timedelta(days=30),
            last_used=datetime.now()
        )
        
        self.credentials[credentials.id] = credentials
        
        await self.monitor.log_event(
            SecurityEvent.CONFIGURATION_CHANGE,
            {'action': 'create_credentials', 'credentials_id': credentials.id}
        )
        
        return credentials

    async def revoke_credentials(self, credentials_id: str) -> None:
        """Revoke security credentials."""
        if credentials_id in self.credentials:
            self.credentials[credentials_id].is_active = False
            
            # Remove any active sessions
            session_ids = [sid for sid, session in self.active_sessions.items()
                         if session.level == self.credentials[credentials_id].access_level]
            for sid in session_ids:
                self.active_sessions.pop(sid)
            
            await self.monitor.log_event(
                SecurityEvent.CONFIGURATION_CHANGE,
                {'action': 'revoke_credentials', 'credentials_id': credentials_id}
            )

class SecureSystem:
    """Base class for implementing security in system components."""
    
    def __init__(self, security_manager: SecurityManager):
        self.security_manager = security_manager
        
    async def secure_operation(self, context: SecurityContext, 
                             level: SecurityLevel,
                             permissions: Set[AccessType],
                             operation: Callable, *args, **kwargs) -> Any:
        """Execute an operation with security checks."""
        if await self.security_manager.authorize(context, level, permissions):
            return await operation(*args, **kwargs)
        raise PermissionError("Unauthorized access attempt")

# Example usage with Training System
class SecureTrainingSystem(SecureSystem):
    def __init__(self, security_manager: SecurityManager, training_system: Any):
        super().__init__(security_manager)
        self.training_system = training_system
        
    async def train(self, context: SecurityContext, training_data: Dict[str, Any]) -> Any:
        """Secure wrapper for training system."""
        return await self.secure_operation(
            context,
            SecurityLevel.HIGH,
            {AccessType.EXECUTE, AccessType.WRITE},
            self.training_system.train,
            training_data
        )
