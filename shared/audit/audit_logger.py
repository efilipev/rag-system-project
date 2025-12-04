"""
Audit Logging System
Tracks security-relevant events and user actions
"""
import logging
import json
from datetime import datetime
from typing import Optional, Dict, Any
from enum import Enum

logger = logging.getLogger(__name__)


class AuditEventType(Enum):
    """Types of audit events"""
    # Authentication events
    LOGIN_SUCCESS = "login_success"
    LOGIN_FAILURE = "login_failure"
    LOGOUT = "logout"
    TOKEN_REFRESH = "token_refresh"
    PASSWORD_CHANGE = "password_change"

    # Authorization events
    ACCESS_GRANTED = "access_granted"
    ACCESS_DENIED = "access_denied"
    PERMISSION_CHANGE = "permission_change"

    # API Key events
    API_KEY_CREATED = "api_key_created"
    API_KEY_REVOKED = "api_key_revoked"
    API_KEY_USED = "api_key_used"

    # Rate limiting events
    RATE_LIMIT_EXCEEDED = "rate_limit_exceeded"
    RATE_LIMIT_RESET = "rate_limit_reset"

    # Query events
    QUERY_SUBMITTED = "query_submitted"
    QUERY_COMPLETED = "query_completed"
    QUERY_FAILED = "query_failed"

    # Security events
    INVALID_INPUT = "invalid_input"
    SQL_INJECTION_ATTEMPT = "sql_injection_attempt"
    XSS_ATTEMPT = "xss_attempt"
    COMMAND_INJECTION_ATTEMPT = "command_injection_attempt"
    SUSPICIOUS_ACTIVITY = "suspicious_activity"

    # Configuration events
    CONFIG_CHANGE = "config_change"
    SERVICE_START = "service_start"
    SERVICE_STOP = "service_stop"


class AuditLogger:
    """
    Audit logger for tracking security-relevant events

    Logs events in structured JSON format for easy parsing and analysis
    """

    def __init__(self, service_name: str, log_to_file: bool = False, audit_log_path: Optional[str] = None):
        """
        Initialize audit logger

        Args:
            service_name: Name of the service generating audit logs
            log_to_file: Whether to log to a separate audit file
            audit_log_path: Path to audit log file (if log_to_file is True)
        """
        self.service_name = service_name
        self.log_to_file = log_to_file

        # Set up file handler if requested
        if log_to_file:
            audit_log_path = audit_log_path or f"audit_{service_name}.log"

            file_handler = logging.FileHandler(audit_log_path)
            file_handler.setLevel(logging.INFO)

            # JSON formatter for structured logs
            formatter = logging.Formatter('%(message)s')
            file_handler.setFormatter(formatter)

            # Create separate logger for audit events
            self.audit_logger = logging.getLogger(f"audit.{service_name}")
            self.audit_logger.setLevel(logging.INFO)
            self.audit_logger.addHandler(file_handler)
            self.audit_logger.propagate = False
        else:
            self.audit_logger = logger

    def log_event(
        self,
        event_type: AuditEventType,
        user_id: Optional[str] = None,
        username: Optional[str] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        resource: Optional[str] = None,
        action: Optional[str] = None,
        status: str = "success",
        details: Optional[Dict[str, Any]] = None,
        correlation_id: Optional[str] = None
    ):
        """
        Log an audit event

        Args:
            event_type: Type of audit event
            user_id: User ID (if applicable)
            username: Username (if applicable)
            ip_address: Client IP address
            user_agent: User agent string
            resource: Resource being accessed
            action: Action being performed
            status: Status (success, failure, denied, etc.)
            details: Additional details as dictionary
            correlation_id: Correlation ID for distributed tracing
        """
        audit_event = {
            "timestamp": datetime.utcnow().isoformat(),
            "service": self.service_name,
            "event_type": event_type.value,
            "status": status,
            "correlation_id": correlation_id
        }

        # Add user information
        if user_id:
            audit_event["user_id"] = user_id
        if username:
            audit_event["username"] = username

        # Add request information
        if ip_address:
            audit_event["ip_address"] = ip_address
        if user_agent:
            audit_event["user_agent"] = user_agent

        # Add resource/action
        if resource:
            audit_event["resource"] = resource
        if action:
            audit_event["action"] = action

        # Add additional details
        if details:
            audit_event["details"] = details

        # Log as JSON
        log_message = json.dumps(audit_event)
        self.audit_logger.info(log_message)

    def log_authentication(
        self,
        event_type: AuditEventType,
        user_id: Optional[str] = None,
        username: Optional[str] = None,
        ip_address: Optional[str] = None,
        status: str = "success",
        reason: Optional[str] = None
    ):
        """
        Log authentication event

        Args:
            event_type: Authentication event type
            user_id: User ID
            username: Username
            ip_address: Client IP
            status: Success or failure
            reason: Failure reason (if applicable)
        """
        details = {}
        if reason:
            details["reason"] = reason

        self.log_event(
            event_type=event_type,
            user_id=user_id,
            username=username,
            ip_address=ip_address,
            action="authenticate",
            status=status,
            details=details if details else None
        )

    def log_authorization(
        self,
        resource: str,
        action: str,
        user_id: Optional[str] = None,
        granted: bool = True,
        required_role: Optional[str] = None
    ):
        """
        Log authorization event

        Args:
            resource: Resource being accessed
            action: Action being performed
            user_id: User ID
            granted: Whether access was granted
            required_role: Required role (if applicable)
        """
        details = {}
        if required_role:
            details["required_role"] = required_role

        self.log_event(
            event_type=AuditEventType.ACCESS_GRANTED if granted else AuditEventType.ACCESS_DENIED,
            user_id=user_id,
            resource=resource,
            action=action,
            status="granted" if granted else "denied",
            details=details if details else None
        )

    def log_security_event(
        self,
        event_type: AuditEventType,
        ip_address: Optional[str] = None,
        user_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        """
        Log security-related event (attacks, suspicious activity)

        Args:
            event_type: Security event type
            ip_address: Client IP address
            user_id: User ID (if known)
            details: Additional details
        """
        self.log_event(
            event_type=event_type,
            user_id=user_id,
            ip_address=ip_address,
            action="security_check",
            status="blocked",
            details=details
        )

    def log_query(
        self,
        query: str,
        user_id: Optional[str] = None,
        correlation_id: Optional[str] = None,
        status: str = "submitted",
        response_time_ms: Optional[float] = None
    ):
        """
        Log query submission

        Args:
            query: Query text (truncated for privacy)
            user_id: User ID
            correlation_id: Correlation ID
            status: Query status
            response_time_ms: Response time in milliseconds
        """
        details = {
            "query_preview": query[:100] + "..." if len(query) > 100 else query,
            "query_length": len(query)
        }

        if response_time_ms:
            details["response_time_ms"] = response_time_ms

        self.log_event(
            event_type=AuditEventType.QUERY_SUBMITTED if status == "submitted" else AuditEventType.QUERY_COMPLETED,
            user_id=user_id,
            correlation_id=correlation_id,
            resource="query",
            action="execute",
            status=status,
            details=details
        )

    def log_rate_limit(
        self,
        user_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        endpoint: Optional[str] = None,
        limit: Optional[int] = None
    ):
        """
        Log rate limit event

        Args:
            user_id: User ID
            ip_address: Client IP
            endpoint: API endpoint
            limit: Rate limit threshold
        """
        details = {}
        if endpoint:
            details["endpoint"] = endpoint
        if limit:
            details["limit"] = limit

        self.log_event(
            event_type=AuditEventType.RATE_LIMIT_EXCEEDED,
            user_id=user_id,
            ip_address=ip_address,
            action="rate_limit_check",
            status="exceeded",
            details=details if details else None
        )


# Global audit logger instance
_audit_logger: Optional[AuditLogger] = None


def get_audit_logger() -> Optional[AuditLogger]:
    """Get global audit logger instance"""
    return _audit_logger


def init_audit_logger(service_name: str, log_to_file: bool = False, audit_log_path: Optional[str] = None):
    """
    Initialize global audit logger

    Args:
        service_name: Service name
        log_to_file: Whether to log to file
        audit_log_path: Path to audit log file
    """
    global _audit_logger
    _audit_logger = AuditLogger(
        service_name=service_name,
        log_to_file=log_to_file,
        audit_log_path=audit_log_path
    )
    return _audit_logger
