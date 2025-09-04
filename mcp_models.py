"""Data models for MCP components."""

from dataclasses import dataclass
from typing import Any, Dict, Optional, List
from enum import Enum


@dataclass
class MCPTool:
    """Represents an MCP tool with server information."""

    name: str
    description: str
    server_name: str
    full_name: str  # server_name.tool_name for conflict resolution
    schema: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        """Generate full_name if not provided."""
        if not self.full_name:
            self.full_name = f"{self.server_name}.{self.name}"


@dataclass
class MCPServerConnection:
    """Represents a connection to an MCP server."""

    name: str
    session: Any  # MCP session object
    connected: bool
    tools: Dict[str, MCPTool]
    command: str
    args: List[str]
    enabled: bool = True
    context_manager: Any = None  # Store the async context manager
    read_stream: Any = None
    write_stream: Any = None

    def __post_init__(self):
        """Initialize tools dict if not provided."""
        if self.tools is None:
            self.tools = {}


@dataclass
class ToolRequest:
    """Represents a request to execute an MCP tool."""

    tool_name: str
    arguments: Dict[str, Any]
    server_name: Optional[str] = None  # For routing to specific server

    def __post_init__(self):
        """Extract server name from full tool name if not provided."""
        if not self.server_name and "." in self.tool_name:
            self.server_name = self.tool_name.split(".", 1)[0]


@dataclass
class ToolResult:
    """Represents the result of an MCP tool execution."""

    success: bool
    result: Any
    error: Optional[str] = None
    tool_name: Optional[str] = None
    server_name: Optional[str] = None
    execution_time: Optional[float] = None

    @classmethod
    def success_result(
        cls,
        result: Any,
        tool_name: str = None,
        server_name: str = None,
        execution_time: float = None,
    ):
        """Create a successful tool result."""
        return cls(
            success=True,
            result=result,
            tool_name=tool_name,
            server_name=server_name,
            execution_time=execution_time,
        )

    @classmethod
    def error_result(
        cls,
        error: str,
        tool_name: str = None,
        server_name: str = None,
        execution_time: float = None,
    ):
        """Create an error tool result."""
        return cls(
            success=False,
            result=None,
            error=error,
            tool_name=tool_name,
            server_name=server_name,
            execution_time=execution_time,
        )


class ServerStatus(Enum):
    """Enumeration of possible MCP server connection statuses."""

    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    ERROR = "error"
    DISABLED = "disabled"


@dataclass
class ServerHealth:
    """Represents the health status of an MCP server."""

    server_name: str
    status: ServerStatus
    last_check: Optional[str] = None
    error_message: Optional[str] = None
    tool_count: int = 0
    uptime: Optional[float] = None

    @property
    def is_healthy(self) -> bool:
        """Check if server is in a healthy state."""
        return self.status == ServerStatus.CONNECTED

    @property
    def can_execute_tools(self) -> bool:
        """Check if server can execute tools."""
        return self.is_healthy and self.tool_count > 0


@dataclass
class MCPManagerConfig:
    """Configuration for the MCP Manager."""

    config_path: str = "mcp_config.json"
    connection_timeout: float = 60.0  # Increased timeout for slower servers
    tool_execution_timeout: float = 30.0
    max_retries: int = 3
    retry_delay: float = 1.0
    health_check_interval: float = 300.0  # 5 minutes
    auto_reconnect: bool = True
    log_tool_executions: bool = True

    def __post_init__(self):
        """Validate configuration values."""
        if self.connection_timeout <= 0:
            raise ValueError("connection_timeout must be positive")
        if self.tool_execution_timeout <= 0:
            raise ValueError("tool_execution_timeout must be positive")
        if self.max_retries < 0:
            raise ValueError("max_retries must be non-negative")
        if self.retry_delay < 0:
            raise ValueError("retry_delay must be non-negative")
        if self.health_check_interval <= 0:
            raise ValueError("health_check_interval must be positive")


@dataclass
class ToolExecutionMetrics:
    """Metrics for tool execution tracking."""

    tool_name: str
    server_name: str
    execution_count: int = 0
    success_count: int = 0
    error_count: int = 0
    total_execution_time: float = 0.0
    last_execution: Optional[str] = None
    last_error: Optional[str] = None

    @property
    def success_rate(self) -> float:
        """Calculate success rate as a percentage."""
        if self.execution_count == 0:
            return 0.0
        return (self.success_count / self.execution_count) * 100.0

    @property
    def average_execution_time(self) -> float:
        """Calculate average execution time."""
        if self.success_count == 0:
            return 0.0
        return self.total_execution_time / self.success_count

    def record_execution(
        self, success: bool, execution_time: float, error: Optional[str] = None
    ):
        """Record a tool execution."""
        self.execution_count += 1
        if success:
            self.success_count += 1
            self.total_execution_time += execution_time
        else:
            self.error_count += 1
            self.last_error = error

        from datetime import datetime

        self.last_execution = datetime.now().isoformat()


@dataclass
class MCPSystemStatus:
    """Overall system status for MCP integration."""

    total_servers: int = 0
    connected_servers: int = 0
    total_tools: int = 0
    available_tools: int = 0
    last_update: Optional[str] = None
    errors: List[str] = None

    def __post_init__(self):
        """Initialize errors list if not provided."""
        if self.errors is None:
            self.errors = []

    @property
    def connection_rate(self) -> float:
        """Calculate server connection rate as a percentage."""
        if self.total_servers == 0:
            return 0.0
        return (self.connected_servers / self.total_servers) * 100.0

    @property
    def tool_availability_rate(self) -> float:
        """Calculate tool availability rate as a percentage."""
        if self.total_tools == 0:
            return 0.0
        return (self.available_tools / self.total_tools) * 100.0

    @property
    def is_healthy(self) -> bool:
        """Check if the MCP system is in a healthy state."""
        return (
            self.connected_servers > 0
            and self.available_tools > 0
            and len(self.errors) == 0
        )

    def add_error(self, error: str):
        """Add an error to the system status."""
        if error not in self.errors:
            self.errors.append(error)

    def clear_errors(self):
        """Clear all errors from the system status."""
        self.errors.clear()

    def update_timestamp(self):
        """Update the last update timestamp."""
        from datetime import datetime

        self.last_update = datetime.now().isoformat()
