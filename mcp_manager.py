"""MCP Manager for handling multiple MCP server connections and tool execution."""

import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
import structlog

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from mcp_config_loader import MCPConfigLoader
from mcp_models import (
    MCPTool,
    MCPServerConnection,
    ToolRequest,
    ToolResult,
    ServerStatus,
    ServerHealth,
    MCPManagerConfig,
    ToolExecutionMetrics,
    MCPSystemStatus,
)

logger = structlog.get_logger(__name__)


class MCPManager:
    """Manages multiple MCP server connections and tool execution."""

    def __init__(
        self,
        config_path: str = "mcp_config.json",
        manager_config: Optional[MCPManagerConfig] = None,
    ):
        """
        Initialize MCPManager.

        Args:
            config_path: Path to MCP configuration file
            manager_config: Manager configuration options
        """
        self.config_path = config_path
        self.config = manager_config or MCPManagerConfig(config_path=config_path)

        # Server connections and tools
        self.servers: Dict[str, MCPServerConnection] = {}
        self.available_tools: Dict[str, MCPTool] = {}

        # Health and metrics tracking
        self.server_health: Dict[str, ServerHealth] = {}
        self.tool_metrics: Dict[str, ToolExecutionMetrics] = {}

        # Thread pool for synchronous MCP operations
        self.executor = ThreadPoolExecutor(max_workers=5, thread_name_prefix="mcp")

        # Configuration and status
        self.raw_config: Optional[Dict[str, Any]] = None
        self.system_status = MCPSystemStatus()

        # Health check task
        self._health_check_task: Optional[asyncio.Task] = None

        logger.info(
            "MCPManager initialized",
            config_path=config_path,
            connection_timeout=self.config.connection_timeout,
            tool_execution_timeout=self.config.tool_execution_timeout,
        )

    async def load_config_and_connect(self) -> None:
        """Load configuration from JSON file and connect to all enabled servers."""
        try:
            # Load configuration
            self.raw_config = MCPConfigLoader.load_config(self.config_path)
            if not self.raw_config:
                logger.warning(
                    "No MCP configuration loaded, MCP features will be disabled",
                    config_path=self.config_path,
                )
                return

            # Get enabled servers
            enabled_servers = MCPConfigLoader.get_enabled_servers(self.raw_config)
            if not enabled_servers:
                logger.info("No enabled MCP servers found")
                return

            # Connect to all enabled servers in parallel
            connection_tasks = []
            for server_name, server_config in enabled_servers.items():
                task = asyncio.create_task(
                    self.connect_server(
                        server_name,
                        server_config["command"],
                        server_config["args"],
                        server_config.get("env", {}),
                    ),
                    name=f"connect_{server_name}",
                )
                connection_tasks.append(task)

            # Wait for all connections with timeout
            if connection_tasks:
                logger.info(
                    "Connecting to MCP servers", server_count=len(connection_tasks)
                )

                try:
                    results = await asyncio.wait_for(
                        asyncio.gather(*connection_tasks, return_exceptions=True),
                        timeout=self.config.connection_timeout,
                    )

                    # Process results
                    successful_connections = sum(1 for r in results if r is True)
                    logger.info(
                        "MCP server connection results",
                        total_servers=len(connection_tasks),
                        successful_connections=successful_connections,
                        failed_connections=len(connection_tasks)
                        - successful_connections,
                    )

                except asyncio.TimeoutError:
                    logger.error(
                        "Timeout connecting to MCP servers",
                        timeout=self.config.connection_timeout,
                    )
                    # Cancel remaining tasks
                    for task in connection_tasks:
                        if not task.done():
                            task.cancel()

            # Discover tools from all connected servers
            await self._discover_all_tools()

            # Update system status
            self._update_system_status()

            # Start health check task if auto-reconnect is enabled
            if self.config.auto_reconnect:
                await self._start_health_check_task()

            logger.info(
                "MCP initialization complete",
                connected_servers=len(
                    [s for s in self.servers.values() if s.connected]
                ),
                total_tools=len(self.available_tools),
            )

        except Exception as e:
            logger.error(
                "Error during MCP initialization",
                error=str(e),
                config_path=self.config_path,
            )
            self.system_status.add_error(f"Initialization error: {str(e)}")

    async def connect_server(
        self,
        server_name: str,
        command: str,
        args: List[str],
        env: Dict[str, str] = None,
    ) -> bool:
        """
        Connect to a single MCP server.

        Args:
            server_name: Name of the server
            command: Command to start the server
            args: Arguments for the server command
            env: Environment variables for the server

        Returns:
            True if connection successful, False otherwise
        """
        try:
            logger.info(
                "Connecting to MCP server",
                server_name=server_name,
                command=command,
                args=args,
                env=env,
            )

            # Update server health status
            self.server_health[server_name] = ServerHealth(
                server_name=server_name, status=ServerStatus.CONNECTING
            )

            # Create server parameters
            server_params = StdioServerParameters(command=command, args=args, env=env)

            # Connect using thread pool (MCP client is synchronous)
            loop = asyncio.get_event_loop()
            session = await loop.run_in_executor(
                self.executor, self._sync_connect_server, server_params
            )

            if session:
                # Create server connection object
                connection = MCPServerConnection(
                    name=server_name,
                    session=session,
                    connected=True,
                    tools={},
                    command=command,
                    args=args,
                    enabled=True,
                )

                self.servers[server_name] = connection

                # Update health status
                self.server_health[server_name] = ServerHealth(
                    server_name=server_name,
                    status=ServerStatus.CONNECTED,
                    last_check=datetime.now().isoformat(),
                    uptime=time.time(),
                )

                logger.info(
                    "Successfully connected to MCP server", server_name=server_name
                )
                return True
            else:
                raise Exception("Failed to establish session")

        except Exception as e:
            error_msg = f"Connection failed: {str(e)}"
            logger.error(
                "Failed to connect to MCP server",
                server_name=server_name,
                error=error_msg,
            )

            # Update health status
            self.server_health[server_name] = ServerHealth(
                server_name=server_name,
                status=ServerStatus.ERROR,
                error_message=error_msg,
                last_check=datetime.now().isoformat(),
            )

            self.system_status.add_error(f"{server_name}: {error_msg}")
            return False

    def _sync_connect_server(
        self, server_params: StdioServerParameters
    ) -> Optional[ClientSession]:
        """
        Synchronously connect to MCP server (runs in thread pool).

        Args:
            server_params: Server parameters for connection

        Returns:
            ClientSession if successful, None otherwise
        """
        try:
            # This is a simplified synchronous connection
            # In a real implementation, you would use the actual MCP client
            session = stdio_client(server_params)
            return session
        except Exception as e:
            logger.error("Sync connection failed", error=str(e))
            return None

    async def discover_tools_from_server(self, server_name: str) -> List[Dict]:
        """
        Discover available tools from a specific MCP server.

        Args:
            server_name: Name of the server to discover tools from

        Returns:
            List of tool dictionaries
        """
        if server_name not in self.servers:
            logger.warning(
                "Cannot discover tools from unknown server", server_name=server_name
            )
            return []

        connection = self.servers[server_name]
        if not connection.connected:
            logger.warning(
                "Cannot discover tools from disconnected server",
                server_name=server_name,
            )
            return []

        try:
            logger.debug("Discovering tools from server", server_name=server_name)

            # Use thread pool for synchronous MCP operations
            loop = asyncio.get_event_loop()
            tools_data = await loop.run_in_executor(
                self.executor, self._sync_discover_tools, connection.session
            )

            if tools_data:
                # Process discovered tools
                for tool_data in tools_data:
                    tool = MCPTool(
                        name=tool_data.get("name", ""),
                        description=tool_data.get("description", ""),
                        server_name=server_name,
                        full_name=f"{server_name}.{tool_data.get('name', '')}",
                        schema=tool_data.get("schema"),
                    )

                    # Handle tool name conflicts
                    if tool.name in self.available_tools:
                        # Use full name for conflicts
                        existing_tool = self.available_tools[tool.name]
                        logger.warning(
                            "Tool name conflict detected",
                            tool_name=tool.name,
                            existing_server=existing_tool.server_name,
                            new_server=server_name,
                        )
                        # Store with full name
                        self.available_tools[tool.full_name] = tool
                        connection.tools[tool.full_name] = tool
                    else:
                        # Store with simple name
                        self.available_tools[tool.name] = tool
                        connection.tools[tool.name] = tool

                # Update server health
                if server_name in self.server_health:
                    self.server_health[server_name].tool_count = len(connection.tools)

                logger.info(
                    "Tools discovered from server",
                    server_name=server_name,
                    tool_count=len(connection.tools),
                )

                return tools_data
            else:
                logger.info("No tools found on server", server_name=server_name)
                return []

        except Exception as e:
            error_msg = f"Tool discovery failed: {str(e)}"
            logger.error(
                "Error discovering tools from server",
                server_name=server_name,
                error=error_msg,
            )

            if server_name in self.server_health:
                self.server_health[server_name].error_message = error_msg

            return []

    def _sync_discover_tools(self, session: ClientSession) -> Optional[List[Dict]]:
        """
        Synchronously discover tools (runs in thread pool).

        Args:
            session: MCP client session

        Returns:
            List of tool dictionaries or None if failed
        """
        try:
            # This is a simplified tool discovery
            # In a real implementation, you would call session.list_tools()
            # For now, return mock data
            return [
                {
                    "name": "example_tool",
                    "description": "Example tool for testing",
                    "schema": {},
                }
            ]
        except Exception as e:
            logger.error("Sync tool discovery failed", error=str(e))
            return None

    async def _discover_all_tools(self) -> None:
        """Discover tools from all connected servers."""
        discovery_tasks = []
        for server_name in self.servers:
            task = asyncio.create_task(
                self.discover_tools_from_server(server_name),
                name=f"discover_{server_name}",
            )
            discovery_tasks.append(task)

        if discovery_tasks:
            try:
                await asyncio.gather(*discovery_tasks, return_exceptions=True)
            except Exception as e:
                logger.error("Error during tool discovery", error=str(e))

    async def execute_tool(self, tool_name: str, arguments: Dict) -> ToolResult:
        """
        Execute an MCP tool with the given arguments.

        Args:
            tool_name: Name of the tool to execute
            arguments: Arguments to pass to the tool

        Returns:
            ToolResult containing execution results
        """
        start_time = time.time()

        try:
            # Find the tool
            tool = self.available_tools.get(tool_name)
            if not tool:
                # Try with server prefix
                for full_name, candidate_tool in self.available_tools.items():
                    if full_name.endswith(f".{tool_name}"):
                        tool = candidate_tool
                        tool_name = full_name
                        break

            if not tool:
                error_msg = f"Tool '{tool_name}' not found"
                logger.warning(
                    error_msg, available_tools=list(self.available_tools.keys())
                )
                return ToolResult.error_result(
                    error=error_msg,
                    tool_name=tool_name,
                    execution_time=time.time() - start_time,
                )

            # Check if server is connected
            server_connection = self.servers.get(tool.server_name)
            if not server_connection or not server_connection.connected:
                error_msg = f"Server '{tool.server_name}' not connected"
                logger.warning(error_msg, tool_name=tool_name)
                return ToolResult.error_result(
                    error=error_msg,
                    tool_name=tool_name,
                    server_name=tool.server_name,
                    execution_time=time.time() - start_time,
                )

            logger.info(
                "Executing MCP tool",
                tool_name=tool_name,
                server_name=tool.server_name,
                arguments=arguments,
            )

            # Execute tool using thread pool
            loop = asyncio.get_event_loop()
            result = await asyncio.wait_for(
                loop.run_in_executor(
                    self.executor,
                    self._sync_execute_tool,
                    server_connection.session,
                    tool.name,
                    arguments,
                ),
                timeout=self.config.tool_execution_timeout,
            )

            execution_time = time.time() - start_time

            # Record metrics
            self._record_tool_execution(
                tool_name, tool.server_name, True, execution_time
            )

            logger.info(
                "Tool execution successful",
                tool_name=tool_name,
                server_name=tool.server_name,
                execution_time=execution_time,
            )

            return ToolResult.success_result(
                result=result,
                tool_name=tool_name,
                server_name=tool.server_name,
                execution_time=execution_time,
            )

        except asyncio.TimeoutError:
            execution_time = time.time() - start_time
            error_msg = (
                f"Tool execution timeout after {self.config.tool_execution_timeout}s"
            )
            logger.error(error_msg, tool_name=tool_name)

            # Record metrics
            self._record_tool_execution(
                tool_name,
                tool.server_name if tool else "unknown",
                False,
                execution_time,
                error_msg,
            )

            return ToolResult.error_result(
                error=error_msg,
                tool_name=tool_name,
                server_name=tool.server_name if tool else None,
                execution_time=execution_time,
            )

        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = f"Tool execution error: {str(e)}"
            logger.error(
                "Tool execution failed",
                tool_name=tool_name,
                error=error_msg,
                execution_time=execution_time,
            )

            # Record metrics
            self._record_tool_execution(
                tool_name,
                tool.server_name if tool else "unknown",
                False,
                execution_time,
                error_msg,
            )

            return ToolResult.error_result(
                error=error_msg,
                tool_name=tool_name,
                server_name=tool.server_name if tool else None,
                execution_time=execution_time,
            )

    def _sync_execute_tool(
        self, session: ClientSession, tool_name: str, arguments: Dict
    ) -> Any:
        """
        Synchronously execute tool (runs in thread pool).

        Args:
            session: MCP client session
            tool_name: Name of the tool
            arguments: Tool arguments

        Returns:
            Tool execution result
        """
        try:
            # This is a simplified tool execution
            # In a real implementation, you would call session.call_tool()
            return f"Mock result for {tool_name} with args {arguments}"
        except Exception as e:
            logger.error("Sync tool execution failed", error=str(e))
            raise

    def get_available_tools(self) -> List[MCPTool]:
        """
        Get list of all available tools.

        Returns:
            List of MCPTool objects
        """
        return list(self.available_tools.values())

    def is_any_server_connected(self) -> bool:
        """
        Check if any MCP server is connected.

        Returns:
            True if at least one server is connected
        """
        return any(server.connected for server in self.servers.values())

    async def reload_config(self) -> None:
        """Reload configuration and reconnect servers."""
        logger.info("Reloading MCP configuration")

        # Stop health check task
        if self._health_check_task:
            self._health_check_task.cancel()
            self._health_check_task = None

        # Disconnect all servers
        await self._disconnect_all_servers()

        # Clear current state
        self.servers.clear()
        self.available_tools.clear()
        self.server_health.clear()
        self.system_status = MCPSystemStatus()

        # Reload configuration and reconnect
        await self.load_config_and_connect()

    def _record_tool_execution(
        self,
        tool_name: str,
        server_name: str,
        success: bool,
        execution_time: float,
        error: Optional[str] = None,
    ) -> None:
        """Record tool execution metrics."""
        if not self.config.log_tool_executions:
            return

        metric_key = f"{server_name}.{tool_name}"
        if metric_key not in self.tool_metrics:
            self.tool_metrics[metric_key] = ToolExecutionMetrics(
                tool_name=tool_name, server_name=server_name
            )

        self.tool_metrics[metric_key].record_execution(success, execution_time, error)

    def _update_system_status(self) -> None:
        """Update overall system status."""
        self.system_status.total_servers = len(self.servers)
        self.system_status.connected_servers = len(
            [s for s in self.servers.values() if s.connected]
        )
        self.system_status.total_tools = len(self.available_tools)
        self.system_status.available_tools = len(
            [
                t
                for t in self.available_tools.values()
                if self.servers.get(
                    t.server_name, MCPServerConnection("", None, False, {}, "", [])
                ).connected
            ]
        )
        self.system_status.update_timestamp()

    async def _start_health_check_task(self) -> None:
        """Start the health check background task."""
        if self._health_check_task:
            return

        self._health_check_task = asyncio.create_task(
            self._health_check_loop(), name="mcp_health_check"
        )

    async def _health_check_loop(self) -> None:
        """Background health check loop."""
        while True:
            try:
                await asyncio.sleep(self.config.health_check_interval)
                await self._perform_health_checks()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Health check error", error=str(e))

    async def _perform_health_checks(self) -> None:
        """Perform health checks on all servers."""
        logger.debug("Performing MCP server health checks")

        for server_name, connection in self.servers.items():
            try:
                # Simple connectivity check
                if connection.connected and connection.session:
                    # In a real implementation, you might ping the server
                    # For now, just update the timestamp
                    if server_name in self.server_health:
                        self.server_health[server_name].last_check = (
                            datetime.now().isoformat()
                        )
                else:
                    # Mark as disconnected
                    if server_name in self.server_health:
                        self.server_health[server_name].status = (
                            ServerStatus.DISCONNECTED
                        )

            except Exception as e:
                logger.warning(
                    "Health check failed for server",
                    server_name=server_name,
                    error=str(e),
                )
                if server_name in self.server_health:
                    self.server_health[server_name].status = ServerStatus.ERROR
                    self.server_health[server_name].error_message = str(e)

        self._update_system_status()

    async def _disconnect_all_servers(self) -> None:
        """Disconnect from all MCP servers."""
        disconnect_tasks = []
        for server_name in self.servers:
            task = asyncio.create_task(
                self._disconnect_server(server_name), name=f"disconnect_{server_name}"
            )
            disconnect_tasks.append(task)

        if disconnect_tasks:
            await asyncio.gather(*disconnect_tasks, return_exceptions=True)

    async def _disconnect_server(self, server_name: str) -> None:
        """Disconnect from a specific server."""
        try:
            connection = self.servers.get(server_name)
            if connection and connection.session:
                # In a real implementation, you would properly close the session
                # For now, just mark as disconnected
                connection.connected = False

            logger.info("Disconnected from MCP server", server_name=server_name)

        except Exception as e:
            logger.error(
                "Error disconnecting from server", server_name=server_name, error=str(e)
            )

    def get_system_status(self) -> MCPSystemStatus:
        """Get current system status."""
        self._update_system_status()
        return self.system_status

    def get_server_health(self, server_name: str) -> Optional[ServerHealth]:
        """Get health status for a specific server."""
        return self.server_health.get(server_name)

    def get_tool_metrics(
        self, tool_name: str = None
    ) -> Dict[str, ToolExecutionMetrics]:
        """Get tool execution metrics."""
        if tool_name:
            return {k: v for k, v in self.tool_metrics.items() if tool_name in k}
        return self.tool_metrics.copy()

    async def shutdown(self) -> None:
        """Shutdown the MCP manager and cleanup resources."""
        logger.info("Shutting down MCP manager")

        # Cancel health check task
        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass

        # Disconnect all servers
        await self._disconnect_all_servers()

        # Shutdown thread pool
        if hasattr(self, "executor"):
            self.executor.shutdown(wait=False)

        logger.info("MCP manager shutdown complete")

    def __del__(self):
        """Cleanup on deletion."""
        if hasattr(self, "executor"):
            self.executor.shutdown(wait=False)
