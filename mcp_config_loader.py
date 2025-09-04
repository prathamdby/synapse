"""JSON configuration loader for MCP servers."""

import json
import logging
import os
import re
from typing import Dict, Any, Optional
import structlog

logger = structlog.get_logger(__name__)


class MCPConfigLoader:
    """Loads and validates MCP server configuration from JSON files."""

    @staticmethod
    def load_config(config_path: str) -> Optional[Dict[str, Any]]:
        """
        Load MCP configuration from JSON file with environment variable support.

        Args:
            config_path: Path to the JSON configuration file

        Returns:
            Configuration dictionary or None if loading fails
        """
        try:
            if not os.path.exists(config_path):
                logger.warning(
                    "MCP configuration file not found", config_path=config_path
                )
                return None

            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)

            if not MCPConfigLoader.validate_config(config):
                logger.error(
                    "Invalid MCP configuration format", config_path=config_path
                )
                return None

            # Process environment variables in configuration
            processed_config = MCPConfigLoader._process_env_variables(config)

            logger.info(
                "MCP configuration loaded successfully",
                config_path=config_path,
                server_count=len(processed_config.get("servers", {})),
            )
            return processed_config

        except json.JSONDecodeError as e:
            logger.error(
                "Invalid JSON in MCP configuration file",
                config_path=config_path,
                error=str(e),
            )
            return None
        except Exception as e:
            logger.error(
                "Error loading MCP configuration", config_path=config_path, error=str(e)
            )
            return None

    @staticmethod
    def _process_env_variables(config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process environment variables in configuration using !VARIABLE_NAME syntax.

        Args:
            config: Configuration dictionary to process

        Returns:
            Configuration with environment variables resolved
        """

        def process_value(value):
            """Recursively process values to resolve environment variables."""
            if isinstance(value, str):
                # Check if value starts with ! (environment variable reference)
                if value.startswith("!"):
                    env_var_name = value[1:]  # Remove the ! prefix
                    env_value = os.getenv(env_var_name)

                    if env_value is None:
                        logger.warning(
                            "Environment variable not found",
                            variable=env_var_name,
                            original_value=value,
                        )
                        # Return the original value if env var not found
                        return value

                    logger.debug(
                        "Resolved environment variable",
                        variable=env_var_name,
                        resolved=True,
                    )
                    return env_value
                else:
                    # Check for inline environment variable references like "prefix_!VAR_suffix"
                    pattern = r"!([A-Z_][A-Z0-9_]*)"
                    matches = re.findall(pattern, value)

                    if matches:
                        processed_value = value
                        for env_var_name in matches:
                            env_value = os.getenv(env_var_name)
                            if env_value is not None:
                                processed_value = processed_value.replace(
                                    f"!{env_var_name}", env_value
                                )
                                logger.debug(
                                    "Resolved inline environment variable",
                                    variable=env_var_name,
                                    in_string=value,
                                )
                            else:
                                logger.warning(
                                    "Inline environment variable not found",
                                    variable=env_var_name,
                                    in_string=value,
                                )
                        return processed_value

                return value
            elif isinstance(value, dict):
                return {k: process_value(v) for k, v in value.items()}
            elif isinstance(value, list):
                return [process_value(item) for item in value]
            else:
                return value

        return process_value(config)

    @staticmethod
    def validate_config(config: Dict[str, Any]) -> bool:
        """
        Validate MCP configuration structure.

        Args:
            config: Configuration dictionary to validate

        Returns:
            True if configuration is valid, False otherwise
        """
        try:
            if not isinstance(config, dict):
                logger.error("Configuration must be a dictionary")
                return False

            if "servers" not in config:
                logger.error("Configuration missing 'servers' section")
                return False

            servers = config["servers"]
            if not isinstance(servers, dict):
                logger.error("'servers' section must be a dictionary")
                return False

            for server_name, server_config in servers.items():
                if not isinstance(server_config, dict):
                    logger.error(
                        "Server configuration must be a dictionary",
                        server_name=server_name,
                    )
                    return False

                # Check required fields
                required_fields = ["command", "args"]
                for field in required_fields:
                    if field not in server_config:
                        logger.error(
                            "Server configuration missing required field",
                            server_name=server_name,
                            field=field,
                        )
                        return False

                # Validate field types
                if not isinstance(server_config["command"], str):
                    logger.error(
                        "Server 'command' must be a string", server_name=server_name
                    )
                    return False

                if not isinstance(server_config["args"], list):
                    logger.error(
                        "Server 'args' must be a list", server_name=server_name
                    )
                    return False

                # Validate optional fields
                if "enabled" in server_config:
                    if not isinstance(server_config["enabled"], bool):
                        logger.error(
                            "Server 'enabled' must be a boolean",
                            server_name=server_name,
                        )
                        return False

                # Validate env field if present
                if "env" in server_config:
                    if not isinstance(server_config["env"], dict):
                        logger.error(
                            "Server 'env' must be a dictionary",
                            server_name=server_name,
                        )
                        return False

            return True

        except Exception as e:
            logger.error("Error validating configuration", error=str(e))
            return False

    @staticmethod
    def get_enabled_servers(config: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """
        Get only enabled servers from configuration.

        Args:
            config: Full configuration dictionary

        Returns:
            Dictionary of enabled server configurations
        """
        if not config or "servers" not in config:
            return {}

        enabled_servers = {}
        for server_name, server_config in config["servers"].items():
            # Default to enabled if not specified
            if server_config.get("enabled", True):
                enabled_servers[server_name] = server_config

        logger.info(
            "Filtered enabled servers",
            total_servers=len(config["servers"]),
            enabled_servers=len(enabled_servers),
        )
        return enabled_servers

    @staticmethod
    def create_example_config(config_path: str) -> bool:
        """
        Create an example MCP configuration file.

        Args:
            config_path: Path where to create the example configuration

        Returns:
            True if example was created successfully, False otherwise
        """
        example_config = {
            "servers": {
                "fetch": {
                    "command": "uvx",
                    "args": ["mcp-server-fetch"],
                    "enabled": True,
                },
                "searxng": {
                    "command": "npx",
                    "args": ["-y", "mcp-searxng"],
                    "env": {"SEARXNG_URL": "!SEARXNG_BASE_URL"},
                    "enabled": True,
                }
            }
        }

        try:
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(example_config, f, indent=2)

            logger.info("Example MCP configuration created", config_path=config_path)
            return True

        except Exception as e:
            logger.error(
                "Error creating example configuration",
                config_path=config_path,
                error=str(e),
            )
            return False
