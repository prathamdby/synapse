# 🧠 Synapse

A sophisticated Telegram bot powered by Cerebras AI that delivers intelligent conversations with persistent memory and lightning-fast responses.

## ✨ Features

- **🚀 Ultra-Fast AI**: Powered by Cerebras's high-performance inference
- **🧠 Conversation Memory**: Maintains context and history for each user
- **⚡ Instant Responses**: Leverages Cerebras's cutting-edge speed
- **👥 Multi-User Support**: Isolated conversations for privacy
- **🎛️ Model Selection**: Switch between available Cerebras models
- **📊 Usage Analytics**: Track interactions and preferences
- **🔒 Rate Limited**: Fair usage controls for optimal performance
- **👍 Smart Reactions**: Bot acknowledges messages with emoji reactions
- **🏢 Group Support**: Full-featured group chat support with admin controls
- **🧵 Thread Management**: Support for forum topics and reply chains
- **⚙️ Configurable**: Flexible group settings and mention policies
- **🔧 MCP Integration**: External tool support via Model Context Protocol
- **🌐 Web Fetching**: Retrieve and analyze web content
- **📁 File Operations**: Read and process files from the filesystem
- **🔄 Git Integration**: Repository operations and version control
- **🛠️ Extensible Tools**: Easy addition of new MCP servers and capabilities

## 🚀 Quick Start

### Prerequisites

- [Telegram Bot Token](https://t.me/botfather)
- [Cerebras API Key](https://cloud.cerebras.ai/)
- [UV Package Manager](https://docs.astral.sh/uv/) (for MCP server installation)

Choose your preferred deployment method:

### Option 1: Docker Deployment (Recommended)

**Prerequisites:**

- Docker & Docker Compose

**Steps:**

1. **Clone the repository**

   ```bash
   git clone https://github.com/prathamdby/synapse.git
   cd synapse
   ```

2. **Configure environment**

   ```bash
   cp .env.example .env
   # Edit .env with your API tokens
   ```

3. **Run with Docker Compose**
   ```bash
   docker-compose up -d
   ```

The bot will automatically start and data will persist in the `./data` directory.

### Option 2: Manual Installation

**Prerequisites:**

- Python 3.11+
- UV package manager

**Steps:**

1. **Clone and setup**

   ```bash
   git clone https://github.com/prathamdby/synapse.git
   cd synapse
   uv sync
   ```

2. **Configure environment**

   ```bash
   cp .env.example .env
   # Edit .env with your API tokens
   ```

3. **Run the bot**
   ```bash
   uv run main.py
   ```

## ⚙️ Configuration

Create a `.env` file:

```env
# Required
TELEGRAM_BOT_TOKEN=your_telegram_bot_token
CEREBRAS_API_KEY=your_cerebras_api_key

# Optional (defaults shown)
DATABASE_PATH=/app/data/bot_database.db  # For Docker deployment
# DATABASE_PATH=./bot_database.db        # For manual installation
LOG_LEVEL=INFO
RATE_LIMIT_MESSAGES_PER_MINUTE=10
RATE_LIMIT_WINDOW_SECONDS=60

# Group Features (optional)
ENABLE_GROUP_FEATURES=true
DEFAULT_GROUP_MODE=shared
DEFAULT_GROUP_MENTION_POLICY=mention_only
GROUP_MAX_CONTEXT=40
DEFAULT_GROUP_RATE_LIMIT=50
DEFAULT_USER_RATE_LIMIT_IN_GROUPS=10

# MCP Integration (optional)
MCP_CONFIG_PATH=./mcp_config.json
ADMIN_USER_IDS=123456789,987654321  # Comma-separated list for MCP admin commands

```

**Note**: For Docker deployment, use `/app/data/bot_database.db` to ensure data persists across container rebuilds.

### Getting API Keys

- **Telegram**: Message [@BotFather](https://t.me/botfather), use `/newbot`
- **Cerebras**: Sign up at [cloud.cerebras.ai](https://cloud.cerebras.ai/)

## 🤖 Bot Commands

### Personal Commands

- `/start` - Welcome message and introduction
- `/help` - Detailed help and usage information
- `/reset` or `/clear` - Clear conversation history
- `/stats` - View usage statistics and preferences
- `/model` - Switch between available AI models
- `/mcp_status` - View MCP server and tool status

### Group Commands (Admin Only)

- `/group_mode` - Set shared or personal memory mode
- `/group_settings` - View all group configuration options
- `/group_reset` - Clear group conversation history
- `/group_stats` - View group statistics and activity

### Admin Commands

- `/mcp_reload` - Reload MCP configuration without restart (admin only)

## 🔧 MCP Integration Setup

### What is MCP?

Model Context Protocol (MCP) allows the bot to use external tools and services to enhance responses with real-time data and functionality. The bot can automatically detect when tools might be helpful and use them seamlessly.

### Installing MCP Servers

Install MCP servers using UV (they run as separate processes):

```bash
# Web content fetching
uvx mcp-server-fetch

# Search integration (requires SearXNG URL)
npx -y mcp-searxng

# File system operations
uvx mcp-server-filesystem

# Git repository operations
uvx mcp-server-git
```

### MCP Configuration

Create or edit `mcp_config.json` in your project root:

```json
{
  "servers": {
    "fetch": {
      "command": "uvx",
      "args": ["mcp-server-fetch"],
      "enabled": true
    },
    "filesystem": {
      "command": "uvx",
      "args": ["mcp-server-filesystem", "/tmp"],
      "enabled": true
    },
    "git": {
      "command": "uvx",
      "args": ["mcp-server-git", "--repository", "/path/to/repo"],
      "enabled": false
    }
  }
}
```

### Configuration Options

- **command**: Executable to run the MCP server
- **args**: Arguments passed to the server command
- **enabled**: Whether this server should be started (true/false)
- **env**: Environment variables to pass to the server (optional)

### Available MCP Servers

#### Web Fetching (`mcp-server-fetch`)

- Retrieves web page content
- Analyzes URLs mentioned in conversations
- Downloads and processes web resources

#### File System (`mcp-server-filesystem`)

- Reads files from specified directories
- Processes file contents for analysis
- Supports various file formats

#### Git Operations (`mcp-server-git`)

- Repository status and information
- Commit history and branch details
- File change tracking

#### Search Integration (`mcp-searxng`)

- Web search through SearXNG instances
- Configurable search endpoints
- Real-time web search capabilities

### Tool Usage Examples

The bot automatically detects when to use tools:

**Web Content**: "What's on this page: https://example.com"

- Bot uses fetch tool to retrieve page content
- Analyzes and summarizes the content

**File Operations**: "Read the file /tmp/config.txt"

- Bot uses filesystem tool to read the file
- Processes and explains the content

**Git Information**: "What's the status of the repository?"

- Bot uses git tool to check repository status
- Reports on branches, commits, and changes

### MCP Management Commands

- **`/mcp_status`**: View all servers and tools status
- **`/mcp_reload`**: Reload configuration (admin only)

### Environment Variables in Configuration

MCP configurations support environment variable substitution using the `!VARIABLE_NAME` syntax:

```json
{
  "servers": {
    "searxng": {
      "command": "npx",
      "args": ["-y", "mcp-searxng"],
      "env": {
        "SEARXNG_URL": "!SEARXNG_BASE_URL",
        "API_KEY": "!MY_API_KEY"
      },
      "enabled": true
    }
  }
}
```

**Features:**

- **Direct substitution**: `"!VAR_NAME"` → value of `VAR_NAME` environment variable
- **Inline substitution**: `"https://!HOST:!PORT/api"` → `"https://localhost:8080/api"`
- **Fallback behavior**: If environment variable not found, original value is kept
- **Automatic logging**: Warns when environment variables are missing

**Environment Variable Setup:**

```bash
# Add to your .env file
SEARXNG_BASE_URL=https://search.example.com
MY_API_KEY=your-secret-key-here
```

### Troubleshooting MCP

1. **Check server status**: Use `/mcp_status` to see connection status
2. **Verify installation**: Ensure MCP servers are installed with `uvx`
3. **Configuration errors**: Check JSON syntax in `mcp_config.json`
4. **Permissions**: Ensure servers have access to required directories/repositories
5. **Reload config**: Use `/mcp_reload` after configuration changes

### Security Considerations

- **File Access**: Limit filesystem server to safe directories only
- **Repository Access**: Ensure git server has appropriate repository permissions
- **Admin Controls**: Only authorized users can reload MCP configuration
- **Tool Filtering**: Bot intelligently selects appropriate tools for requests

## 🏗️ Architecture

```
synapse/
├── main.py                 # Core bot logic and Telegram integration
├── database.py            # SQLite database management
├── cerebras_client.py     # Cerebras AI API client with async support
├── langchain_cerebras.py  # LangChain wrapper for Cerebras integration
├── mcp_manager.py         # MCP server connection and tool management
├── mcp_models.py          # Data models for MCP components
├── mcp_config_loader.py   # JSON configuration loading and validation
├── mcp_config.json        # MCP server configuration
├── pyproject.toml         # Project configuration and dependencies
└── README.md              # Documentation
```

### Key Components

- **TelegramBot**: Main bot class handling commands and messages
- **DatabaseManager**: Persistent storage for users and conversations
- **CerebrasClient**: Async API client for Cerebras AI models
- **CerebrasLLM**: LangChain integration for advanced workflows
- **MCPManager**: Manages multiple MCP server connections and tool execution
- **MCPConfigLoader**: Loads and validates JSON configuration for MCP servers

## 🔧 Development

### Code Quality

The project follows Python best practices:

- **Modular Design**: Clean separation of concerns
- **SOLID Principles**: Maintainable and extensible architecture
- **Async/Await**: Non-blocking I/O for optimal performance
- **Structured Logging**: Comprehensive debugging and monitoring
- **Type Hints**: Enhanced code clarity and IDE support

## 📊 Database Schema

### Core Tables

- **users**: User profiles and preferences
- **conversations**: Chat history and context per user
- **message_reactions**: Bot interaction tracking
- **rate_limits**: Fair usage enforcement with scope support

### Group Tables

- **groups**: Group chat information and metadata
- **group_settings**: Configurable group behavior and preferences
- **group_conversations**: Threaded conversation history for groups

## 🚨 Production Notes

For production deployment:

1. **Security**: Use proper secret management for API keys
2. **Monitoring**: Set up structured logging and alerts
3. **Backups**: Regular database backups for conversation data
4. **Scaling**: Monitor memory usage for large conversation histories
5. **Rate Limits**: Adjust based on usage patterns and API quotas

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Cerebras](https://cerebras.ai/) for ultra-fast AI inference
- [python-telegram-bot](https://github.com/python-telegram-bot/python-telegram-bot) for excellent Telegram integration
- [LangChain](https://langchain.com/) for LLM framework support
