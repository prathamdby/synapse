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

## 🚀 Quick Start

### Prerequisites

- [Telegram Bot Token](https://t.me/botfather)
- [Cerebras API Key](https://cloud.cerebras.ai/)

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

### Group Commands (Admin Only)

- `/group_mode` - Set shared or personal memory mode
- `/group_settings` - View all group configuration options
- `/group_reset` - Clear group conversation history
- `/group_stats` - View group statistics and activity

## 🏗️ Architecture

```
synapse/
├── main.py                 # Core bot logic and Telegram integration
├── database.py            # SQLite database management
├── cerebras_client.py     # Cerebras AI API client with async support
├── langchain_cerebras.py  # LangChain wrapper for Cerebras integration
├── pyproject.toml         # Project configuration and dependencies
└── README.md              # Documentation
```

### Key Components

- **TelegramBot**: Main bot class handling commands and messages
- **DatabaseManager**: Persistent storage for users and conversations
- **CerebrasClient**: Async API client for Cerebras AI models
- **CerebrasLLM**: LangChain integration for advanced workflows

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
