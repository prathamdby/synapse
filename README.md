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

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- UV package manager
- [Telegram Bot Token](https://t.me/botfather)
- [Cerebras API Key](https://cloud.cerebras.ai/)

### Installation

#### Option 1: Local Development

1. **Clone and setup**

   ```bash
   git clone https://github.com/prathamdby/synapse.git
   cd synapse
   uv sync
   ```

2. **Configure environment**

   ```bash
   cp .env.example .env
   # Edit .env with your tokens
   ```

3. **Run the bot**
   ```bash
   uv run python main.py
   ```

#### Option 2: Docker Deployment

1. **Clone the repository**

   ```bash
   git clone https://github.com/prathamdby/synapse.git
   cd synapse
   ```

2. **Configure environment**

   ```bash
   cp .env.example .env
   # Edit .env with your tokens
   ```

3. **Run with Docker Compose** (Recommended)

   ```bash
   docker-compose up -d
   ```

4. **Or build and run manually**

   ```bash
   # Build the image
   docker build -t synapse-bot .
   
   # Run the container
   docker run -d \
     --name synapse-bot \
     --env-file .env \
     -v $(pwd)/data:/app/data \
     synapse-bot
   ```

5. **Using UV in Docker** (Alternative)

   If you prefer to use UV in Docker:
   ```bash
   docker build -f Dockerfile.uv -t synapse-bot:uv .
   ```

## 🐳 Docker Deployment

### Quick Start with Docker

The easiest way to deploy Synapse is using Docker:

```bash
# Clone the repository
git clone https://github.com/prathamdby/synapse.git
cd synapse

# Copy and configure environment
cp .env.example .env
# Edit .env with your API keys

# Start with docker-compose (recommended)
docker-compose up -d

# View logs
docker-compose logs -f synapse-bot
```

### Docker Environment Variables

The container supports all the same environment variables as the local installation:

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `TELEGRAM_BOT_TOKEN` | Yes | - | Your Telegram bot token |
| `CEREBRAS_API_KEY` | Yes | - | Your Cerebras API key |
| `DATABASE_PATH` | No | `/app/data/bot_database.db` | Database file path |
| `LOG_LEVEL` | No | `INFO` | Logging level |
| `RATE_LIMIT_MESSAGES_PER_MINUTE` | No | `10` | Rate limiting |
| `RATE_LIMIT_WINDOW_SECONDS` | No | `60` | Rate limit window |

### Docker Volumes

- `/app/data` - Database and persistent data storage

### Building Custom Images

To build your own image:

```bash
# Standard build using pip
docker build -t my-synapse-bot .

# Alternative build using UV (requires internet access)
docker build -f Dockerfile.uv -t my-synapse-bot:uv .
```

### Development with Docker

For development, you can use the override configuration:

```bash
# Copy the development override
cp docker-compose.override.yml.example docker-compose.override.yml

# Start in development mode
docker-compose up
```

## ⚙️ Configuration

Create a `.env` file:

```env
# Required
TELEGRAM_BOT_TOKEN=your_telegram_bot_token
CEREBRAS_API_KEY=your_cerebras_api_key

# Optional
DATABASE_PATH=./bot_database.db
LOG_LEVEL=INFO
RATE_LIMIT_MESSAGES_PER_MINUTE=10
RATE_LIMIT_WINDOW_SECONDS=60
```

### Getting API Keys

- **Telegram**: Message [@BotFather](https://t.me/botfather), use `/newbot`
- **Cerebras**: Sign up at [cloud.cerebras.ai](https://cloud.cerebras.ai/)

## 🤖 Bot Commands

- `/start` - Welcome message and introduction
- `/help` - Detailed help and usage information
- `/reset` or `/clear` - Clear conversation history
- `/stats` - View usage statistics and preferences
- `/model` - Switch between available AI models

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

- **users**: User profiles and preferences
- **conversations**: Chat history and context per user
- **message_reactions**: Bot interaction tracking
- **rate_limits**: Fair usage enforcement

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
