#!/bin/bash
# Docker management script for Synapse Telegram Bot

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if .env file exists
check_env() {
    if [ ! -f ".env" ]; then
        print_warning ".env file not found. Creating from .env.example..."
        if [ -f ".env.example" ]; then
            cp .env.example .env
            print_warning "Please edit .env file with your API keys before running the bot."
            exit 1
        else
            print_error ".env.example file not found!"
            exit 1
        fi
    fi
    
    # Check if required variables are set
    if ! grep -q "TELEGRAM_BOT_TOKEN=your_telegram_bot_token" .env && ! grep -q "CEREBRAS_API_KEY=your_cerebras_api_key" .env; then
        print_status ".env file appears to be configured."
    else
        print_warning "Please configure your API keys in .env file."
        exit 1
    fi
}

# Build Docker image
build() {
    print_status "Building Docker image..."
    docker build -t synapse-bot:latest .
    print_status "Docker image built successfully!"
}

# Build with UV
build_uv() {
    print_status "Building Docker image with UV..."
    docker build -f Dockerfile.uv -t synapse-bot:uv .
    print_status "Docker image with UV built successfully!"
}

# Run with docker-compose
start() {
    check_env
    print_status "Starting Synapse bot with docker-compose..."
    docker-compose up -d
    print_status "Bot started! Use 'docker-compose logs -f synapse-bot' to view logs."
}

# Stop docker-compose
stop() {
    print_status "Stopping Synapse bot..."
    docker-compose down
    print_status "Bot stopped."
}

# View logs
logs() {
    docker-compose logs -f synapse-bot
}

# Run manually
run() {
    check_env
    print_status "Running Synapse bot manually..."
    docker run -d \
        --name synapse-bot \
        --env-file .env \
        -v "$(pwd)/data:/app/data" \
        synapse-bot:latest
    print_status "Bot started! Use 'docker logs -f synapse-bot' to view logs."
}

# Show help
help() {
    echo "Synapse Bot Docker Management Script"
    echo ""
    echo "Usage: $0 [command]"
    echo ""
    echo "Commands:"
    echo "  build     Build Docker image using pip"
    echo "  build-uv  Build Docker image using UV"
    echo "  start     Start bot using docker-compose"
    echo "  stop      Stop bot using docker-compose"
    echo "  logs      View bot logs"
    echo "  run       Run bot manually (without docker-compose)"
    echo "  help      Show this help message"
    echo ""
}

# Main script logic
case "$1" in
    build)
        build
        ;;
    build-uv)
        build_uv
        ;;
    start)
        start
        ;;
    stop)
        stop
        ;;
    logs)
        logs
        ;;
    run)
        run
        ;;
    help|--help|-h)
        help
        ;;
    *)
        print_error "Unknown command: $1"
        help
        exit 1
        ;;
esac