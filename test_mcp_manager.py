#!/usr/bin/env python3
"""Test the MCP manager with the fixes."""

import asyncio
import os
from dotenv import load_dotenv

# Load environment
load_dotenv()

from mcp_manager import MCPManager

async def test_mcp_manager():
    """Test the MCP manager initialization."""
    print("🧪 Testing MCP Manager...")
    
    try:
        # Create MCP manager
        manager = MCPManager(config_path="./mcp_config.json")
        
        print("✅ MCPManager created")
        
        # Load config and connect
        await manager.load_config_and_connect()
        
        print("✅ Configuration loaded and connections attempted")
        
        # Check status
        status = manager.get_system_status()
        print(f"📊 Connected servers: {status.connected_servers}/{status.total_servers}")
        print(f"📊 Available tools: {status.available_tools}")
        
        # List tools
        tools = manager.get_available_tools()
        print(f"🛠️  Tools found: {len(tools)}")
        
        for tool in tools:
            print(f"  📋 {tool.name} ({tool.server_name}): {tool.description[:100]}...")
        
        # Test tool execution if tools are available
        if tools:
            print("\n🧪 Testing tool execution...")
            first_tool = tools[0]
            
            if first_tool.name == "fetch":
                result = await manager.execute_tool("fetch", {"url": "https://httpbin.org/json"})
                print(f"✅ Tool execution result: {result.success}")
                if result.success:
                    print(f"📄 Result preview: {str(result.result)[:200]}...")
                else:
                    print(f"❌ Error: {result.error}")
        
        # Cleanup
        await manager.shutdown()
        print("✅ Manager shutdown complete")
        
        return status.available_tools > 0
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_mcp_manager())
    print(f"\n🎯 Test result: {'✅ SUCCESS' if success else '❌ FAILED'}")
    exit(0 if success else 1)
