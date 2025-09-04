#!/usr/bin/env python3
import asyncio
from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

async def test():
    try:
        async with stdio_client(
            StdioServerParameters(command="uvx", args=["mcp-server-fetch"])
        ) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                tools = await session.list_tools()
                print(f"✅ Tools found: {len(tools.tools)}")
                for tool in tools.tools:
                    print(f"  📋 {tool.name}")
                    # Test the schema access
                    if hasattr(tool, "inputSchema"):
                        print(f"  📄 Schema type: {type(tool.inputSchema)}")
                        print(f"  📄 Schema: {tool.inputSchema}")
                return True
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    success = asyncio.run(test())
    print("✅ SUCCESS" if success else "❌ FAILED")
