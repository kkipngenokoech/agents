from mcp.server.fastmcp import FastMCP

## create an MCP sever
server = FastMCP()

## tool implementation
@server.tool()
def hello(name: str) -> str:
    """Greet a person."""
    return f"Hello, {name}!"

## resource implementation
@server.resource()
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b

## prompt implementation
@server.prompt()
def greet_prompt(name: str) -> str:
    """Generate a greeting prompt."""
    return f"Please greet {name} warmly."

## 

if __name__ == "__main__":
    ## run the server
    server.run()
    
