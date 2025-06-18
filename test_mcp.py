import requests
import json

def discover_api():
    """Discover available endpoints of the MCP server."""
    base_url = "http://localhost:8000"
    
    # Try to get the API info
    try:
        response = requests.get(f"{base_url}/")
        print("Root endpoint response:", response.status_code)
        if response.status_code == 200:
            print(response.text[:200] + "..." if len(response.text) > 200 else response.text)
    except Exception as e:
        print(f"Error accessing root endpoint: {e}")
    
    # Try to get the OpenAPI docs
    try:
        response = requests.get(f"{base_url}/openapi.json")
        print("\nOpenAPI docs response:", response.status_code)
        if response.status_code == 200:
            api_spec = response.json()
            print("Available paths:")
            for path in api_spec.get("paths", {}):
                print(f"  {path}")
    except Exception as e:
        print(f"Error accessing OpenAPI docs: {e}")
    
    # Try common MCP endpoints
    common_endpoints = [
        "/mcp",
        "/mcp/",
        "/mcp/tools",
        "/mcp/tool/list_files",
        "/tools",
        "/docs",
        "/redoc"
    ]
    
    print("\nTesting common endpoints:")
    for endpoint in common_endpoints:
        try:
            response = requests.get(f"{base_url}{endpoint}")
            print(f"  {endpoint}: {response.status_code}")
        except Exception as e:
            print(f"  {endpoint}: Error - {e}")

def test_list_files():
    """Test the list_files tool of the MCP server."""
    # Try different possible URLs for the list_files tool
    base_url = "http://localhost:8000"
    possible_urls = [
        f"{base_url}/mcp/tool/list_files",
        f"{base_url}/tools/list_files",
        f"{base_url}/mcp/tools/list_files",
        f"{base_url}/api/mcp/tool/list_files"
    ]
    
    for url in possible_urls:
        print(f"\nTrying URL: {url}")
        try:
            # Call without any arguments
            response = requests.post(url, json={})
            print("Response status:", response.status_code)
            
            if response.status_code == 200:
                print("Success! List files tool works without arguments.")
                files = response.json()
                print(f"Found {len(files)} files.")
                return
            else:
                print("Error:", response.text[:200] + "..." if len(response.text) > 200 else response.text)
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    print("=== MCP Server API Discovery ===")
    discover_api()
    
    print("\n=== Testing list_files tool ===")
    test_list_files()
