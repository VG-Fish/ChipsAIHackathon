---
name: dashboard
description: Verify the MCP server is reachable and open the Dashboard in a browser tab. Use when the user invokes /dashboard or wants to start the dashboard.
---

# Dashboard

1. Call `dashboard_ping`. If the tool is unavailable or returns an error, tell the operator the MCP server is not connected and stop.
2. Open `http://localhost:3002` in the browser.
3. Tell the operator the dashboard is ready.
