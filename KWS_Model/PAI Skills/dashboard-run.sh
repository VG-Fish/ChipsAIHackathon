#!/bin/sh
# Runtime launcher for the dashboard MCP server.
#
# Claude Code launches this over stdio (JSON-RPC on stdout), so stdout must stay
# byte-for-byte untouched. We only append the container's stderr to a log file
# next to this script — that log is the one readable trace of failures that
# otherwise vanish into Claude Code's invisible MCP subprocess stderr.
LOG="$(dirname "$0")/dashboard.log"
exec docker "$@" 2>>"$LOG"
