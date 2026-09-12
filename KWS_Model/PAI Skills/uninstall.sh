#!/bin/sh
set -e

SELF_DIR="$(cd "$(dirname "$0")" && pwd)"
MCP_FILE=".mcp.json"
SKILLS_DIR=".claude/skills"
ARTIFACT_ROOT=""

while [ $# -gt 0 ]; do
  case "$1" in
    --artifact-root)
      ARTIFACT_ROOT="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1" >&2
      exit 1
      ;;
  esac
done

PWD_ABS="$(pwd)"
if [ -z "$ARTIFACT_ROOT" ]; then
  ARTIFACT_ROOT="${PAI_ARTIFACT_ROOT:-$PWD_ABS/.perforated_tools}"
fi
case "$ARTIFACT_ROOT" in
  /*) ;;
  *) ARTIFACT_ROOT="$PWD_ABS/$ARTIFACT_ROOT" ;;
esac

if [ -f "$MCP_FILE" ]; then
  python3 - "$MCP_FILE" <<'EOF'
import json
import sys

file = sys.argv[1]
with open(file) as f:
    config = json.load(f)
if "mcpServers" in config:
    config["mcpServers"].pop("dashboard", None)
with open(file, "w") as f:
    json.dump(config, f, indent=2)
    f.write("\n")
EOF
  echo "Removed dashboard entry from $MCP_FILE"
fi

for skill_dir in "$SELF_DIR"/skills/*/; do
  skill_name="$(basename "$skill_dir")"
  if [ -d "$SKILLS_DIR/$skill_name" ]; then
    rm -rf "$SKILLS_DIR/$skill_name"
    echo "Removed $SKILLS_DIR/$skill_name"
  fi
done

# Remove the runtime launcher + its log, but preserve .perforated_tools and any
# user data (exports, training runs) it holds.
rm -f "$ARTIFACT_ROOT/dashboard-run.sh" "$ARTIFACT_ROOT/dashboard.log"

docker rmi rorrybrenner/perforated_dashboard_mcp:v0.1.0
