#!/bin/sh
set -e

PORT=3002
ARTIFACT_ROOT=""

while [ $# -gt 0 ]; do
  case "$1" in
    --port)
      PORT="$2"
      shift 2
      ;;
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

SELF_DIR="$(cd "$(dirname "$0")" && pwd)"
MCP_FILE=".mcp.json"
SKILLS_DIR=".claude/skills"
PWD_ABS="$(pwd)"
IMAGE="rorrybrenner/perforated_dashboard_mcp:v0.1.0"
if [ -z "$ARTIFACT_ROOT" ]; then
  ARTIFACT_ROOT="${PAI_ARTIFACT_ROOT:-$PWD_ABS/.perforated_tools}"
fi
case "$ARTIFACT_ROOT" in
  /*) ;;
  *) ARTIFACT_ROOT="$PWD_ABS/$ARTIFACT_ROOT" ;;
esac

# --- pull the published multi-arch image from Docker Hub. Docker selects the
# manifest matching this host's architecture, so an exec-format mismatch can't
# happen. Surface network/registry failures here, at install time, rather than
# in the invisible MCP subprocess later.
if ! docker pull "$IMAGE"; then
  printf 'error: failed to pull %s — check your network and that Docker is running\n' "$IMAGE" >&2
  exit 1
fi

# --- smoke test: verify the image actually runs AND its deps import on this
# machine, before we wire up the MCP config. Importing the module exercises the
# heavy imports (torch/onnx/ml plugins) without starting the blocking server.
if ! docker run --rm --entrypoint python "$IMAGE" -c "import mcp_server.server" >/dev/null 2>&1; then
  echo "error: $IMAGE failed to start on this machine. Details:" >&2
  docker run --rm --entrypoint python "$IMAGE" -c "import mcp_server.server" 2>&1 | sed 's/^/  /' >&2
  exit 1
fi

# --- install the runtime launcher that logs container stderr to a readable file.
mkdir -p "$ARTIFACT_ROOT"
cp "$SELF_DIR/dashboard-run.sh" "$ARTIFACT_ROOT/dashboard-run.sh"
chmod +x "$ARTIFACT_ROOT/dashboard-run.sh"

if [ ! -f "$MCP_FILE" ]; then
  echo '{}' > "$MCP_FILE"
fi

# Write or overwrite the dashboard mcpServers entry, preserving other keys.
python3 - "$MCP_FILE" "$PORT" "$PWD_ABS" "$IMAGE" "$ARTIFACT_ROOT" <<'EOF'
import json
import sys

file, port, cwd, image, artifact_root = sys.argv[1:6]
with open(file) as f:
    config = json.load(f)
config.setdefault("mcpServers", {})
config["mcpServers"]["dashboard"] = {
    "command": f"{artifact_root}/dashboard-run.sh",
    "args": [
        "run", "--rm", "-i",
        "-p", f"{port}:{port}",
        "-v", f"{cwd}:/workspace:ro",
        "-v", f"{artifact_root}:/perforated_tools:rw",
        "-e", "PAI_ARTIFACT_ROOT=/perforated_tools",
        image,
    ],
}
with open(file, "w") as f:
    json.dump(config, f, indent=2)
    f.write("\n")
EOF

mkdir -p "$SKILLS_DIR"
for skill_dir in "$SELF_DIR"/skills/*/; do
  skill_name="$(basename "$skill_dir")"
  rm -rf "$SKILLS_DIR/$skill_name"
  cp -R "$skill_dir" "$SKILLS_DIR/$skill_name"
done

echo "Installed dashboard MCP server on port $PORT"
echo "MCP config written to $MCP_FILE"
echo "Installed skills to $SKILLS_DIR"
echo "Runtime artifacts written to $ARTIFACT_ROOT"
