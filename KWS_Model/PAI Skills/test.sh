#!/bin/sh
# Shell tests for install.sh / uninstall.sh
# No external dependencies — stubs docker via PATH.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
INSTALL="$SCRIPT_DIR/install.sh"
UNINSTALL="$SCRIPT_DIR/uninstall.sh"

PASS=0
FAIL=0

_pass() { PASS=$((PASS+1)); printf 'ok %d - %s\n' "$((PASS+FAIL))" "$1"; }
_fail() { FAIL=$((FAIL+1)); printf 'not ok %d - %s\n' "$((PASS+FAIL))" "$1"; [ -n "$2" ] && printf '  # %s\n' "$2"; }

assert_file_exists() { [ -f "$1" ] && _pass "$2" || _fail "$2" "file not found: $1"; }
assert_dir_exists()  { [ -d "$1" ] && _pass "$2" || _fail "$2" "dir not found: $1"; }
assert_not_exists()  { [ ! -e "$1" ] && _pass "$2" || _fail "$2" "expected absent: $1"; }
assert_contains() {
  printf '%s' "$1" | grep -qF "$2" && _pass "$3" || _fail "$3" "expected to contain: $2"
}
assert_not_contains() {
  ! printf '%s' "$1" | grep -qF "$2" && _pass "$3" || _fail "$3" "expected NOT to contain: $2"
}

# Creates an isolated tmpdir, sets up fake docker on PATH, and creates the
# package-level fixture (skills/dashboard/SKILL.md) that install.sh expects to
# find next to itself.
# Must be called before running any script. Exports TMP and ORIG_PATH.
_setup() {
  ORIG_PATH="$PATH"
  TMP=$(mktemp -d)
  mkdir -p "$TMP/bin"
  # Fake docker, env-controlled so tests can simulate a failed pull / bad image:
  #   pull … -> exit $FAKE_PULL_EXIT (default 0)
  #   run  … -> exit $FAKE_RUN_EXIT  (default 0)
  #   everything else (rmi) -> logged to docker-calls, exit 0
  cat > "$TMP/bin/docker" <<STUB
#!/bin/sh
printf '%s\n' "\$*" >> "$TMP/docker-calls"
case "\$1" in
  pull) exit "\${FAKE_PULL_EXIT:-0}" ;;
  run)  exit "\${FAKE_RUN_EXIT:-0}" ;;
esac
STUB
  chmod +x "$TMP/bin/docker"
  export PATH="$TMP/bin:$ORIG_PATH"
  # Reset failure-injection vars between tests.
  unset FAKE_PULL_EXIT FAKE_RUN_EXIT
  # Package asset install.sh references via $(dirname $0)/…
  mkdir -p "$SCRIPT_DIR/skills/dashboard"
  [ -f "$SCRIPT_DIR/skills/dashboard/SKILL.md" ] || \
    printf '# Dashboard skill (test fixture)\n' \
      > "$SCRIPT_DIR/skills/dashboard/SKILL.md"
}

_teardown() {
  rm -rf "$TMP"
  export PATH="$ORIG_PATH"
}

# ---------------------------------------------------------------------------
# Test 1 — tracer bullet: install creates .mcp.json with correct port
# ---------------------------------------------------------------------------
t1_install_writes_port() {
  _setup
  cd "$TMP"
  "$INSTALL" --port 3010 >/dev/null 2>&1
  MCP=$(cat "$TMP/.mcp.json")
  assert_file_exists "$TMP/.mcp.json" "t1: .mcp.json created"
  assert_contains "$MCP" "3010" "t1: port 3010 in .mcp.json"
  _teardown
}

# ---------------------------------------------------------------------------
# Test 2 — install pulls the published image from Docker Hub
# ---------------------------------------------------------------------------
t2_install_pulls_image() {
  _setup
  cd "$TMP"
  "$INSTALL" --port 3002 >/dev/null 2>&1
  CALLS=$(cat "$TMP/docker-calls" 2>/dev/null || true)
  assert_contains "$CALLS" "pull rorrybrenner/perforated_dashboard_mcp:v0.1.0" \
    "t2: docker pull called with pinned image"
  _teardown
}

# ---------------------------------------------------------------------------
# Test 3 — install copies SKILL.md into .claude/skills/dashboard/
# ---------------------------------------------------------------------------
t3_install_copies_skill() {
  _setup
  cd "$TMP"
  "$INSTALL" --port 3002 >/dev/null 2>&1
  assert_file_exists "$TMP/.claude/skills/dashboard/SKILL.md" "t3: SKILL.md copied"
  _teardown
}

# ---------------------------------------------------------------------------
# Test 4 — install is idempotent (run twice, no duplicate key)
# ---------------------------------------------------------------------------
t4_install_idempotent() {
  _setup
  cd "$TMP"
  "$INSTALL" --port 3002 >/dev/null 2>&1
  "$INSTALL" --port 3002 >/dev/null 2>&1
  COUNT=$(grep -c '"dashboard"' "$TMP/.mcp.json" || true)
  [ "$COUNT" -eq 1 ] && _pass "t4: dashboard key appears exactly once" \
    || _fail "t4: dashboard key appears exactly once" "found $COUNT occurrences"
  _teardown
}

# ---------------------------------------------------------------------------
# Test 5 — install preserves other keys in .mcp.json
# ---------------------------------------------------------------------------
t5_install_preserves_keys() {
  _setup
  cd "$TMP"
  printf '{"someOtherKey":"value123"}\n' > "$TMP/.mcp.json"
  "$INSTALL" --port 3002 >/dev/null 2>&1
  MCP=$(cat "$TMP/.mcp.json")
  assert_contains "$MCP" "someOtherKey" "t5: other key preserved"
  assert_contains "$MCP" "value123"    "t5: other value preserved"
  _teardown
}

# ---------------------------------------------------------------------------
# Test 6 — uninstall removes dashboard key, leaves others intact
# ---------------------------------------------------------------------------
t6_uninstall_removes_dashboard_key() {
  _setup
  cd "$TMP"
  printf '{"mcpServers":{"dashboard":{"command":"docker"},"other":{"command":"foo"}}}\n' \
    > "$TMP/.mcp.json"
  "$UNINSTALL" >/dev/null 2>&1
  MCP=$(cat "$TMP/.mcp.json")
  assert_not_contains "$MCP" '"dashboard"' "t6: dashboard key removed"
  assert_contains "$MCP" '"other"'         "t6: other mcpServer preserved"
  _teardown
}

# ---------------------------------------------------------------------------
# Test 7 — uninstall removes .claude/skills/dashboard/
# ---------------------------------------------------------------------------
t7_uninstall_removes_skill_dir() {
  _setup
  cd "$TMP"
  mkdir -p "$TMP/.claude/skills/dashboard"
  printf '# skill\n' > "$TMP/.claude/skills/dashboard/SKILL.md"
  printf '{"mcpServers":{"dashboard":{}}}\n' > "$TMP/.mcp.json"
  "$UNINSTALL" >/dev/null 2>&1
  assert_not_exists "$TMP/.claude/skills/dashboard" "t7: skill dir removed"
  _teardown
}

# ---------------------------------------------------------------------------
# Test 8 — uninstall calls `docker rmi rorrybrenner/perforated_dashboard_mcp:v0.1.0`
# ---------------------------------------------------------------------------
t8_uninstall_calls_docker_rmi() {
  _setup
  cd "$TMP"
  printf '{"mcpServers":{"dashboard":{}}}\n' > "$TMP/.mcp.json"
  "$UNINSTALL" >/dev/null 2>&1
  CALLS=$(cat "$TMP/docker-calls" 2>/dev/null || true)
  assert_contains "$CALLS" "rmi rorrybrenner/perforated_dashboard_mcp:v0.1.0" "t8: docker rmi called"
  _teardown
}

# ---------------------------------------------------------------------------
# Test 9 — install writes writable perforated_tools mount
# ---------------------------------------------------------------------------
t9_install_writes_perforated_tools_mount() {
  _setup
  cd "$TMP"
  "$INSTALL" --port 3002 >/dev/null 2>&1
  MCP=$(cat "$TMP/.mcp.json")
  assert_contains "$MCP" ".perforated_tools:/perforated_tools:rw" \
    "t9: perforated_tools writable mount in args"
  _teardown
}

# ---------------------------------------------------------------------------
# Test 10 — workspace mount remains read-only after adding second mount
# ---------------------------------------------------------------------------
t10_install_workspace_still_readonly() {
  _setup
  cd "$TMP"
  "$INSTALL" --port 3002 >/dev/null 2>&1
  MCP=$(cat "$TMP/.mcp.json")
  assert_contains "$MCP" "/workspace:ro" "t10: workspace mount still ro"
  _teardown
}

# ---------------------------------------------------------------------------
# Test 11 — install creates an executable wrapper in .perforated_tools/
# ---------------------------------------------------------------------------
t11_install_creates_wrapper() {
  _setup
  cd "$TMP"
  "$INSTALL" --port 3002 >/dev/null 2>&1
  assert_file_exists "$TMP/.perforated_tools/dashboard-run.sh" "t11: wrapper created"
  [ -x "$TMP/.perforated_tools/dashboard-run.sh" ] \
    && _pass "t11: wrapper is executable" \
    || _fail "t11: wrapper is executable" "not executable"
  _teardown
}

# ---------------------------------------------------------------------------
# Test 12 — .mcp.json command points at the wrapper, not docker directly
# ---------------------------------------------------------------------------
t12_mcp_command_is_wrapper() {
  _setup
  cd "$TMP"
  "$INSTALL" --port 3002 >/dev/null 2>&1
  MCP=$(cat "$TMP/.mcp.json")
  assert_contains "$MCP" ".perforated_tools/dashboard-run.sh" "t12: command is wrapper path"
  _teardown
}

# ---------------------------------------------------------------------------
# Test 13 — a failed pull aborts install (no .mcp.json written)
# ---------------------------------------------------------------------------
t13_pull_failure_aborts() {
  _setup
  cd "$TMP"
  FAKE_PULL_EXIT=1 "$INSTALL" --port 3002 >/dev/null 2>&1
  assert_not_exists "$TMP/.mcp.json" "t13: install aborted on failed pull"
  _teardown
}

# ---------------------------------------------------------------------------
# Test 14 — smoke-test failure aborts install (no .mcp.json written)
# ---------------------------------------------------------------------------
t14_smoke_failure_aborts() {
  _setup
  cd "$TMP"
  FAKE_RUN_EXIT=1 "$INSTALL" --port 3002 >/dev/null 2>&1
  assert_not_exists "$TMP/.mcp.json" "t14: install aborted on smoke-test failure"
  _teardown
}

# ---------------------------------------------------------------------------
# Test 15 — uninstall removes the wrapper but preserves sibling user data
# ---------------------------------------------------------------------------
t15_uninstall_removes_wrapper_keeps_data() {
  _setup
  cd "$TMP"
  mkdir -p "$TMP/.perforated_tools/exports"
  printf '#!/bin/sh\n' > "$TMP/.perforated_tools/dashboard-run.sh"
  printf 'model-data\n' > "$TMP/.perforated_tools/exports/model.onnx"
  printf '{"mcpServers":{"dashboard":{}}}\n' > "$TMP/.mcp.json"
  "$UNINSTALL" >/dev/null 2>&1
  assert_not_exists "$TMP/.perforated_tools/dashboard-run.sh" "t15: wrapper removed"
  assert_file_exists "$TMP/.perforated_tools/exports/model.onnx" "t15: user data preserved"
  _teardown
}

# ---------------------------------------------------------------------------
# Test 16 — custom artifact root is mounted and used by the launcher
# ---------------------------------------------------------------------------
t16_install_custom_artifact_root() {
  _setup
  cd "$TMP"
  mkdir -p "$TMP/outputs/pai"
  "$INSTALL" --artifact-root "$TMP/outputs/pai" >/dev/null 2>&1
  MCP=$(cat "$TMP/.mcp.json")
  assert_contains "$MCP" "$TMP/outputs/pai/dashboard-run.sh" "t16: custom launcher path"
  assert_contains "$MCP" "$TMP/outputs/pai:/perforated_tools:rw" "t16: custom runtime mount"
  assert_file_exists "$TMP/outputs/pai/dashboard-run.sh" "t16: custom wrapper created"
  _teardown
}

# ---------------------------------------------------------------------------
# Test 17 — environment fallback selects the custom artifact root
# ---------------------------------------------------------------------------
t17_install_env_artifact_root() {
  _setup
  cd "$TMP"
  mkdir -p "$TMP/env-root"
  PAI_ARTIFACT_ROOT="$TMP/env-root" "$INSTALL" >/dev/null 2>&1
  MCP=$(cat "$TMP/.mcp.json")
  assert_contains "$MCP" "$TMP/env-root:/perforated_tools:rw" "t17: environment runtime mount"
  _teardown
}

# ---------------------------------------------------------------------------
# Test 18 — custom uninstall preserves user artifacts
# ---------------------------------------------------------------------------
t18_uninstall_custom_preserves_artifacts() {
  _setup
  cd "$TMP"
  mkdir -p "$TMP/outputs/pai/exports"
  printf 'user-data\n' > "$TMP/outputs/pai/exports/model.onnx"
  printf '#!/bin/sh\n' > "$TMP/outputs/pai/dashboard-run.sh"
  printf 'log\n' > "$TMP/outputs/pai/dashboard.log"
  printf '{"mcpServers":{"dashboard":{}}}\n' > "$TMP/.mcp.json"
  "$UNINSTALL" --artifact-root "$TMP/outputs/pai" >/dev/null 2>&1
  assert_file_exists "$TMP/outputs/pai/exports/model.onnx" "t18: custom user artifact preserved"
  assert_not_exists "$TMP/outputs/pai/dashboard-run.sh" "t18: custom wrapper removed"
  _teardown
}

# ---------------------------------------------------------------------------
# Run all tests
# ---------------------------------------------------------------------------
t1_install_writes_port
t2_install_pulls_image
t3_install_copies_skill
t4_install_idempotent
t5_install_preserves_keys
t6_uninstall_removes_dashboard_key
t7_uninstall_removes_skill_dir
t8_uninstall_calls_docker_rmi
t9_install_writes_perforated_tools_mount
t10_install_workspace_still_readonly
t11_install_creates_wrapper
t12_mcp_command_is_wrapper
t13_pull_failure_aborts
t14_smoke_failure_aborts
t15_uninstall_removes_wrapper_keeps_data
t16_install_custom_artifact_root
t17_install_env_artifact_root
t18_uninstall_custom_preserves_artifacts

printf '\n%d passed, %d failed\n' "$PASS" "$FAIL"
[ "$FAIL" -eq 0 ]
