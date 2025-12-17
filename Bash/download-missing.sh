#!/usr/bin/env bash
set -Eeuo pipefail
IFS=$'\n\t'

########################################
# Defaults (override via env or flags)
########################################
MEGA_LINK="${MEGA_LINK:-https://mega.nz/folder/RLx0XSaS#VH3EfBKktThKpgoudE69xg}"
DEST_DIR="${DEST_DIR:-$HOME/mega_sync/Adept/Adventurer}"

# Mode:
#   default (SAFE): download directly from the public link (no mega-login/mega-logout)
#   session mode:   mega-login to the folder link, then operate on REMOTE_PATH (needed for mega-find counts)
USE_SESSION="${USE_SESSION:-0}"
REMOTE_PATH="${REMOTE_PATH:-/}"

DRY_RUN="${DRY_RUN:-0}"
VERBOSE="${VERBOSE:-0}"
COUNT_REMOTE="${COUNT_REMOTE:-0}"

########################################
# Pretty output helpers
########################################
if [[ -t 1 ]]; then
  BOLD="$(tput bold 2>/dev/null || true)"
  RESET="$(tput sgr0 2>/dev/null || true)"
  GREEN="$(tput setaf 2 2>/dev/null || true)"
  YELLOW="$(tput setaf 3 2>/dev/null || true)"
  RED="$(tput setaf 1 2>/dev/null || true)"
  BLUE="$(tput setaf 4 2>/dev/null || true)"
else
  BOLD=""; RESET=""; GREEN=""; YELLOW=""; RED=""; BLUE=""
fi

log_info()  { printf "%s[%sINFO%s]%s %s\n"  "$BOLD" "$BLUE"   "$RESET" "$RESET" "$*"; }
log_warn()  { printf "%s[%sWARN%s]%s %s\n"  "$BOLD" "$YELLOW" "$RESET" "$RESET" "$*"; }
log_error() { printf "%s[%sERROR%s]%s %s\n" "$BOLD" "$RED"    "$RESET" "$RESET" "$*"; }
log_ok()    { printf "%s[%sOK%s]%s %s\n"    "$BOLD" "$GREEN"  "$RESET" "$RESET" "$*"; }

die() { log_error "$*"; exit 1; }

need() {
  command -v "$1" >/dev/null 2>&1 || die "Missing dependency: $1"
}

usage() {
  cat <<EOF
Usage: $(basename "$0") [options]

Options:
  -l, --link URL          Public MEGA folder link (default: env MEGA_LINK)
  -d, --dest DIR          Destination dir (default: env DEST_DIR)
  -n, --dry-run           Print what would run, do nothing
  -v, --verbose           Verbose (also enables bash xtrace when run twice nicely)
  --count-remote          Try to count remote files (requires --session)
  --session               Use mega-login/mega-logout + REMOTE_PATH (NOT default; may affect your MEGAcmd session)
  -r, --remote-path PATH  Remote path in session mode (default: /)
  -h, --help              Show help

Examples:
  $(basename "$0")
  DRY_RUN=1 $(basename "$0")
  $(basename "$0") --session --count-remote
EOF
}

########################################
# Arg parsing
########################################
while [[ $# -gt 0 ]]; do
  case "$1" in
    -l|--link)        MEGA_LINK="$2"; shift 2 ;;
    -d|--dest)        DEST_DIR="$2"; shift 2 ;;
    -r|--remote-path) REMOTE_PATH="$2"; shift 2 ;;
    -n|--dry-run)     DRY_RUN=1; shift ;;
    -v|--verbose)     VERBOSE=1; shift ;;
    --count-remote)   COUNT_REMOTE=1; shift ;;
    --session)        USE_SESSION=1; shift ;;
    -h|--help)        usage; exit 0 ;;
    *) die "Unknown arg: $1 (use --help)" ;;
  esac
done

[[ "$VERBOSE" == "1" ]] && set -x

########################################
# Better error reporting than bare `set -e`
########################################
on_err() {
  local ec=$?
  set +x
  log_error "Failed (exit $ec) at line $1: $2"
  exit "$ec"
}
trap 'on_err "$LINENO" "$BASH_COMMAND"' ERR

########################################
# Checks / setup
########################################
need mega-get

# Only require login tooling if session mode is explicitly requested
if [[ "$USE_SESSION" == "1" ]]; then
  need mega-login
  need mega-logout
fi

mkdir -p "$DEST_DIR" || die "Cannot create destination: $DEST_DIR"

# Lock to prevent two runs trampling each other
LOCKFILE="$DEST_DIR/.download-missing.lock"
if command -v flock >/dev/null 2>&1; then
  exec 9>"$LOCKFILE"
  flock -n 9 || die "Another run is active (lock: $LOCKFILE)"
else
  log_warn "flock not found; skipping lock."
fi

cleanup() {
  # Only logout if we logged into a public folder session in this run
  if [[ "${_DID_LOGIN:-0}" == "1" ]]; then
    mega-logout >/dev/null 2>&1 || true
    log_info "Logged out of MEGA session."
  fi
}
trap cleanup EXIT INT TERM

printf "\n%s==> MEGA incremental download for Adept/Adventurer%s\n\n" "$BOLD" "$RESET"
log_info "Destination directory: $DEST_DIR"

LOCAL_FILES="$(find "$DEST_DIR" -type f 2>/dev/null | wc -l | tr -d ' ')"
log_info "Current local files: $LOCAL_FILES"
echo

########################################
# Optional: session login + remote counts
########################################
REMOTE_FILES="?"
if [[ "$COUNT_REMOTE" == "1" ]]; then
  if [[ "$USE_SESSION" != "1" ]]; then
    log_warn "--count-remote requires --session (skipping remote count)."
  else
    log_info "Logging into MEGA public folder session (session mode enabled)..."
    mega-login "$MEGA_LINK" >/dev/null
    _DID_LOGIN=1
    log_ok "Logged into MEGA folder."
    echo

    if command -v mega-ls >/dev/null 2>&1; then
      log_info "Verifying access to remote path ($REMOTE_PATH)..."
      mega-ls "$REMOTE_PATH" >/dev/null 2>&1 && log_ok "Remote folder is reachable." || log_warn "Could not list remote path."
      echo
    fi

    if command -v mega-find >/dev/null 2>&1; then
      # mega-find supports --type=d|f :contentReference[oaicite:2]{index=2}
      REMOTE_FILES="$(mega-find "$REMOTE_PATH" --type=f 2>/dev/null | wc -l | tr -d ' ')" || REMOTE_FILES="?"
    else
      REMOTE_FILES="?"
    fi

    if [[ "$REMOTE_FILES" != "?" ]]; then
      log_info "Approx. remote files in MEGA: $REMOTE_FILES"
    else
      log_warn "Could not count remote files (mega-find missing or failed)."
    fi
    echo
  fi
fi

########################################
# Dry run
########################################
if [[ "$DRY_RUN" == "1" ]]; then
  log_warn "DRY_RUN=1 set – not downloading."
  echo
  if [[ "$USE_SESSION" == "1" ]]; then
    printf "%smega-get -m %q %q%s\n" "$BOLD" "$REMOTE_PATH" "$DEST_DIR" "$RESET"
  else
    # MEGAcmd supports downloading shared links directly (no login/logout required). :contentReference[oaicite:3]{index=3}
    printf "%smega-get -m %q %q%s\n" "$BOLD" "$MEGA_LINK" "$DEST_DIR" "$RESET"
  fi
  echo
  exit 0
fi

########################################
# Actual download / merge
########################################
log_info "Starting MEGA merge download..."
log_info "Existing local files are kept; only new/missing files will be fetched."
echo

if [[ "$USE_SESSION" == "1" ]]; then
  mega-get -m "$REMOTE_PATH" "$DEST_DIR"
else
  mega-get -m "$MEGA_LINK" "$DEST_DIR"
fi

echo
log_ok "Download/merge completed successfully."
NEW_LOCAL_FILES="$(find "$DEST_DIR" -type f 2>/dev/null | wc -l | tr -d ' ')"
log_info "Local files before: $LOCAL_FILES"
log_info "Local files now   : $NEW_LOCAL_FILES"

