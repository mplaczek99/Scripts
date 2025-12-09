#!/usr/bin/env bash
set -euo pipefail

########################################
# Config
########################################

# Public MEGA folder link (with built-in key)
MEGA_LINK="https://mega.nz/folder/RLx0XSaS#VH3EfBKktThKpgoudE69xg"

# Fixed destination directory
DEST_DIR="${HOME}/mega_sync/Adept/Adventurer"

########################################
# Pretty output helpers
########################################

if [[ -t 1 ]]; then
  BOLD="$(tput bold || true)"
  RESET="$(tput sgr0 || true)"
  GREEN="$(tput setaf 2 || true)"
  YELLOW="$(tput setaf 3 || true)"
  RED="$(tput setaf 1 || true)"
  BLUE="$(tput setaf 4 || true)"
else
  BOLD=""; RESET=""; GREEN=""; YELLOW=""; RED=""; BLUE=""
fi

log_info()  { printf "%s[%sINFO%s]%s %s\n"  "$BOLD" "$BLUE"   "$RESET" "$RESET" "$1"; }
log_warn()  { printf "%s[%sWARN%s]%s %s\n"  "$BOLD" "$YELLOW" "$RESET" "$RESET" "$1"; }
log_error() { printf "%s[%sERROR%s]%s %s\n" "$BOLD" "$RED"    "$RESET" "$RESET" "$1"; }
log_ok()    { printf "%s[%sOK%s]%s %s\n"    "$BOLD" "$GREEN"  "$RESET" "$RESET" "$1"; }

die() {
  log_error "$1"
  exit 1
}

########################################
# Checks
########################################

command -v mega-get   >/dev/null 2>&1 || die "mega-get (MEGAcmd) not found in PATH."
command -v mega-login >/dev/null 2>&1 || die "mega-login (MEGAcmd) not found in PATH."

mkdir -p "$DEST_DIR"

printf "\n%s==> MEGA incremental download for Adept/Adventurer%s\n\n" "$BOLD" "$RESET"
log_info "Destination directory: $DEST_DIR"

# Local stats
LOCAL_FILES="$(find "$DEST_DIR" -type f 2>/dev/null | wc -l | tr -d ' ')"
log_info "Current local files: $LOCAL_FILES"
echo

########################################
# Login to the public folder
########################################

log_info "Logging into MEGA public folder session..."
mega-logout >/dev/null 2>&1 || true

if ! mega-login "$MEGA_LINK" >/dev/null 2>&1; then
  die "Failed to login using the folder link. Check the URL or your network."
fi

log_ok "Logged into MEGA folder."
echo

########################################
# Verify remote + count files
########################################

if command -v mega-ls >/dev/null 2>&1; then
  log_info "Verifying access to remote root (/)..."
  if mega-ls / >/dev/null 2>&1; then
    log_ok "Remote folder is reachable."
  else
    log_warn "Could not list remote root with mega-ls; continuing anyway."
  fi
  echo
fi

REMOTE_FILES="?"
if command -v mega-find >/dev/null 2>&1; then
  if REMOTE_LIST="$(mega-find / -t file 2>/dev/null)"; then
    REMOTE_FILES="$(printf '%s\n' "$REMOTE_LIST" | wc -l | tr -d ' ')"
  fi
fi

if [[ "$REMOTE_FILES" != "?" ]]; then
  log_info "Approx. remote files in MEGA: $REMOTE_FILES"
else
  log_warn "Could not count remote files (mega-find not available or failed)."
fi
echo

########################################
# Dry run mode
########################################

if [[ "${DRY_RUN:-0}" = "1" ]]; then
  log_warn "DRY_RUN=1 set – not downloading, just showing the command."
  echo
  printf "%smega-get -m / \"%s\"%s\n" \
    "$BOLD" "$DEST_DIR" "$RESET"
  echo

  # Clean up session even in dry run
  mega-logout >/dev/null 2>&1 || true
  exit 0
fi

########################################
# Actual download / merge
########################################

log_info "Starting MEGA merge download..."
log_info "Existing local files are kept; only new/missing files will be fetched."
echo

if mega-get -m / "$DEST_DIR"; then
  echo
  log_ok "Download/merge completed successfully."
  NEW_LOCAL_FILES="$(find "$DEST_DIR" -type f 2>/dev/null | wc -l | tr -d ' ')"
  log_info "Local files before: $LOCAL_FILES"
  log_info "Local files now   : $NEW_LOCAL_FILES"
else
  echo
  die "mega-get reported an error."
fi

# Log out so the session isn't left around
mega-logout >/dev/null 2>&1 || true
log_info "Logged out of MEGA session."

