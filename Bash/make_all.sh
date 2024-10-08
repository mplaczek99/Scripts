#!/bin/bash

# Replace with the path to your AUR directory
AUR_DIR="$HOME/AUR"

# Check if the AUR directory exists
if [ ! -d "$AUR_DIR" ]; then
    echo "Error: AUR directory '$AUR_DIR' does not exist."
    exit 1
fi

# Ask for sudo password upfront
echo "Requesting sudo access..."
if sudo -v; then
    echo "Sudo credentials cached."
else
    echo "Failed to obtain sudo credentials."
    exit 1
fi

# Keep-alive: update existing sudo time stamp until script has finished
( while true; do sudo -n true; sleep 60; done ) 2>/dev/null &

# PID of the keep-alive process
SUDO_KEEPALIVE_PID=$!

# Function to clean up the keep-alive process on exit
cleanup() {
    kill "$SUDO_KEEPALIVE_PID"
}
trap cleanup EXIT

# Find all Git repositories containing a PKGBUILD file and store them in an array
mapfile -t git_dirs < <(find "$AUR_DIR" -type d -name ".git" 2>/dev/null)

# Iterate over each Git repository
for git_dir in "${git_dirs[@]}"; do
    # Get the parent directory of the .git folder
    repo_dir="$(dirname "$git_dir")"

    # Check if PKGBUILD exists in the repository
    if [ -f "$repo_dir/PKGBUILD" ]; then
        echo "Processing repository: $repo_dir"
        cd "$repo_dir" || { echo "Failed to enter directory $repo_dir"; continue; }

        # Execute makepkg with the specified flags
        makepkg -srcif --noconfirm

        # Check if makepkg was successful
        if [ $? -ne 0 ]; then
            echo "makepkg failed in $repo_dir"
        else
            echo "makepkg succeeded in $repo_dir"
        fi
    else
        echo "No PKGBUILD found in $repo_dir, skipping."
    fi
done

# Clean up the sudo keep-alive process
cleanup
