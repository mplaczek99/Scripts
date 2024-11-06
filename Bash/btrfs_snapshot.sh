#!/bin/bash
set -e

SNAPSHOT_DIR="/btrfs_snapshots"
ROOT_SUBVOL="/"  # Adjust if your root subvolume is different
MAX_SNAPSHOTS=5

# Ensure the snapshot directory exists
mkdir -p "$SNAPSHOT_DIR"

# Generate snapshot name with day of the month
SNAPSHOT_NAME="$SNAPSHOT_DIR/snapshot-$(date +%d)"

# If today's snapshot already exists, delete it
if [ -d "$SNAPSHOT_NAME" ]; then
    echo "Deleting existing snapshot: $SNAPSHOT_NAME"
    btrfs subvolume delete "$SNAPSHOT_NAME"
fi

# Create the new snapshot
echo "Creating snapshot: $SNAPSHOT_NAME"
btrfs subvolume snapshot -r "$ROOT_SUBVOL" "$SNAPSHOT_NAME"

# Rotate snapshots, keeping only the most recent $MAX_SNAPSHOTS
cd "$SNAPSHOT_DIR"

# Get a list of snapshots, sorted by modification time (newest first)
readarray -t SNAPSHOTS < <(ls -1dt snapshot-*)

# Delete snapshots exceeding the maximum allowed
if [ "${#SNAPSHOTS[@]}" -gt "$MAX_SNAPSHOTS" ]; then
    SNAPSHOTS_TO_DELETE=("${SNAPSHOTS[@]:$MAX_SNAPSHOTS}")
    for SNAPSHOT in "${SNAPSHOTS_TO_DELETE[@]}"; do
        echo "Deleting old snapshot: $SNAPSHOT"
        btrfs subvolume delete "$SNAPSHOT"
    done
fi

echo "Snapshot rotation complete. Current snapshots:"
ls -1d snapshot-*
