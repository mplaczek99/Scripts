#!/bin/bash
set -e

# Configuration
SNAPSHOT_DIR="/.snapshots"       # Local snapshot directory on root filesystem
DEST_DIR="/mnt/backup/snapshots" # Destination directory on separate disk
ROOT_SUBVOL="/"                  # Root subvolume path
MAX_SNAPSHOTS=10                 # Number of snapshots to keep

# Ensure the local snapshot directory exists
mkdir -p "$SNAPSHOT_DIR"

# Generate snapshot name with timestamp
TIMESTAMP=$(date +%Y%m%d%H%M%S)
SNAPSHOT_NAME="snapshot-$TIMESTAMP"
SNAPSHOT_PATH="$SNAPSHOT_DIR/$SNAPSHOT_NAME"

# Create the snapshot locally
echo "Creating local snapshot: $SNAPSHOT_PATH"
sudo btrfs subvolume snapshot -r "$ROOT_SUBVOL" "$SNAPSHOT_PATH"

# Ensure the destination directory exists
sudo mkdir -p "$DEST_DIR"

# Send the snapshot to the separate disk
echo "Sending snapshot to $DEST_DIR"
sudo btrfs send "$SNAPSHOT_PATH" | sudo btrfs receive "$DEST_DIR"

# Optionally, delete the local snapshot if not needed
# echo "Deleting local snapshot: $SNAPSHOT_PATH"
# sudo btrfs subvolume delete "$SNAPSHOT_PATH"

# Rotate snapshots on the destination disk, keeping only the most recent $MAX_SNAPSHOTS
cd "$DEST_DIR"

# Get a list of snapshots, sorted by modification time (newest first)
readarray -t SNAPSHOTS < <(ls -1dt snapshot-*)

# Delete snapshots exceeding the maximum allowed
if [ "${#SNAPSHOTS[@]}" -gt "$MAX_SNAPSHOTS" ]; then
    SNAPSHOTS_TO_DELETE=("${SNAPSHOTS[@]:$MAX_SNAPSHOTS}")
    for SNAPSHOT in "${SNAPSHOTS_TO_DELETE[@]}"; do
        echo "Deleting old snapshot on destination: $SNAPSHOT"
        sudo btrfs subvolume delete "$DEST_DIR/$SNAPSHOT"
    done
fi

echo "Snapshot transfer and rotation completed successfully."
