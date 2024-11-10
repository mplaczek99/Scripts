#!/bin/bash
set -e

# Configuration
SNAPSHOT_DIR="/.snapshots"       # Local snapshot directory on root filesystem
DEST_DIR="/mnt/backup/snapshots" # Destination directory on separate disk
ROOT_SUBVOL="/"                  # Root subvolume path
MAX_SNAPSHOTS=10                 # Number of snapshots to keep (minimum 2)

# Ensure the local snapshot directory exists
mkdir -p "$SNAPSHOT_DIR"

# Generate snapshot name with timestamp
TIMESTAMP=$(date +%Y%m%d%H%M%S)
SNAPSHOT_NAME="snapshot-$TIMESTAMP"
SNAPSHOT_PATH="$SNAPSHOT_DIR/$SNAPSHOT_NAME"

# Create the snapshot locally
echo "Creating local snapshot: $SNAPSHOT_PATH"
sudo btrfs subvolume snapshot -r "$ROOT_SUBVOL" "$SNAPSHOT_PATH"

# Find the previous snapshot
cd "$SNAPSHOT_DIR"
LOCAL_SNAPSHOTS=(snapshot-*)
PREVIOUS_SNAPSHOT="${LOCAL_SNAPSHOTS[1]}"

# Ensure the destination directory exists
sudo mkdir -p "$DEST_DIR"

# Check if parent snapshot exists on destination
if [ -n "$PREVIOUS_SNAPSHOT" ] && [ -d "$DEST_DIR/$PREVIOUS_SNAPSHOT" ]; then
    echo "Sending incremental snapshot to $DEST_DIR"
    sudo btrfs send -p "$SNAPSHOT_DIR/$PREVIOUS_SNAPSHOT" "$SNAPSHOT_PATH" | sudo btrfs receive "$DEST_DIR"
else
    echo "Parent snapshot not found on destination. Sending full snapshot."
    sudo btrfs send "$SNAPSHOT_PATH" | sudo btrfs receive "$DEST_DIR"
fi

# Delete older local snapshots, keeping the latest two
cd "$SNAPSHOT_DIR"
readarray -t LOCAL_SNAPSHOTS < <(ls -1dt snapshot-*)
if [ "${#LOCAL_SNAPSHOTS[@]}" -gt 2 ]; then
    LOCAL_SNAPSHOTS_TO_DELETE=("${LOCAL_SNAPSHOTS[@]:2}")
    for SNAPSHOT in "${LOCAL_SNAPSHOTS_TO_DELETE[@]}"; do
        if [ "$SNAPSHOT" != "$PREVIOUS_SNAPSHOT" ]; then
            echo "Deleting old local snapshot: $SNAPSHOT"
            sudo btrfs subvolume delete "$SNAPSHOT_DIR/$SNAPSHOT"
        else
            echo "Retaining parent snapshot locally: $SNAPSHOT"
        fi
    done
fi

# Rotate snapshots on the destination disk, keeping at least two snapshots
cd "$DEST_DIR"
readarray -t DEST_SNAPSHOTS < <(ls -1dt snapshot-*)

# Delete snapshots exceeding the maximum allowed, but keep necessary snapshots
if [ "${#DEST_SNAPSHOTS[@]}" -gt "$MAX_SNAPSHOTS" ]; then
    DEST_SNAPSHOTS_TO_DELETE=("${DEST_SNAPSHOTS[@]:$MAX_SNAPSHOTS}")
    for SNAPSHOT in "${DEST_SNAPSHOTS_TO_DELETE[@]}"; do
        if [ "$SNAPSHOT" != "$PREVIOUS_SNAPSHOT" ]; then
            echo "Deleting old snapshot on destination: $SNAPSHOT"
            sudo btrfs subvolume delete "$DEST_DIR/$SNAPSHOT"
        else
            echo "Retaining parent snapshot on destination: $SNAPSHOT"
        fi
    done
fi

echo "Snapshot transfer and rotation completed successfully."
