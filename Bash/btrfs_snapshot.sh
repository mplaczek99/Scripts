#!/bin/bash
set -e

# Configuration
SNAPSHOT_DIR="/.snapshots"       # Local snapshot directory on root filesystem
DEST_DIR="/mnt/backup/snapshots" # Destination directory on separate disk
ROOT_SUBVOL="/"                  # Root subvolume path
MAX_SNAPSHOTS=10                 # Number of snapshots to keep (minimum 2)

# Determine if sudo is needed
if [ "$EUID" -ne 0 ]; then
    SUDO_CMD="sudo"
else
    SUDO_CMD=""
fi

# Ensure the local and destination snapshot directories exist
mkdir -p "$SNAPSHOT_DIR"
$SUDO_CMD mkdir -p "$DEST_DIR"

# Generate snapshot name with timestamp
TIMESTAMP=$(date +"%Y%m%d%H%M%S")
SNAPSHOT_NAME="snapshot-$TIMESTAMP"
SNAPSHOT_PATH="$SNAPSHOT_DIR/$SNAPSHOT_NAME"

# Function to create a snapshot
create_snapshot() {
    echo "Creating local snapshot: $SNAPSHOT_PATH"
    $SUDO_CMD btrfs subvolume snapshot -r "$ROOT_SUBVOL" "$SNAPSHOT_PATH"
}

# Function to find the previous snapshot
find_previous_snapshot() {
    readarray -t LOCAL_SNAPSHOTS < <(ls -1dt "$SNAPSHOT_DIR"/snapshot-* 2>/dev/null)
    if [ "${#LOCAL_SNAPSHOTS[@]}" -gt 1 ]; then
        PREVIOUS_SNAPSHOT=$(basename "${LOCAL_SNAPSHOTS[1]}")
    else
        PREVIOUS_SNAPSHOT=""
    fi
}

# Function to send the snapshot
send_snapshot() {
    if [ -n "$PREVIOUS_SNAPSHOT" ] && [ -d "$DEST_DIR/$PREVIOUS_SNAPSHOT" ]; then
        echo "Sending incremental snapshot to $DEST_DIR"
        $SUDO_CMD btrfs send -p "$SNAPSHOT_DIR/$PREVIOUS_SNAPSHOT" "$SNAPSHOT_PATH" | $SUDO_CMD btrfs receive "$DEST_DIR"
    else
        echo "Parent snapshot not found on destination. Sending full snapshot."
        $SUDO_CMD btrfs send "$SNAPSHOT_PATH" | $SUDO_CMD btrfs receive "$DEST_DIR"
    fi
}

# Function to delete old snapshots
delete_old_snapshots() {
    # Delete older local snapshots, keeping the latest two
    readarray -t LOCAL_SNAPSHOTS < <(ls -1dt "$SNAPSHOT_DIR"/snapshot-* 2>/dev/null)
    if [ "${#LOCAL_SNAPSHOTS[@]}" -gt 2 ]; then
        LOCAL_SNAPSHOTS_TO_DELETE=("${LOCAL_SNAPSHOTS[@]:2}")
        for SNAPSHOT_PATH in "${LOCAL_SNAPSHOTS_TO_DELETE[@]}"; do
            SNAPSHOT=$(basename "$SNAPSHOT_PATH")
            if [ "$SNAPSHOT" != "$PREVIOUS_SNAPSHOT" ]; then
                echo "Deleting old local snapshot: $SNAPSHOT"
                $SUDO_CMD btrfs subvolume delete "$SNAPSHOT_PATH"
            else
                echo "Retaining parent snapshot locally: $SNAPSHOT"
            fi
        done
    fi

    # Rotate snapshots on the destination disk, keeping at most MAX_SNAPSHOTS
    readarray -t DEST_SNAPSHOTS < <(ls -1dt "$DEST_DIR"/snapshot-* 2>/dev/null)
    if [ "${#DEST_SNAPSHOTS[@]}" -gt "$MAX_SNAPSHOTS" ]; then
        DEST_SNAPSHOTS_TO_DELETE=("${DEST_SNAPSHOTS[@]:$MAX_SNAPSHOTS}")
        for SNAPSHOT_PATH in "${DEST_SNAPSHOTS_TO_DELETE[@]}"; do
            SNAPSHOT=$(basename "$SNAPSHOT_PATH")
            if [ "$SNAPSHOT" != "$PREVIOUS_SNAPSHOT" ]; then
                echo "Deleting old snapshot on destination: $SNAPSHOT"
                $SUDO_CMD btrfs subvolume delete "$DEST_DIR/$SNAPSHOT"
            else
                echo "Retaining parent snapshot on destination: $SNAPSHOT"
            fi
        done
    fi
}

# Main script execution
create_snapshot
find_previous_snapshot
send_snapshot
delete_old_snapshots

echo "Snapshot transfer and rotation completed successfully."
