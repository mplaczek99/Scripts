#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Variables
VG_NAME="data"
LV_NAME="root"
SNAP_NAME="snap"
SNAP_SIZE="100G"
SNAP_LIMIT=5  # Number of snapshots to keep

# Create a timestamp
TIMESTAMP=$(date +"%F")
SNAPSHOT_NAME="${SNAP_NAME}_${TIMESTAMP}"

# Ensure the script is run as root
if [[ $EUID -ne 0 ]]; then
    echo "This script must be run as root." >&2
    exit 1
fi

# Cleanup: Remove older snapshots, keeping only the last $SNAP_LIMIT
echo "Cleaning up old snapshots. Keeping the last $SNAP_LIMIT snapshots."

# Get list of snapshots sorted by creation time, excluding the ones to keep
SNAPSHOTS_TO_REMOVE=$(lvs --noheadings -o lv_name,lv_time --sort lv_time \
    --select "lv_name =~ ^${SNAP_NAME}_.* && vg_name='${VG_NAME}'" | \
    awk '{print $1}' | tail -n +$((SNAP_LIMIT+1)))

# Only proceed if there are snapshots to remove
if [[ -n "$SNAPSHOTS_TO_REMOVE" ]]; then
    echo "$SNAPSHOTS_TO_REMOVE" | xargs -I {} lvremove -f "/dev/$VG_NAME/{}" && \
    echo "Old snapshots removed successfully."
else
    echo "No old snapshots to remove."
fi

# Now, create the snapshot after cleaning up old ones
echo "Creating snapshot $SNAPSHOT_NAME for volume ${VG_NAME}/${LV_NAME}"
lvcreate -L "$SNAP_SIZE" -s -n "$SNAPSHOT_NAME" "/dev/$VG_NAME/$LV_NAME"
echo "Snapshot $SNAPSHOT_NAME created successfully."
