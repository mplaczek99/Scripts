#!/bin/bash

# Variables
VG_NAME="data"
LV_NAME="root"
SNAP_NAME="snap"
SNAP_SIZE="100G"
SNAP_LIMIT=9 # Number of snapshots to keep
LOG_FILE="/var/log/auto_lvm_snapshot.log"

# Create a timestamp
TIMESTAMP=$(date +"%F")

# Function to log messages
log() {
    echo "$(date +"%F %T") - $1" >> $LOG_FILE
}

# Create the snapshot
log "Creating snapshot ${SNAP_NAME}_${TIMESTAMP} for volume ${VG_NAME}/${LV_NAME}"
sudo lvcreate -L $SNAP_SIZE -s -n ${SNAP_NAME}_${TIMESTAMP} /dev/$VG_NAME/$LV_NAME &>> $LOG_FILE

# Check if snapshot creation was successful
if [ $? -eq 0 ]; then
    log "Snapshot ${SNAP_NAME}_${TIMESTAMP} created successfully."
else
    log "Error: Failed to create snapshot ${SNAP_NAME}_${TIMESTAMP}."
    exit 1
fi

# Remove older snapshots, keeping only the last $SNAP_LIMIT snapshots
log "Cleaning up old snapshots. Keeping the last $SNAP_LIMIT snapshots."

SNAPSHOTS=$(lvs --noheadings -o lv_name,lv_time --sort lv_time | grep "^$SNAP_NAME" | awk '{print $1}' | head -n -$SNAP_LIMIT)

if [ -n "$SNAPSHOTS" ]; then
    for SNAPSHOT in $SNAPSHOTS; do
        sudo lvremove -f /dev/$VG_NAME/$SNAPSHOT &>> $LOG_FILE
        if [ $? -eq 0 ]; then
            log "Snapshot $SNAPSHOT removed successfully."
        else
            log "Error: Failed to remove snapshot $SNAPSHOT."
        fi
    done
else
    log "No snapshots to remove."
fi
