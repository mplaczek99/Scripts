#!/bin/bash

# Check to make sure the user has entered exactly two arguments.
if [ $# -ne 2 ]
then
    echo "Usage: backup.sh <source_directory> <target_directory>"
    echo "Please try again."
    exit 1
fi

# Check to see if rsync is installed.
if ! command -v rsync > /dev/null 2>&1
then
    echo "This script requires rsync to be installed."
    echo "Please use your distribution's package manager to install it and try again."
    exit 2
fi

# Capture the current date, and store it in the format YYYY-MM-DD
current_date=$(date +%Y-%m-%d)

rsync_options="-avb --backup-dir $2/$current_date --delete"

$(which rsync) $rsync_options $1 $2/current >> backup_$current_date.log
