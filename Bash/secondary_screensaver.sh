#!/bin/bash

# --- Configuration ---
IDLE_TIMEOUT_MS=300000  # Idle time in milliseconds (e.g., 300000 = 5 minutes)
CHECK_INTERVAL_S=10     # How often to check idle time (seconds)

# --- Second Monitor Geometry (!!! MODIFY THESE from xrandr output !!!) ---
MONITOR_WIDTH=1920
MONITOR_HEIGHT=1080
MONITOR_X_OFFSET=1920
MONITOR_Y_OFFSET=0
# Construct the geometry string for wmctrl's -e option
# Format: gravity,X,Y,Width,Height (0 for default gravity)
MONITOR_GEOMETRY="0,${MONITOR_X_OFFSET},${MONITOR_Y_OFFSET},${MONITOR_WIDTH},${MONITOR_HEIGHT}"

# --- Screensaver Program (!!! MODIFY THIS if needed !!!) ---
SCREENSAVER_CMD="/usr/libexec/xscreensaver/glmatrix" # Command to run the screensaver
SCREENSAVER_NAME="glmatrix" # The process name or a unique part of the command to find it later

# --- Internal Variables ---
screensaver_pid=0 # Stores the PID of the running screensaver

# --- Functions ---
start_screensaver() {
    if [[ $screensaver_pid -eq 0 ]]; then
        echo "Starting screensaver..."
        # Launch in the background
        $SCREENSAVER_CMD &
        # Store the PID of the last background process
        screensaver_pid=$!
        echo "Screensaver PID: $screensaver_pid"

        # Give it a moment to create its window
        sleep 2

        # Find the window ID using wmctrl (might need adjustment based on window title)
        # Try matching by PID first if wmctrl supports it, otherwise by name
        # Use -F for exact title match, or just part of the title. Adjust "glmatrix" if needed.
        window_id=$(wmctrl -lp | grep $screensaver_pid | awk '{print $1}')

        if [[ -z "$window_id" ]]; then
             echo "Could not find window ID for PID $screensaver_pid. Trying title..."
             # Fallback: Match window title (adjust 'glmatrix' if the window title is different)
             window_id=$(wmctrl -l | grep -i "$SCREENSAVER_NAME" | head -n 1 | awk '{print $1}')
        fi


        if [[ -n "$window_id" ]]; then
            echo "Found window ID: $window_id"
            # Remove decorations (optional, might help with fullscreen)
            # wmctrl -ir "$window_id" -b add,hidden,shaded
            # Move and resize to the target monitor
            wmctrl -ir "$window_id" -e "$MONITOR_GEOMETRY"
            sleep 0.5 # Give window manager time to move/resize
            # Make fullscreen
            wmctrl -ir "$window_id" -b add,fullscreen
            echo "Screensaver moved and fullscreened."
        else
            echo "ERROR: Could not find window ID for the screensaver. Killing process."
            kill $screensaver_pid
            screensaver_pid=0
        fi
    fi
}

stop_screensaver() {
    if [[ $screensaver_pid -ne 0 ]]; then
        echo "Stopping screensaver (PID: $screensaver_pid)..."
        # Check if the process still exists before trying to kill
        if ps -p $screensaver_pid > /dev/null; then
           kill $screensaver_pid
           # Wait briefly for termination
           sleep 1
           # Force kill if it didn't terminate gracefully
           if ps -p $screensaver_pid > /dev/null; then
              echo "Process $screensaver_pid did not terminate gracefully, sending SIGKILL..."
              kill -9 $screensaver_pid
           fi
        else
           echo "Process $screensaver_pid not found (already stopped?)."
        fi
        screensaver_pid=0
    fi
}

# --- Main Loop ---
echo "Secondary monitor screensaver script started."
echo "Watching for inactivity longer than $((IDLE_TIMEOUT_MS / 1000)) seconds."

# Ensure any previous instances are stopped on script start
pkill -f "$SCREENSAVER_CMD"
screensaver_pid=0

# Trap SIGINT and SIGTERM to stop the screensaver on script exit
trap "stop_screensaver; exit 0" SIGINT SIGTERM

while true; do
    idle_ms=$(xprintidle)

    if [[ $idle_ms -ge $IDLE_TIMEOUT_MS ]]; then
        # Idle threshold reached, start if not running
        if [[ $screensaver_pid -eq 0 ]]; then
           # Double check process isn't somehow running without PID tracked
           if ! pgrep -f "$SCREENSAVER_CMD" > /dev/null; then
               start_screensaver
           else
               echo "Screensaver process found running unexpectedly. Attempting to manage..."
               # Try to find PID again and manage/kill if necessary
               screensaver_pid=$(pgrep -f "$SCREENSAVER_CMD" | head -n 1)
               # We'll let the activity check handle killing if needed,
               # or the next loop iteration will try to start if it died.
           fi
        fi
    else
        # Active, stop if running
        if [[ $screensaver_pid -ne 0 ]]; then
            stop_screensaver
        fi
    fi

    sleep $CHECK_INTERVAL_S
done
