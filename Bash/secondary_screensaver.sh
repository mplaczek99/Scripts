#!/bin/bash

# --- Configuration ---
TIMEOUT_SECONDS=300      # Trigger screensaver after mouse is off Monitor 2 for this many seconds (e.g., 300 = 5 minutes)
CHECK_INTERVAL_S=5       # How often to check mouse position (seconds) - shorter interval = more responsive

# --- Second Monitor Geometry (!!! MODIFY THESE from xrandr output !!!) ---
MONITOR_WIDTH=1920
MONITOR_HEIGHT=1080
MONITOR_X_OFFSET=1920
MONITOR_Y_OFFSET=0
# Calculate boundaries
MONITOR_X_MAX=$((MONITOR_X_OFFSET + MONITOR_WIDTH))
MONITOR_Y_MAX=$((MONITOR_Y_OFFSET + MONITOR_HEIGHT))

# Construct the geometry string for wmctrl's -e option
# Format: gravity,X,Y,Width,Height (0 for default gravity)
MONITOR_GEOMETRY="0,${MONITOR_X_OFFSET},${MONITOR_Y_OFFSET},${MONITOR_WIDTH},${MONITOR_HEIGHT}"

# --- Screensaver Program (!!! MODIFY THIS if needed !!!) ---
SCREENSAVER_CMD="/usr/libexec/xscreensaver/glmatrix -geometry 1920x1080+1920+0 -fullscreen" # Command to run the screensaver
SCREENSAVER_NAME="glmatrix" # The process name or a unique part of the command/window title

# --- Internal Variables ---
screensaver_pid=0       # Stores the PID of the running screensaver
mouse_off_monitor2_since=0 # Timestamp (seconds since epoch) when mouse left monitor 2

# --- Functions ---
start_screensaver() {
    # Only start if PID is not tracked (or if tracked PID doesn't exist anymore)
    if [[ $screensaver_pid -eq 0 ]] || ! ps -p $screensaver_pid > /dev/null; then
        echo "Starting screensaver..."
        # Double-check process isn't already running somehow
        if pgrep -f "$SCREENSAVER_CMD" > /dev/null; then
            echo "Screensaver process detected running unexpectedly. Killing old instances..."
            pkill -f "$SCREENSAVER_CMD"
            sleep 1 # Give it a moment to die
        fi

        # Launch in the background
        $SCREENSAVER_CMD &
        screensaver_pid=$!
        echo "Screensaver launched with PID: $screensaver_pid"

        # Give it time to create its window
        sleep 2

        # Find the window ID using wmctrl (try PID first, then title)
        window_id=$(wmctrl -lp | grep $screensaver_pid | awk '{print $1}')
        if [[ -z "$window_id" ]]; then
             echo "Could not find window ID via PID $screensaver_pid. Trying title '$SCREENSAVER_NAME'..."
             window_id=$(wmctrl -l | grep -i "$SCREENSAVER_NAME" | head -n 1 | awk '{print $1}')
        fi

        if [[ -n "$window_id" ]]; then
            echo "Found window ID: $window_id"
            wmctrl -ir "$window_id" -e "$MONITOR_GEOMETRY"
            sleep 0.5
            wmctrl -ir "$window_id" -b add,fullscreen
            echo "Screensaver moved and fullscreened."
        else
            echo "ERROR: Could not find window ID for the screensaver. Killing process $screensaver_pid."
            kill $screensaver_pid &> /dev/null # Kill silently if window not found
            screensaver_pid=0 # Reset PID tracking
        fi
    else
         echo "Screensaver start requested, but process $screensaver_pid seems to be running."
    fi
}

stop_screensaver() {
    if [[ $screensaver_pid -ne 0 ]]; then
        echo "Stopping screensaver (PID: $screensaver_pid)..."
        if ps -p $screensaver_pid > /dev/null; then
           kill $screensaver_pid
           sleep 1
           if ps -p $screensaver_pid > /dev/null; then
              echo "Process $screensaver_pid did not terminate gracefully, sending SIGKILL..."
              kill -9 $screensaver_pid
           fi
        else
           echo "Process $screensaver_pid not found (already stopped?)."
        fi
        screensaver_pid=0 # Reset PID tracking
    fi
    # Also reset the timer state
    mouse_off_monitor2_since=0
}

# --- Main Loop ---
echo "Secondary monitor screensaver script started."
echo "Watching for mouse pointer off Monitor 2 (bounds X:${MONITOR_X_OFFSET}-${MONITOR_X_MAX}, Y:${MONITOR_Y_OFFSET}-${MONITOR_Y_MAX}) for longer than ${TIMEOUT_SECONDS} seconds."

# Ensure any previous instances are stopped on script start
pkill -f "$SCREENSAVER_CMD"
screensaver_pid=0
mouse_off_monitor2_since=0

# Trap SIGINT and SIGTERM to stop the screensaver on script exit
trap "echo 'Termination signal received.'; stop_screensaver; echo 'Exiting.'; exit 0" SIGINT SIGTERM

while true; do
    # Get current mouse coordinates
    # eval $(xdotool getmouselocation --shell) # Old method, may fail if output has unexpected chars
    coords=$(xdotool getmouselocation --shell)
    X=$(echo "$coords" | grep '^X=' | cut -d= -f2)
    Y=$(echo "$coords" | grep '^Y=' | cut -d= -f2)
    # SCREEN=$(echo "$coords" | grep '^SCREEN=' | cut -d= -f2) # We don't need screen number here
    # WINDOW=$(echo "$coords" | grep '^WINDOW=' | cut -d= -f2) # We don't need window ID here

    # Check if coordinates are numeric (xdotool might fail sometimes)
    if [[ "$X" =~ ^[0-9]+$ ]] && [[ "$Y" =~ ^[0-9]+$ ]]; then

        # Check if mouse is within the bounds of the second monitor
        if [[ $X -ge $MONITOR_X_OFFSET && $X -lt $MONITOR_X_MAX && $Y -ge $MONITOR_Y_OFFSET && $Y -lt $MONITOR_Y_MAX ]]; then
            # Mouse is ON Monitor 2
            if [[ $mouse_off_monitor2_since -ne 0 ]]; then
                echo "Mouse moved onto Monitor 2."
                stop_screensaver # Stop the saver if it was running
                mouse_off_monitor2_since=0 # Reset timer
            fi
            # If it wasn't running, do nothing.
        else
            # Mouse is OFF Monitor 2
            current_time=$(date +%s)
            if [[ $mouse_off_monitor2_since -eq 0 ]]; then
                # Timer not started yet, start it now
                # echo "Mouse moved off Monitor 2. Starting timer." # Can be noisy
                mouse_off_monitor2_since=$current_time
            else
                # Timer is running, check elapsed time
                elapsed=$((current_time - mouse_off_monitor2_since))
                # echo "Mouse off Monitor 2 for ${elapsed}s" # Debugging line

                if [[ $elapsed -ge $TIMEOUT_SECONDS ]]; then
                    # Timeout reached, start screensaver if not already running
                    if [[ $screensaver_pid -eq 0 ]] || ! ps -p $screensaver_pid > /dev/null; then
                         echo "Mouse off Monitor 2 timeout (${TIMEOUT_SECONDS}s) reached."
                         start_screensaver
                    fi
                fi
            fi
        fi
    else
         echo "Warning: Could not get valid mouse coordinates from xdotool."
         # Decide how to handle this - maybe stop the saver? Or do nothing?
         # Let's do nothing for now, it might be a transient error.
         sleep 1 # Add a small delay if error occurs
    fi

    sleep $CHECK_INTERVAL_S
done
