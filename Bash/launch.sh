#!/usr/bin/env bash

# Name of your Polybar bar (check your polybar config, e.g., example, top, main)
BAR_NAME="bar"
# If you have multiple bars launched by a launch.sh script,

if pgrep -x polybar > /dev/null; then
    # Polybar is running, toggle its visibility
    polybar-msg cmd toggle
else
    # Polybar is not running, launch it
    polybar "$BAR_NAME" &
fi
