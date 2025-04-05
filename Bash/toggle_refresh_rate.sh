#!/bin/bash

# Define your display and the two refresh rates
DISPLAY_NAME="DP-4"
MODE="1920x1080"
RATE1="143.98"
RATE2="60.00"

# Extract the current refresh rate
CURRENT_RATE=$(xrandr | grep -Po '\d+\.\d+(?=\*)')

# Toggle between the two refresh rates
NEW_RATE=$RATE2
[ "$CURRENT_RATE" == "$RATE2" ] && NEW_RATE=$RATE1

xrandr --output "$DISPLAY_NAME" --mode "$MODE" --rate "$NEW_RATE"
notify-send "Refresh Rate" "Switched to $NEW_RATE Hz"
