#!/bin/bash

# Directory containing Arch Linux wallpapers
WALLPAPER_DIR="/usr/share/backgrounds/archlinux"

# Set a random wallpaper from the directory
feh --bg-scale "$(find "$WALLPAPER_DIR" -type f | shuf -n 1)"
