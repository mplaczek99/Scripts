#!/bin/bash

# Get the kernel version
kernel_version=$(uname -r)

# Define colors
LINUX_COLOR="#4682B4"   # Steel Blue for Linux
ZEN_COLOR="#B22222"     # Firebrick Red for Zen

# Check if it's Zen or standard Linux kernel and output with color
if [[ $kernel_version == *"zen"* ]]; then
    echo -e "<span color=\"$ZEN_COLOR\">Zen Kernel</span>"
else
    echo -e "<span color=\"$LINUX_COLOR\">Linux Kernel</span>"
fi
