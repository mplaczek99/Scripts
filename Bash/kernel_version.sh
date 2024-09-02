#!/bin/bash

# Get the kernel version
kernel_version=$(uname -r)

# Define colors
LTS_COLOR="#4682B4"   # Steel Blue for LTS
ZEN_COLOR="#B22222"   # Firebrick Red for Zen

# Check if it's LTS or Zen and output with color
if [[ $kernel_version == *"lts"* ]]; then
    echo -e "<span color=\"$LTS_COLOR\">LTS Kernel</span>"
elif [[ $kernel_version == *"zen"* ]]; then
    echo -e "<span color=\"$ZEN_COLOR\">Zen Kernel</span>"
fi
