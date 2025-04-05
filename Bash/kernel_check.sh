#!/bin/bash

# Get the kernel version
kernel_version=$(uname -r)

# Check if it's Zen or LTS kernel and output with color
if [[ $kernel_version == *"lts"* ]]; then
    echo "LTS Kernel"
else
    echo "Zen Kernel"
fi
