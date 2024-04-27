#!/bin/bash

# Get the current kernel version
kernel_version=$(uname -r)

# Check if it's an LTS kernel
if [[ $kernel_version == *"lts"* ]]; then
  echo "LTS Kernel"
elif [[ $kernel_version == *"zen"* ]]; then
  echo "Zen Kernel"
else
  echo "Other Kernel"
fi
