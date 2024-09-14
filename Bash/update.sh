#!/bin/bash

# Define the root directory containing your AUR packages
AUR_ROOT="$HOME/AUR"

# Check if the AUR_ROOT directory exists
if [ ! -d "$AUR_ROOT" ]; then
    echo "Error: AUR directory '$AUR_ROOT' does not exist."
    exit 1
fi

# Define color variables
COLOR_RESET="\e[0m"
COLOR_BLUE="\e[1;34m"
COLOR_GREEN="\e[1;32m"
COLOR_YELLOW="\e[1;33m"
COLOR_RED="\e[1;31m"

# Arrays to store categorized results
declare -a up_to_date updated failed

# Function to display a progress bar
show_progress() {
    local current=$1 total=$2 width=50
    local progress=$((current * width / total))
    printf "\r[%-${width}s] %d%%" "$(printf "%0.s#" $(seq 1 $progress))" "$((current * 100 / total))"
}

# Function to perform git pull and categorize results
perform_git_pull() {
    local dir="$1"
    local repo_name subdir

    repo_name=$(basename "$dir")
    subdir=$(basename "$(dirname "$dir")") # Get the parent directory name (e.g., System76, Other)

    # Check if the directory is a git repository
    if [ ! -d "$dir/.git" ]; then
        failed+=("${COLOR_RED}[$subdir] Not a git repository: $repo_name${COLOR_RESET}")
        return
    fi

    # Perform git pull
    output=$(git -C "$dir" pull --ff-only 2>&1)
    exit_code=$?

    # Categorize the results based on exit code and output
    if [ $exit_code -eq 0 ]; then
        if [[ "$output" == *"Already up to date."* ]]; then
            up_to_date+=("${COLOR_GREEN}[$subdir] Already up to date: $repo_name${COLOR_RESET}")
        else
            updated+=("${COLOR_YELLOW}[$subdir] Updated: $repo_name${COLOR_RESET}")
        fi
    else
        failed+=("${COLOR_RED}[$subdir] Failed to update: $repo_name\n$output${COLOR_RESET}")
    fi
}

# Gather all git repositories
mapfile -t gitdirs < <(find "$AUR_ROOT" -type d -name ".git" -prune 2>/dev/null)
total_repos=${#gitdirs[@]}

# Inform user and start processing with progress bar
echo -e "${COLOR_BLUE}Checking AUR Repositories:${COLOR_RESET}\n"
for i in "${!gitdirs[@]}"; do
    perform_git_pull "$(dirname "${gitdirs[i]}")"
    show_progress $((i + 1)) "$total_repos"
done

# Function to print a category summary
print_summary() {
    local category="$1" items=("${!2}")
    local count=${#items[@]}

    if (( count > 0 )); then
        echo -e "\n$category ($count):"
        printf "  - %b\n" "${items[@]}"
    fi
}

# Output the categorized summary
echo -e "\n\n${COLOR_BLUE}Summary of Actions:${COLOR_RESET}"
print_summary "Repositories already up to date" up_to_date[@]
print_summary "Repositories updated" updated[@]
print_summary "Repositories failed to update" failed[@]

# Reset terminal color
echo -e "${COLOR_RESET}"
