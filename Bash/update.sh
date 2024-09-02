#!/bin/bash

# Define the root directory containing your AUR packages
AUR_ROOT="$HOME/AUR"

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
    local repo_name subdir output

    repo_name=$(basename "$dir")
    subdir=$(basename "$(dirname "$dir")") # Get the parent directory name (e.g., System76, Other)
    output=$(git -C "$dir" pull 2>&1)

    # Categorize the results using case
    case "$output" in
        *"Already up to date."*)
            up_to_date+=("${COLOR_GREEN}[$subdir] Already up to date: $repo_name${COLOR_RESET}")
            ;;
        *"Updating"*|*"Fast-forward"*)
            updated+=("${COLOR_YELLOW}[$subdir] Updated: $repo_name${COLOR_RESET}")
            ;;
        *)
            failed+=("${COLOR_RED}[$subdir] Failed to update: $repo_name${COLOR_RESET}")
            ;;
    esac
}

# Gather all .git directories
gitdirs=($(find "$AUR_ROOT" -type d -name ".git"))
total_repos=${#gitdirs[@]}

# Inform user and start processing with progress bar
echo -e "${COLOR_BLUE}Checking AUR:${COLOR_RESET}\n"
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
echo -e "\n\n${COLOR_BLUE}Summary of actions:${COLOR_RESET}"
print_summary "Repositories already up to date" up_to_date[@]
print_summary "Repositories updated" updated[@]
print_summary "Repositories failed to update" failed[@]
echo
