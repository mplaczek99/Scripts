#!/bin/bash

# Define the root directory containing your AUR packages
AUR_ROOT="$HOME/AUR"

# Check if the AUR_ROOT directory exists
if [ ! -d "$AUR_ROOT" ]; then
    echo "Error: AUR directory '$AUR_ROOT' does not exist."
    exit 1
fi

# Define color variables using tput for portability
COLOR_RESET=$(tput sgr0)
COLOR_BLUE=$(tput setaf 4; tput bold)
COLOR_GREEN=$(tput setaf 2; tput bold)
COLOR_YELLOW=$(tput setaf 3; tput bold)
COLOR_RED=$(tput setaf 1; tput bold)

# Associative arrays to store categorized results
declare -A up_to_date updated failed

# Function to perform git fetch and categorize results
perform_git_pull() {
    local dir="$1"
    local repo_name="${dir##*/}"
    local subdir="${dir%/*}"
    subdir="${subdir##*/}"

    # Verify git repository
    if [ ! -d "$dir/.git" ] || ! git -C "$dir" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
        echo "failed|${COLOR_RED}[$subdir] Not a git repository: $repo_name${COLOR_RESET}"
        return
    fi

    # Ensure the branch has an upstream
    if ! git -C "$dir" rev-parse --abbrev-ref --symbolic-full-name @{u} >/dev/null 2>&1; then
        echo "failed|${COLOR_RED}[$subdir] No upstream set: $repo_name${COLOR_RESET}"
        return
    fi

    # Check for uncommitted changes using git status
    if [ -n "$(git -C "$dir" status --porcelain --untracked-files=no)" ]; then
        echo "failed|${COLOR_RED}[$subdir] Uncommitted changes in: $repo_name${COLOR_RESET}"
        return
    fi

    # Fetch and attempt to fast-forward merge
    git -C "$dir" fetch --quiet

    # Attempt to merge
    if git -C "$dir" merge --ff-only --quiet >/dev/null 2>&1; then
        echo "updated|${COLOR_YELLOW}[$subdir] Updated: $repo_name${COLOR_RESET}"
    else
        # If merge fails, check if already up to date
        local LOCAL REMOTE
        LOCAL=$(git -C "$dir" rev-parse @)
        REMOTE=$(git -C "$dir" rev-parse @{u})

        if [ "$LOCAL" = "$REMOTE" ]; then
            echo "up_to_date|${COLOR_GREEN}[$subdir] Already up to date: $repo_name${COLOR_RESET}"
        else
            echo "failed|${COLOR_RED}[$subdir] Failed to update: $repo_name${COLOR_RESET}"
        fi
    fi
}

# Gather all git repositories
mapfile -t gitdirs < <(find "$AUR_ROOT" -type d -name ".git" 2>/dev/null)
repos=("${gitdirs[@]/%\/.git/}")  # Remove '/.git' from paths
total_repos=${#repos[@]}

# Inform user and start processing
echo -e "${COLOR_BLUE}Checking AUR Repositories:${COLOR_RESET}\n"

# Initialize progress variables
progress=0

# Function to update the progress bar
update_progress_bar() {
    local progress=$1
    local total=$2
    local width=50
    local percent=$(( progress * 100 / total ))
    local filled=$(( progress * width / total ))
    local empty=$(( width - filled ))
    local bar
    printf -v bar "%*s" "$filled"
    bar=${bar// /#}
    printf -v spaces "%*s" "$empty"
    printf "\rProgress: [${bar// /#}${spaces}] %3d%% (%d/%d)" "$percent" "$progress" "$total"
}

# Run the updates
for dir in "${repos[@]}"; do
    # Run perform_git_pull and capture output
    result=$(perform_git_pull "$dir")
    status="${result%%|*}"
    message="${result#*|}"

    # Categorize the result
    case "$status" in
        up_to_date)
            up_to_date["$message"]=1
            ;;
        updated)
            updated["$message"]=1
            ;;
        failed)
            failed["$message"]=1
            ;;
    esac

    # Update progress
    ((progress++))
    update_progress_bar "$progress" "$total_repos"
done

# Move to a new line after progress bar
echo -e "\n"

# Function to print a category summary
print_summary() {
    local category="$1"
    declare -n items="$2"
    local count=${#items[@]}

    if (( count > 0 )); then
        echo -e "$category ($count):"
        for message in "${!items[@]}"; do
            echo -e "  - $message"
        done | sort
        echo
    fi
}

# Output the categorized summary
echo -e "${COLOR_BLUE}Summary of Actions:${COLOR_RESET}"
print_summary "Repositories already up to date" up_to_date
print_summary "Repositories updated" updated
print_summary "Repositories failed to update" failed

# Reset terminal color
echo -e "${COLOR_RESET}"
