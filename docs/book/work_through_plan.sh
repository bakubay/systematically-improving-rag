#!/bin/bash

# Script to work through plan.md systematically using the Cursor agent
# Each iteration works on the next unchecked section and commits to a git stack

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Get the repo root (two levels up from docs/book/)
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PLAN_FILE="$SCRIPT_DIR/plan.md"
MAX_ITERATIONS=50  # Safety limit to prevent infinite loops
ITERATION=0

# Change to repo root so agent works in the right context
cd "$REPO_ROOT" || exit 1

# Check if plan.md exists
if [ ! -f "$PLAN_FILE" ]; then
    echo "Error: $PLAN_FILE not found"
    exit 1
fi

# Main loop
while [ $ITERATION -lt $MAX_ITERATIONS ]; do
    ITERATION=$((ITERATION + 1))
    echo ""
    echo "========================================="
    echo "Iteration $ITERATION"
    echo "========================================="
    echo ""
    
    # Prompt for the agent
    PROMPT="Read $PLAN_FILE and work through it systematically:

1. Find the next unchecked section/item in the plan (look for sections without [x] or checkmarks)
2. Work on completing that specific section only - do the immediate next thing
3. When you finish the section, check it off in plan.md (mark with [x] or ✓)
4. Phase notes and changelog:
   - For each phase you work on, create notes in docs/book/changelog/{phase_name}_notes.md
   - Phase names should be numbered like chapter-phase format: '1-2' (Chapter 1 Phase 2), '3-1' (Chapter 3 Phase 1), '0-3' (Chapter 0 Phase 3), etc.
   - Include any important notes, decisions, blockers, or observations about the phase work in the changelog file
   - Create the changelog file if it doesn't exist, or append to it if it does
5. Git stack management per stage and part:
   - For each stage/section/part, create a new git stack branch if one doesn't exist yet (use 'git-stack branch create <name>' if available, otherwise 'git checkout -b <name>')
   - Use a descriptive branch name like 'plan/section-name' or 'plan/part-name' based on what you worked on
   - Keep iterating on the same graphite/git-stack branch for that stage/part - make multiple commits as you work through the section or part
   - Only create a new branch when moving to a completely different stage/section/part
   - Commit your changes frequently with clear messages describing what you completed
   - Make sure to commit the updated plan.md when you check off a section or part
   - Commit changelog files along with your work

Important: Only work on ONE section or part per iteration. Don't jump ahead. If a section or part is already checked off, move to the next unchecked one. Keep iterating on the same graphite stack per stage/part until that stage/part is complete. Be thorough but focused."

    echo "Calling agent with prompt..."
    echo ""
    
    # Call the agent in exec mode (default) with --print for non-interactive use
    agent --print --output-format text "$PROMPT"
    
    EXIT_CODE=$?
    
    if [ $EXIT_CODE -ne 0 ]; then
        echo "Agent exited with code $EXIT_CODE"
        echo "Stopping loop"
        break
    fi
    
    echo ""
    echo "Iteration $ITERATION completed"
    echo "Press Enter to continue to next iteration, or Ctrl+C to stop"
    read -r
done

echo ""
echo "Reached maximum iterations ($MAX_ITERATIONS) or stopped by user"
echo "Script completed"
