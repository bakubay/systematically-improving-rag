#!/bin/bash
# Read input from stdin (required by hooks protocol)
cat > /dev/null

# Return follow-up message to keep the agent going
echo '{"followup_message": "keep going until there is no more next steps. If you are done, say 'done'"}'
