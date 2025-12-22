if [ -f ".env.zai" ]; then
    source .env.zai
fi

claude --dangerously-skip-permissions