# TeeTotalled

A Telegram bot that runs a trust game to measure how much people trust TEE (Trusted Execution Environment) privacy guarantees.

## How It Works

1. Add the bot to a Telegram group
2. Use `/start` to begin a game
3. Participants privately message the bot with offensive content
4. An LLM scores each message on a 1-100 offensiveness scale
5. After 10 minutes (or `/stop`), the bot shows aggregate statistics and a moral reflection
6. Individual messages are never revealed

The willingness to submit offensive content serves as a "trust thermometer" for dstack's TEE privacy guarantees.

## Running Locally

```bash
# Install dependencies
uv sync

# Set environment variables
export TELEGRAM_BOT_TOKEN="your-token"
export TEE_ENV=development

# Run the bot
uv run python -m tee_totalled
```

## Production Deployment

Deploy to Phala Network using dstack. The bot will automatically detect the TEE environment and enable attestation.

## Source

https://github.com/sangaline/tee-totalled/
