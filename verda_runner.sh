[[ "$*" == *'"'* ]] && echo "Error: command must not contain double quotes" && exit 1

# sleep 5  # needed to avoid verda deploying blank instances?

# # DeepSeek needs the bigger GPU (B200 180GB); otherwise use RTX PRO 6000 96GB
# # Exception: DeepSeek+RepSelect fits on H200 141GB (cheaper)
# if [[ "$*" == *DeepSeek*RepSelect* || "$*" == *RepSelect*DeepSeek* ]]; then
#   URL="https://tasks.datacrunch.io/open-unlearning3/run"  # H200 141GB
# elif [[ "$*" == *DeepSeek* ]]; then
#   URL="https://tasks.datacrunch.io/open-unlearning/run"  # B200 180GB
# elif [[ "$*" == *Qwen3.5-9B* ]]; then
#   URL="https://tasks.datacrunch.io/open-unlearning3/run"  # H200 141GB
# else
#   URL="https://tasks.datacrunch.io/open-unlearning2/run"  # RTX PRO 6000 96GB
# fi

# repselect ablations:
if [[ "$*" == *DeepSeek* ]]; then
  # need H200
  URL="https://tasks.datacrunch.io/open-unlearning/run"
elif [[ "$*" == *"trainer=NPO"* || "$*" == *"trainer=UNDIAL"* ]]; then
  # NPO and UNDIAL need H100 141GB (matching trainer= so SimNPO stays on 96GB)
  URL="https://tasks.datacrunch.io/open-unlearning3/run"
else
  # RTX PRO 6000 96GB is enough
  URL="https://tasks.datacrunch.io/open-unlearning2/run"
fi

curl -X POST "$URL" \
  -H "Authorization: Bearer $(cat verda_token.txt)" \
  -H "Content-Type: application/json" \
  -d "{\"command\": \"$*\"}"
  # -d "{\"command\": \"git checkout COMMIT_HASH && $*\"}"
