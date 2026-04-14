[[ "$*" == *'"'* ]] && echo "Error: command must not contain double quotes" && exit 1

curl -X POST https://tasks.datacrunch.io/open-unlearning/run \
  -H "Authorization: Bearer $(cat verda_token.txt)" \
  -H "Content-Type: application/json" \
  -d "{\"command\": \"$*\"}"