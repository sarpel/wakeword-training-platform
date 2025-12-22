# GLM-4.6 ortam değişkenlerini ayarla (bash / zsh)

# SECURITY: Token'ı environment variable'dan oku
if [ -f ".env.zai" ]; then
    source .env.zai
else
    echo "⚠️  .env.zai dosyası bulunamadı. Token'ınızı güvenli bir şekilde ekleyin!"
    echo "Örnek .env.zai dosyası:"
    echo "export ANTHROPIC_BASE_URL=\"https://api.z.ai/api/anthropic\""
    echo "export ANTHROPIC_AUTH_TOKEN=\"YOUR_TOKEN_HERE\""
    echo "export ANTHROPIC_MODEL=\"GLM-4.6\""
    echo "export ANTHROPIC_SMALL_FAST_MODEL=\"GLM-4.6\""
    exit 1
fi

claude --dangerously-skip-permissions