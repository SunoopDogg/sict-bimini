curl -fsSL https://ollama.com/install.sh | sh

apt update
apt install -y pciutils lshw
rm -rf /var/lib/apt/lists/*

ollama serve -d &
ollama pull gpt-oss:20b