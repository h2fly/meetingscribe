#!/bin/bash
# install.sh — MeetingScribe 一键安装脚本
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo ""
echo "=== MeetingScribe 安装向导 ==="
echo ""

# ── 1. 检查 Homebrew ────────────────────────────────────────────────────────
if ! command -v brew &>/dev/null; then
  echo -e "${RED}✗${NC}  未检测到 Homebrew，请先安装："
  echo "   /bin/bash -c \"\$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\""
  exit 1
fi
echo -e "${GREEN}✓${NC} Homebrew 已安装"

# ── 2. 安装 BlackHole 虚拟音频驱动 ─────────────────────────────────────────
if brew list --cask blackhole-2ch &>/dev/null 2>&1; then
  echo -e "${GREEN}✓${NC} BlackHole 2ch 已安装"
else
  echo ""
  echo "安装 BlackHole 2ch 虚拟音频驱动..."
  brew install --cask blackhole-2ch
  echo -e "${GREEN}✓${NC} BlackHole 2ch 安装完成"
fi

# ── 3. 安装 ffmpeg ──────────────────────────────────────────────────────────
if command -v ffmpeg &>/dev/null; then
  echo -e "${GREEN}✓${NC} ffmpeg 已安装"
else
  echo ""
  echo "安装 ffmpeg（FunASR 音频解码依赖）..."
  brew install ffmpeg
  echo -e "${GREEN}✓${NC} ffmpeg 安装完成"
fi

# ── 4. 安装 Python 依赖 ─────────────────────────────────────────────────────
echo ""
echo "安装 Python 依赖..."
pip3 install -r "$SCRIPT_DIR/requirements.txt" -q
echo -e "${GREEN}✓${NC} Python 依赖安装完成"

# ── 5. 创建 meetingscribe 全局命令 ──────────────────────────────────────────
LOCAL_BIN="$HOME/.local/bin"
BIN_PATH="$LOCAL_BIN/meetingscribe"
mkdir -p "$LOCAL_BIN"
echo ""
echo "创建全局命令 meetingscribe → $BIN_PATH"
cat > "$BIN_PATH" <<EOF
#!/bin/bash
python3 "$SCRIPT_DIR/meetingscribe.py" "\$@"
EOF
chmod +x "$BIN_PATH"
echo -e "${GREEN}✓${NC} 全局命令创建完成: $BIN_PATH"

if [[ ":$PATH:" != *":$LOCAL_BIN:"* ]]; then
  echo "" >> "$HOME/.zshrc"
  echo "export PATH=\"\$HOME/.local/bin:\$PATH\"" >> "$HOME/.zshrc"
  echo -e "${YELLOW}!${NC}  已将 ~/.local/bin 写入 ~/.zshrc，请运行："
  echo "   source ~/.zshrc"
  export PATH="$LOCAL_BIN:$PATH"
fi

# ── 6. 检查 Claude Code CLI ─────────────────────────────────────────────────
echo ""
if command -v claude &>/dev/null; then
  echo -e "${GREEN}✓${NC} claude CLI 已安装：$(which claude)"
else
  echo -e "${YELLOW}⚠${NC}  未检测到 claude 命令，请安装 Claude Code CLI："
  echo "   npm install -g @anthropic-ai/claude-code"
  echo "   claude login"
fi

# ── 7. 音频路由配置说明 ─────────────────────────────────────────────────────
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo -e "${YELLOW}【需手动完成】macOS 音频路由配置${NC}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "步骤一：打开「音频 MIDI 设置」"
echo "  open \"/System/Applications/Utilities/Audio MIDI Setup.app\""
echo ""
echo "步骤二：创建「多输出设备」"
echo "  1. 左下角点「+」→「创建多输出设备」"
echo "  2. 勾选你的扬声器（MacBook Air/Pro 扬声器 或外接耳机）"
echo "     同时勾选「BlackHole 2ch」"
echo "  3. 勾选 BlackHole 2ch 那行的「漂移校正」"
echo "  4. 右键该设备 →「将此设备用作系统声音输出」"
echo ""
echo "步骤三：更新配置文件 config.jsonc"
echo "  运行以下命令查看设备名称："
echo "  python3 meetingscribe.py devices"
echo ""
echo "  根据输出修改 config.jsonc 中的："
echo "    output_record  → 多输出设备的名称"
echo "    output_restore → 你的扬声器名称"
echo "    device_mic     → 麦克风名称"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo -e "${GREEN}安装完成！${NC} 配置好音频路由后运行："
echo ""
echo "  python3 meetingscribe.py ui       # 图形界面"
echo "  python3 meetingscribe.py record   # 命令行录音"
echo ""
