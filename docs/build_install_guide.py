"""Build packaging/install-guide.html (中文) and packaging/install-guide.en.html
(English) — the install guide for people who got MeetingScribe as a DMG (not
the source checkout; that is README.md).

Both languages, three places: build_dmg.sh copies them into the DMG side by
side as 安装说明.html and Install Guide.html,
the spec bundles it into the app (Settings → Environment "查看步骤" links and
Help → 安装说明 open it at a section anchor), and README links to it. Images
are base64-embedded so the file works offline and survives being copied.

Screenshots of the app are taken from the real native rendering
(`QT_QPA_PLATFORM=cocoa`, `QMainWindow.show` patched out, `grab()`), so no
window ever appears on screen. The Audio MIDI Setup screenshot is the
existing annotated one in docs/tutorial-assets — capturing it again would
open that app on the user's desktop.

Run: python3 docs/build_install_guide.py
"""

from __future__ import annotations

import base64
import html
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
ASSETS = REPO / "docs" / "tutorial-assets"
OUT = {"zh": REPO / "packaging" / "install-guide.html",
       "en": REPO / "packaging" / "install-guide.en.html"}

# Section anchors. The app links to these (see `_GUIDE_ANCHORS` in
# meetingscribe.py) — renaming one breaks a link, so keep them stable.
ANCHORS = ("install", "permissions", "blackhole", "multi-output", "ai-cli",
           "ffmpeg", "models", "optional", "faq")


def _capture_app() -> "dict[str, dict[str, bytes]]":
    """PNG bytes of the recording view and the Environment card, per UI
    language — the English guide must not show a Chinese UI."""
    os.environ["QT_QPA_PLATFORM"] = "cocoa"
    sys.path.insert(0, str(REPO))
    sys.argv = ["meetingscribe.py", "ui"]
    import meetingscribe as ms
    from PyQt6.QtCore import QBuffer, QByteArray, QIODevice, QTimer
    from PyQt6.QtWidgets import QApplication, QFrame, QMainWindow

    QMainWindow.show = lambda self: None   # never put a window on screen
    shots: "dict[str, dict[str, bytes]]" = {"zh": {}, "en": {}}

    def png(pixmap) -> bytes:
        data = QByteArray()
        buf = QBuffer(data)
        buf.open(QIODevice.OpenModeFlag.WriteOnly)
        pixmap.save(buf, "PNG")
        return bytes(data)

    def capture():
        app = QApplication.instance()
        win = next(w for w in app.topLevelWidgets()
                   if type(w).__name__ == "MainWindow")
        win.resize(1280, 800)
        for lang in ("zh", "en"):
            win._set_language(lang)
            win.stack.setCurrentWidget(win.recording_view)
            app.processEvents()
            shots[lang]["recording"] = png(win.grab())
            win.stack.setCurrentWidget(win.config_view)
            app.processEvents()
            cards = [f for f in win.config_view.findChildren(QFrame)
                     if f.objectName() == "groupCard"]
            if cards:
                shots[lang]["environment"] = png(cards[0].grab())
        win._set_language("zh")
        app.quit()

    orig_exec = QApplication.exec
    QApplication.exec = staticmethod(
        lambda *a: (QTimer.singleShot(2500, capture), orig_exec())[1])
    try:
        ms.main()
    except SystemExit:
        pass
    return shots


def _data_uri(data: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(data).decode("ascii")


COPY = {"zh": ("复制", "已复制"), "en": ("Copy", "Copied")}


def _cmd(text: str, lang: str = "zh") -> str:
    """A command line with a copy button (works from file://, no network)."""
    t = html.escape(text)
    return (f'<div class="cmd"><code>{t}</code>'
            f'<button onclick="copyCmd(this)" data-cmd="{t}">{COPY[lang][0]}</button></div>')


def _figure(uri: str, caption: str, width: str = "100%") -> str:
    return (f'<figure><img src="{uri}" style="max-width:{width}" alt="">'
            f'<figcaption>{html.escape(caption)}</figcaption></figure>')


def _body_zh(shots: dict, midi: bytes) -> str:
    _c = lambda t: _cmd(t, "zh")
    rec_fig = _figure(_data_uri(shots["recording"]),
                      "录音页：左边是录音和处理按钮，右边是实时字幕") \
        if "recording" in shots else ""
    env_fig = _figure(_data_uri(shots["environment"]),
                      "配置 → 运行环境：绿点表示已就绪", "760px") \
        if "environment" in shots else ""
    midi_fig = _figure(_data_uri(midi),
                       "音频 MIDI 设置：① 左下角「+」创建；② 勾选 BlackHole 2ch 和你的扬声器 / 耳机",
                       "820px")
    return f"""
<header>
  <h1>MeetingScribe 安装说明</h1>
  <p class="lead">会议录音、转写、自动生成纪要的 macOS 应用。按下面的顺序做一遍，大约 15 分钟。
  装好后打开 app，在 <b>配置 → 运行环境</b> 里能看到每一项是否就绪。</p>
  <nav>
    <a href="#install">1 安装 app</a><a href="#permissions">2 权限</a>
    <a href="#blackhole">3 BlackHole</a><a href="#multi-output">4 多输出设备</a>
    <a href="#ai-cli">5 AI 命令行</a><a href="#ffmpeg">6 ffmpeg</a>
    <a href="#models">7 模型下载</a><a href="#optional">8 可选设置</a>
    <a href="#faq">常见问题</a>
  </nav>
</header>

<section id="install">
  <h2><span class="n">1</span>安装 app</h2>
  <p>系统要求：Apple 芯片（M1 及以后）的 Mac，macOS 13 或更新。</p>
  <ol>
    <li>双击 <code>MeetingScribe-x.y.z.dmg</code>，把 <b>MeetingScribe</b> 拖到右边的 <b>Applications</b>。</li>
    <li>第一次打开：在「应用程序」里<b>右键 MeetingScribe → 打开</b>，弹窗里再点一次「打开」。
      app 没有经过苹果公证，直接双击会被系统拦下，这一步只需要做一次。</li>
    <li>如果右键打开也被拦，在「终端」里运行下面这行，再正常打开：</li>
  </ol>
  {_c("xattr -dr com.apple.quarantine /Applications/MeetingScribe.app")}
  <p class="note">第一次启动大约需要 20 秒，之后会快一些。</p>
  {rec_fig}
</section>

<section id="permissions">
  <h2><span class="n">2</span>允许麦克风权限</h2>
  <p>第一次点录音按钮时，macOS 会询问 MeetingScribe 能否使用麦克风，点 <b>允许</b>。</p>
  <p>如果之前点了「不允许」：打开 <b>系统设置 → 隐私与安全性 → 麦克风</b>，打开 MeetingScribe 的开关，然后重启 app。</p>
</section>

<section id="blackhole">
  <h2><span class="n">3</span>安装 BlackHole 2ch（录制对方的声音）</h2>
  <p>麦克风只能录到你自己。会议软件里对方的声音是从电脑播放出来的，要靠 BlackHole 这个免费的虚拟声卡才能录下来。
  不装它，录音里就只有你一个人的声音。</p>
  <p><b>方式一（推荐）</b>：用 Homebrew 安装。还没装 Homebrew 的，先按 <a href="https://brew.sh">brew.sh</a> 首页的一行命令装好。</p>
  {_c("brew install --cask blackhole-2ch")}
  <p><b>方式二</b>：到官网 <a href="https://existential.audio/blackhole/">existential.audio/blackhole</a>
  下载 <b>BlackHole 2ch</b> 的安装包，双击安装。</p>
  <p class="warn">装完请<b>重启电脑</b>。BlackHole 是系统音频驱动，重启后才会出现在下一步的设备列表里。安装时会要求输入电脑密码，这是正常的。</p>
</section>

<section id="multi-output">
  <h2><span class="n">4</span>配置「多输出设备」（边听边录）</h2>
  <p>录音时，电脑的声音要同时送到两个地方：你的扬声器 / 耳机（你要听），以及 BlackHole（app 要录）。
  macOS 自带的「多输出设备」就是做这件事的。</p>
  <ol>
    <li>打开 <b>音频 MIDI 设置</b>：在「启动台 → 其他」里，或者用 Spotlight（⌘ 空格）搜索「音频 MIDI 设置」。</li>
    <li>点左下角的 <b>「+」</b>，选择 <b>创建多输出设备</b>。（下图 ①）</li>
    <li>在右侧列表里勾选 <b>BlackHole 2ch</b> 和你平时用的 <b>扬声器或耳机</b>，例如 MacBook 扬声器。（下图 ②）</li>
    <li>在 BlackHole 2ch 那一行勾上 <b>漂移校正</b>，避免录音时间一长出现杂音。「主设备」保持默认即可。</li>
    <li>名字保持默认的「多输出设备」，app 会自动找到它。</li>
  </ol>
  {midi_fig}
  <p>录音时 app 会自动把系统输出切到多输出设备，停止后再切回来，平时不需要手动切换。</p>
  <p class="tip"><b>想避免录音开始时音乐 App 被暂停？</b>可以直接把「多输出设备」设为系统默认输出（右键它 → 将此设备用于声音输出），
  这样开始和停止录音时就不会切换设备。代价是：用多输出设备时，键盘上的音量键不起作用，要用 app 里的音量滑块调节。</p>
  <p class="tip">如果你有时用扬声器、有时用耳机，可以把两者都勾上。录音时 app 会自动静音你当前没在用的那个，不会两边同时出声。</p>
</section>

<section id="ai-cli">
  <h2><span class="n">5</span>安装并登录 AI 命令行（生成纪要用）</h2>
  <p>校对文字稿和生成纪要用的是 Claude（或 Codex）的命令行工具，用的是你自己的账号。不装的话，录音和转写照常，只是不能生成纪要。</p>
  <ol>
    <li>安装 Node.js：到 <a href="https://nodejs.org">nodejs.org</a> 下载 LTS 版本安装；或者用 Homebrew：</li>
  </ol>
  {_c("brew install node")}
  <ol start="2">
    <li>安装 Claude Code 命令行：</li>
  </ol>
  {_c("npm install -g @anthropic-ai/claude-code")}
  <ol start="3">
    <li>在「终端」里运行一次 <code>claude</code>，按提示登录你的账号，登录完成后退出即可。</li>
  </ol>
  {_c("claude")}
  <p>用 Codex 的同事，把上面两步换成：</p>
  {_c("npm install -g @openai/codex")}
  {_c("codex login")}
  <p>然后在 app 的 <b>配置 → 常用设置 → AI 模型</b> 里，把「校对文字稿」和「生成纪要」改成 Codex。</p>
  <p class="tip">装好后，在 <b>配置 → 运行环境</b> 里点 <b>测试连接</b>，显示「已登录，能正常回复」就说明可以用了。</p>
  {env_fig}
</section>

<section id="ffmpeg">
  <h2><span class="n">6</span>ffmpeg（可选）</h2>
  <p>只有导入 mp3、m4a 等非 WAV 格式的音频（比如手机录音）时才需要。app 自己录的音不需要它。</p>
  {_c("brew install ffmpeg")}
</section>

<section id="models">
  <h2><span class="n">7</span>第一次使用会下载模型</h2>
  <p>语音识别模型不在安装包里，第一次用到时自动下载，只下载一次。开始下载前 app 会提示要下载哪些、大概多大。</p>
  <table>
    <tr><th>什么时候</th><th>下载什么</th><th>大小</th></tr>
    <tr><td>第一次「语音转文字」或生成纪要</td><td>中文语音识别、断句、标点模型</td><td>约 2 GB</td></tr>
    <tr><td>第一次打开实时字幕（英文识别）</td><td>英文流式识别 + 翻译模型</td><td>约 1.7 GB</td></tr>
    <tr><td>第一次打开实时字幕（中文识别）</td><td>中文流式识别（+ 上面的中文模型）</td><td>约 0.2 GB</td></tr>
  </table>
  <p>下载期间请保持联网，进度可能会停留一段时间，属于正常现象。磁盘请预留 5 GB 以上的空间。</p>
</section>

<section id="optional">
  <h2><span class="n">8</span>可选设置</h2>
  <ul>
    <li><b>日程关联</b>：录音前选中今天的日程，录音按日程标题命名，参会人会交给纪要环节对上真实姓名。
      需要 Claude 账号里已连接 Google 日历，在 <b>配置 → 日程关联</b> 里打开，打开时会先检查连接。</li>
    <li><b>热词</b>：同事姓名、项目代号等专有名词。每次生成纪要后会自动积累，也可以在 <b>配置 → 常用设置 → 编辑热词表</b> 里手工添加，改完重启 app 生效。</li>
    <li><b>自定义目标</b>：录音页的「自定义目标 ▾」，用一句话描述你想要的输出（如「列出客户提出的所有需求」），AI 会帮你扩写成完整的提示词。</li>
  </ul>
  <p>app 的所有数据都在 <code>~/Documents/meetingscribe/</code>：配置、热词、录音和生成的纪要、日志。</p>
</section>

<section id="faq">
  <h2>常见问题</h2>
  <dl>
    <dt>录音里没有对方的声音</dt>
    <dd>① 确认已装 BlackHole 并重启过电脑；② 确认多输出设备里勾选了 BlackHole 2ch；③ 打开 <b>配置 → 运行环境</b>，看前两项是否都是绿点。
      开始录音时如果没找到 BlackHole，app 顶部也会提示。</dd>
    <dt>录音时自己听不到声音</dt>
    <dd>多输出设备里没有勾选扬声器或耳机。打开「音频 MIDI 设置」把它勾上；运行环境里这一项会显示橙色提示。</dd>
    <dt>录音时键盘音量键不起作用</dt>
    <dd>这是 macOS 对多输出设备的限制。录音期间请用 app 录音页上的「输出音量」滑块调节。</dd>
    <dt>提示「无法打开」或「无法验证开发者」</dt>
    <dd>见第 1 步：右键 → 打开，或运行那条 <code>xattr</code> 命令。</dd>
    <dt>生成纪要失败</dt>
    <dd>到 <b>配置 → 运行环境</b> 点「测试连接」，按提示安装或重新登录 claude / codex。</dd>
    <dt>转写一直停在开头不动</dt>
    <dd>第一次转写要先下载约 2 GB 的模型，网络慢时需要几分钟。之后就不会了。</dd>
    <dt>需要把问题反馈给别人</dt>
    <dd>日志在 <code>~/Documents/meetingscribe/logs/</code>，按日期一天一个文件，把出问题那天的日志发过去即可。</dd>
  </dl>
</section>
"""


def _body_en(shots: dict, midi: bytes) -> str:
    _c = lambda t: _cmd(t, "en")
    rec_fig = _figure(_data_uri(shots["recording"]),
                      "Recording page: recording and processing on the left, live captions on the right") \
        if "recording" in shots else ""
    env_fig = _figure(_data_uri(shots["environment"]),
                      "Settings → Environment: a green dot means ready", "760px") \
        if "environment" in shots else ""
    midi_fig = _figure(_data_uri(midi),
                       "Audio MIDI Setup: ① “+” at the bottom left; ② tick BlackHole 2ch and your speakers / headphones "
                       "(this screenshot is from a Chinese-language Mac — the layout is identical)",
                       "820px")
    return f"""
<header>
  <h1>MeetingScribe Install Guide</h1>
  <p class="lead">A macOS app that records meetings, transcribes them and writes the notes. Follow the steps in order —
  about 15 minutes. Once installed, <b>Settings → Environment</b> in the app shows whether each piece is ready.</p>
  <nav>
    <a href="#install">1 Install</a><a href="#permissions">2 Permissions</a>
    <a href="#blackhole">3 BlackHole</a><a href="#multi-output">4 Multi-Output</a>
    <a href="#ai-cli">5 AI CLI</a><a href="#ffmpeg">6 ffmpeg</a>
    <a href="#models">7 Model downloads</a><a href="#optional">8 Optional</a>
    <a href="#faq">Troubleshooting</a>
  </nav>
</header>

<section id="install">
  <h2><span class="n">1</span>Install the app</h2>
  <p>Requires a Mac with Apple silicon (M1 or later) and macOS 13 or newer.</p>
  <ol>
    <li>Open <code>MeetingScribe-x.y.z.dmg</code> and drag <b>MeetingScribe</b> onto <b>Applications</b>.</li>
    <li>First launch: in Applications, <b>right-click MeetingScribe → Open</b>, then click <b>Open</b> again.
      The app is not notarized by Apple, so a plain double-click is blocked the first time. You only do this once.</li>
    <li>If right-click → Open is blocked too, run this in Terminal and open the app normally:</li>
  </ol>
  {_c("xattr -dr com.apple.quarantine /Applications/MeetingScribe.app")}
  <p class="note">The first launch takes about 20 seconds; later launches are faster.</p>
  {rec_fig}
</section>

<section id="permissions">
  <h2><span class="n">2</span>Allow the microphone</h2>
  <p>The first time you press record, macOS asks whether MeetingScribe may use the microphone. Click <b>Allow</b>.</p>
  <p>If you clicked “Don’t Allow” earlier: open <b>System Settings → Privacy &amp; Security → Microphone</b>,
  turn MeetingScribe on, then restart the app.</p>
</section>

<section id="blackhole">
  <h2><span class="n">3</span>Install BlackHole 2ch (to record the other side)</h2>
  <p>The microphone only hears you. The other people in a call come out of your computer’s speakers, and recording
  them needs BlackHole, a free virtual audio device. Without it the recording contains only your own voice.</p>
  <p><b>Option 1 (recommended)</b>: with Homebrew. If you don’t have Homebrew, install it first with the one-line command
  on <a href="https://brew.sh">brew.sh</a>.</p>
  {_c("brew install --cask blackhole-2ch")}
  <p><b>Option 2</b>: download the <b>BlackHole 2ch</b> installer from
  <a href="https://existential.audio/blackhole/">existential.audio/blackhole</a> and run it.</p>
  <p class="warn"><b>Restart your Mac</b> afterwards. BlackHole is a system audio driver and only appears in the next
  step’s device list after a restart. The installer asks for your password — that is expected.</p>
</section>

<section id="multi-output">
  <h2><span class="n">4</span>Create a Multi-Output Device (hear and record at once)</h2>
  <p>While recording, your computer’s sound has to go to two places: your speakers / headphones (so you hear it)
  and BlackHole (so the app records it). macOS’s built-in Multi-Output Device does exactly that.</p>
  <ol>
    <li>Open <b>Audio MIDI Setup</b>: Launchpad → Other, or search “Audio MIDI Setup” with Spotlight (⌘ Space).</li>
    <li>Click <b>“+”</b> at the bottom left and choose <b>Create Multi-Output Device</b>. (① below)</li>
    <li>In the list on the right, tick <b>BlackHole 2ch</b> and the <b>speakers or headphones</b> you normally use,
      e.g. the MacBook speakers. (② below)</li>
    <li>Tick <b>Drift Correction</b> on the BlackHole 2ch row so long recordings don’t pick up crackles.
      Leave the primary device at its default.</li>
    <li>Keep the default name “Multi-Output Device” — the app finds it automatically.</li>
  </ol>
  {midi_fig}
  <p>When you record, the app switches the system output to the Multi-Output Device and switches it back when you
  stop. You don’t need to switch anything by hand.</p>
  <p class="tip"><b>Want music apps not to pause when recording starts?</b> Make the Multi-Output Device your default
  output (right-click it → Use This Device For Sound Output). Then starting and stopping a recording switches nothing.
  The catch: with the Multi-Output Device selected, the keyboard volume keys don’t work — use the volume slider in the app.</p>
  <p class="tip">If you switch between speakers and headphones, tick both. While recording, the app mutes whichever one
  you aren’t using, so sound never plays from both.</p>
</section>

<section id="ai-cli">
  <h2><span class="n">5</span>Install and log in to the AI CLI (for notes)</h2>
  <p>Polishing the transcript and writing the notes use the Claude (or Codex) command-line tool with your own account.
  Without it, recording and transcription still work — only the notes step can’t run.</p>
  <ol>
    <li>Install Node.js: download the LTS version from <a href="https://nodejs.org">nodejs.org</a>, or with Homebrew:</li>
  </ol>
  {_c("brew install node")}
  <ol start="2">
    <li>Install the Claude Code CLI:</li>
  </ol>
  {_c("npm install -g @anthropic-ai/claude-code")}
  <ol start="3">
    <li>Run <code>claude</code> once in Terminal, log in to your account when prompted, then quit it.</li>
  </ol>
  {_c("claude")}
  <p>If you use Codex instead, replace the two steps above with:</p>
  {_c("npm install -g @openai/codex")}
  {_c("codex login")}
  <p>Then, in <b>Settings → Common settings → AI models</b>, set “Polish transcript” and “Notes / reports” to Codex.</p>
  <p class="tip">Afterwards, click <b>Test</b> in <b>Settings → Environment</b>. “logged in and replying” means it works.</p>
  {env_fig}
</section>

<section id="ffmpeg">
  <h2><span class="n">6</span>ffmpeg (optional)</h2>
  <p>Only needed to import mp3, m4a and other non-WAV audio (e.g. phone recordings). Recordings the app makes itself
  don’t need it.</p>
  {_c("brew install ffmpeg")}
</section>

<section id="models">
  <h2><span class="n">7</span>Models download on first use</h2>
  <p>The speech models aren’t in the installer. Each downloads automatically the first time it’s needed, once.
  The app tells you what it is about to download and roughly how big before it starts.</p>
  <table>
    <tr><th>When</th><th>What</th><th>Size</th></tr>
    <tr><td>First “Transcribe” or notes</td><td>Chinese speech recognition, segmentation and punctuation models</td><td>~2 GB</td></tr>
    <tr><td>First live captions (English)</td><td>English streaming recognition + translation model</td><td>~1.7 GB</td></tr>
    <tr><td>First live captions (Chinese)</td><td>Chinese streaming recognition (+ the Chinese models above)</td><td>~0.2 GB</td></tr>
  </table>
  <p>Stay online during the download; progress may sit still for a while, which is normal. Keep at least 5 GB of disk free.</p>
</section>

<section id="optional">
  <h2><span class="n">8</span>Optional settings</h2>
  <ul>
    <li><b>Calendar linkage</b>: pick today’s event before recording. The recording is named after it, and its
      attendees help the notes put real names on speakers. Needs Google Calendar connected to your Claude account;
      turn it on in <b>Settings → Calendar linkage</b>, which checks the connection first.</li>
    <li><b>Hotwords</b>: colleague names, project codenames and other proper nouns. They accumulate automatically after
      each set of notes; you can also add them under <b>Settings → Common settings → Edit hotword list</b>
      (restart the app afterwards).</li>
    <li><b>Custom goals</b>: “Custom goals ▾” on the recording page. Describe the output you want in one sentence
      (e.g. “list every requirement the customer raised”) and the AI expands it into a full prompt.</li>
  </ul>
  <p>All of the app’s data lives in <code>~/Documents/meetingscribe/</code>: settings, hotwords, recordings and
  generated notes, and logs.</p>
</section>

<section id="faq">
  <h2>Troubleshooting</h2>
  <dl>
    <dt>The other side isn’t in the recording</dt>
    <dd>① Check BlackHole is installed and you restarted afterwards; ② check the Multi-Output Device has BlackHole 2ch
      ticked; ③ open <b>Settings → Environment</b> and make sure the first two rows are green. The app also warns at the
      top if BlackHole is missing when you start recording.</dd>
    <dt>I can’t hear anything while recording</dt>
    <dd>The Multi-Output Device has no speakers or headphones ticked. Tick them in Audio MIDI Setup; Settings → Environment
      shows this row in orange.</dd>
    <dt>The volume keys don’t work while recording</dt>
    <dd>A macOS limitation of Multi-Output Devices. Use the “Output volume” slider on the recording page.</dd>
    <dt>“cannot be opened” / “developer cannot be verified”</dt>
    <dd>See step 1: right-click → Open, or run the <code>xattr</code> command.</dd>
    <dt>Generating notes fails</dt>
    <dd>Click “Test” in <b>Settings → Environment</b> and install or log in to claude / codex as it suggests.</dd>
    <dt>Transcription sits at the start</dt>
    <dd>The first transcription downloads ~2 GB of models first, which takes a few minutes on a slow network.
      It won’t happen again.</dd>
    <dt>I need to report a problem</dt>
    <dd>Logs are in <code>~/Documents/meetingscribe/logs/</code>, one file per day — send the file for the day it happened.</dd>
  </dl>
</section>
"""


def build() -> None:
    shots = _capture_app()
    midi = (ASSETS / "audio-midi-setup.annotated.png").read_bytes()
    for lang, body in (("zh", _body_zh(shots["zh"], midi)),
                       ("en", _body_en(shots["en"], midi))):
        _write(lang, body, sorted(shots[lang]))


def _write(lang: str, body: str, shot_names: list) -> None:
    css = """
:root { --bg:#f5f5f7; --card:#fff; --text:#1d1d1f; --muted:#6e6e73; --line:#e3e3e8;
        --accent:#007aff; --code:#f2f2f5; --tip:#eef5ff; --warn:#fff4e5; }
@media (prefers-color-scheme: dark) {
  :root { --bg:#1c1c1e; --card:#2c2c2e; --text:#f2f2f7; --muted:#a1a1a6; --line:#3a3a3c;
          --accent:#0a84ff; --code:#3a3a3c; --tip:#1d2b3f; --warn:#3a2d17; }
}
* { box-sizing:border-box; }
body { margin:0; background:var(--bg); color:var(--text); line-height:1.7;
       font:15px/1.7 -apple-system, BlinkMacSystemFont, "PingFang SC", "Helvetica Neue", sans-serif; }
main { max-width:900px; margin:0 auto; padding:32px 20px 64px; }
header h1 { font-size:30px; margin:8px 0 6px; }
.lead { color:var(--muted); margin:0 0 16px; }
nav { display:flex; flex-wrap:wrap; gap:8px; }
nav a { background:var(--card); border:1px solid var(--line); border-radius:999px; padding:4px 12px;
        color:var(--text); text-decoration:none; font-size:13px; }
nav a:hover { border-color:var(--accent); color:var(--accent); }
section { background:var(--card); border:1px solid var(--line); border-radius:14px;
          padding:22px 26px; margin-top:18px; scroll-margin-top:16px; }
section:target { border-color:var(--accent); box-shadow:0 0 0 3px rgba(0,122,255,.15); }
h2 { font-size:20px; margin:0 0 10px; display:flex; align-items:center; gap:10px; }
.n { background:var(--accent); color:#fff; border-radius:50%; width:26px; height:26px; font-size:14px;
     display:inline-flex; align-items:center; justify-content:center; flex:none; }
a { color:var(--accent); }
code { background:var(--code); border-radius:5px; padding:1px 6px; font:13px Menlo, monospace; }
.cmd { display:flex; align-items:center; gap:10px; background:var(--code); border-radius:9px;
       padding:9px 10px 9px 14px; margin:8px 0 12px; }
.cmd code { background:none; padding:0; flex:1; overflow-x:auto; white-space:nowrap; }
.cmd button { border:1px solid var(--line); background:var(--card); color:var(--text); border-radius:7px;
              padding:3px 12px; font-size:13px; cursor:pointer; }
.cmd button.done { color:#34c759; border-color:#34c759; }
.note { color:var(--muted); font-size:14px; }
.tip, .warn { border-radius:9px; padding:10px 14px; }
.tip { background:var(--tip); } .warn { background:var(--warn); }
figure { margin:16px 0 4px; text-align:center; }
figure img { width:100%; border-radius:10px; border:1px solid var(--line); }
figcaption { color:var(--muted); font-size:13px; margin-top:6px; }
table { border-collapse:collapse; width:100%; font-size:14px; }
th, td { text-align:left; padding:8px 10px; border-bottom:1px solid var(--line); }
th { color:var(--muted); font-weight:600; }
dt { font-weight:600; margin-top:12px; } dd { margin:4px 0 0 0; color:var(--text); }
.lang { text-align:right; margin:0; font-size:13px; }
"""
    copy, copied = COPY[lang]
    js = """
function copyCmd(btn) {
  const text = btn.dataset.cmd;
  const done = () => { btn.textContent = '__COPIED__'; btn.classList.add('done');
                       setTimeout(() => { btn.textContent = '__COPY__'; btn.classList.remove('done'); }, 1500); };
  if (navigator.clipboard && window.isSecureContext) {
    navigator.clipboard.writeText(text).then(done, () => fallback(text, done));
  } else { fallback(text, done); }
}
function fallback(text, done) {
  const ta = document.createElement('textarea'); ta.value = text; document.body.appendChild(ta);
  ta.select(); try { document.execCommand('copy'); done(); } finally { ta.remove(); }
}
"""
    js = js.replace("__COPIED__", copied).replace("__COPY__", copy)
    other = OUT["en" if lang == "zh" else "zh"].name
    switch = (f'<p class="lang"><a href="{other}">English</a></p>' if lang == "zh"
              else f'<p class="lang"><a href="{other}">中文</a></p>')
    title = "MeetingScribe 安装说明" if lang == "zh" else "MeetingScribe Install Guide"
    page = (
        f'<!doctype html><html lang="{"zh-CN" if lang == "zh" else "en"}"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        f"<title>{title}</title>"
        f"<style>{css}</style></head><body><main>{switch}{body}</main>"
        f"<script>{js}</script></body></html>"
    )
    for a in ANCHORS:
        assert f'id="{a}"' in page, f"missing anchor #{a} ({lang})"
    out = OUT[lang]
    out.write_text(page, encoding="utf-8")
    print(f"wrote {out} ({out.stat().st_size // 1024} KB, screenshots: {shot_names})")


if __name__ == "__main__":
    build()
