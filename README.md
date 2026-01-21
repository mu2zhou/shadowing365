# Shadowing365: Intelligent Shadowing Video Generator
# 智能影子跟读视频生成器

Does your language learning feel dry? Shadowing365 turns any PDF book into an engaging, "high-frequency" shadowing video. With Solarized visuals, karaoke-style highlighting, and AI-powered insights, it transforms reading into an immersive audio-visual experience.

你的语言学习是否枯燥乏味？Shadowing365 将任何 PDF 书籍转化为引人入胜的“高频”影子跟读视频。配合 Solarized 护眼视觉、卡拉OK式的高亮指读以及 AI 生成的趣味知识点，让阅读变成沉浸式的视听体验。

---

## 🎯 Purpose & Design Philosophy (设计初衷)
The core goal is **Deep Immersion (深度沉浸)** and **Eye Comfort (用眼舒适)**.

*   **Visual Ergonomics**: We use the **Solarized Light** theme (Cream background + Dark Blue/Grey text) to minimize eye strain during long practice sessions.
*   **Shadowing Optimization**:
    *   **Karaoke Highlighting**: Words light up as they are spoken, guiding your rhythm and focus.
    *   **Natural Pacing**: TTS is slowed by 10% to match a comfortable shadowing speed.
    *   **Inspiration Cards (灵感卡片)**: A stylish card displays English idioms, slang, or cultural nuggets found in the text, explained in Chinese. Help you learn English in depth.

核心目标是实现 **深度沉浸** 和 **用眼舒适**。
*   **视觉人体工学**: 采用 **Solarized Light** 主题 (奶酪色背景 + 深灰蓝文字)，最大程度减少长时间练习的眼部疲劳。
*   **跟读优化**:
    *   **卡拉OK高亮**: 单词随语音逐个点亮，引导你的节奏和注意力。
    *   **自然语速**: 语音速度降低 10%，匹配舒适的跟读节奏。
    *   **灵感卡片**: 屏幕底部会出现精心设计的“知识卡片”，深度解析句中的英语地道表达（俚语、文化梗），挖掘语言背后的趣味。

---

## 🛠 Features (功能特性)
*   **PDF to Video**: Extracts text from PDF, segments it into sentences.
*   **AI Translation**: Integrates **Ollama** (offline) or online APIs for context-aware translation.
*   **Bilingual Display**: Source (English/German) + Target (Chinese) subtitles.
*   **Karaoke Logic**: Pseudo-alignment algorithm estimates word timing for dynamic highlighting.
*   **Inspiration Cards**: Auto-generates insights based on the **Inspiration Prompt**:

> **Inspiration Card Prompt (灵感卡片 Prompt)**:
> ```
> Analyze this English sentence for learners: '[SOURCE]'. 
> Find something TRULY fascinating or a common usage trap. Share: 
> - A core mental model (how native speakers 'see' this word), OR 
> - A vivid metaphor/idiom, OR 
> - A structural pattern that makes them sound smart. 
> Avoid generic facts. Be inspired and natural. 
> NEVER say '无固定习语'. If nothing obvious, connect it to a related concept. 
> Max 25 words in Chinese. Output ONLY the insight.
> ```

---

## 💻 Technical Stack (技术栈)
*   **Language**: Python 3.11+
*   **Core Libraries**:
    *   `PyMuPDF` (Extraction)
    *   `spaCy` (NLP/Segmentation)
    *   `MoviePy` & `Pillow` (Video Synthesis)
    *   `Ollama` (Local LLM Integration)
    *   `Edge-TTS` (Natural Speech Synthesis)

---

## ⚙️ Hardware & Model Recommendations (硬件与模型推荐)

### GPU Requirements (GPU 要求)
*   **Lightweight**: CPU-only is possible but slow for translation.
*   **Recommended**: Nvidia GPU (8GB+ VRAM).
*   **User Setup (A5500 / 24GB VRAM)**:
    *   You have an **Nvidia RTX A5500 (24GB)**. This is powerful but has limits.

### AI Model Strategy (AI 模型策略)
To get the best translation results:

1.  **DeepSeek-V3 (The "Best" So Far)**:
    *   **Offline**: Impossible on single 24GB card (Requires ~350GB+ VRAM).
    *   **Solution**: Use the **DeepSeek API** (Online). It is extremely cheap and effective.
    *   **Config**: Set `translation_provider: openai` and use DeepSeek base URL.

2.  **Qwen-2.5-14B (Best Offline Option)**:
    *   Fits comfortably in 24GB VRAM.
    *   Excellent bilingual capability.
    *   Fast inference.

3.  **Qwen-2.5-32B-Int4**:
    *   Fits tightly in 20-22GB VRAM.
    *   Better reasoning than 14B.

---

## 🚀 Usage (使用方法)

### 1. Installation 
```bash
pip install -r requirements.txt
# Ensure you have 'ollama' installed system-wide
```

### 2. Configuration (`config.yaml`)
Customize your experience:
```yaml
input_file: "input/my_book.pdf"
source_lang: "en"
translation_provider: "ollama"  # or 'openai' for DeepSeek API
ollama_model: "qwen2.5:14b"
enable_trivia: true # Turn on/off inspiration text
theme: "solarized_light"
```

### 3. Run
```bash
python pdf_to_video.py
```
Check `output/` for your video!

---

## 🔮 Roadmap (未来展望)
*   **Scrolling Subtitles (Teleprompter Mode)**: Continuous scrolling text for fluid reading.
*   **Strict Alignment**: Integrate `Aeneas` or `Montreal Forced Aligner` for millisecond-perfect karaoke.
*   **Multi-Speaker**: Assign different voices to different characters in fiction books.
*   **Mobile App**: Package as a Flutter/React Native app for on-the-go generation.

---

## 🐟 Fish Speech Integration (SOTA Voice Cloning)

We have integrated **Fish Speech 1.5** (SOTA Open Source TTS) for cinema-grade voice quality.

### Pros & Cons (优缺点)
*   **✅ Pros**: 
    *   **Incredible Realism**: Far superior to standard TTS. Sounds like a real human breathing and pausing.
    *   **Voice Cloning**: Clone *any* voice from a 15s reference audio (British, American, Anime characters, etc.).
    *   **Context Aware**: Understands emotion and prosody better.
*   **❌ Cons**:
    *   **Heavy Resource Usage**: Requires Nvidia GPU (8GB+ VRAM recommended).
    *   **Slower Generation**: Unlike Edge-TTS (instant), Fish Speech takes time (~2-5s per sentence on A5500).
    *   **Complex Deployment**: Requires a dedicated Docker API server.

### 🛡️ Reliability & Deploying (部署难点与方案)
Fish Speech is notoriously hard to deploy due to dependency conflicts. We solved this by:
1.  **Docker Isolation**: Running the API server in a validated container (`fishaudio/fish-speech:latest-server-cuda`).
2.  **Critical Server Patches**: We successfully patched the server core to fix:
    *   **Crash on WAV loading**: Switched from `ffmpeg` to `soundfile` backend.
    *   **415 Errors**: Implemented manual Base64 decoding for robust API communication.
3.  **Resume Capability (断点续传)**: 
    *   **Smart Checkpointing**: The script now saves `step1_translated.json` and `step2_audio.json`.
    *   **Crash Protection**: If the 5-hour task crashes at 99%, simply restart. It will **Skip** already translated text and **Skip** already generated audio files, finishing the rest in seconds.
    *   **Visual Progress**: Added `tqdm` progress bars to tell you exactly how long is left.

### Quick Start with Fish Speech
1.  **Start Server**: `docker compose -f fish_speech/docker-compose.yml up -d`
2.  **Config**: Set `tts_provider: "fish_speech"` in `config.yaml`.
3.  **Choose Your Voice**: We have included several high-quality samples in `input/`:
    *   `ref_voice_obama.wav`: Iconic rhetorical American (Barack Obama).
    *   `ref_voice_pure_american.wav`: Clean, patient narration (Phil Chenevert).
    *   `ref_voice_us_broadcast.wav`: Standard broadcast style.
    *   `ref_voice.wav`: Professional British accent.
4.  **Legal Attribution**: Please refer to [CREDITS.md](CREDITS.md) for licensing and source information for these voice samples.
5.  **Run**: `python pdf_to_video.py`
# shadowing365
