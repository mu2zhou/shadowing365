import os
import yaml
import logging
import argparse
from typing import List, Dict
from tqdm import tqdm
import shutil

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ShadowingVideoGenerator:
    def __init__(self, config_path: str):
        self.config = self._load_config(config_path)
        self.output_dir = self.config.get('output_dir', 'output')
        os.makedirs(self.output_dir, exist_ok=True)
        
    def _load_config(self, path: str) -> dict:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Config file not found: {path}")
        with open(path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)

    def extract_text(self, pdf_path: str) -> str:
        """Extracts raw text from PDF using PyMuPDF (or PyMuPDF4LLM for better markdown)."""
        logging.info(f"Extracting text from {pdf_path}...")
        
        # Try PyMuPDF4LLM first for smarter extraction (markdown)
        try:
            import pymupdf4llm
            logging.info("Using pymupdf4llm for smart extraction...")
            # Check if we need to limit pages
            max_pages = self.config.get('max_pages', 0)
            pages_arg = None
            if max_pages > 0:
                # pymupdf4llm supports `pages` argument as a list of page indices (0-based)
                pages_arg = list(range(max_pages))
                logging.info(f"Limiting to first {max_pages} pages.")
            
            # to_markdown returns string
            text = pymupdf4llm.to_markdown(pdf_path, pages=pages_arg)
            logging.info(f"Extracted {len(text)} characters using pymupdf4llm.")
            if len(text) < 500:
                logging.info(f"Extracted text preview: {text}")
            return text
        except ImportError:
            pass

        try:
            import fitz  # PyMuPDF
        except ImportError:
            logging.error("PyMuPDF (fitz) not installed.")
            return ""

        doc = fitz.open(pdf_path)
        text = ""
        max_pages = self.config.get('max_pages', 0)
        # ... (fitz logic) ...
        return text

    def segment_text(self, text: str) -> List[str]:
        """Segments text into sentences/phrases suitable for subtitles."""
        logging.info("Segmenting text...")
        if not text:
             logging.warning("No text to segment!")
             return []
             
        # ...
        for i, page in enumerate(doc):
            if max_pages > 0 and i >= max_pages:
                logging.info(f"Limiting to first {max_pages} pages.")
                break
            text += page.get_text()
        return text

    def segment_text(self, raw_text: str) -> List[str]:
        """Segments text into sentences using spaCy."""
        logging.info("Segmenting text...")
        try:
            import spacy
        except ImportError:
            logging.error("spaCy not installed.")
            return []

        # Load spaCy model (ensure it's downloaded: python -m spacy download en_core_web_sm)
        # For German, we would need 'de_core_news_sm'
        model_name = "en_core_web_sm" if self.config.get('source_lang') == 'en' else "de_core_news_sm"
        try:
            nlp = spacy.load(model_name)
        except OSError:
            logging.warning(f"Model {model_name} not found, downloading...")
            from spacy.cli import download
            download(model_name)
            nlp = spacy.load(model_name)

        # Fix hyphenation across line breaks (e.g., "ha- \n ving" -> "having") and remove soft hyphens
        import re
        raw_text = raw_text.replace('\u00AD', '')  # Remove soft hyphens (soft hyphen)
        # Remove hyphen followed by whitespace/newline and a letter (likely line-break hyphenation)
        raw_text = re.sub(r'([A-Za-z])-\s+([A-Za-z])', r'\1\2', raw_text)
        # Basic cleaning before segmentation (collapse whitespace)
        clean_text = " ".join(raw_text.split())  # Remove parsing artifacts like multiple spaces/newlines

        # Normalize Chinese punctuation to English equivalents
        punctuation_map = {
            '，': ',',   # Chinese comma to English comma
            '。': '.',   # Chinese period to English period
            '；': ';',   # Chinese semicolon
            '：': ':',   # Chinese colon
            '？': '?',   # Chinese question mark
            '！': '!',   # Chinese exclamation mark
            '“': '"',   # Chinese left quote
            '”': '"',   # Chinese right quote
            '‘': "'",   # Chinese left single quote
            '’': "'",   # Chinese right single quote
            '—': '-',   # Chinese em dash
            '…': '...',  # Chinese ellipsis
        }
        for cn_punct, en_punct in punctuation_map.items():
            clean_text = clean_text.replace(cn_punct, en_punct)
        
        doc = nlp(clean_text)
        sentences = [sent.text.strip() for sent in doc.sents if len(sent.text.strip()) > 0]
        
        # Filter by length if needed
        min_len = self.config.get('min_segment_len', 20)  # Default minimum 20 chars
        max_len = self.config.get('max_segment_len', 9999)
        
        filtered_sentences = []
        for s in sentences:
            # Skip very short fragments
            if len(s) < min_len:
                # Try to merge with previous sentence if exists
                if filtered_sentences:
                    filtered_sentences[-1] = filtered_sentences[-1] + " " + s
                continue
            
            # Split long sentences at clause boundaries
            if len(s) > max_len:
                import re
                # Split at comma, semicolon, colon followed by space
                parts = re.split(r'(?<=[,;:])\s+', s)
                current_chunk = ""
                for part in parts:
                    if len(current_chunk) + len(part) <= max_len:
                        current_chunk += (" " if current_chunk else "") + part
                    else:
                        # Only add chunk if it's meaningful (> 40 chars)
                        if current_chunk and len(current_chunk) >= 40:
                            filtered_sentences.append(current_chunk.strip())
                        elif current_chunk and filtered_sentences:
                            # Merge short chunk with previous
                            filtered_sentences[-1] = filtered_sentences[-1] + " " + current_chunk.strip()
                        current_chunk = part
                # Handle remaining chunk
                if current_chunk:
                    if len(current_chunk) >= 40:
                        filtered_sentences.append(current_chunk.strip())
                    elif filtered_sentences:
                        filtered_sentences[-1] = filtered_sentences[-1] + " " + current_chunk.strip()
                    else:
                        filtered_sentences.append(current_chunk.strip())
            else:
                filtered_sentences.append(s)
            
        logging.info(f"Segmented into {len(filtered_sentences)} sentences.")
        return filtered_sentences

    async def _translate_with_ollama(self, segments: List[str]) -> List[str]:
        import ollama
        host = self.config.get('ollama_base_url', 'http://localhost:11434')
        model = self.config.get('ollama_model', 'qwen2.5:7b')
        client = ollama.Client(host=host)
        
        translated = []
        for seg in tqdm(segments, desc="Translating", unit="seg"):
            try:
                # Prompt engineering for better translation
                prompt = (
                    f"Translate the following sentence into {self.config.get('target_lang', 'zh')}. "
                    "Output ONLY the translation, no explanation.\n\n"
                    f"{seg}"
                )
                response = client.generate(model=model, prompt=prompt)
                translated.append(response['response'].strip())
            except Exception as e:
                logging.error(f"Ollama translation failed: {e}")
                translated.append("[Trans Error]")
        return translated

    def translate_text(self, segments: List[str]) -> List[Dict[str, str]]:
        """Translates segments. Returns list of {'source': ..., 'target': ...}"""
        logging.info("Translating text...")
        provider = self.config.get('translation_provider', 'mock')
        results = []
        
        if provider == 'mock':
            for seg in segments:
                results.append({
                    "source": seg,
                    "target": f"[译] {seg}" 
                })
        elif provider == 'ollama':
            import ollama
            host = self.config.get('ollama_base_url', 'http://localhost:11434')
            model = self.config.get('ollama_model', 'qwen2.5:7b')
            try:
                client = ollama.Client(host=host)
            except Exception as e:
                logging.error(f"Failed to connect to Ollama: {e}")
                return [{"source": s, "target": "[Error]"} for s in segments]
            
            logging.info(f"Using Ollama ({model}) at {host}")
            
            # Use tqdm for progress
            for i, seg in enumerate(tqdm(segments, desc="Translating", unit="seg")):
                try:
                    prompt = (
                        f"Translate the following sentence into Chinese (Simplified). "
                        "Output ONLY the translation, do not include original text or notes.\n\n"
                        f"Sentence: {seg}"
                    )
                    resp = client.generate(model=model, prompt=prompt)
                    results.append({
                        "source": seg,
                        "target": resp['response'].strip()
                    })
                except Exception as e:
                    logging.error(f"Ollama error: {e}")
                    results.append({"source": seg, "target": "[Error]"})

            # Trivia Generation (Optional)
            if self.config.get('enable_trivia', False):
                logging.info("Generating trivia...")
                for i, item in enumerate(tqdm(results, desc="Generating Trivia", unit="item")):
                    try:
                        # Improved Prompt for "English Learning Inspiration"
                        prompt_trivia = (
                            f"Analyze this English sentence for learners: '{item['source']}'. "
                            "Find something TRULY fascinating or a common usage trap. Share: "
                            "- A core mental model (how native speakers 'see' this word), OR "
                            "- A vivid metaphor/idiom, OR "
                            "- A structural pattern that makes them sound smart. "
                            "Avoid generic facts. Be inspired and natural. "
                            "NEVER say '无固定习语'. If nothing obvious, connect it to a related concept. "
                            "Max 25 words in Chinese. Output ONLY the insight."
                        )
                        resp = client.generate(model=model, prompt=prompt_trivia)
                        triv = resp['response'].strip()
                        # Simple filter: if too long or empty, skip
                        if len(triv) > 1 and len(triv) < 100:
                            item['trivia'] = triv
                        else:
                            item['trivia'] = ""
                    except Exception as e:
                        logging.warning(f"Trivia failed: {e}")
                        item['trivia'] = ""

        logging.info(f"Translated {len(results)} segments.")
        return results

    async def _generate_audio_edge_tts(self, text: str, voice: str, output_path: str) -> None:
        import edge_tts
        # Slower rate for shadowing (-10%) + slight pitch adjustment for naturalness
        rate = self.config.get('tts_rate', '-10%') 
        pitch = self.config.get('tts_pitch', '+0Hz')
        communicate = edge_tts.Communicate(text, voice, rate=rate, pitch=pitch)
        await communicate.save(output_path)

    async def _fish_post(self, payload: dict):
        """Low-level HTTP POST for Fish Speech. Separated for easier testing/mocking."""
        import httpx
        url = self.config.get('fish_speech_url', 'http://localhost:8000/v1/tts')
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(url, json=payload)
            return resp

    async def _generate_audio_fish_speech(self, text: str, output_path: str, segment_index: int = None) -> bool:
        """Generates audio using Fish Speech API server with robust retries and chunking fallback.
        Strategy:
        1) Try several attempts with exponential backoff + jitter
        2) If all attempts fail, split text into clause/sentence-sized chunks and synthesize each chunk, then concatenate
        3) Save raw responses and errors under output/debug_fish/<segment_index>/ for post-mortem
        Returns True if final MP3 file written and non-empty, False otherwise.
        """
        import base64, random, tempfile, subprocess, asyncio

        ref_audio_path = self.config.get('fish_speech_ref_audio')
        if not ref_audio_path or not os.path.exists(ref_audio_path):
            logging.error(f"Fish Speech reference audio not found: {ref_audio_path}")
            return False

        # Read and prepare reference audio
        with open(ref_audio_path, 'rb') as f:
            audio_bytes = f.read()
        ref_audio_b64 = base64.b64encode(audio_bytes).decode('utf-8')

        # Retry/backoff configuration
        max_attempts = int(self.config.get('fish_retry_attempts', 4))
        backoff_base = float(self.config.get('fish_retry_backoff_base', 0.5))
        jitter = float(self.config.get('fish_retry_jitter', 0.3))
        chunk_word_limit = int(self.config.get('fish_chunk_word_limit', 45))

        debug_dir = os.path.join(self.output_dir, 'debug_fish', f'seg_{segment_index:04d}' if segment_index is not None else 'seg_unknown')
        os.makedirs(debug_dir, exist_ok=True)

        async def attempt_request(payload, attempt_no):
            try:
                resp = await self._fish_post(payload)
                with open(os.path.join(debug_dir, f'attempt_{attempt_no}.bin'), 'wb') as bf:
                    bf.write(resp.content)
                if resp.status_code == 200:
                    return True, resp.content
                else:
                    with open(os.path.join(debug_dir, f'attempt_{attempt_no}_error.txt'), 'w', encoding='utf-8') as ef:
                        ef.write(resp.text)
                    return False, resp.text
            except Exception as e:
                with open(os.path.join(debug_dir, f'attempt_{attempt_no}_exc.txt'), 'w', encoding='utf-8') as ef:
                    ef.write(str(e))
                return False, str(e)

        # Build base payload
        base_payload = {
            'text': text,
            'chunk_length': int(self.config.get('fish_chunk_length', 200)),
            'format': 'mp3',
            'references': [
                {
                    'text': '',
                    'audio': ref_audio_b64
                }
            ],
            'normalize': True,
            'top_p': 0.7,
            'repetition_penalty': 1.2,
            'temperature': 0.7,
            'streaming': False
        }

        # 1) Simple retry loop for the whole text
        for attempt in range(1, max_attempts + 1):
            ok, data = await attempt_request(base_payload, attempt)
            if ok:
                # Write mp3 content
                with open(output_path, 'wb') as out:
                    out.write(data)
                return True
            # backoff with jitter
            delay = backoff_base * (2 ** (attempt - 1)) * (1.0 + random.uniform(0.0, jitter))
            logging.warning(f"Fish Speech attempt {attempt}/{max_attempts} failed for segment {segment_index}; retrying in {delay:.2f}s")
            await asyncio.sleep(delay)

        # 2) If full requests fail, try chunked approach
        logging.info(f"Fish Speech full-text attempts failed for segment {segment_index}; trying chunked requests")
        # Split into clauses/sentences by punctuation and spaces
        import re
        parts = re.split(r'(?<=[\.;,:?])\s+', text)
        # Recombine into word-limited chunks
        chunks = []
        cur = ''
        for p in parts:
            if not cur:
                cur = p
            elif len((cur + ' ' + p).split()) <= chunk_word_limit:
                cur = cur + ' ' + p
            else:
                chunks.append(cur.strip())
                cur = p
        if cur:
            chunks.append(cur.strip())

        if len(chunks) == 1:
            logging.warning(f"Chunking did not produce smaller pieces for segment {segment_index}")
            return False

        chunk_files = []
        for ci, chunk in enumerate(chunks):
            chunk_payload = dict(base_payload)
            chunk_payload['text'] = chunk
            success = False
            for attempt in range(1, max_attempts + 1):
                ok, data = await attempt_request(chunk_payload, f'chunk{ci}_att{attempt}')
                if ok:
                    chunk_path = os.path.join(debug_dir, f'chunk_{ci}.mp3')
                    with open(chunk_path, 'wb') as cf:
                        cf.write(data)
                    chunk_files.append(chunk_path)
                    success = True
                    break
                delay = backoff_base * (2 ** (attempt - 1)) * (1.0 + random.uniform(0.0, jitter))
                await asyncio.sleep(delay)
            if not success:
                logging.error(f"Chunk {ci} failed for segment {segment_index}; aborting chunked approach")
                return False

        # Concatenate chunks into final output via ffmpeg
        try:
            tmpdir = tempfile.mkdtemp(prefix='fish_concat_')
            concat_txt = os.path.join(tmpdir, 'concat.txt')
            with open(concat_txt, 'w', encoding='utf-8') as cf:
                for p in chunk_files:
                    cf.write(f"file '{p}'\n")
            # Re-encode concatenated chunks directly to MP3 to ensure a valid container and correct timestamps
            final_tmp = os.path.join(tmpdir, 'out_final.mp3')
            subprocess.check_call([
                'ffmpeg', '-y', '-hide_banner', '-nostats', '-f', 'concat', '-safe', '0', '-i', concat_txt,
                '-c:a', 'libmp3lame', '-q:a', '4', final_tmp
            ])
            os.replace(final_tmp, output_path)
            return True
        except Exception as e:
            logging.error(f"Failed to concatenate Fish Speech chunks for segment {segment_index}: {e}")
            return False

    async def _generate_audio_kokoro(self, text: str, output_path: str) -> bool:
        """Generates audio using Kokoro TTS."""
        try:
            from kokoro import KPipeline
            import soundfile as sf
            import numpy as np
        except ImportError:
            logging.error("Kokoro or soundfile not installed. Run: pip install kokoro soundfile numpy torch")
            return False

        lang_code = self.config.get('kokoro_lang_code', 'a')  # 'a' for American English, 'b' for British
        voice = self.config.get('kokoro_voice', 'af_heart')
        model_id = self.config.get('kokoro_model_path', 'hexgrad/Kokoro-82M')
        
        # Use a singleton/cached pipeline if possible to avoid reloading model
        if not hasattr(self, '_kokoro_pipeline'):
            logging.info(f"Initializing Kokoro pipeline (lang={lang_code}, model={model_id})...")
            # Ensure CUDA if available, else CPU
            import torch
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            # Check if model_id is a local existing path
            kmodel = None
            if os.path.exists(model_id):
                 logging.info(f"Using local Kokoro model at {model_id}")
                 try:
                     from kokoro import KModel
                     config_path = os.path.join(model_id, 'config.json')
                     pth_path = os.path.join(model_id, 'kokoro-v1_0.pth')
                     
                     if not os.path.exists(pth_path):
                         # Fallback search for .pth
                         for f in os.listdir(model_id):
                             if f.endswith('.pth') and 'kokoro' in f:
                                 pth_path = os.path.join(model_id, f)
                                 break
                     
                     if os.path.exists(config_path) and os.path.exists(pth_path):
                         logging.info(f"Found config: {config_path} and model: {pth_path}")
                         # Initialize KModel manually to avoid HF download
                         kmodel = KModel(repo_id=None, config=config_path, model=pth_path).to(device).eval()
                     else:
                         logging.warning(f"Could not find config.json or .pth in {model_id}")
                 except Exception as e:
                     logging.error(f"Failed to load local KModel: {e}")

            if kmodel:
                self._kokoro_pipeline = KPipeline(lang_code=lang_code, model=kmodel, device=device)
            else:
                self._kokoro_pipeline = KPipeline(lang_code=lang_code, device=device, repo_id=model_id)
        
        pipeline = self._kokoro_pipeline
        
        # Resolve voice path if using local model
        if os.path.isdir(model_id) and not voice.endswith('.pt'):
             # Try to find voice in voices subdir
             voice_path = os.path.join(model_id, 'voices', f"{voice}.pt")
             if os.path.exists(voice_path):
                 logging.info(f"Using local voice: {voice_path}")
                 voice = voice_path
        
        # Generator returns chunks: (graphemes, phonemes, audio_tensor)
        generator = pipeline(text, voice=voice, speed=1, split_pattern=r'\n+')
        
        all_audio = []
        for i, (gs, ps, audio) in enumerate(generator):
            if audio is not None:
                all_audio.append(audio)
                logging.info(f"Kokoro chunk {i}: {len(audio)} samples")
            else:
                logging.warning(f"Kokoro chunk {i}: No audio")
        
        if not all_audio:
            logging.warning("Kokoro produced no audio chunks")
            return False
            
        final_audio = np.concatenate(all_audio)
        logging.info(f"Writing Kokoro audio to {output_path}, shape={final_audio.shape}")
        
        # Kokoro usually outputs at 24000 Hz
        sf.write(output_path, final_audio, 24000)
        return True

    def generate_audio(self, bilingual_data: List[Dict[str, str]]) -> List[Dict[str, any]]:
        """Generates audio for each segment and updates data with audio paths."""
        logging.info("Generating audio...")
        import asyncio
        
        provider = self.config.get('tts_provider', 'edge')
        voice = self.config.get('tts_voice', 'en-US-ChristopherNeural')
        temp_audio_dir = os.path.join(self.output_dir, "temp_audio")
        os.makedirs(temp_audio_dir, exist_ok=True)
        
        updated_data = []
        
        async def process_batch():
            # Use tqdm for progress tracking
            for i, item in enumerate(tqdm(bilingual_data, desc="Generating Audio", unit="seg")):
                file_name = f"seg_{i:04d}.mp3"
                file_path = os.path.join(temp_audio_dir, file_name)
                
                # Check if file exists to save time (RESUME capability)
                text_to_speak = item['source']
                need_generate = (not os.path.exists(file_path)) or os.path.getsize(file_path) == 0
                # Define providers for potential fallback/validity retry even if no generation is needed initially
                # Default fallback providers if user didn't configure any
                providers_to_try = [provider] + list(self.config.get('tts_fallback_providers', ['edge', 'mock']))

                if need_generate:
                    # Try primary provider and any configured fallback providers, with retries and exponential backoff
                    retry_attempts = int(self.config.get('tts_retry_attempts', 2))
                    backoff_base = float(self.config.get('tts_retry_backoff_base', 0.5))
                    generated = False
                    errors = []

                    for prov in providers_to_try:
                        for attempt in range(1, retry_attempts + 1):
                            logging.info(f"Attempting TTS provider '{prov}' (attempt {attempt}/{retry_attempts}) for segment {i}")
                            try:
                                if prov == 'fish_speech':
                                    success = await self._generate_audio_fish_speech(text_to_speak, file_path)
                                    if not success:
                                        msg = f"Fish Speech attempt {attempt} failed for segment {i}"
                                        logging.warning(msg)
                                        errors.append({'provider': prov, 'attempt': attempt, 'error': msg})
                                        # exponential backoff before retrying
                                        if attempt < retry_attempts:
                                            await asyncio.sleep(backoff_base * (2 ** (attempt - 1)))
                                        continue
                                        if attempt < retry_attempts:
                                            await asyncio.sleep(backoff_base * (2 ** (attempt - 1)))
                                        continue
                                elif prov == 'kokoro':
                                    success = await self._generate_audio_kokoro(text_to_speak, file_path)
                                    if not success:
                                        msg = f"Kokoro attempt {attempt} failed for segment {i}"
                                        logging.warning(msg)
                                        errors.append({'provider': prov, 'attempt': attempt, 'error': msg})
                                        if attempt < retry_attempts:
                                            await asyncio.sleep(backoff_base * (2 ** (attempt - 1)))
                                        continue
                                elif prov == 'mock':
                                    # Create a short 0.8s sine wave WAV as mock audio so the pipeline can run without external TTS
                                    try:
                                        import wave, struct, math
                                        sample_rate = 22050
                                        duration = self.config.get('mock_tts_duration', 0.8)
                                        freq = 220.0
                                        n_samples = int(sample_rate * duration)
                                        with wave.open(file_path, 'w') as wf:
                                            wf.setnchannels(1)
                                            wf.setsampwidth(2)  # 16-bit
                                            wf.setframerate(sample_rate)
                                            for s in range(n_samples):
                                                t = float(s) / sample_rate
                                                val = int(16000.0 * math.sin(2.0 * math.pi * freq * t))
                                                wf.writeframes(struct.pack('<h', val))
                                    except Exception as e:
                                        msg = f"Mock TTS write failed on attempt {attempt} for segment {i}: {e}"
                                        logging.warning(msg)
                                        errors.append({'provider': prov, 'attempt': attempt, 'error': str(e)})
                                        if attempt < retry_attempts:
                                            await asyncio.sleep(backoff_base * (2 ** (attempt - 1)))
                                        continue
                                else:
                                    # Use edge-tts for other providers; if text is long, chunk into sentence/phrase pieces and concat
                                    try:
                                        edge_chunk_limit = int(self.config.get('edge_chunk_word_limit', 60))
                                        words = len(text_to_speak.split())
                                        if words > edge_chunk_limit:
                                            # split into sentence-ish parts
                                            import re, tempfile, subprocess
                                            parts = re.split(r'(?<=[\.;,:?])\s+', text_to_speak)
                                            chunks = []
                                            cur = ''
                                            for p in parts:
                                                if not cur:
                                                    cur = p
                                                elif len((cur + ' ' + p).split()) <= edge_chunk_limit:
                                                    cur = cur + ' ' + p
                                                else:
                                                    chunks.append(cur.strip())
                                                    cur = p
                                            if cur:
                                                chunks.append(cur.strip())

                                            tmpdir = tempfile.mkdtemp(prefix='edge_chunks_')
                                            chunk_files = []
                                            for ci, ch in enumerate(chunks):
                                                chunk_path = os.path.join(tmpdir, f'chunk_{ci}.mp3')
                                                await self._generate_audio_edge_tts(ch, voice, chunk_path)
                                                chunk_files.append(chunk_path)

                                            concat_txt = os.path.join(tmpdir, 'concat.txt')
                                            with open(concat_txt, 'w', encoding='utf-8') as cf:
                                                for p in chunk_files:
                                                    cf.write(f"file '{p}'\n")
                                            # Re-encode concatenated chunks directly to MP3 to ensure valid container and correct timestamps
                                            final_tmp = os.path.join(tmpdir, 'out_final.mp3')
                                            subprocess.check_call([
                                                'ffmpeg', '-y', '-hide_banner', '-nostats', '-f', 'concat', '-safe', '0', '-i', concat_txt,
                                                '-c:a', 'libmp3lame', '-q:a', '4', final_tmp
                                            ])
                                            os.replace(final_tmp, file_path)
                                        else:
                                            await self._generate_audio_edge_tts(text_to_speak, voice, file_path)
                                    except Exception as e:
                                        msg = f"Edge TTS attempt {attempt} failed for segment {i}: {e}"
                                        logging.warning(msg)
                                        errors.append({'provider': prov, 'attempt': attempt, 'error': str(e)})
                                        if attempt < retry_attempts:
                                            await asyncio.sleep(backoff_base * (2 ** (attempt - 1)))
                                        continue

                                # If we reach here, a file was written — record provider and validate it below (outside providers loop)
                                generated = True
                                last_provider = prov
                                break
                            except Exception as e:
                                msg = f"TTS provider '{prov}' failed on attempt {attempt} for segment {i}: {e}"
                                logging.warning(msg)
                                errors.append({'provider': prov, 'attempt': attempt, 'error': str(e)})
                                if attempt < retry_attempts:
                                    await asyncio.sleep(backoff_base * (2 ** (attempt - 1)))
                        if generated:
                            break

                    if not generated:
                        # After trying providers and retries, abort — we require valid audio for each segment
                        logging.error(f"TTS generation failed for segment {i} after retries and fallbacks: providers={providers_to_try}, errors={errors}")
                        # Write errors for debugging
                        try:
                            import json
                            err_file = os.path.join(self.output_dir, 'failed_tts.json')
                            with open(err_file, 'a', encoding='utf-8') as ef:
                                ef.write(json.dumps({'segment_index': i, 'text': text_to_speak, 'errors': errors}, ensure_ascii=False) + '\n')
                        except Exception:
                            pass
                        raise RuntimeError(f"TTS generation failed for segment {i}")


                # Validate the generated audio isn't empty or too short
                try:
                    # Short post-processing: compress any long internal silences if configured
                    max_internal = float(self.config.get('max_internal_silence', 1.0))
                    replacement = float(self.config.get('internal_silence_replacement', 0.18))
                    if max_internal > 0:
                        changed = self._compress_internal_silences(file_path, max_internal, replacement)
                        if changed:
                            logging.info(f"Compressed internal silences for {file_path}")

                    # Prefer MoviePy to validate duration if available
                    try:
                        from moviepy.editor import AudioFileClip
                        moviepy_available = True
                    except Exception:
                        moviepy_available = False

                    # Validate audio using the robust validator (duration + RMS checks)
                    estimate = self._estimate_duration_from_text(text_to_speak)
                    audio_stats = self._is_audio_valid(file_path, estimated_duration=estimate)

                    # If this was produced by a fallback provider (e.g., Edge), enforce a stricter duration requirement
                    fallback_min_ratio = float(self.config.get('fallback_min_duration_ratio', 0.85))
                    if audio_stats.get('valid', False) and 'last_provider' in locals() and last_provider != 'fish_speech':
                        if audio_stats.get('dur', 0.0) < max(self.config.get('min_audio_duration', 0.05), estimate * fallback_min_ratio):
                            logging.warning(f"Fallback provider '{last_provider}' produced audio shorter than expected (dur={audio_stats.get('dur'):.2f}s, est={estimate:.2f}s); marking as invalid and attempting retry/fallback...")
                            audio_stats['valid'] = False

                    if not audio_stats.get('valid', False):
                        logging.warning(f"Audio file for segment {i} failed validation (dur={audio_stats.get('dur')}, rms={audio_stats.get('rms')}, non_silent_ratio={audio_stats.get('non_silent_ratio')}). Attempting one retry/fallback...")

                        # If fallback Edge produced the short/invalid output, save the input text for debugging
                        try:
                            if 'last_provider' in locals() and last_provider == 'edge':
                                dbg_dir = os.path.join(self.output_dir, 'debug_edge', f'seg_{i:04d}')
                                os.makedirs(dbg_dir, exist_ok=True)
                                with open(os.path.join(dbg_dir, 'text.txt'), 'w', encoding='utf-8') as tf:
                                    tf.write(text_to_speak[:2000])
                        except Exception:
                            pass

                        # Try one extra provider retry cycle (respecting providers_to_try loop outer logic)
                        # Re-run the provider loop in the immediate context to attempt regeneration
                        regenerated = False
                        for prov in providers_to_try:
                            try:
                                if prov == 'fish_speech':
                                    success = await self._generate_audio_fish_speech(text_to_speak, file_path)
                                    if not success:
                                        continue
                                elif prov == 'mock':
                                    import wave, struct, math
                                    sample_rate = 22050
                                    duration = self.config.get('mock_tts_duration', 0.8)
                                    freq = 220.0
                                    n_samples = int(sample_rate * duration)
                                    with wave.open(file_path, 'w') as wf:
                                        wf.setnchannels(1)
                                        wf.setsampwidth(2)
                                        wf.setframerate(sample_rate)
                                        for s in range(n_samples):
                                            t = float(s) / sample_rate
                                            val = int(16000.0 * math.sin(2.0 * math.pi * freq * t))
                                            wf.writeframes(struct.pack('<h', val))
                                elif prov == 'kokoro':
                                    success = await self._generate_audio_kokoro(text_to_speak, file_path)
                                    if not success:
                                        continue
                                else:
                                    try:
                                        await self._generate_audio_edge_tts(text_to_speak, voice, file_path)
                                    except Exception:
                                        continue

                                audio_stats = self._is_audio_valid(file_path, estimated_duration=estimate)
                                if audio_stats.get('valid', False):
                                    regenerated = True
                                    break
                            except Exception as e:
                                logging.warning(f"Retry attempt failed for segment {i} with provider {prov}: {e}")
                                continue

                        if not regenerated:
                            logging.error(f"Audio invalid after retry for segment {i}; aborting pipeline. stats={audio_stats}")
                            raise RuntimeError(f"Invalid audio generated for segment {i}")

                    item['audio_path'] = file_path
                except Exception as e:
                    logging.error(f"Audio validation failed for segment {i}: {e}")
                    raise RuntimeError(f"Audio validation failed for segment {i}: {e}")

                updated_data.append(item)
        
        asyncio.run(process_batch())
        logging.info(f"Generated audio for {len(updated_data)} segments.")
        return updated_data

    # ... (Keep _wrap_text) ...

    def _wrap_text(self, text: str, font, max_width: int, draw) -> str:
        """Wraps text to fit within a maximum width.
        Handles CJK by breaking between characters when there are no spaces.
        """
        if not text:
            return ""

        # Quick heuristic to detect CJK characters
        import re
        has_cjk = bool(re.search(r"[\u4E00-\u9FFF]", text))

        lines = []
        if has_cjk and ' ' not in text:
            # Break into characters and fit as many as will fit per line
            chars = list(text)
            current_line = ''
            for ch in chars:
                test_line = current_line + ch
                w = draw.textbbox((0, 0), test_line, font=font)[2]
                if w <= max_width:
                    current_line = test_line
                else:
                    if current_line:
                        lines.append(current_line)
                    current_line = ch
            if current_line:
                lines.append(current_line)
        else:
            # Use whitespace-based wrapping (suitable for English and mixed text)
            words = text.split()
            current_line = []
            for word in words:
                test_line = " ".join(current_line + [word])
                w = draw.textbbox((0, 0), test_line, font=font)[2]
                if w <= max_width:
                    current_line.append(word)
                else:
                    lines.append(" ".join(current_line))
                    current_line = [word]
            lines.append(" ".join(current_line))

        return "\n".join(lines)

    def _is_audio_valid(self, file_path: str, estimated_duration: float = None) -> dict:
        """Validate audio file is non-empty, has sufficient duration, and non-trivial amplitude (RMS).
        Returns a dict with keys: valid (bool), dur (float), rms (float), non_silent_ratio (float)
        """
        res = {'valid': False, 'dur': 0.0, 'rms': 0.0, 'non_silent_ratio': 0.0}
        min_dur_cfg = self.config.get('min_audio_duration', 0.05)
        min_rms = self.config.get('min_audio_rms', 1e-3)
        min_non_silent_ratio = self.config.get('min_non_silent_ratio', 0.05)
        min_duration_ratio = self.config.get('min_duration_ratio', 0.6)

        try:
            try:
                from moviepy.editor import AudioFileClip
                moviepy_available = True
            except Exception:
                moviepy_available = False

            if moviepy_available:
                ac = AudioFileClip(file_path)
                dur = ac.duration
                res['dur'] = dur or 0.0
                if not dur or dur < min_dur_cfg:
                    ac.close()
                    return res
                sample_len = min(0.5, dur)
                try:
                    arr = ac.subclip(0, sample_len).to_soundarray(fps=22050)
                    ac.close()
                    import numpy as _np
                    if arr.size == 0:
                        return res
                    # arr is (n, channels); flatten
                    flat = _np.mean(_np.abs(arr), axis=1)
                    rms = float(_np.sqrt((flat ** 2).mean()))
                    res['rms'] = rms
                    # compute non-silent ratio (fraction of frames above small threshold)
                    noise_thresh = self.config.get('silence_threshold', 0.02)
                    non_silent = float((flat > noise_thresh).sum()) / max(1, flat.size)
                    res['non_silent_ratio'] = non_silent
                except Exception:
                    # If sound array extraction fails, fall back to duration check
                    return res
            else:
                # Fallback to wave header for WAV files
                import wave
                try:
                    with wave.open(file_path, 'rb') as wf:
                        nframes = wf.getnframes()
                        fr = wf.getframerate()
                        dur = nframes / float(fr) if fr else 0
                        res['dur'] = dur
                        if not dur or dur < min_dur_cfg:
                            return res
                        # Read a small chunk and compute RMS
                        wf.rewind()
                        frames = wf.readframes(min(nframes, int(fr * 0.5)))
                        if not frames:
                            return res
                        import numpy as _np
                        # Assuming 16-bit samples
                        samples = _np.frombuffer(frames, dtype=_np.int16).astype(_np.float32) / 32768.0
                        if samples.size == 0:
                            return res
                        rms = float(_np.sqrt((samples ** 2).mean()))
                        res['rms'] = rms
                        noise_thresh = self.config.get('silence_threshold', 0.02)
                        non_silent = float((abs(samples) > noise_thresh).sum()) / float(max(1, samples.size))
                        res['non_silent_ratio'] = non_silent
                except Exception:
                    # Unknown format or failure; as a last resort, check file size
                    try:
                        res['dur'] = 0.0
                        if os.path.getsize(file_path) <= 100:
                            return res
                        else:
                            res['rms'] = 0.0
                            res['non_silent_ratio'] = 0.0
                            # accept based on size only as last resort
                            res['valid'] = True
                            return res
                    except Exception:
                        return res

            # Apply estimated duration requirement if provided
            if estimated_duration:
                req_dur = max(min_dur_cfg, estimated_duration * min_duration_ratio)
            else:
                req_dur = min_dur_cfg

            # Determine validity
            if res['dur'] >= req_dur and res['rms'] >= min_rms and res['non_silent_ratio'] >= min_non_silent_ratio:
                res['valid'] = True
            return res
        except Exception:
            return res


    def _compress_internal_silences(self, file_path: str, max_silence: float = 1.0, replacement: float = 0.2) -> bool:
        """Detects internal silences longer than `max_silence` and replaces them with a short replacement silence.
        Returns True if the file was modified, False otherwise.
        """
        import subprocess, re, tempfile, os
        silence_db = self.config.get('silence_threshold_db', -35)
        try:
            # Run silencedetect to find silence ranges
            cmd = [
                'ffmpeg', '-hide_banner', '-nostats', '-i', file_path,
                '-af', f'silencedetect=noise={silence_db}dB:d=0.1', '-f', 'null', '-'
            ]
            out = subprocess.check_output(cmd, stderr=subprocess.STDOUT).decode('utf-8', errors='ignore')

            # Parse silence_start and silence_end pairs
            starts = [float(m.group(1)) for m in re.finditer(r'silence_start: ([0-9\.]+)', out)]
            ends = [float(m.group(1)) for m in re.finditer(r'silence_end: ([0-9\.]+)', out)]

            # Pair them; silencedetect typically reports start then end
            silences = []
            for s,e in zip(starts, ends):
                dur = e - s
                if dur >= max_silence:
                    silences.append((s, e))

            if not silences:
                return False

            # Get total duration
            try:
                dur_raw = subprocess.check_output(['ffprobe','-v','error','-show_entries','format=duration','-of','default=noprint_wrappers=1:nokey=1', file_path])
                total_dur = float(dur_raw.decode().strip())
            except Exception:
                total_dur = None

            # Build non-silent segments
            segments = []
            cur = 0.0
            for s,e in silences:
                if s > cur + 1e-6:
                    segments.append((cur, s))
                cur = e
            if total_dur and cur < total_dur - 1e-6:
                segments.append((cur, total_dur))

            if not segments:
                return False

            tmpdir = tempfile.mkdtemp(prefix='silence_fix_')
            part_files = []
            idx = 0
            # Extract non-silent parts as WAV
            for (a,b) in segments:
                out_part = os.path.join(tmpdir, f'part_{idx:03d}.wav')
                idx += 1
                subprocess.check_call(['ffmpeg', '-y', '-hide_banner', '-nostats', '-ss', str(a), '-to', str(b), '-i', file_path, '-ar', '44100', '-ac', '1', out_part])
                part_files.append(out_part)
                # Insert replacement silence between parts (except after last)
                if idx <= len(segments) - 1:
                    sil_path = os.path.join(tmpdir, f'sil_{idx:03d}.wav')
                    subprocess.check_call(['ffmpeg', '-y', '-hide_banner', '-nostats', '-f', 'lavfi', '-i', f'anullsrc=r=44100:cl=mono', '-t', str(replacement), sil_path])
                    part_files.append(sil_path)

            # Create concat file
            concat_txt = os.path.join(tmpdir, 'concat.txt')
            with open(concat_txt, 'w', encoding='utf-8') as cf:
                for p in part_files:
                    cf.write(f"file '{p}'\n")

            # Concatenate into WAV then re-encode to MP3, overwrite original
            concat_wav = os.path.join(tmpdir, 'out_concat.wav')
            subprocess.check_call(['ffmpeg', '-y', '-hide_banner', '-nostats', '-f', 'concat', '-safe', '0', '-i', concat_txt, '-c', 'copy', concat_wav])
            # If copying as WAV failed (some ffmpeg builds), re-encode
            final_tmp = os.path.join(tmpdir, 'out_final.mp3')
            subprocess.check_call(['ffmpeg', '-y', '-hide_banner', '-nostats', '-i', concat_wav, '-codec:a', 'libmp3lame', '-q:a', '4', final_tmp])

            # Replace original
            os.replace(final_tmp, file_path)
            # Cleanup tmpdir
            try:
                for p in part_files:
                    os.remove(p)
                os.remove(concat_txt)
                os.remove(concat_wav)
            except Exception:
                pass
            return True
        except Exception as e:
            logging.warning(f"_compress_internal_silences failed for {file_path}: {e}")
            return False


    def _clean_text(self, raw_text: str) -> str:
        """Cleans raw text: removes soft hyphens and fixes line-break hyphenation, collapses whitespace, and normalizes punctuation."""
        import re
        text = raw_text.replace('\u00AD', '')  # remove soft hyphen
        # Join broken words split like 'ha-\n ving' into 'having'
        text = re.sub(r'([A-Za-z])-\s+([A-Za-z])', r'\1\2', text)
        clean_text = " ".join(text.split())

        # Normalize Chinese punctuation to English equivalents
        punctuation_map = {
            '，': ',',   # Chinese comma to English comma
            '。': '.',   # Chinese period to English period
            '；': ';',   # Chinese semicolon
            '：': ':',   # Chinese colon
            '？': '?',   # Chinese question mark
            '！': '!',   # Chinese exclamation mark
            '“': '"',   # Chinese left quote
            '”': '"',   # Chinese right quote
            '‘': "'",   # Chinese left single quote
            '’': "'",   # Chinese right single quote
            '—': '-',   # Chinese em dash
            '…': '...',  # Chinese ellipsis
        }
        for cn_punct, en_punct in punctuation_map.items():
            clean_text = clean_text.replace(cn_punct, en_punct)

        return clean_text

    def _compute_text_layout(self, item: Dict[str, str], draw, resolution, font_path: str, font_idx: int, font_size_src: int, font_size_tgt: int) -> Dict[str, any]:
        """Compute wrapped text, scale fonts if necessary, and return font objects and positions to avoid overlap.
        Returns: {
            'font_src': ImageFont,
            'font_tgt': ImageFont,
            'src_text': str,
            'tgt_text': str,
            'src_pos': (x,y),
            'tgt_pos': (x,y),
            'src_bbox': bbox,
            'tgt_bbox': bbox
        }
        """
        from PIL import ImageFont

        tgt_text = self._wrap_text(item.get('target', ''), ImageFont.truetype(font_path, font_size_tgt, index=font_idx) if os.path.exists(font_path) else ImageFont.load_default(), resolution[0] - 200, draw)
        src_text = self._wrap_text(item.get('source', ''), ImageFont.truetype(font_path, font_size_src, index=font_idx) if os.path.exists(font_path) else ImageFont.load_default(), resolution[0] - 200, draw)

        center_x = resolution[0] // 2
        center_y = resolution[1] // 2
        spacing = self.config.get('line_spacing', 24)

        def bbox_for(text, font):
            return draw.textbbox((0, 0), text, font=font)

        max_content_height = resolution[1] - 200  # keep 100px margin top/bottom
        src_font_size = font_size_src
        tgt_font_size = font_size_tgt
        try:
            font_src_cur = ImageFont.truetype(font_path, src_font_size, index=font_idx)
            font_tgt_cur = ImageFont.truetype(font_path, tgt_font_size, index=font_idx)
        except Exception:
            font_src_cur = ImageFont.load_default()
            font_tgt_cur = ImageFont.load_default()

        # Try scaling down up to 3 times if content too big vertically
        for _attempt in range(3):
            src_bbox = bbox_for(src_text, font_src_cur)
            tgt_bbox = bbox_for(tgt_text, font_tgt_cur)
            src_h = src_bbox[3] - src_bbox[1]
            tgt_h = tgt_bbox[3] - tgt_bbox[1]
            total_h = src_h + spacing + tgt_h
            if total_h <= max_content_height:
                break
            # scale down fonts proportionally
            scale = (max_content_height / total_h) * 0.95
            src_font_size = max(20, int(src_font_size * scale))
            tgt_font_size = max(12, int(tgt_font_size * scale))
            try:
                font_src_cur = ImageFont.truetype(font_path, src_font_size, index=font_idx)
                font_tgt_cur = ImageFont.truetype(font_path, tgt_font_size, index=font_idx)
            except Exception:
                font_src_cur = ImageFont.load_default()
                font_tgt_cur = ImageFont.load_default()
            # Re-wrap with potentially different font widths
            tgt_text = self._wrap_text(item.get('target', ''), font_tgt_cur, resolution[0] - 200, draw)
            src_text = self._wrap_text(item.get('source', ''), font_src_cur, resolution[0] - 200, draw)

        # Ensure no single line exceeds available width; if so, scale down that block's font size
        max_width = resolution[0] - 200
        def max_line_width(text, font):
            w = 0
            for line in str(text).split('\n'):
                bbox = draw.textbbox((0, 0), line, font=font)
                w = max(w, bbox[2] - bbox[0])
            return w

        # Try to fit source text horizontally
        src_try = 0
        while max_line_width(src_text, font_src_cur) > max_width and src_try < 5:
            src_try += 1
            src_font_size = max(14, int(src_font_size * 0.9))
            try:
                font_src_cur = ImageFont.truetype(font_path, src_font_size, index=font_idx)
            except Exception:
                font_src_cur = ImageFont.load_default()
            src_text = self._wrap_text(item.get('source', ''), font_src_cur, max_width, draw)

        # Try to fit target (translation) text horizontally
        tgt_try = 0
        while max_line_width(tgt_text, font_tgt_cur) > max_width and tgt_try < 5:
            tgt_try += 1
            tgt_font_size = max(12, int(tgt_font_size * 0.9))
            try:
                font_tgt_cur = ImageFont.truetype(font_path, tgt_font_size, index=font_idx)
            except Exception:
                font_tgt_cur = ImageFont.load_default()
            tgt_text = self._wrap_text(item.get('target', ''), font_tgt_cur, max_width, draw)

        # Final bboxes after potential scaling
        src_bbox = bbox_for(src_text, font_src_cur)
        tgt_bbox = bbox_for(tgt_text, font_tgt_cur)
        src_h = src_bbox[3] - src_bbox[1]
        src_w = src_bbox[2] - src_bbox[0]
        tgt_h = tgt_bbox[3] - tgt_bbox[1]
        tgt_w = tgt_bbox[2] - tgt_bbox[0]

        src_x = center_x - src_w // 2
        src_y = int(center_y - (src_h + spacing + tgt_h) / 2)
        tgt_x = center_x - tgt_w // 2
        tgt_y = src_y + src_h + spacing

        return {
            'font_src': font_src_cur,
            'font_tgt': font_tgt_cur,
            'src_text': src_text,
            'tgt_text': tgt_text,
            'src_pos': (src_x, src_y),
            'tgt_pos': (tgt_x, tgt_y),
            'src_bbox': src_bbox,
            'tgt_bbox': tgt_bbox
        }

    def _estimate_duration_from_text(self, text: str) -> float:
        """Estimate duration (seconds) from text length using words-per-minute baseline."""
        words_per_min = float(self.config.get('words_per_minute', 150))
        wps = words_per_min / 60.0
        words = len(text.split()) if text else 0
        est = max(0.5, words / wps if wps > 0 else 0.5)
        return est

    def create_video(self, processed_data: List[Dict[str, any]]):
        """Synthesizes video from text and audio using MoviePy and PIL."""
        logging.info("Creating video... This may take a while.")
        has_moviepy = True
        try:
            from moviepy.editor import AudioFileClip, ImageClip, CompositeVideoClip, concatenate_videoclips
            MOVIEPY_V2 = False
        except Exception:
            try:
                from moviepy import AudioFileClip, ImageClip, CompositeVideoClip, concatenate_videoclips
                MOVIEPY_V2 = True
            except Exception as e:
                logging.warning(f"MoviePy not available ({e}). Falling back to ffmpeg-based rendering if possible.")
                has_moviepy = False
                AudioFileClip = None
                ImageClip = None
                CompositeVideoClip = None
                concatenate_videoclips = None

        try:
            from PIL import Image, ImageDraw, ImageFont
            import numpy as np
        except ImportError:
            logging.error("PIL (Pillow) or Numpy not installed.")
            return

        clips = []
        ffmpeg_clips = []
        resolution = tuple(self.config.get('resolution', [1920, 1080]))
        bg_color_hex = self.config.get('background_color', '#002b36')
        
        # Helper to hex to rgb
        def hex_to_rgb(hex_code):
            hex_code = hex_code.lstrip('#')
            return tuple(int(hex_code[i:i+2], 16) for i in (0, 2, 4))
        
        bg_color_rgb = hex_to_rgb(bg_color_hex)
        
        # ... (Font loading logic omitted for brevity, assuming generic font)
        # Re-implementing font loading for context
        font_size_src = self.config.get('font_size_source', 70)
        font_size_tgt = self.config.get('font_size_target', 45)
        
        try:
            # Try a common linux font
            font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
            if self.config.get('font_path') and os.path.exists(self.config.get('font_path')):
                font_path = self.config.get('font_path')
            
            font_idx = self.config.get('font_index', 0)
            font_src = ImageFont.truetype(font_path, font_size_src, index=font_idx)
            font_tgt = ImageFont.truetype(font_path, font_size_tgt, index=font_idx)
            font_trivia = ImageFont.truetype(font_path, 30, index=font_idx)
        except Exception as e:
            logging.warning(f"Font loading failed: {e}. Using default.")
            font_src = ImageFont.load_default()
            font_tgt = ImageFont.load_default()
            font_trivia = ImageFont.load_default()

        # Generate clips with progress bar
        logging.info("Rendering video clips...")
        temp_audio_dir = os.path.join(self.output_dir, "temp_audio")
        os.makedirs(temp_audio_dir, exist_ok=True)

        if has_moviepy:
            for i, item in enumerate(tqdm(processed_data, desc="Rendering Clips", unit="clip")):
                audio_path = item.get('audio_path')
                # If audio missing, abort — we require every clip to have valid audio
                if not audio_path or not os.path.exists(audio_path):
                    logging.error(f"Missing audio for segment {i} during video rendering; aborting.")
                    raise RuntimeError(f"Missing audio for segment {i}; ensure generate_audio succeeded")

                try:
                    audio_clip = AudioFileClip(audio_path)
                    duration = audio_clip.duration

                    # --- Drawing Logic (Re-used) ---
                    img = Image.new('RGB', resolution, color=bg_color_rgb)
                    draw = ImageDraw.Draw(img)

                    # Determine layout for source and target text to avoid overlap
                    layout = self._compute_text_layout(item, draw, resolution, font_path, font_idx, font_size_src, font_size_tgt)
                    draw.multiline_text(layout['src_pos'], layout['src_text'], font=layout['font_src'], fill=self.config.get('text_color_highlight', 'white'), align="center")
                    draw.multiline_text(layout['tgt_pos'], layout['tgt_text'], font=layout['font_tgt'], fill=self.config.get('text_color_dim', 'grey'), align="center")

                    if item.get('trivia'):
                        triv_text = "★ " + item['trivia']
                        draw.text((resolution[0]//2, resolution[1] - 80), triv_text, font=font_trivia, fill="#b58900", anchor="mm", align="center")

                    img_np = np.array(img)
                    txt_clip = ImageClip(img_np)

                    if MOVIEPY_V2:
                        txt_clip = txt_clip.with_duration(duration).with_audio(audio_clip)
                    else:
                        txt_clip = txt_clip.set_duration(duration).set_audio(audio_clip)

                    clips.append(txt_clip)
                except Exception as e:
                    logging.error(f"Error creating clip for segment: {e}")
                    continue

            if not clips:
                logging.warning("No clips created.")
                return

            final_video = concatenate_videoclips(clips)
            import datetime
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(self.output_dir, f"{self.config.get('project_name', 'video')}_{timestamp}.mp4")

            logging.info(f"Writing video to {output_path}...")
            final_video.write_videofile(
                output_path,
                fps=self.config.get('fps', 24),
                codec='libx264',
                audio_codec='aac',
                logger='bar'  # built-in moviepy progress bar
            )
        else:
            # Fallback rendering using ffmpeg (image + audio per segment, then concat)
            import shutil, subprocess, datetime
            if not shutil.which('ffmpeg'):
                logging.error('ffmpeg not found; cannot render video without moviepy')
                return

            tmp_dir = os.path.join(self.output_dir, 'ffmpeg_temp')
            os.makedirs(tmp_dir, exist_ok=True)
            frame_dir = os.path.join(tmp_dir, 'frames')
            os.makedirs(frame_dir, exist_ok=True)
            clip_paths = []

            for idx, item in enumerate(tqdm(processed_data, desc="Rendering Clips (ffmpeg)", unit="clip")):
                audio_path = item.get('audio_path')
                # If missing, abort — we require valid audio for every segment
                if not audio_path or not os.path.exists(audio_path):
                    logging.error(f"Missing audio for segment {idx} during ffmpeg rendering; aborting.")
                    raise RuntimeError(f"Missing audio for segment {idx}; ensure generate_audio succeeded")

                # Determine duration using ffprobe (works for MP3/WAV/others)
                duration = 1.0
                try:
                    dur_raw = subprocess.check_output([
                        'ffprobe', '-v', 'error', '-show_entries', 'format=duration',
                        '-of', 'default=noprint_wrappers=1:nokey=1', audio_path
                    ])
                    duration = float(dur_raw.decode().strip())
                except Exception:
                    logging.warning(f"Could not determine duration for {audio_path}; defaulting to {duration}s")

                # Draw frame image
                img = Image.new('RGB', resolution, color=bg_color_rgb)
                draw = ImageDraw.Draw(img)
                layout = self._compute_text_layout(item, draw, resolution, font_path, font_idx, font_size_src, font_size_tgt)
                draw.multiline_text(layout['src_pos'], layout['src_text'], font=layout['font_src'], fill=self.config.get('text_color_highlight', 'white'), align="center")
                draw.multiline_text(layout['tgt_pos'], layout['tgt_text'], font=layout['font_tgt'], fill=self.config.get('text_color_dim', 'grey'), align="center")
                if item.get('trivia'):
                    draw.text((resolution[0]//2, resolution[1] - 80), "★ " + item['trivia'], font=font_trivia, fill="#b58900", anchor="mm", align="center")

                frame_path = os.path.join(frame_dir, f"frame_{idx:04d}.png")
                img.save(frame_path)

                # Use ffmpeg to create a video segment from the single frame + audio
                clip_path = os.path.join(tmp_dir, f"clip_{idx:04d}.mp4")
                cmd = [
                    'ffmpeg', '-y', '-loop', '1', '-i', frame_path, '-i', audio_path,
                    '-c:v', 'libx264', '-t', str(duration), '-pix_fmt', 'yuv420p', '-c:a', 'aac', clip_path
                ]
                try:
                    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    clip_paths.append(clip_path)
                except subprocess.CalledProcessError as e:
                    logging.error(f"ffmpeg failed to create clip for segment {idx}: {e}")
                    continue

            if not clip_paths:
                logging.warning("No ffmpeg clips created.")
                return

            # Create concat list
            concat_list = os.path.join(tmp_dir, 'clips.txt')
            with open(concat_list, 'w', encoding='utf-8') as f:
                for p in clip_paths:
                    f.write(f"file '{p}'\n")

            output_path = os.path.join(self.output_dir, f"{self.config.get('project_name', 'video')}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4")
            cmd = ['ffmpeg', '-y', '-f', 'concat', '-safe', '0', '-i', concat_list, '-c', 'copy', output_path]
            try:
                subprocess.run(cmd, check=True)
                logging.info(f"Wrote video to {output_path}")
            except subprocess.CalledProcessError as e:
                logging.error(f"ffmpeg concat failed: {e}")
                return


    def run(self, clean: bool = False):
        """Executes the full pipeline with checkpointing."""
        import json
        
        if clean:
            logging.info("Cleaning cache...")
            step1_file = os.path.join(self.output_dir, 'step1_translated.json')
            step2_file = os.path.join(self.output_dir, 'step2_audio.json')
            temp_audio_dir = os.path.join(self.output_dir, "temp_audio")
            
            if os.path.exists(step1_file):
                os.remove(step1_file)
                logging.info(f"Removed {step1_file}")
            if os.path.exists(step2_file):
                os.remove(step2_file)
                logging.info(f"Removed {step2_file}")
            if os.path.exists(temp_audio_dir):
                shutil.rmtree(temp_audio_dir)
                logging.info(f"Removed {temp_audio_dir}")
        
        input_file = self.config.get('input_file')
        if not input_file or not os.path.exists(input_file):
            logging.error(f"Input file not found: {input_file}")
            return
            
        step1_file = os.path.join(self.output_dir, 'step1_translated.json')
        step2_file = os.path.join(self.output_dir, 'step2_audio.json')
        
        # --- Step 1: Text & Translation ---
        if os.path.exists(step1_file):
            logging.info("Loading cached translation...")
            with open(step1_file, 'r', encoding='utf-8') as f:
                bilingual_data = json.load(f)
        else:
            raw_text = self.extract_text(input_file)
            segments = self.segment_text(raw_text)
            bilingual_data = self.translate_text(segments)
            # Save Checkpoint
            with open(step1_file, 'w', encoding='utf-8') as f:
                json.dump(bilingual_data, f, ensure_ascii=False, indent=2)
                
        # --- Step 2: Audio Generation ---
        if os.path.exists(step2_file):
             # Check if we should re-generate based on config?
             # For now, trust checkpoint. User can delete json to force retry.
             logging.info("Loading cached audio data...")
             with open(step2_file, 'r', encoding='utf-8') as f:
                audio_data = json.load(f)
        else:
            audio_data = self.generate_audio(bilingual_data)
            # Save Checkpoint
            with open(step2_file, 'w', encoding='utf-8') as f:
                json.dump(audio_data, f, ensure_ascii=False, indent=2)
        
        # --- Step 3: Video Creation ---
        # Video creation is the final step, usually no checkpoint needed after this,
        # but inputs are preserved in step2_file if we need to re-run video only.
        self.create_video(audio_data)
        logging.info("Process completed successfully!")

if __name__ == "__main__":
    from tqdm import tqdm # lazy import
    parser = argparse.ArgumentParser(description="Shadowing Video Generator")
    parser.add_argument("--config", default="config.yaml", help="Path to config file")
    parser.add_argument("--clean", action="store_true", help="Clean cache files before starting")
    args = parser.parse_args()

    generator = ShadowingVideoGenerator(args.config)
    generator.run(clean=args.clean)

