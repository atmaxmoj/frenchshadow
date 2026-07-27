const SENTENCE = "Le petit chat noir mange une pomme rouge.";
const LANGUAGE = "fr-fr";
const SENTENCE_WORDS = SENTENCE.split(/\s+/);

const els = {
  sentence: document.getElementById("sentence"),
  btnStart: document.getElementById("btnStart"),
  btnStop: document.getElementById("btnStop"),
  btnRetry: document.getElementById("btnRetry"),
  btnPlay: document.getElementById("btnPlay"),
  status: document.getElementById("status"),
  timer: document.getElementById("timer"),
  vuMeter: document.getElementById("vuMeter"),
  audioWrap: document.getElementById("audioWrap"),
  audioPlayer: document.getElementById("audioPlayer"),
  results: document.getElementById("results"),
  overallScore: document.getElementById("overallScore"),
  resultList: document.getElementById("resultList"),
};

let mediaRecorder = null;
let recordedChunks = [];
let recordingStartTime = 0;
let timerInterval = null;
let analysisData = null;
let recordedBlob = null;
let audioContext = null;
let analyser = null;
let silenceTimer = null;
let vadFrameId = null;
let recognition = null;
let nextWordIndex = 0;
let isRecording = false;
let liaisonPairs = new Set();

const VU_BARS = 24;
// Threshold and duration tuned for shadow-reading.  Time-domain RMS is less
// jumpy than frequency-domain energy, and a longer silence window avoids
// stopping in the middle of a short phrase like "le petit".
const SILENCE_THRESHOLD = 0.018;
const SILENCE_DURATION_MS = 3200;
const MIN_RECORDING_MS = 1200;

function renderSentence() {
  els.sentence.innerHTML = "";
  SENTENCE_WORDS.forEach((word, idx) => {
    const span = document.createElement("span");
    span.className = "word";
    span.dataset.index = idx;
    span.textContent = word;
    els.sentence.appendChild(span);
    // Mark liaison: if this word initiates a liaison with the next word,
    // add a small connector instead of a plain space.
    const nextWord = SENTENCE_WORDS[idx + 1];
    if (nextWord && liaisonPairs.has(`${word.toLowerCase()}|${nextWord.toLowerCase()}`)) {
      const connector = document.createElement("span");
      connector.className = "liaison-connector";
      connector.textContent = "‿";
      els.sentence.appendChild(connector);
    }
    els.sentence.appendChild(document.createTextNode(" "));
  });
}

async function loadLiaisons() {
  if (!LANGUAGE.startsWith("fr")) return;
  try {
    const res = await fetch(`/liaisons?sentence=${encodeURIComponent(SENTENCE)}&language=${encodeURIComponent(LANGUAGE)}`);
    if (!res.ok) return;
    const data = await res.json();
    liaisonPairs = new Set((data.liaisons || []).map(([a, b]) => `${a.toLowerCase()}|${b.toLowerCase()}`));
    renderSentence();
  } catch (err) {
    console.warn("liaison load failed", err);
  }
}

function createVuMeter() {
  els.vuMeter.innerHTML = "";
  for (let i = 0; i < VU_BARS; i++) {
    const bar = document.createElement("div");
    bar.className = "vu-bar";
    bar.style.height = "4px";
    els.vuMeter.appendChild(bar);
  }
}

function setStatus(html) {
  els.status.innerHTML = html;
}

function startTimer() {
  recordingStartTime = Date.now();
  els.timer.textContent = "00:00.0";
  timerInterval = setInterval(() => {
    const ms = Date.now() - recordingStartTime;
    const secs = Math.floor(ms / 1000);
    const tenths = Math.floor((ms % 1000) / 100);
    els.timer.textContent = `${String(secs).padStart(2, "0")}:${String(tenths)}`;
  }, 100);
}

function stopTimer() {
  clearInterval(timerInterval);
  els.timer.textContent = "";
}

function normalizeWord(word) {
  return word.toLowerCase().replace(/[^a-z0-9]/g, "");
}

function initSpeechRecognition() {
  const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!SpeechRecognition) return null;
  const rec = new SpeechRecognition();
  rec.continuous = true;
  rec.interimResults = true;
  rec.lang = "fr-FR";

  rec.onresult = (event) => {
    if (!isRecording) return;
    for (let i = event.resultIndex; i < event.results.length; i++) {
      const transcript = event.results[i][0].transcript;
      const words = transcript.split(/\s+/).map(normalizeWord).filter(Boolean);
      for (const word of words) {
        if (nextWordIndex >= SENTENCE_WORDS.length) break;
        const target = normalizeWord(SENTENCE_WORDS[nextWordIndex]);
        if (word === target || word.includes(target)) {
          highlightWord(nextWordIndex);
          nextWordIndex++;
          if (nextWordIndex >= SENTENCE_WORDS.length) {
            setTimeout(autoStop, 300);
          }
        }
      }
    }
  };

  rec.onerror = (e) => {
    // Ignore no-speech and aborted errors during active recording.
    if (e.error === "no-speech" || e.error === "aborted") return;
    console.warn("Speech recognition error:", e.error);
  };

  return rec;
}

function highlightWord(index) {
  const spans = els.sentence.querySelectorAll(".word");
  spans.forEach((s) => s.classList.remove("active"));
  const span = spans[index];
  if (span) {
    span.classList.add("active");
    span.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }
}

function clearActiveHighlight() {
  els.sentence.querySelectorAll(".word").forEach((s) => s.classList.remove("active"));
}

function setupVAD(stream) {
  audioContext = new (window.AudioContext || window.webkitAudioContext)();
  const source = audioContext.createMediaStreamSource(stream);
  analyser = audioContext.createAnalyser();
  analyser.fftSize = 256;
  source.connect(analyser);

  const dataArray = new Uint8Array(analyser.fftSize);
  let silenceStart = null;

  function tick() {
    if (!isRecording) return;
    analyser.getByteTimeDomainData(dataArray);

    // True RMS over the time-domain waveform (values centred at 128).
    let sum = 0;
    for (let i = 0; i < dataArray.length; i++) {
      const v = (dataArray[i] - 128) / 128;
      sum += v * v;
    }
    const rms = Math.sqrt(sum / dataArray.length);

    updateVuMeter(rms);

    const now = Date.now();
    if (rms < SILENCE_THRESHOLD) {
      if (silenceStart === null) silenceStart = now;
      else if (now - silenceStart > SILENCE_DURATION_MS && now - recordingStartTime > MIN_RECORDING_MS) {
        autoStop();
        return;
      }
    } else {
      silenceStart = null;
    }

    vadFrameId = requestAnimationFrame(tick);
  }

  vadFrameId = requestAnimationFrame(tick);
}

function updateVuMeter(level) {
  const bars = els.vuMeter.querySelectorAll(".vu-bar");
  const activeCount = Math.min(VU_BARS, Math.ceil(level * VU_BARS * 2.5));
  bars.forEach((bar, idx) => {
    const h = idx < activeCount ? 4 + (idx / VU_BARS) * 28 : 4;
    bar.style.height = `${h}px`;
    bar.classList.toggle("hot", idx > VU_BARS * 0.75 && idx < activeCount);
  });
}

function resetVuMeter() {
  const bars = els.vuMeter.querySelectorAll(".vu-bar");
  bars.forEach((bar) => {
    bar.style.height = "4px";
    bar.classList.remove("hot");
  });
}

async function startRecording() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        echoCancellation: false,
        noiseSuppression: true,
        autoGainControl: false,
      },
    });
    mediaRecorder = new MediaRecorder(stream, { mimeType: "audio/webm" });
    recordedChunks = [];
    nextWordIndex = 0;
    isRecording = true;

    mediaRecorder.ondataavailable = (e) => {
      if (e.data.size > 0) recordedChunks.push(e.data);
    };

    mediaRecorder.onstop = () => {
      recordedBlob = new Blob(recordedChunks, { type: "audio/webm" });
      const url = URL.createObjectURL(recordedBlob);
      els.audioPlayer.src = url;
      els.audioWrap.classList.add("visible");
      uploadAndAnalyze(recordedBlob);
    };

    mediaRecorder.start(100);
    setupVAD(stream);
    startTimer();

    recognition = initSpeechRecognition();
    if (recognition) {
      try { recognition.start(); } catch (e) { console.warn(e); }
    }

    els.btnStart.classList.add("hidden");
    els.btnStop.disabled = false;
    els.btnRetry.classList.add("hidden");
    els.btnPlay.classList.add("hidden");
    els.results.classList.add("hidden");
    resetWordStyles();
    setStatus('<span class="recording-dot"></span>正在听… 读完自动停止');
  } catch (err) {
    console.error(err);
    setStatus("无法访问麦克风，请检查浏览器权限。");
  }
}

function autoStop() {
  if (!isRecording) return;
  stopRecording();
}

function stopRecording() {
  isRecording = false;
  if (vadFrameId) cancelAnimationFrame(vadFrameId);
  if (silenceTimer) clearTimeout(silenceTimer);
  if (audioContext && audioContext.state !== "closed") audioContext.close();

  if (recognition) {
    try { recognition.stop(); } catch (e) { /* ignore */ }
    recognition = null;
  }

  if (mediaRecorder && mediaRecorder.state !== "inactive") {
    mediaRecorder.stop();
    mediaRecorder.stream.getTracks().forEach((t) => t.stop());
  }

  stopTimer();
  resetVuMeter();
  clearActiveHighlight();
  els.btnStart.classList.remove("hidden");
  els.btnStop.disabled = true;
  els.btnRetry.classList.remove("hidden");
  setStatus("录音结束，正在分析…");
}

function resetWordStyles() {
  els.sentence.querySelectorAll(".word").forEach((span) => {
    span.className = "word";
    span.title = "";
  });
}

function resetAll() {
  stopRecording();
  recordedBlob = null;
  analysisData = null;
  nextWordIndex = 0;
  els.audioPlayer.src = "";
  els.audioWrap.classList.remove("visible");
  els.results.classList.add("hidden");
  els.btnPlay.classList.add("hidden");
  els.btnRetry.classList.add("hidden");
  els.btnStart.classList.remove("hidden");
  els.btnStart.disabled = false;
  els.btnStop.disabled = true;
  resetWordStyles();
  resetVuMeter();
  clearActiveHighlight();
  setStatus("点击「开始朗读」，然后读出上面的句子");
}

async function uploadAndAnalyze(blob) {
  const form = new FormData();
  form.append("audio", blob, "recording.webm");
  form.append("target_text", SENTENCE);
  form.append("language", LANGUAGE);

  try {
    const res = await fetch("/transcribe", { method: "POST", body: form });
    const data = await res.json();
    if (!res.ok) {
      throw new Error(data.detail || `HTTP ${res.status}`);
    }
    analysisData = data;
    renderAnalysis(data);
    setStatus("分析完成。点击「播放并高亮」可回放跟读效果。");
  } catch (err) {
    console.error(err);
    setStatus("分析失败：" + err.message);
  }
}

function scoreClass(score) {
  if (score >= 0.99) return "good";
  if (score >= 0.5) return "warn";
  return "bad";
}

async function playReference(word) {
  try {
    const q = new URLSearchParams({ text: word, language: LANGUAGE, sentence: SENTENCE });
    const res = await fetch(`/reference_audio?${q.toString()}`);
    if (!res.ok) throw new Error("reference audio failed");
    const blob = await res.blob();
    const audio = new Audio(URL.createObjectURL(blob));
    audio.play();
  } catch (err) {
    console.error(err);
  }
}

async function playUserSlice(startTime, endTime) {
  if (!recordedBlob || startTime >= endTime) return;
  try {
    const arrayBuffer = await recordedBlob.arrayBuffer();
    const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    const decoded = await audioCtx.decodeAudioData(arrayBuffer);
    const duration = endTime - startTime;
    const offline = new OfflineAudioContext(decoded.numberOfChannels, Math.ceil(decoded.sampleRate * duration), decoded.sampleRate);
    const src = offline.createBufferSource();
    src.buffer = decoded;
    src.connect(offline.destination);
    src.start(0, startTime, duration);
    const rendered = await offline.startRendering();
    const dest = audioCtx.createBufferSource();
    dest.buffer = rendered;
    dest.connect(audioCtx.destination);
    dest.start();
  } catch (err) {
    console.error(err);
  }
}

function diagramImg(url, phone, fallbackText) {
  if (url) {
    return `<img src="${url}&_=${Date.now()}" alt="/${phone}/" class="mouth-diagram" onerror="this.style.display='none'" />`;
  }
  const svg = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 300 210" width="300" height="210">
    <rect x="30" y="25" width="240" height="160" rx="14" fill="none" stroke="#5f6368" stroke-width="2" stroke-dasharray="6 4"/>
    <text x="150" y="112" text-anchor="middle" fill="#9aa0a6" font-size="20" font-family="sans-serif">${fallbackText}</text>
  </svg>`;
  return `<img src="data:image/svg+xml,${encodeURIComponent(svg)}" alt="${fallbackText}" class="mouth-diagram" />`;
}

function renderAnalysis(data) {
  const analysis = data.analysis || {};
  const words = analysis.words || [];

  els.overallScore.textContent = Math.round((analysis.overall_score || 0) * 100);
  els.overallScore.style.color =
    (analysis.overall_score || 0) >= 0.8 ? "var(--good)" :
    (analysis.overall_score || 0) >= 0.5 ? "var(--warn)" : "var(--bad)";

  const wordSpans = els.sentence.querySelectorAll(".word");
  words.forEach((w, i) => {
    const span = wordSpans[i];
    if (!span) return;
    span.classList.add(scoreClass(w.score));
    span.title = `${w.word} 目标: /${w.target_ipa}/ 你说: /${w.learner_ipa}/ 得分: ${Math.round(w.score * 100)}`;
  });

  els.resultList.innerHTML = "";
  words.forEach((w) => {
    const item = document.createElement("div");
    const isPerfect = w.score >= 0.99 && w.errors.length === 0;
    item.className = "result-item";
    const safeWord = w.word.replace(/"/g, "&quot;");
    const start = typeof w.start_time === "number" ? w.start_time : 0;
    const end = typeof w.end_time === "number" ? w.end_time : 0;
    let errs = "";
    if (!isPerfect) {
      w.errors.forEach((e) => {
        const tips = e.tips || {};
        const expectedImg = diagramImg(tips.diagram_expected, e.expected || "-", "无图示");
        const actualImg = diagramImg(tips.diagram_actual, e.actual || "-", e.actual ? "无图示" : "未发音");
        errs += `
          <div class="tip-box">
            <h4>${e.label} <span style="color:var(--muted);font-weight:400;">(${e.expected || "-"} → ${e.actual || "-"})</span></h4>
            <div class="mouth-diagrams">
              <div class="mouth-diagram-wrap">
                <span class="mouth-diagram-label correct">正确 /${e.expected || "-"}/</span>
                ${expectedImg}
              </div>
              <div class="mouth-diagram-wrap">
                <span class="mouth-diagram-label user">你的 /${e.actual || "-"}/</span>
                ${actualImg}
              </div>
            </div>
            <p><span class="label">问题：</span>${tips.description || ""}</p>
            <p><span class="label">舌头：</span>${tips.tongue || ""}</p>
            <p><span class="label">嘴唇：</span>${tips.lips || ""}</p>
            <p><span class="label">下巴：</span>${tips.jaw || ""}</p>
            <p><span class="label">练习：</span>${tips.practice || ""}</p>
          </div>
        `;
      });
    }
    item.innerHTML = `
      <div class="result-header">
        <span class="result-word">${w.word}</span>
        <span class="result-ipa">/${w.target_ipa}/</span>
        <span class="result-score" style="color:${scoreClass(w.score) === "good" ? "var(--good)" : scoreClass(w.score) === "warn" ? "var(--warn)" : "var(--bad)"}">${Math.round(w.score * 100)}</span>
        <span class="word-play-btns">
          <button class="btn-ref" onclick="window.playReference('${safeWord}')" title="标准发音">🔊</button>
          <button class="btn-user" onclick="window.playUserSlice(${start}, ${end})" title="我的录音">🎤</button>
        </span>
      </div>
      ${isPerfect ? '<p style="color:var(--good);margin:4px 0 0;">✓ 发音良好</p>' : errs}
    `;
    els.resultList.appendChild(item);
  });

  els.results.classList.remove("hidden");
  els.btnPlay.classList.remove("hidden");
}

// expose for inline onclick handlers
window.playReference = playReference;
window.playUserSlice = playUserSlice;

function playWithHighlight() {
  if (!analysisData || !analysisData.analysis) return;
  const words = analysisData.analysis.words || [];
  const player = els.audioPlayer;

  function update() {
    const t = player.currentTime;
    const spans = els.sentence.querySelectorAll(".word");
    spans.forEach((span) => span.classList.remove("active"));

    for (let i = 0; i < words.length; i++) {
      const w = words[i];
      if (t >= w.start_time && t < w.end_time) {
        spans[i]?.classList.add("active");
        break;
      }
    }
  }

  player.ontimeupdate = update;
  player.onended = () => {
    player.ontimeupdate = null;
    els.sentence.querySelectorAll(".word").forEach((s) => s.classList.remove("active"));
  };
  player.play();
}

els.btnStart.addEventListener("click", startRecording);
els.btnStop.addEventListener("click", stopRecording);
els.btnRetry.addEventListener("click", resetAll);
els.btnPlay.addEventListener("click", playWithHighlight);

renderSentence();
loadLiaisons();
createVuMeter();
