const els = {
  landing: document.getElementById("landing"),
  urlInput: document.getElementById("urlInput"),
  langSelect: document.getElementById("langSelect"),
  btnLoad: document.getElementById("btnLoad"),
  landingStatus: document.getElementById("landingStatus"),
  videoInfo: document.getElementById("videoInfo"),
  practice: document.getElementById("practice"),
  sentenceList: document.getElementById("sentenceList"),
  subtitleBox: document.getElementById("subtitleBox"),
  analysisPanel: document.getElementById("analysisPanel"),
  practiceStatus: document.getElementById("practiceStatus"),
  btnPlaySource: document.getElementById("btnPlaySource"),
  btnRecord: document.getElementById("btnRecord"),
  btnPlayRef: document.getElementById("btnPlayRef"),
  btnPlayUser: document.getElementById("btnPlayUser"),
  btnRepeat: document.getElementById("btnRepeat"),
  btnContinue: document.getElementById("btnContinue"),
  topNav: document.getElementById("topNav"),
  dashboard: document.getElementById("dashboard"),
  statsGrid: document.getElementById("statsGrid"),
  recentVideos: document.getElementById("recentVideos"),
};

const state = {
  mode: "landing",
  url: "",
  videoId: null,
  language: "fr-fr",
  videoInfo: null,
  sentences: [],
  currentIdx: 0,
  player: null,
  playerReady: false,
  isPlayingSource: false,
  sourcePollId: null,
  wordAudios: {}, // { sentenceIdx: { word: blobUrl } }
  sentenceAudios: {}, // { sentenceIdx: blobUrl }
  recordedBlob: null,
  analysis: null,
  tokenTimes: [],
  selectedWord: null,
  lastDurationS: 0,
};

// ---------- recording globals (reused per sentence) ----------
let mediaRecorder = null;
let recordedChunks = [];
let recordingStartTime = 0;
let timerInterval = null;
let audioContext = null;
let analyser = null;
let silenceTimer = null;
let vadFrameId = null;
let isRecording = false;
let sourceStopTimer = null;

const VU_BARS = 24;
const SILENCE_THRESHOLD = 0.018;
const SILENCE_DURATION_MS = 3200;
const MIN_RECORDING_MS = 1200;

// ---------- YouTube API ----------
function ensureYouTubePlayer(videoId) {
  if (!window.YT || !window.YT.Player) return false;

  // If a player already exists, just cue the new video.
  if (state.player && state.player.cueVideoById) {
    state.player.cueVideoById(videoId);
    state.playerReady = true;
    return true;
  }

  const container = document.getElementById("player");
  if (!container) return false;
  container.innerHTML = "";

  state.player = new YT.Player("player", {
    width: "100%",
    height: "100%",
    videoId: videoId,
    playerVars: {
      rel: 0,
      modestbranding: 1,
      enablejsapi: 1,
      origin: window.location.origin,
    },
    events: {
      onReady: () => {
        state.playerReady = true;
      },
      onStateChange: (event) => {
        if (event.data === YT.PlayerState.PAUSED || event.data === YT.PlayerState.ENDED) {
          stopSourcePoll();
        }
      },
      onError: (event) => {
        console.error("YouTube player error", event.data);
        setStatus("视频播放失败 (code " + event.data + ")");
      },
    },
  });
  return true;
}

if (window.YT && window.YT.Player) {
  // Defer player creation until practice view is shown.
  window.onYouTubeIframeAPIReady = () => {};
} else {
  window.onYouTubeIframeAPIReady = () => {};
}

// ---------- landing ----------
async function loadVideoInfo() {
  const url = els.urlInput.value.trim();
  if (!url) {
    setLandingStatus("请输入 YouTube 链接");
    return;
  }
  state.url = url;
  state.language = els.langSelect.value;
  setLandingStatus("解析中…");
  try {
    const res = await fetch(`/youtube/info?url=${encodeURIComponent(url)}&language=${encodeURIComponent(state.language)}`);
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || `HTTP ${res.status}`);
    state.videoInfo = data;
    state.videoId = data.video_id;
    renderVideoInfo(data);
    setLandingStatus("");
    if (state.playerReady && state.player.cueVideoById) {
      state.player.cueVideoById(state.videoId);
    }
  } catch (err) {
    setLandingStatus("解析失败：" + err.message);
  }
}

function renderVideoInfo(info) {
  const hasLang = info.has_preferred_language
    ? ""
    : `<div class="langs">注意：该视频可能没有 ${state.language} 字幕，解析后可能没有文本。</div>`;
  const langs = info.available_languages.slice(0, 8).map((l) => `${l.name} (${l.code})`).join(", ") || "未知";
  els.videoInfo.innerHTML = `
    <img src="${info.thumbnail}" alt="" />
    <div class="meta">
      <div class="title">${escapeHtml(info.title)}</div>
      <div class="author">${escapeHtml(info.author)}</div>
      <div class="langs">可用字幕：${escapeHtml(langs)}</div>
      ${hasLang}
      <button id="btnStartPractice" class="btn-primary">开始练习</button>
    </div>
  `;
  els.videoInfo.classList.remove("hidden");
  document.getElementById("btnStartPractice").addEventListener("click", startPractice);
}

async function startPractice() {
  if (!state.videoId) return;
  setLandingStatus("加载字幕…");
  try {
    const res = await fetch(`/youtube/transcript?video_id=${encodeURIComponent(state.videoId)}&language=${encodeURIComponent(state.language)}`);
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || `HTTP ${res.status}`);
    state.sentences = data.sentences || [];
    if (!state.sentences.length) {
      setLandingStatus("没有可用的字幕，无法练习。");
      return;
    }
    state.currentIdx = 0;
    switchView("practice");
    renderSentenceList();
    setCurrentSentence(0);
    prebakeAudiosFor(0);
    prebakeAudiosFor(1);
  } catch (err) {
    setLandingStatus("字幕加载失败：" + err.message);
  }
}

function setLandingStatus(html) {
  els.landingStatus.innerHTML = html;
}

// ---------- navigation / dashboard ----------
function switchView(view) {
  state.mode = view === "practice" ? "practice" : view;
  els.landing.classList.toggle("hidden", view !== "landing");
  els.dashboard.classList.toggle("hidden", view !== "dashboard");
  els.practice.classList.toggle("hidden", view !== "practice");
  els.topNav.querySelectorAll("button").forEach((btn) => {
    btn.classList.toggle("active", btn.dataset.view === view);
  });
  if (view === "dashboard") loadDashboard();
}

async function loadDashboard() {
  try {
    const [statsRes, videosRes] = await Promise.all([
      fetch("/stats"),
      fetch("/recent_videos"),
    ]);
    const stats = statsRes.ok ? await statsRes.json() : {};
    const videosData = videosRes.ok ? await videosRes.json() : { videos: [] };
    renderStats(stats);
    renderRecentVideos(videosData.videos || []);
  } catch (err) {
    console.error(err);
    els.statsGrid.innerHTML = `<div class="empty-state">加载失败：${escapeHtml(err.message)}</div>`;
    els.recentVideos.innerHTML = "";
  }
}

function renderStats(stats) {
  const items = [
    { label: "视频数", value: stats.videos || 0 },
    { label: "总尝试", value: stats.attempts || 0 },
    { label: "句子数", value: stats.sentences || 0 },
    { label: "分钟", value: stats.total_minutes || 0 },
    { label: "天数", value: stats.days || 0 },
    { label: "连续", value: stats.streak || 0 },
  ];
  els.statsGrid.innerHTML = items
    .map(
      (s) => `
      <div class="stat-card">
        <div class="stat-value">${escapeHtml(String(s.value))}</div>
        <div class="stat-label">${escapeHtml(s.label)}</div>
      </div>
    `
    )
    .join("");
}

function renderRecentVideos(videos) {
  if (!videos.length) {
    els.recentVideos.innerHTML = `<div class="empty-state" style="grid-column:1/-1">还没有练习记录，先去输入链接练一段吧。</div>`;
    return;
  }
  els.recentVideos.innerHTML = videos
    .map((v) => {
      const total = v.total_sentences || 0;
      const last = v.last_sentence_idx || 0;
      const pct = total > 0 ? Math.min(100, Math.round((last / Math.max(total - 1, 1)) * 100)) : 0;
      const langLabel = v.language === "fr-fr" ? "Français" : v.language === "en-us" ? "English" : v.language;
      return `
        <div class="video-card">
          <img src="${escapeHtml(v.thumbnail || "")}" alt="" />
          <div class="meta">
            <div class="title">${escapeHtml(v.title || v.video_id)}</div>
            <div class="progress">
              ${escapeHtml(langLabel)} · ${v.attempt_count || 0} 次尝试
              <div class="progress-bar"><div style="width:${pct}%"></div></div>
            </div>
            <button class="btn-primary" onclick="window.continueVideo('${escapeHtml(v.video_id)}', '${escapeHtml(v.language || "fr-fr")}', ${last})">继续跟读</button>
          </div>
        </div>
      `;
    })
    .join("");
}

window.continueVideo = async function (videoId, language, startIdx) {
  state.videoId = videoId;
  state.language = language || "fr-fr";
  setLandingStatus("恢复进度…");
  try {
    const infoRes = await fetch(`/youtube/info?url=${encodeURIComponent(`https://www.youtube.com/watch?v=${videoId}`)}&language=${encodeURIComponent(state.language)}`);
    const info = await infoRes.json();
    if (!infoRes.ok) throw new Error(info.detail || `HTTP ${infoRes.status}`);
    state.videoInfo = info;

    const transRes = await fetch(`/youtube/transcript?video_id=${encodeURIComponent(videoId)}&language=${encodeURIComponent(state.language)}`);
    const trans = await transRes.json();
    if (!transRes.ok) throw new Error(trans.detail || `HTTP ${transRes.status}`);
    state.sentences = trans.sentences || [];
    if (!state.sentences.length) {
      setLandingStatus("没有可用的字幕，无法继续。");
      return;
    }

    switchView("practice");
    const idx = Math.max(0, Math.min(startIdx, state.sentences.length - 1));
    state.currentIdx = idx;
    setCurrentSentence(idx);
    playSourceSentence(idx);
    prebakeAudiosFor(idx + 1);
  } catch (err) {
    setLandingStatus("继续失败：" + err.message);
  }
};

function setStatus(html) {
  els.practiceStatus.innerHTML = html;
}

// ---------- sentence navigation ----------
function renderSentenceList() {
  if (!state.sentences.length) {
    els.sentenceList.innerHTML = "";
    return;
  }
  els.sentenceList.innerHTML = state.sentences
    .map(
      (s, i) => `
      <span class="sentence-item ${i === state.currentIdx ? "current" : ""}" data-idx="${i}">
        <span class="idx">${i + 1}</span>${escapeHtml(s.text)}
      </span><span class="sentence-sep"> </span>
    `
    )
    .join("");
  els.sentenceList.querySelectorAll(".sentence-item").forEach((el) => {
    el.addEventListener("click", () => {
      const idx = parseInt(el.dataset.idx, 10);
      setCurrentSentence(idx);
      playSourceSentence(idx);
    });
  });
  scrollCurrentSentenceIntoView();
}

function scrollCurrentSentenceIntoView() {
  const current = els.sentenceList.querySelector(".sentence-item.current");
  if (current) current.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

function setCurrentSentence(idx) {
  if (idx < 0 || idx >= state.sentences.length) return;
  state.currentIdx = idx;
  state.analysis = null;
  state.recordedBlob = null;
  state.tokenTimes = [];
  state.selectedWord = null;
  renderSentenceList();
  renderSubtitle(idx);
  renderAnalysis(null);
  loadAttempts();
  setStatus("按 ▶ 原句 听一遍，或按 🎤 跟读");
  els.btnPlayUser.disabled = true;
  const s = state.sentences[idx];
  if (state.mode === "practice") {
    ensureYouTubePlayer(state.videoId);
    if (state.playerReady && state.player.cueVideoById) {
      state.player.cueVideoById({ videoId: state.videoId, startSeconds: s.start });
    }
  }
}

function renderSubtitle(idx) {
  const s = state.sentences[idx];
  if (!s) return;
  const html = s.words
    .map(
      (w) => `<span class="subtitle-word" data-word="${escapeHtml(w.text)}" data-start="${w.start}" data-end="${w.end}">${escapeHtml(w.text)}</span>`
    )
    .join(" ");
  els.subtitleBox.innerHTML = html;
  els.subtitleBox.querySelectorAll(".subtitle-word").forEach((el) => {
    el.addEventListener("click", () => {
      const text = el.dataset.word;
      const wordObj = s.words.find((w) => w.text === text);
      const wordResult = state.analysis?.words?.find((w) => w.word === text);
      showWordDetail(wordObj, wordResult);
    });
  });
}

// ---------- source playback ----------
function playSourceSentence(idx) {
  if (idx !== undefined) state.currentIdx = idx;
  const s = state.sentences[state.currentIdx];
  if (!s) return;

  ensureYouTubePlayer(state.videoId);
  if (!state.playerReady) {
    setStatus("播放器准备中…");
    setTimeout(() => playSourceSentence(state.currentIdx), 400);
    return;
  }

  stopRecording();
  stopSourcePoll();
  state.isPlayingSource = true;
  setStatus("播放原句…");
  state.player.seekTo(s.start, true);
  state.player.playVideo();

  const durationMs = Math.max((s.end - s.start) * 1000, 200);

  // Poll as a safety net in case the timeout drifts.
  state.sourcePollId = setInterval(() => {
    const t = state.player.getCurrentTime();
    if (t >= s.end - 0.08 || t < s.start - 0.5) {
      state.player.pauseVideo();
      stopSourcePoll();
      setTimeout(autoStartRecording, 400);
    }
  }, 60);

  // Primary stop: pause when the sentence should have finished.
  sourceStopTimer = setTimeout(() => {
    if (state.isPlayingSource) {
      state.player.pauseVideo();
      stopSourcePoll();
      setTimeout(autoStartRecording, 400);
    }
  }, durationMs + 250);
}

function playSourceWord(wordObj) {
  if (!wordObj) return;
  ensureYouTubePlayer(state.videoId);
  if (!state.playerReady) {
    setTimeout(() => playSourceWord(wordObj), 400);
    return;
  }
  stopSourcePoll();
  state.isPlayingSource = true;
  state.player.seekTo(wordObj.start, true);
  state.player.playVideo();

  const durationMs = Math.max((wordObj.end - wordObj.start) * 1000, 150);
  sourceStopTimer = setTimeout(() => {
    state.player.pauseVideo();
    stopSourcePoll();
  }, durationMs + 120);
}

function stopSourcePoll() {
  state.isPlayingSource = false;
  if (state.sourcePollId) {
    clearInterval(state.sourcePollId);
    state.sourcePollId = null;
  }
  if (sourceStopTimer) {
    clearTimeout(sourceStopTimer);
    sourceStopTimer = null;
  }
}

// ---------- recording ----------
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
  // no-op: VU meter removed in YouTube layout; kept for compatibility
}

function resetVuMeter() {}

function startTimer() {
  recordingStartTime = Date.now();
  timerInterval = setInterval(() => {
    const ms = Date.now() - recordingStartTime;
    const secs = Math.floor(ms / 1000);
    const tenths = Math.floor((ms % 1000) / 100);
    setStatus(`正在听… ${String(secs).padStart(2, "0")}:${String(tenths)}`);
  }, 100);
}

function stopTimer() {
  clearInterval(timerInterval);
}

async function startRecording() {
  try {
    const s = state.sentences[state.currentIdx];
    const stream = await navigator.mediaDevices.getUserMedia({
      audio: { echoCancellation: false, noiseSuppression: true, autoGainControl: false },
    });
    mediaRecorder = new MediaRecorder(stream, { mimeType: "audio/webm" });
    recordedChunks = [];
    isRecording = true;
    mediaRecorder.ondataavailable = (e) => {
      if (e.data.size > 0) recordedChunks.push(e.data);
    };
    mediaRecorder.onstop = () => {
      state.recordedBlob = new Blob(recordedChunks, { type: "audio/webm" });
      uploadAndAnalyze(state.recordedBlob);
    };
    mediaRecorder.start(100);
    setupVAD(stream);
    startTimer();
    els.btnRecord.disabled = true;
    setStatus('<span class="recording-dot"></span>请跟读当前句子');
  } catch (err) {
    console.error(err);
    setStatus("无法访问麦克风");
  }
}

function autoStartRecording() {
  if (state.mode !== "practice") return;
  setStatus("轮到你了…");
  setTimeout(startRecording, 300);
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
  if (mediaRecorder && mediaRecorder.state !== "inactive") {
    mediaRecorder.stop();
    mediaRecorder.stream.getTracks().forEach((t) => t.stop());
  }
  stopTimer();
  els.btnRecord.disabled = false;
}

async function uploadAndAnalyze(blob) {
  const s = state.sentences[state.currentIdx];
  setStatus("分析中…");
  const form = new FormData();
  form.append("audio", blob, "recording.webm");
  form.append("target_text", s.text);
  form.append("language", state.language);
  try {
    const res = await fetch("/transcribe", { method: "POST", body: form });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || `HTTP ${res.status}`);
    state.analysis = data.analysis || {};
    state.tokenTimes = data.tokens || [];
    state.lastDurationS = data.duration_s || 0;
    renderAnalysis(data);
    await saveAttempt(blob, data);
    await loadAttempts();
    setStatus("分析完成。r=重读，c=继续");
    els.btnPlayUser.disabled = false;
  } catch (err) {
    console.error(err);
    setStatus("分析失败：" + err.message);
  }
}

async function saveAttempt(blob, data) {
  const s = state.sentences[state.currentIdx];
  const info = state.videoInfo || {};
  const form = new FormData();
  form.append("audio", blob, "recording.webm");
  form.append("video_id", state.videoId);
  form.append("sentence_idx", state.currentIdx);
  form.append("sentence_text", s.text);
  form.append("language", state.language);
  form.append("analysis", JSON.stringify(data.analysis || {}));
  form.append("duration_s", state.lastDurationS || data.duration_s || 0);
  form.append("title", info.title || "");
  form.append("thumbnail", info.thumbnail || "");
  form.append("total_sentences", state.sentences.length);
  try {
    await fetch("/attempts", { method: "POST", body: form });
  } catch (err) {
    console.warn("save attempt failed", err);
  }
}

async function loadAttempts() {
  if (!state.videoId) return;
  try {
    const res = await fetch(`/attempts?video_id=${encodeURIComponent(state.videoId)}&sentence_idx=${state.currentIdx}`);
    if (!res.ok) return;
    const data = await res.json();
    renderHistory(data.attempts || []);
  } catch (err) {
    console.warn("load attempts failed", err);
  }
}

function renderHistory(attempts) {
  const existing = document.getElementById("historySection");
  if (existing) existing.remove();
  if (!attempts.length) return;
  const best = Math.max(...attempts.map((a) => a.overall_score || 0));
  const list = attempts
    .map(
      (a) => `
      <div class="history-item">
        <span class="history-score" style="color:var(--${scoreClass(a.overall_score)})">${Math.round((a.overall_score || 0) * 100)}</span>
        <span class="history-time">${formatTime(a.created_at)}</span>
        <button class="btn-secondary" onclick="window.playAttemptAudio('${a.id}')">▶</button>
        <button class="btn-secondary" onclick="window.loadAttemptAnalysis('${a.id}')">分析</button>
      </div>
    `
    )
    .join("");
  const section = document.createElement("div");
  section.id = "historySection";
  section.className = "history-section";
  section.innerHTML = `
    <h4>本句历史（最佳 ${Math.round(best * 100)}）</h4>
    ${list}
  `;
  els.analysisPanel.appendChild(section);
}

window.playAttemptAudio = async function (attemptId) {
  try {
    const res = await fetch(`/attempts/${attemptId}/audio`);
    if (!res.ok) return;
    const blob = await res.blob();
    new Audio(URL.createObjectURL(blob)).play();
  } catch (err) {
    console.error(err);
  }
};

window.loadAttemptAnalysis = async function (attemptId) {
  try {
    const res = await fetch(`/attempts?video_id=${encodeURIComponent(state.videoId)}&sentence_idx=${state.currentIdx}`);
    if (!res.ok) return;
    const data = await res.json();
    const attempt = (data.attempts || []).find((a) => a.id === attemptId);
    if (!attempt) return;
    // Reconstruct a data object compatible with renderAnalysis.
    const full = { analysis: attempt.analysis };
    state.analysis = attempt.analysis || {};
    renderAnalysis(full);
  } catch (err) {
    console.error(err);
  }
};

function formatTime(iso) {
  try {
    const d = new Date(iso);
    return `${d.getMonth() + 1}/${d.getDate()} ${String(d.getHours()).padStart(2, "0")}:${String(d.getMinutes()).padStart(2, "0")}`;
  } catch (e) {
    return iso;
  }
}

// ---------- analysis rendering ----------
function scoreClass(score) {
  if (score >= 0.99) return "good";
  if (score >= 0.5) return "warn";
  return "bad";
}

function renderAnalysis(data) {
  if (!data || !data.analysis) {
    els.analysisPanel.innerHTML = '<p class="analysis-empty">跟读后会显示逐词分析</p>';
    highlightSubtitleWords(null);
    return;
  }
  const analysis = data.analysis;
  const words = analysis.words || [];
  highlightSubtitleWords(words);

  let html = `<div class="overall-score" style="color:var(--${scoreClass(analysis.overall_score || 0)})">${Math.round((analysis.overall_score || 0) * 100)}</div>`;
  for (const w of words) {
    const errs = (w.errors || []).slice(0, 3);
    let tipsHtml = "";
    if (errs.length) {
      tipsHtml = errs
        .map((e) => {
          const t = e.tips || {};
          const expectedImg = t.diagram_expected
            ? `<img src="${t.diagram_expected}&_=${Date.now()}" class="mouth-diagram" alt="" />`
            : "";
          const actualImg = t.diagram_actual
            ? `<img src="${t.diagram_actual}&_=${Date.now()}" class="mouth-diagram" alt="" />`
            : "";
          return `
            <div class="tip-box">
              <h4>${e.label}</h4>
              <div class="mouth-diagrams">
                <div class="mouth-diagram-wrap"><span class="mouth-diagram-label correct">正确</span>${expectedImg}</div>
                <div class="mouth-diagram-wrap"><span class="mouth-diagram-label user">你的</span>${actualImg}</div>
              </div>
              <p><span class="label">舌头：</span>${escapeHtml(t.tongue || "")}</p>
              <p><span class="label">嘴唇：</span>${escapeHtml(t.lips || "")}</p>
              <p><span class="label">下巴：</span>${escapeHtml(t.jaw || "")}</p>
            </div>
          `;
        })
        .join("");
    }
    html += `
      <div class="word-result" style="cursor:pointer" data-word="${escapeHtml(w.word)}">
        <div class="word-result-header">
          <span class="word-result-word">${escapeHtml(w.word)}</span>
          <span class="word-result-ipa">/${escapeHtml(w.target_ipa)}/</span>
          <span class="word-result-score" style="color:var(--${scoreClass(w.score)})">${Math.round(w.score * 100)}</span>
        </div>
        ${tipsHtml}
      </div>
    `;
  }
  els.analysisPanel.innerHTML = html;
  els.analysisPanel.querySelectorAll(".word-result").forEach((el) => {
    el.addEventListener("click", () => {
      const word = el.dataset.word;
      window.showWordDetailByWord(word);
    });
  });
}

function highlightSubtitleWords(words) {
  els.subtitleBox.querySelectorAll(".subtitle-word").forEach((el) => {
    el.classList.remove("good", "warn", "bad");
  });
  if (!words) return;
  for (const w of words) {
    const el = Array.from(els.subtitleBox.querySelectorAll(".subtitle-word")).find(
      (e) => e.dataset.word === w.word
    );
    if (el) el.classList.add(scoreClass(w.score));
  }
}

// ---------- word detail ----------
window.showWordDetailByWord = function (word) {
  const s = state.sentences[state.currentIdx];
  const wordObj = s?.words.find((w) => w.text === word);
  const wordResult = state.analysis?.words?.find((w) => w.word === word);
  showWordDetail(wordObj, wordResult);
};

async function fetchWordIpa(word) {
  try {
    const res = await fetch(`/word_ipa?word=${encodeURIComponent(word)}&language=${encodeURIComponent(state.language)}`);
    if (!res.ok) return "";
    const data = await res.json();
    return data.ipa || "";
  } catch (err) {
    return "";
  }
}

async function showWordDetail(wordObj, wordResult) {
  if (!wordObj) return;
  state.selectedWord = wordObj;
  const ipa = wordResult?.target_ipa || (await fetchWordIpa(wordObj.text));
  let html = `
    <div class="word-detail">
      <h4>${escapeHtml(wordObj.text)}</h4>
      <div class="ipa">/${escapeHtml(ipa)}/</div>
      <div class="mini-btns">
        <button class="btn-secondary" onclick="window.playSourceSelected()">▶ 视频原声</button>
        <button class="btn-secondary" onclick="window.playRefSelected()">🔊 标准音</button>
  `;
  if (wordResult) {
    html += `<button class="btn-secondary" onclick="window.playUserSelected()">🎧 我的</button>`;
  }
  html += `</div>`;
  if (wordResult && wordResult.errors?.length) {
    const e = wordResult.errors[0];
    const t = e.tips || {};
    html += `
      <div class="tip-box">
        <h4>${e.label}</h4>
        <p><span class="label">舌头：</span>${escapeHtml(t.tongue || "")}</p>
        <p><span class="label">嘴唇：</span>${escapeHtml(t.lips || "")}</p>
        <p><span class="label">下巴：</span>${escapeHtml(t.jaw || "")}</p>
      </div>
    `;
  }
  html += `</div>`;
  els.analysisPanel.innerHTML = html;
}

window.playSourceSelected = function () {
  if (state.selectedWord) playSourceWord(state.selectedWord);
};

window.playRefSelected = async function () {
  if (!state.selectedWord) return;
  const url = await ensureWordAudio(state.currentIdx, state.selectedWord.text);
  if (url) new Audio(url).play();
};

window.playUserSelected = function () {
  if (!state.selectedWord || !state.analysis) return;
  const wordResult = state.analysis.words?.find((w) => w.word === state.selectedWord.text);
  if (wordResult) playUserWord(wordResult);
};

function playUserWord(wordResult) {
  if (!state.recordedBlob || !state.tokenTimes.length) return;
  const startIdx = wordResult.learner_start || 0;
  const endIdx = Math.max(startIdx + 1, wordResult.learner_end || startIdx + 1);
  const startTime = state.tokenTimes[startIdx] || 0;
  const avg = state.tokenTimes.length > 1
    ? (state.tokenTimes[state.tokenTimes.length - 1] - state.tokenTimes[0]) / state.tokenTimes.length
    : 0.1;
  const endTime = (state.tokenTimes[endIdx - 1] || startTime) + avg;
  playUserSlice(startTime, endTime);
}

async function playUserSlice(startTime, endTime) {
  if (!state.recordedBlob || startTime >= endTime) return;
  try {
    const arrayBuffer = await state.recordedBlob.arrayBuffer();
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

window.playUserSlice = playUserSlice;

// ---------- audio caching ----------
function normalizeWordKey(word) {
  return String(word).toLowerCase().replace(/[.,!?;:"'()\[\]{}«»]/g, "");
}

async function ensureWordAudio(sentenceIdx, word) {
  const key = normalizeWordKey(word);
  if (state.wordAudios[sentenceIdx]?.[key]) return state.wordAudios[sentenceIdx][key];
  await prebakeAudiosFor(sentenceIdx);
  return state.wordAudios[sentenceIdx]?.[key] || null;
}

async function prebakeAudiosFor(idx) {
  if (idx < 0 || idx >= state.sentences.length) return;
  if (state.wordAudios[idx]) return;
  const s = state.sentences[idx];
  try {
    // Word-level reference audios
    const q = new URLSearchParams({ sentence: s.text, language: state.language });
    const res = await fetch(`/prebake_reference?${q.toString()}`);
    if (res.ok) {
      const data = await res.json();
      const map = {};
      for (const [w, b64] of Object.entries(data.audios || {})) {
        const bytes = Uint8Array.from(atob(b64), (c) => c.charCodeAt(0));
        const key = w.toLowerCase().replace(/[.,!?;:"'()\[\]{}«»]/g, "");
        if (key) map[key] = URL.createObjectURL(new Blob([bytes], { type: "audio/wav" }));
      }
      state.wordAudios[idx] = map;
    }
    // Full-sentence reference audio
    const refRes = await fetch(`/reference_audio?text=${encodeURIComponent(s.text)}&language=${encodeURIComponent(state.language)}`);
    if (refRes.ok) {
      state.sentenceAudios[idx] = URL.createObjectURL(await refRes.blob());
    }
  } catch (err) {
    console.warn("prebake failed for sentence", idx, err);
  }
}

async function playReferenceSentence() {
  const idx = state.currentIdx;
  if (!state.sentenceAudios[idx]) await prebakeAudiosFor(idx);
  const url = state.sentenceAudios[idx];
  if (url) new Audio(url).play();
}

async function playReference(word) {
  const url = await ensureWordAudio(state.currentIdx, word);
  if (url) {
    new Audio(url).play();
    return;
  }
  // fallback
  try {
    const q = new URLSearchParams({ text: word, language: state.language, sentence: state.sentences[state.currentIdx]?.text || "" });
    const res = await fetch(`/reference_audio?${q.toString()}`);
    if (!res.ok) return;
    new Audio(URL.createObjectURL(await res.blob())).play();
  } catch (err) {
    console.error(err);
  }
}

// ---------- utilities ----------
function escapeHtml(s) {
  return String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
}

// ---------- controls ----------
function repeatSentence() {
  stopRecording();
  stopSourcePoll();
  setCurrentSentence(state.currentIdx);
  playSourceSentence(state.currentIdx);
}

function continueNext() {
  stopRecording();
  stopSourcePoll();
  const next = state.currentIdx + 1;
  if (next >= state.sentences.length) {
    setStatus("全部完成 🎉");
    return;
  }
  setCurrentSentence(next);
  playSourceSentence(next);
  prebakeAudiosFor(next + 1);
}

// ---------- event listeners ----------
els.topNav.querySelectorAll("button").forEach((btn) => {
  btn.addEventListener("click", () => switchView(btn.dataset.view));
});

els.btnLoad.addEventListener("click", loadVideoInfo);
els.urlInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter") loadVideoInfo();
});
els.btnPlaySource.addEventListener("click", () => playSourceSentence(state.currentIdx));
els.btnRecord.addEventListener("click", startRecording);
els.btnPlayRef.addEventListener("click", playReferenceSentence);
els.btnPlayUser.addEventListener("click", () => {
  if (state.recordedBlob) new Audio(URL.createObjectURL(state.recordedBlob)).play();
});
els.btnRepeat.addEventListener("click", repeatSentence);
els.btnContinue.addEventListener("click", continueNext);

document.addEventListener("keydown", (e) => {
  if (state.mode !== "practice") return;
  const tag = document.activeElement?.tagName;
  if (tag === "INPUT" || tag === "TEXTAREA") return;
  if (e.key === "r" || e.key === "R") {
    e.preventDefault();
    repeatSentence();
  } else if (e.key === "c" || e.key === "C") {
    e.preventDefault();
    continueNext();
  }
});

window.playReference = playReference;
