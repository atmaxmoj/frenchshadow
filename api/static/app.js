const SENTENCE = "The quick brown fox jumps over the lazy dog.";

const els = {
  sentence: document.getElementById("sentence"),
  btnStart: document.getElementById("btnStart"),
  btnStop: document.getElementById("btnStop"),
  btnPlay: document.getElementById("btnPlay"),
  status: document.getElementById("status"),
  timer: document.getElementById("timer"),
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

function renderSentence() {
  els.sentence.innerHTML = "";
  SENTENCE.split(/\s+/).forEach((word, idx) => {
    const span = document.createElement("span");
    span.className = "word";
    span.dataset.index = idx;
    span.textContent = word;
    els.sentence.appendChild(span);
    els.sentence.appendChild(document.createTextNode(" "));
  });
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

async function startRecording() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    mediaRecorder = new MediaRecorder(stream, { mimeType: "audio/webm" });
    recordedChunks = [];

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

    mediaRecorder.start();
    startTimer();
    els.btnStart.disabled = true;
    els.btnStop.disabled = false;
    els.btnPlay.classList.add("hidden");
    els.results.classList.add("hidden");
    resetHighlights();
    setStatus('<span class="recording-dot"></span>正在录音，请读出上面的句子');
  } catch (err) {
    console.error(err);
    setStatus("无法访问麦克风，请检查浏览器权限。");
  }
}

function stopRecording() {
  if (mediaRecorder && mediaRecorder.state !== "inactive") {
    mediaRecorder.stop();
    mediaRecorder.stream.getTracks().forEach((t) => t.stop());
  }
  stopTimer();
  els.btnStart.disabled = false;
  els.btnStop.disabled = true;
  setStatus("录音结束，正在分析...");
}

async function uploadAndAnalyze(blob) {
  const form = new FormData();
  form.append("audio", blob, "recording.webm");
  form.append("target_text", SENTENCE);

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
    if (w.score >= 0.99 && w.errors.length === 0) return;
    const item = document.createElement("div");
    item.className = "result-item";
    let errs = "";
    w.errors.forEach((e) => {
      const tips = e.tips || {};
      errs += `
        <div class="tip-box">
          <h4>${e.label} <span style="color:var(--muted);font-weight:400;">(${e.expected || "-"} → ${e.actual || "-"})</span></h4>
          <p><span class="label">问题：</span>${tips.description || ""}</p>
          <p><span class="label">舌头：</span>${tips.tongue || ""}</p>
          <p><span class="label">嘴唇：</span>${tips.lips || ""}</p>
          <p><span class="label">下巴：</span>${tips.jaw || ""}</p>
          <p><span class="label">练习：</span>${tips.practice || ""}</p>
        </div>
      `;
    });
    item.innerHTML = `
      <div class="result-header">
        <span class="result-word">${w.word}</span>
        <span class="result-ipa">/${w.target_ipa}/</span>
        <span class="result-score" style="color:${scoreClass(w.score) === "good" ? "var(--good)" : scoreClass(w.score) === "warn" ? "var(--warn)" : "var(--bad)"}">${Math.round(w.score * 100)}</span>
      </div>
      ${errs}
    `;
    els.resultList.appendChild(item);
  });

  els.results.classList.remove("hidden");
  els.btnPlay.classList.remove("hidden");
}

function resetHighlights() {
  els.sentence.querySelectorAll(".word").forEach((span) => {
    span.classList.remove("active", "good", "warn", "bad");
    span.title = "";
  });
}

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
els.btnPlay.addEventListener("click", playWithHighlight);

renderSentence();
