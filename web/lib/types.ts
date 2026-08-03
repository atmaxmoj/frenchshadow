// Shared data shapes returned by the FastAPI backend.

export interface WordToken {
  text: string;
  start: number;
  end: number;
}

export interface Sentence {
  text: string;
  start: number;
  end: number;
  words: WordToken[];
}

export interface Transcript {
  video_id: string;
  language: string;
  sentence_count: number;
  sentences: Sentence[];
}

export interface CaptionLanguage {
  code: string;
  name: string;
  generated: boolean;
}

export interface VideoInfo {
  video_id: string;
  title: string;
  author: string;
  thumbnail: string;
  available_languages: CaptionLanguage[];
  preferred_language: string;
  has_preferred_language: boolean;
}

export interface ArticulatoryTip {
  description?: string;
  tongue?: string;
  lips?: string;
  jaw?: string;
  practice?: string;
  diagram_expected?: string | null;
  diagram_actual?: string | null;
}

export interface WordError {
  position: number;
  ref_index: number;
  expected: string | null;
  actual: string | null;
  label: string;
  l1_pattern: boolean;
  confidence: string;
  tips?: ArticulatoryTip | null;
}

export interface WordResult {
  word: string;
  target_ipa: string;
  target_phones: string[];
  learner_ipa: string;
  score: number;
  coverage: number;
  intelligibility?: number; // 达意: would a listener recover this word (Whisper)
  learner_start: number;
  learner_end: number;
  errors: WordError[];
  start_time?: number;
  end_time?: number;
}

export interface Analysis {
  overall_score: number; // 发音: pronunciation quality (native-likeness)
  coverage: number; // legacy GOP completeness (fallback for 达意)
  intelligibility?: number; // 达意: would a listener understand (Whisper), sentence-level
  heard?: string; // what Whisper (a context-aware listener) actually heard
  sentences: string[];
  words: WordResult[];
}

export interface TranscribeResult {
  duration_s: number;
  tokens: string[];
  analysis?: Analysis;
  analysis_error?: string;
}
