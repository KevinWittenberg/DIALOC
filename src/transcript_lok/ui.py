from __future__ import annotations

import os
import json
import hashlib
import html
import threading
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any
from urllib.parse import urlparse

import gradio as gr
import numpy as np
import pandas as pd

from .asr import ASRConfig, WhisperTranscriber, loaded_model_keys
from .audio import TARGET_SAMPLE_RATE, audio_metadata, resample_audio, to_mono_float32, write_wav
from .diagnostics import RuntimeDiagnostics, collect_diagnostics, collect_readiness, diagnostics_markdown, readiness_markdown
from .diarization import DEFAULT_DIARIZATION_MODEL, DiarizationConfig, SpeakerDiarizer
from .llama_cpp import (
    DEFAULT_LLAMA_CPP_CHAT_TEMPLATE,
    DEFAULT_LLAMA_CPP_CONTEXT_SIZE,
    DEFAULT_LLAMA_CPP_GPU_LAYERS,
    DEFAULT_LLAMA_CPP_MODEL,
    DEFAULT_LLAMA_CPP_PORT,
    LlamaCppServerConfig,
    default_llama_cpp_dir,
    default_llama_cpp_endpoint,
    start_llama_cpp_server,
    stop_llama_cpp_server,
)
from .llm import (
    TEMPLATE_TASKS,
    SummaryConfig,
    correct_transcript,
    llm_provenance,
    provider_health,
    run_llm_task,
)
from .models import TRANSCRIPT_COLUMNS, TranscriptDocument, format_timestamp
from .policy import TranscriptLokPolicy, load_policy
from .sessions import load_session, save_session


ROOT = Path(__file__).resolve().parents[2]
ACTIVE_POLICY = load_policy()
OUTPUT_DIR = Path(os.getenv("TRANSCRIPT_LOK_OUTPUT_DIR", str(ROOT / "outputs"))).expanduser()
LANGUAGES = {"Auto": None, "Dutch": "nl", "English": "en"}
ARTIFACT_LANGUAGES = {"Dutch": "nl", "English": "en"}
ASR_MODELS = ["large-v3", "distil-large-v3", "medium", "small", "base", "tiny"]
COMPUTE_TYPES = ["auto", "float16", "int8_float16", "int8", "float32"]
DEVICES = ["auto", "cuda", "cpu"]
SUMMARY_PROVIDERS = ["ollama", "llama.cpp", "openai-compatible"] + (["openai"] if ACTIVE_POLICY.feature_enabled("cloud_providers") else [])
MODEL_PRESETS = {
    "Fast live": ("tiny", "auto", "auto"),
    "Production CPU": ("base", "cpu", "int8"),
    "Balanced file": ("small", "auto", "auto"),
    "Accurate final": ("medium", "cpu", "int8"),
    "Fast draft": ("tiny", "auto", "auto"),
    "Production": ("base", "cpu", "int8"),
    "Balanced": ("small", "auto", "auto"),
    "Accurate CPU": ("medium", "cpu", "int8"),
    "Accurate GPU": ("large-v3", "cuda", "float16"),
}
WORKFLOW_PROFILES = {
    "Fast live": {
        "asr": ("tiny", "auto", "auto"),
        "llm": ("llama.cpp", "gemma-4", default_llama_cpp_endpoint()),
        "description": "Lowest latency local meeting capture.",
    },
    "Production CPU": {
        "asr": ("base", "cpu", "int8"),
        "llm": ("llama.cpp", "gemma-3-1b", default_llama_cpp_endpoint()),
        "description": "2-thread CPU profile for OpenOnDemand production.",
    },
    "Balanced file": {
        "asr": ("small", "auto", "auto"),
        "llm": ("llama.cpp", "gemma-4", default_llama_cpp_endpoint()),
        "description": "Everyday file transcription and local artifacts.",
    },
    "Accurate final": {
        "asr": ("medium", "cpu", "int8"),
        "llm": ("llama.cpp", "gemma-4", default_llama_cpp_endpoint()),
        "description": "Slower final local pass with stronger ASR.",
    },
    "Light LLM": {
        "asr": ("small", "auto", "auto"),
        "llm": ("llama.cpp", "gemma-4", default_llama_cpp_endpoint()),
        "description": "Local Gemma/GGUF artifact generation.",
    },
    "Best local LLM": {
        "asr": ("small", "auto", "auto"),
        "llm": ("llama.cpp", "gemma-4", default_llama_cpp_endpoint()),
        "description": "Prefer the best local Gemma model available on the llama.cpp server.",
    },
}
DEFAULT_MODEL_PRESET = "Production CPU" if ACTIVE_POLICY.runtime.production_cpu else "Balanced"
DEFAULT_WORKFLOW_PROFILE = "Production CPU" if ACTIVE_POLICY.runtime.production_cpu else "Balanced file"
DEFAULT_SUMMARY_PROVIDER = SUMMARY_PROVIDERS[0] if SUMMARY_PROVIDERS else "ollama"
DEFAULT_SUMMARY_MODEL = "gemma-3-1b" if ACTIVE_POLICY.runtime.production_cpu else "llama3.1"
ARTIFACT_TILES = [
    {"feature": "correction", "label": "Correction pass", "kind": "correction"},
    {"feature": "meeting_summary", "label": "Meeting summary", "kind": "summary", "template": "Meeting summary", "primary": True},
    {"feature": "action_points", "label": "Action points", "kind": "summary", "template": "Action points"},
    {
        "feature": "decision_summaries",
        "label": "Internal anonymous decisions",
        "kind": "summary",
        "template": "Decision summary - internal anonymous",
    },
    {
        "feature": "decision_summaries",
        "label": "External anonymous decisions",
        "kind": "summary",
        "template": "Decision summary - external anonymous",
    },
    {"feature": "follow_up_email", "label": "Follow-up email", "kind": "follow_up"},
]
PROGRESS_BAR_WIDTH = 20
PROGRESS_BUCKET_PERCENT = 5
PROGRESS_BUCKET_SECONDS = 30
FILE_CANCEL_EVENT = threading.Event()
APP_CSS = """
.gradio-container {
  max-width: 100% !important;
  color: #17213a;
  background: #f6f9fb;
}
.lok-shell {
  gap: 0 !important;
}
.lok-topbar {
  height: 64px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 0 28px;
  border-bottom: 1px solid #dce4ec;
  background: linear-gradient(180deg, #ffffff 0%, #f9fbfd 100%);
}
.lok-brand {
  display: flex;
  align-items: center;
  gap: 12px;
  font-size: 22px;
  font-weight: 750;
}
.lok-wave {
  width: 28px;
  height: 28px;
  border-radius: 8px;
  background: repeating-linear-gradient(90deg, #008c93 0 3px, transparent 3px 7px);
}
.lok-topmeta {
  display: flex;
  align-items: center;
  gap: 18px;
  color: #4c5b73;
  font-size: 13px;
}
.lok-ready {
  padding: 7px 13px;
  border: 1px solid #bfe7d2;
  border-radius: 8px;
  color: #087a3c;
  background: #ecfff5;
  font-weight: 700;
}
.lok-grid {
  display: grid !important;
  grid-template-columns: minmax(250px, 300px) minmax(560px, 1fr);
  gap: 0 !important;
  align-items: stretch;
}
.lok-sidebar {
  min-height: calc(100vh - 72px);
  padding: 22px 22px;
  background: #ffffff;
}
.lok-sidebar {
  border-right: 1px solid #dce4ec;
}
.lok-settings {
  position: fixed !important;
  top: 64px;
  right: 0;
  z-index: 60;
  width: min(390px, calc(100vw - 44px));
  max-width: calc(100vw - 44px);
  height: calc(100vh - 64px);
  display: flex !important;
  flex-direction: column !important;
  flex-wrap: nowrap !important;
  align-items: stretch !important;
  gap: 12px !important;
  overflow-y: auto;
  overflow-x: hidden;
  padding: 22px 22px 32px;
  background: #ffffff;
  border-left: 1px solid #dce4ec;
  box-shadow: -18px 0 36px rgba(22, 32, 54, 0.14);
  transform: translateX(calc(100% + 2px));
  opacity: 0;
  pointer-events: none;
  transition: transform 160ms ease, opacity 160ms ease;
}
.lok-settings.lok-settings-open {
  transform: translateX(0);
  opacity: 1;
  pointer-events: auto;
}
#lok-settings-toggle {
  position: fixed !important;
  right: 0;
  top: 86px;
  z-index: 70;
  width: 44px !important;
  min-width: 44px !important;
  max-width: 44px !important;
  height: 64px !important;
  padding: 0 !important;
  border-radius: 8px 0 0 8px !important;
  border: 1px solid #bfd7e4 !important;
  border-right: 0 !important;
  background: #e9f6f7 !important;
  color: transparent !important;
  font-size: 0 !important;
  line-height: 0 !important;
  overflow: hidden !important;
  box-shadow: 0 6px 16px rgba(22, 32, 54, 0.12);
}
#lok-settings-toggle > button {
  position: absolute !important;
  inset: 0 !important;
  width: 44px !important;
  min-width: 44px !important;
  height: 64px !important;
  padding: 0 !important;
  border: 0 !important;
  background: transparent !important;
  color: transparent !important;
  font-size: 0 !important;
  line-height: 0 !important;
  box-shadow: none !important;
}
#lok-settings-toggle::before,
#lok-settings-toggle > button::before {
  content: "";
  display: block;
  width: 20px;
  height: 16px;
  margin: 24px auto 0;
  background:
    linear-gradient(#075f66, #075f66) 0 1px / 20px 2px no-repeat,
    linear-gradient(#075f66, #075f66) 0 7px / 20px 2px no-repeat,
    linear-gradient(#075f66, #075f66) 0 13px / 20px 2px no-repeat;
}
#lok-settings-toggle::after,
#lok-settings-toggle > button::after {
  content: "";
  position: absolute;
  left: 12px;
  top: 19px;
  width: 4px;
  height: 4px;
  border-radius: 999px;
  background: #075f66;
  box-shadow: 8px 6px 0 #075f66, 2px 12px 0 #075f66;
}
.lok-workbench {
  min-width: 0;
  padding: 22px 20px;
}
.lok-panel,
.lok-card,
.lok-artifact-link,
.lok-status-card {
  border: 1px solid #dde6ef;
  border-radius: 8px;
  background: #ffffff;
  box-shadow: 0 1px 2px rgba(20, 32, 56, 0.03);
}
.lok-panel {
  padding: 16px;
}
.lok-card,
.lok-status-card {
  padding: 14px;
}
.lok-artifact-link {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 12px 14px;
  margin: 8px 0;
  color: #263553;
}
.lok-muted {
  color: #607089;
  font-size: 13px;
}
.lok-accent {
  color: #008c93;
}
.lok-meter {
  height: 76px;
  border: 1px solid #dce4ec;
  border-radius: 6px;
  background:
    linear-gradient(90deg, transparent 0 8px, rgba(0, 140, 147, 0.25) 8px 10px, transparent 10px 18px),
    linear-gradient(180deg, transparent 0 44%, #0b9ca3 44% 56%, transparent 56%);
}
.lok-runtime-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
  gap: 14px;
  margin-bottom: 16px;
}
.lok-status-card h3 {
  margin: 0 0 10px;
  font-size: 18px;
}
.lok-status-card p {
  margin: 0;
  line-height: 1.55;
}
.lok-chip {
  display: inline-block;
  padding: 3px 8px;
  border-radius: 999px;
  color: #087a3c;
  background: #dff8e8;
  font-size: 12px;
  font-weight: 700;
}
.lok-table-note {
  display: flex;
  justify-content: space-between;
  color: #607089;
  font-size: 13px;
  padding-top: 8px;
}
.lok-settings .gradio-accordion {
  border-radius: 8px !important;
}
.lok-flash {
  animation: lok-flash 1.5s ease-out;
}
@keyframes lok-flash {
  0% { box-shadow: 0 0 0 3px rgba(0, 140, 147, 0.22); }
  100% { box-shadow: 0 0 0 0 rgba(0, 140, 147, 0); }
}
.lok-artifact-grid {
  display: grid !important;
  grid-template-columns: repeat(auto-fit, minmax(190px, 1fr));
  gap: 12px !important;
  align-items: stretch;
}
.lok-artifact-tile {
  min-width: 0 !important;
}
.lok-artifact-tile button {
  min-height: 78px !important;
  white-space: normal !important;
  text-align: left !important;
  justify-content: flex-start !important;
  border: 1px solid #d7e4ed !important;
  border-radius: 8px !important;
  background: #ffffff !important;
  color: #17213a !important;
  box-shadow: 0 1px 2px rgba(20, 32, 56, 0.04);
}
.lok-artifact-tile button:hover {
  border-color: #008c93 !important;
  background: #f1fbfc !important;
}
.lok-workbench .tab-nav {
  border-bottom: 1px solid #dce4ec;
}
button.primary,
.gradio-button.primary {
  background: #008c93 !important;
  border-color: #008c93 !important;
}
@media (max-width: 1200px) {
  .lok-grid {
    grid-template-columns: 1fr;
  }
  .lok-sidebar {
    min-height: auto;
    border: 0;
  }
  .lok-settings {
    top: 64px;
    height: calc(100vh - 64px);
  }
}
"""

SETTINGS_SEARCH_OPTIONS = [
    "Model: Model & Profile",
    "Runtime: Readiness",
    "Audio: Live transcription",
    "Transcript: Search and edits",
    "Artifacts: Summaries and corrections",
    "Glossary: Custom wordlist",
    "Provider: LLM endpoint",
]

SETTINGS_SEARCH_JS = """
(value) => {
  const targets = {
    "Model: Model & Profile": "setting-model-profile",
    "Runtime: Readiness": "setting-workflow-profile",
    "Audio: Live transcription": "setting-live-controls",
    "Transcript: Search and edits": "setting-transcript-tools",
    "Artifacts: Summaries and corrections": "setting-artifact-language",
    "Glossary: Custom wordlist": "setting-glossary",
    "Provider: LLM endpoint": "setting-llm-provider"
  };
  const targetId = targets[value];
  if (targetId) {
    const target = document.getElementById(targetId);
    if (target) {
      const details = target.closest("details");
      if (details) {
        details.open = true;
      }
      window.setTimeout(() => {
        target.scrollIntoView({ behavior: "smooth", block: "center" });
        target.classList.add("lok-flash");
        window.setTimeout(() => target.classList.remove("lok-flash"), 1600);
      }, 80);
    }
  }
  return value;
}
"""

SETTINGS_TOGGLE_JS = """
(opened) => {
  const nextOpen = !Boolean(opened);
  const panel = document.querySelector(".lok-settings");
  if (panel) {
    panel.classList.toggle("lok-settings-open", nextOpen);
  }
  return opened;
}
"""


def build_app() -> gr.Blocks:
    policy = ACTIVE_POLICY
    with gr.Blocks(title="Transcript LOK") as app:
        doc_state = gr.State({})
        live_audio_state = gr.State(None)
        live_pause_state = gr.State(False)
        settings_open_state = gr.State(False)
        meeting_summary_template = gr.State("Meeting summary")
        action_points_template = gr.State("Action points")
        internal_decision_template = gr.State("Decision summary - internal anonymous")
        external_decision_template = gr.State("Decision summary - external anonymous")
        summary_state = gr.State("")
        exports_state = gr.State([])
        llm_outputs_state = gr.State({})

        with gr.Column(elem_classes=["lok-shell"]):
            gr.HTML(f"<style>{APP_CSS}</style>")
            gr.HTML(_topbar_html())
            settings_toggle = gr.Button("Settings", elem_id="lok-settings-toggle")
            with gr.Row(elem_classes=["lok-grid"]):
                with gr.Column(elem_classes=["lok-sidebar"]):
                    gr.Markdown("### Live Meeting\n<span class='lok-accent'>Connected</span>")
                    gr.Markdown("**Product Sync**\n\n00:00:00", elem_classes=["lok-muted"])
                    live_mic = gr.Audio(label="Audio input", sources=["microphone"], type="numpy", streaming=True, visible=policy.feature_enabled("live_transcription"))
                    with gr.Row():
                        pause_live = gr.Button("Pause", variant="primary", visible=policy.feature_enabled("live_transcription"))
                        reset_live = gr.Button("Reset live", visible=policy.feature_enabled("live_transcription"))
                    live_behavior_note = gr.Markdown("Live transcription is literal. Run correction after the meeting for cleanup.")
                    live_chunk_seconds = gr.Slider(4, 30, value=10, step=1, label="Chunk seconds", elem_id="setting-live-controls")
                    live_rolling_context = gr.Checkbox(value=True, label="Rolling context")
                    live_autosave_every = gr.Number(value=10, label="Auto-save every N segments", precision=0)
                    gr.HTML("<div class='lok-meter' aria-label='Audio level placeholder'></div>")
                    gr.HTML(
                        "<div class='lok-artifact-link'><span>Meeting Artifacts</span><strong>3</strong></div>"
                        "<div class='lok-artifact-link'><span>Action Items</span><strong>7</strong></div>"
                        "<div class='lok-artifact-link'><span>Summary</span><strong>1</strong></div>"
                    )
                    open_outputs_button = gr.Button("Open Artifacts")

                with gr.Column(elem_classes=["lok-workbench"]):
                    job_status = gr.Markdown(_job_status("Idle", "Load audio or start a live session."), elem_classes=["lok-panel"])
                    diagnostics_panel = gr.Markdown()
                    audio_preflight = gr.Markdown()

                    with gr.Tabs():
                        with gr.Tab("Runtime"):
                            runtime_cards = gr.HTML(
                                _runtime_cards_html(
                                    collect_diagnostics(
                                        ACTIVE_POLICY.runtime.default_model,
                                        ACTIVE_POLICY.runtime.default_device,
                                        ACTIVE_POLICY.runtime.default_compute_type,
                                        cpu_threads=ACTIVE_POLICY.runtime.cpu_threads,
                                    )
                                )
                            )
                            workflow_profile = gr.Dropdown(
                                list(WORKFLOW_PROFILES.keys()),
                                value=DEFAULT_WORKFLOW_PROFILE,
                                label="Workflow profile",
                                elem_id="setting-workflow-profile",
                            )
                            with gr.Row():
                                apply_workflow_profile = gr.Button("Apply profile", variant="primary")
                                readiness_button = gr.Button("Refresh readiness")
                            readiness_panel = gr.Markdown("Press Refresh readiness to inspect the local runtime.", label="Readiness", elem_classes=["lok-panel"])

                        with gr.Tab("File"):
                            with gr.Column(elem_classes=["lok-panel"]):
                                audio_file = gr.Audio(label="Audio file", type="filepath")
                                transcription_mode = gr.Radio(
                                    ["Raw", "Corrected"] if policy.feature_enabled("correction") else ["Raw"],
                                    value="Raw",
                                    label="Transcript mode",
                                    visible=policy.feature_enabled("file_transcription"),
                                )
                                with gr.Row():
                                    run_file = gr.Button("Transcribe file", variant="primary", visible=policy.feature_enabled("file_transcription"))
                                    run_file_with_diarization = gr.Button(
                                        "Transcribe + diarize",
                                        visible=policy.feature_enabled("file_transcription") and policy.feature_enabled("diarization"),
                                    )
                                    cancel_job = gr.Button("Cancel", visible=policy.feature_enabled("file_transcription"))

                        with gr.Tab("Live"):
                            with gr.Column(elem_classes=["lok-panel"]):
                                gr.Markdown("### Live Controls")
                                gr.Markdown("Live output stays literal. Summaries and corrections are post-meeting artifact actions.")
                                with gr.Row():
                                    commit_live = gr.Button("Commit live chunk", variant="primary", visible=policy.feature_enabled("live_transcription"))
                                    discard_live = gr.Button("Discard last chunk", visible=policy.feature_enabled("live_transcription"))
                                    diarize_live = gr.Button(
                                        "Diarize live recording",
                                        visible=policy.feature_enabled("live_transcription") and policy.feature_enabled("diarization"),
                                    )

                        with gr.Tab("Transcript"):
                            with gr.Row():
                                search_text = gr.Textbox(label="Search transcript")
                                search_speaker = gr.Textbox(label="Speaker")
                                search_start = gr.Number(label="Start", precision=2)
                                search_end = gr.Number(label="End", precision=2)
                            with gr.Row():
                                apply_filter = gr.Button("Filter")
                                clear_filter = gr.Button("Clear filter")
                                apply_edits = gr.Button("Apply edits")
                            transcript_table = gr.Dataframe(
                                headers=TRANSCRIPT_COLUMNS,
                                datatype=["number", "number", "str", "str", "number"],
                                label="Transcript",
                                interactive=True,
                                row_count=(10, "dynamic"),
                                column_count=(5, "fixed"),
                                elem_classes=["lok-panel"],
                            )
                            gr.HTML("<div class='lok-table-note'><span>Rows per page 25</span><span>1-25 of current transcript</span></div>")
                            with gr.Row():
                                row_index = gr.Number(value=1, label="Row #", precision=0)
                                split_time = gr.Number(label="Split at seconds", precision=2)
                                merge_next = gr.Button("Merge with next")
                                split_row = gr.Button("Split row")
                            speaker_names_table = gr.Dataframe(
                                headers=["speaker", "name"],
                                datatype=["str", "str"],
                                label="Speaker names",
                                interactive=True,
                                row_count=(1, "dynamic"),
                                column_count=(2, "fixed"),
                            )
                            with gr.Row():
                                refresh_speakers = gr.Button("Find speakers")
                                apply_names = gr.Button("Apply speaker names")
                                export_files = gr.Button("Export")
                            with gr.Row():
                                save_session_button = gr.Button("Save session")
                                load_session_file = gr.File(label="Load session", file_count="single")
                                load_session_button = gr.Button("Load")
                            session_output = gr.File(label="Session file")
                            export_output = gr.File(label="Exports", file_count="multiple")
                            copy_transcript_button = gr.Button("Copy transcript text")
                            transcript_copy = gr.Textbox(label="Transcript text", lines=8)

                        with gr.Tab("Artifacts"):
                            gr.Markdown("### Meeting Artifacts")
                            with gr.Row(elem_classes=["lok-artifact-grid"]):
                                run_correction = gr.Button("Correction pass", elem_classes=["lok-artifact-tile"], visible=policy.feature_enabled("correction"))
                                run_meeting_summary = gr.Button(
                                    "Meeting summary",
                                    variant="primary",
                                    elem_classes=["lok-artifact-tile"],
                                    visible=policy.feature_enabled("meeting_summary"),
                                )
                                run_action_points = gr.Button("Action points", elem_classes=["lok-artifact-tile"], visible=policy.feature_enabled("action_points"))
                                run_internal_decisions = gr.Button(
                                    "Internal anonymous decisions",
                                    elem_classes=["lok-artifact-tile"],
                                    visible=policy.feature_enabled("decision_summaries"),
                                )
                                run_external_decisions = gr.Button(
                                    "External anonymous decisions",
                                    elem_classes=["lok-artifact-tile"],
                                    visible=policy.feature_enabled("decision_summaries"),
                                )
                                run_follow_up = gr.Button("Follow-up email", elem_classes=["lok-artifact-tile"], visible=policy.feature_enabled("follow_up_email"))
                            with gr.Row():
                                export_llm_artifacts = gr.Button("Export LLM artifacts")
                                copy_summary_button = gr.Button("Copy summary")
                            summary_output = gr.Markdown(label="Summary", elem_classes=["lok-panel"])
                            llm_provenance_panel = gr.Markdown("No LLM outputs yet.", label="LLM provenance", elem_classes=["lok-panel"])
                            llm_artifact_output = gr.File(label="LLM artifact files", file_count="multiple")
                            summary_copy = gr.Textbox(label="Summary text", lines=8)

                    transcript_preview = gr.Markdown(label="Preview", elem_classes=["lok-panel"])

                with gr.Column(elem_classes=["lok-settings"]):
                    gr.Markdown("### Model & Profile")
                    settings_search = gr.Dropdown(
                        SETTINGS_SEARCH_OPTIONS,
                        label="Search settings",
                        value=None,
                        allow_custom_value=True,
                    )
                    with gr.Accordion("ASR", open=True):
                        model_preset = gr.Dropdown(list(MODEL_PRESETS.keys()), value=DEFAULT_MODEL_PRESET, label="Model preset", elem_id="setting-model-profile")
                        model_name = gr.Dropdown(ASR_MODELS, value=policy.runtime.default_model, label="Whisper model", elem_id="setting-whisper-model")
                        language = gr.Dropdown(list(LANGUAGES.keys()), value="Auto", label="Transcript language", elem_id="setting-transcript-language")
                        device = gr.Dropdown(DEVICES, value=policy.runtime.default_device, label="Device", elem_id="setting-device")
                        compute_type = gr.Dropdown(COMPUTE_TYPES, value=policy.runtime.default_compute_type, label="Compute type", elem_id="setting-compute-type")
                        chunk_enabled = gr.Checkbox(value=policy.runtime.chunk_long_files, label="Chunk long files", elem_id="setting-transcript-tools")
                        chunk_seconds = gr.Slider(60, 1800, value=policy.runtime.chunk_seconds, step=30, label="Chunk seconds")
                        overlap_seconds = gr.Slider(0, 30, value=policy.runtime.overlap_seconds, step=1, label="Overlap seconds")
                        diagnostics_button = gr.Button("Refresh diagnostics")

                    with gr.Accordion("Diarization", open=False, visible=policy.feature_enabled("diarization")):
                        hf_token = gr.Textbox(label="Hugging Face token", type="password")
                        diarization_model = gr.Textbox(label="Diarization model or local path", value=DEFAULT_DIARIZATION_MODEL, elem_id="setting-diarization")
                        with gr.Row():
                            min_speakers = gr.Number(label="Min speakers", precision=0)
                            max_speakers = gr.Number(label="Max speakers", precision=0)

                    with gr.Accordion("Glossary", open=False):
                        glossary = gr.Textbox(label="Glossary", lines=6, placeholder="Names, jargon, acronyms", elem_id="setting-glossary")
                        glossary_file = gr.File(label="Import glossary", file_count="single")
                        with gr.Row():
                            load_glossary = gr.Button("Load")
                            export_glossary = gr.Button("Export")
                        glossary_export = gr.File(label="Glossary file")

                    with gr.Accordion("LLM Provider", open=True):
                        summary_provider = gr.Dropdown(SUMMARY_PROVIDERS, value=DEFAULT_SUMMARY_PROVIDER, label="Provider", elem_id="setting-llm-provider")
                        summary_model = gr.Textbox(label="Model", value=DEFAULT_SUMMARY_MODEL, elem_id="setting-llm-model")
                        artifact_language = gr.Dropdown(list(ARTIFACT_LANGUAGES.keys()), value="Dutch", label="Artifact language", elem_id="setting-artifact-language")
                        summary_prompt = gr.Textbox(label="Custom prompt", lines=4)
                        summary_endpoint = gr.Textbox(label="Endpoint")
                        summary_api_key = gr.Textbox(label="API key", type="password")
                        with gr.Accordion("llama.cpp server", open=False):
                            llama_cpp_dir = gr.Textbox(label="llama.cpp folder", value=default_llama_cpp_dir(), elem_id="setting-llama-cpp")
                            llama_cpp_model_source = gr.Textbox(
                                label="GGUF path or Hugging Face repo",
                                value=os.getenv("LLAMA_CPP_MODEL_SOURCE", ""),
                                placeholder=r"C:\models\gemma-4-4b-it-Q4_K_M.gguf or ggml-org/gemma-4-E4B-it-GGUF:Q4_K_M",
                            )
                            with gr.Row():
                                llama_cpp_port = gr.Number(value=DEFAULT_LLAMA_CPP_PORT, label="Port", precision=0)
                                llama_cpp_context_size = gr.Number(value=DEFAULT_LLAMA_CPP_CONTEXT_SIZE, label="Context", precision=0)
                            with gr.Row():
                                llama_cpp_gpu_layers = gr.Textbox(label="GPU layers", value=DEFAULT_LLAMA_CPP_GPU_LAYERS)
                                llama_cpp_chat_template = gr.Textbox(label="Chat template", value=DEFAULT_LLAMA_CPP_CHAT_TEMPLATE)
                            with gr.Row():
                                start_llama_cpp = gr.Button("Start llama.cpp")
                                stop_llama_cpp = gr.Button("Stop llama.cpp")
                        health_check = gr.Button("Health check")

        settings_toggle.click(
            _toggle_settings_panel,
            inputs=[settings_open_state],
            outputs=[settings_open_state, settings_toggle],
            js=SETTINGS_TOGGLE_JS,
            queue=False,
            show_progress="hidden",
        )
        settings_search.change(
            _clear_settings_search,
            inputs=[settings_search],
            outputs=[settings_search],
            js=SETTINGS_SEARCH_JS,
            queue=False,
            show_progress="hidden",
        )
        model_preset.change(_apply_model_preset, inputs=[model_preset], outputs=[model_name, device, compute_type])
        apply_workflow_profile.click(
            _apply_workflow_profile,
            inputs=[workflow_profile],
            outputs=[model_name, device, compute_type, summary_provider, summary_model, summary_endpoint, job_status],
        )
        readiness_button.click(
            _readiness,
            inputs=[
                model_name,
                device,
                compute_type,
                diarization_model,
                summary_provider,
                summary_model,
                summary_endpoint,
                llama_cpp_dir,
                llama_cpp_model_source,
            ],
            outputs=[runtime_cards, readiness_panel, job_status],
        )
        diagnostics_button.click(
            _diagnostics,
            inputs=[model_name, device, compute_type, diarization_model],
            outputs=[runtime_cards, diagnostics_panel, job_status],
        )
        audio_file.change(
            _preflight_audio,
            inputs=[audio_file, model_name, device, compute_type, chunk_enabled, chunk_seconds],
            outputs=[audio_preflight, job_status],
        )

        run_event = run_file.click(
            fn=_transcribe_file,
            inputs=[
                audio_file,
                model_name,
                language,
                device,
                compute_type,
                glossary,
                transcription_mode,
                summary_provider,
                summary_model,
                summary_endpoint,
                summary_api_key,
                artifact_language,
                gr.State(False),
                hf_token,
                diarization_model,
                min_speakers,
                max_speakers,
                chunk_enabled,
                chunk_seconds,
                overlap_seconds,
            ],
            outputs=[doc_state, transcript_table, transcript_preview, job_status],
        )
        diarize_event = run_file_with_diarization.click(
            fn=_transcribe_file,
            inputs=[
                audio_file,
                model_name,
                language,
                device,
                compute_type,
                glossary,
                transcription_mode,
                summary_provider,
                summary_model,
                summary_endpoint,
                summary_api_key,
                artifact_language,
                gr.State(True),
                hf_token,
                diarization_model,
                min_speakers,
                max_speakers,
                chunk_enabled,
                chunk_seconds,
                overlap_seconds,
            ],
            outputs=[doc_state, transcript_table, transcript_preview, job_status],
        )
        cancel_job.click(_cancel_file_job, outputs=[job_status], cancels=[run_event, diarize_event])

        live_mic.stream(
            fn=_stream_live,
            inputs=[
                live_mic,
                live_audio_state,
                doc_state,
                model_name,
                language,
                device,
                compute_type,
                glossary,
                live_chunk_seconds,
                live_pause_state,
                live_rolling_context,
                live_autosave_every,
            ],
            outputs=[live_audio_state, doc_state, transcript_table, transcript_preview, job_status],
            stream_every=1,
        )
        pause_live.click(_toggle_live_pause, inputs=[live_pause_state], outputs=[live_pause_state, pause_live, job_status])
        commit_live.click(
            fn=_flush_live,
            inputs=[live_audio_state, doc_state, model_name, language, device, compute_type, glossary, live_rolling_context, live_autosave_every],
            outputs=[live_audio_state, doc_state, transcript_table, transcript_preview, job_status],
        )
        discard_live.click(
            fn=_discard_last_live_chunk,
            inputs=[live_audio_state, doc_state],
            outputs=[live_audio_state, doc_state, transcript_table, transcript_preview, job_status],
        )
        diarize_live.click(
            fn=_diarize_live,
            inputs=[live_audio_state, doc_state, hf_token, diarization_model, device, min_speakers, max_speakers],
            outputs=[doc_state, transcript_table, transcript_preview, job_status],
        )
        reset_live.click(fn=_reset_live, outputs=[live_audio_state, live_pause_state, pause_live, doc_state, transcript_table, transcript_preview, job_status])

        load_glossary.click(_load_glossary, inputs=[glossary_file], outputs=[glossary, job_status])
        export_glossary.click(_export_glossary, inputs=[glossary], outputs=[glossary_export, job_status])
        apply_edits.click(_apply_edits, inputs=[transcript_table, doc_state], outputs=[doc_state, transcript_table, transcript_preview, job_status])
        apply_filter.click(_filter_transcript, inputs=[transcript_table, search_text, search_speaker, search_start, search_end], outputs=[transcript_table, job_status])
        clear_filter.click(_clear_filter, inputs=[doc_state], outputs=[transcript_table, job_status])
        merge_next.click(_merge_next, inputs=[transcript_table, doc_state, row_index], outputs=[doc_state, transcript_table, transcript_preview, job_status])
        split_row.click(_split_row, inputs=[transcript_table, doc_state, row_index, split_time], outputs=[doc_state, transcript_table, transcript_preview, job_status])
        refresh_speakers.click(_speaker_table, inputs=[transcript_table, doc_state], outputs=[speaker_names_table, job_status])
        apply_names.click(_apply_names, inputs=[transcript_table, doc_state, speaker_names_table], outputs=[doc_state, transcript_table, transcript_preview, job_status])
        export_files.click(_export, inputs=[transcript_table, doc_state], outputs=[export_output, exports_state, job_status])
        save_session_button.click(
            _save_session,
            inputs=[
                transcript_table,
                doc_state,
                model_name,
                language,
                device,
                compute_type,
                glossary,
                speaker_names_table,
                summary_state,
                exports_state,
                llm_outputs_state,
            ],
            outputs=[session_output, job_status],
        )
        load_session_button.click(
            _load_session,
            inputs=[load_session_file],
            outputs=[
                doc_state,
                transcript_table,
                transcript_preview,
                glossary,
                speaker_names_table,
                summary_output,
                summary_state,
                llm_outputs_state,
                llm_provenance_panel,
                job_status,
            ],
        )
        run_correction.click(
            _run_correction_pass,
            inputs=[
                transcript_table,
                doc_state,
                summary_provider,
                summary_model,
                summary_endpoint,
                summary_api_key,
                glossary,
                artifact_language,
                llm_outputs_state,
            ],
            outputs=[doc_state, transcript_table, transcript_preview, summary_output, summary_state, llm_outputs_state, llm_provenance_panel, job_status],
        )
        run_meeting_summary.click(
            _summarize,
            inputs=[
                transcript_table,
                doc_state,
                summary_provider,
                summary_model,
                summary_endpoint,
                summary_api_key,
                meeting_summary_template,
                summary_prompt,
                glossary,
                artifact_language,
                llm_outputs_state,
            ],
            outputs=[summary_output, summary_state, llm_outputs_state, llm_provenance_panel, job_status],
        )
        run_action_points.click(
            _summarize,
            inputs=[
                transcript_table,
                doc_state,
                summary_provider,
                summary_model,
                summary_endpoint,
                summary_api_key,
                action_points_template,
                summary_prompt,
                glossary,
                artifact_language,
                llm_outputs_state,
            ],
            outputs=[summary_output, summary_state, llm_outputs_state, llm_provenance_panel, job_status],
        )
        run_internal_decisions.click(
            _summarize,
            inputs=[
                transcript_table,
                doc_state,
                summary_provider,
                summary_model,
                summary_endpoint,
                summary_api_key,
                internal_decision_template,
                summary_prompt,
                glossary,
                artifact_language,
                llm_outputs_state,
            ],
            outputs=[summary_output, summary_state, llm_outputs_state, llm_provenance_panel, job_status],
        )
        run_external_decisions.click(
            _summarize,
            inputs=[
                transcript_table,
                doc_state,
                summary_provider,
                summary_model,
                summary_endpoint,
                summary_api_key,
                external_decision_template,
                summary_prompt,
                glossary,
                artifact_language,
                llm_outputs_state,
            ],
            outputs=[summary_output, summary_state, llm_outputs_state, llm_provenance_panel, job_status],
        )
        run_follow_up.click(
            _follow_up_email,
            inputs=[
                transcript_table,
                doc_state,
                summary_provider,
                summary_model,
                summary_endpoint,
                summary_api_key,
                glossary,
                artifact_language,
                llm_outputs_state,
            ],
            outputs=[summary_output, summary_state, llm_outputs_state, llm_provenance_panel, job_status],
        )
        summary_provider.change(
            _apply_summary_provider_defaults,
            inputs=[summary_provider, summary_model, summary_endpoint],
            outputs=[summary_model, summary_endpoint],
        )
        start_llama_cpp.click(
            _start_llama_cpp,
            inputs=[
                summary_model,
                llama_cpp_dir,
                llama_cpp_model_source,
                llama_cpp_port,
                llama_cpp_gpu_layers,
                llama_cpp_context_size,
                llama_cpp_chat_template,
            ],
            outputs=[summary_provider, summary_model, summary_endpoint, job_status],
        )
        stop_llama_cpp.click(_stop_llama_cpp, outputs=[job_status])
        health_check.click(_summary_health, inputs=[summary_provider, summary_model, summary_endpoint, summary_api_key], outputs=[job_status])
        copy_transcript_button.click(_copy_transcript, inputs=[transcript_table, doc_state], outputs=[transcript_copy, job_status])
        copy_summary_button.click(_copy_summary, inputs=[summary_state], outputs=[summary_copy, job_status])
        export_llm_artifacts.click(
            _export_llm_artifacts,
            inputs=[transcript_table, doc_state, llm_outputs_state],
            outputs=[llm_artifact_output, job_status],
        )
        open_outputs_button.click(_open_outputs_folder, outputs=[job_status])

    return app


def _topbar_html() -> str:
    return (
        "<div class='lok-topbar'>"
        "<div class='lok-brand'><span class='lok-wave'></span><span>Transcript LOK</span></div>"
        "<div class='lok-topmeta'>"
        "<span class='lok-ready'>Local Ready</span>"
        "<span>Runtime: Local</span>"
        "<span>CPU --</span>"
        "<span>RAM --</span>"
        "<span>VRAM --</span>"
        "<span>No active job</span>"
        "</div>"
        "</div>"
    )


def _runtime_cards_html(diagnostics: RuntimeDiagnostics | None = None) -> str:
    diagnostics = diagnostics or collect_diagnostics(
        ACTIVE_POLICY.runtime.default_model,
        ACTIVE_POLICY.runtime.default_device,
        ACTIVE_POLICY.runtime.default_compute_type,
        cpu_threads=ACTIVE_POLICY.runtime.cpu_threads,
    )
    cuda_label = "CUDA usable" if diagnostics.cuda_runtime_usable else "CPU fallback"
    cuda_detail = (
        f"CT2 devices {diagnostics.cuda_device_count}; torch {_yes_no(diagnostics.torch_cuda_available)}; "
        f"cuBLAS {_yes_no(diagnostics.cublas_available)}; cuDNN {_yes_no(diagnostics.cudnn_available)}"
    )
    llama_label = "Ready" if diagnostics.llama_cpp_server and diagnostics.llama_cpp_model_source_ok else "Check"
    disabled = ACTIVE_POLICY.disabled_features()
    disabled_text = "None" if not disabled else ", ".join(disabled)
    return (
        "<div class='lok-runtime-grid'>"
        f"<div class='lok-status-card'><h3>System <span class='lok-chip'>{html.escape(str(diagnostics.effective_cpu_threads))} threads</span></h3>"
        f"<p class='lok-muted'>{html.escape(diagnostics.platform)}<br>CPU logical {html.escape(str(diagnostics.cpu_logical_count or 'unknown'))}<br>RAM {html.escape(_format_bytes(diagnostics.ram_total_bytes))}</p></div>"
        f"<div class='lok-status-card'><h3>ASR <span class='lok-chip'>{html.escape(diagnostics.expected_runtime)}</span></h3>"
        f"<p class='lok-muted'>Model {html.escape(diagnostics.requested_model)}<br>PyAV {_yes_no(diagnostics.pyav_available)}<br>Python {html.escape(diagnostics.python)}</p></div>"
        f"<div class='lok-status-card'><h3>Acceleration <span class='lok-chip'>{html.escape(cuda_label)}</span></h3>"
        f"<p class='lok-muted'>{html.escape(cuda_detail)}</p></div>"
        f"<div class='lok-status-card'><h3>Local LLM <span class='lok-chip'>{html.escape(llama_label)}</span></h3>"
        f"<p class='lok-muted'>llama.cpp {html.escape(diagnostics.llama_cpp_server or 'not found')}<br>Source {html.escape(diagnostics.llama_cpp_model_source)}</p></div>"
        f"<div class='lok-status-card'><h3>Admin Policy</h3><p class='lok-muted'>Source {html.escape(ACTIVE_POLICY.source_path or 'defaults/env')}<br>Disabled {html.escape(disabled_text)}</p></div>"
        "</div>"
    )


def _yes_no(value: bool) -> str:
    return "yes" if value else "no"


def _transcribe_file(
    audio_path: str | None,
    model_name: str,
    language_label: str,
    device: str,
    compute_type: str,
    glossary: str,
    transcription_mode: str,
    provider: str,
    llm_model: str,
    endpoint: str,
    api_key: str,
    artifact_language: str,
    diarize: bool,
    hf_token: str,
    diarization_model: str,
    min_speakers: float | None,
    max_speakers: float | None,
    chunk_enabled: bool,
    chunk_seconds: float,
    overlap_seconds: float,
) -> Iterator[tuple[dict[str, Any], pd.DataFrame, str, str]]:
    if not audio_path:
        raise gr.Error("Choose an audio file first.")

    FILE_CANCEL_EVENT.clear()
    started = datetime.now()
    started_perf = perf_counter()
    yield _empty_outputs(_job_status("Loading model", f"Loading `{model_name}`..."))
    loaded_before = set(loaded_model_keys())
    transcriber = WhisperTranscriber(_asr_config(model_name, language_label, device, compute_type))
    cache_key = (
        model_name,
        transcriber.bundle.device,
        transcriber.bundle.compute_type,
        int(transcriber.config.cpu_threads or 0),
        int(transcriber.config.num_workers or 1),
    )
    cached_note = "already loaded" if cache_key in loaded_before else "loaded"
    yield _empty_outputs(
        _job_status(
            "Transcribing",
            f"Model {cached_note} on `{transcriber.bundle.device}/{transcriber.bundle.compute_type}`.",
            elapsed=started,
        )
    )

    document = TranscriptDocument()
    iterator = (
        transcriber.transcribe_file_chunked_iter(
            audio_path,
            glossary,
            chunk_seconds=float(chunk_seconds),
            overlap_seconds=float(overlap_seconds),
            cancel_check=FILE_CANCEL_EVENT.is_set,
        )
        if chunk_enabled
        else transcriber.transcribe_file_iter(audio_path, glossary, cancel_check=FILE_CANCEL_EVENT.is_set)
    )
    last_progress_bucket = -1
    last_progress_seconds = -PROGRESS_BUCKET_SECONDS
    for update in iterator:
        if update.is_final:
            document = update.document or TranscriptDocument()
            break
        should_emit, last_progress_bucket, last_progress_seconds = _should_emit_progress(
            update.completed_seconds,
            update.total_seconds,
            last_progress_bucket,
            last_progress_seconds,
        )
        if should_emit:
            yield _empty_outputs(
                _progress_status(
                    update.completed_seconds,
                    update.total_seconds,
                    transcriber.bundle.device,
                    transcriber.bundle.compute_type,
                    started,
                )
            )

    cancelled = bool(document.metadata.get("cancelled"))
    _record_run_metadata(
        document,
        "file_transcription",
        started,
        started_perf,
        model=model_name,
        language=language_label,
        requested_device=device,
        requested_compute_type=compute_type,
        runtime=f"{document.metadata.get('asr_device')}/{document.metadata.get('asr_compute_type')}",
        cpu_threads=document.metadata.get("asr_cpu_threads"),
        num_workers=document.metadata.get("asr_num_workers"),
        beam_size=document.metadata.get("asr_beam_size"),
        chunked=bool(chunk_enabled),
        chunk_seconds=_optional_float_value(chunk_seconds),
        overlap_seconds=_optional_float_value(overlap_seconds),
        audio_duration_seconds=document.metadata.get("audio_duration_seconds"),
        segment_count=len(document.segments),
        cancelled=cancelled,
    )
    phase = "Cancelled" if cancelled else "Transcribed"
    message = (
        f"{len(document.segments)} segments with `{document.metadata.get('asr_model')}` on "
        f"`{document.metadata.get('asr_device')}/{document.metadata.get('asr_compute_type')}`.\n\n"
        "Last safe state: transcript table is current in memory; save the session before closing."
    )
    yield _document_outputs(document, _job_status(phase, message, percent=100 if not cancelled else None, elapsed=started))
    if cancelled:
        return

    if diarize:
        yield _document_outputs(document, _job_status("Diarization", "Loading diarization model...", elapsed=started))
        try:
            diarize_started = datetime.now()
            diarize_started_perf = perf_counter()
            diarizer = SpeakerDiarizer(_diarization_config(hf_token, diarization_model, device, min_speakers, max_speakers))
            yield _document_outputs(document, _job_status("Diarization", "Assigning speaker labels...", elapsed=started))
            document = diarizer.apply(audio_path, document)
            _record_run_metadata(
                document,
                "diarization",
                diarize_started,
                diarize_started_perf,
                model=diarization_model or DEFAULT_DIARIZATION_MODEL,
                requested_device=device,
                speaker_count=len(document.speaker_labels()),
                segment_count=len(document.segments),
            )
            yield _document_outputs(
                document,
                _job_status("Complete", f"Diarized `{len(document.speaker_labels())}` speakers.", percent=100, elapsed=started),
            )
        except Exception as exc:
            yield _document_outputs(document, _job_status("Diarization skipped", str(exc), elapsed=started))

    if _corrected_mode_selected(transcription_mode):
        raw_document = TranscriptDocument.from_dict(document.to_dict())
        yield _document_outputs(document, _job_status("Correction pass", "Running local transcript cleanup...", elapsed=started))
        try:
            corrected_document, correction_status = _run_file_correction(
                raw_document,
                provider,
                llm_model,
                endpoint,
                api_key,
                glossary,
                artifact_language,
            )
        except Exception as exc:
            yield _document_outputs(
                document,
                _job_status("Correction skipped", f"Raw transcript preserved. {exc}", percent=100, elapsed=started),
            )
            return
        yield _document_outputs(corrected_document, _job_status("Corrected", correction_status, percent=100, elapsed=started))


def _stream_live(
    audio: tuple[int, np.ndarray] | None,
    audio_state: dict[str, Any] | None,
    doc_state: dict[str, Any] | None,
    model_name: str,
    language_label: str,
    device: str,
    compute_type: str,
    glossary: str,
    chunk_seconds: float,
    paused: bool,
    rolling_context: bool,
    autosave_every: float | None,
) -> tuple[dict[str, Any] | None, dict[str, Any], pd.DataFrame, str, str]:
    if audio is None:
        return _live_outputs(audio_state, doc_state, _job_status("Live", "Waiting for microphone audio."))
    if paused:
        return _live_outputs(audio_state, doc_state, _job_status("Live paused", _live_status_message(audio_state, "Microphone input is currently paused.")))

    sample_rate, samples = audio
    state = _ensure_live_state(audio_state, sample_rate)
    mono = to_mono_float32(samples)
    state["full_audio"] = _append_audio(state["full_audio"], mono)
    state["buffer"] = _append_audio(state["buffer"], mono)
    state["recording_seconds"] = len(state["full_audio"]) / float(state["sample_rate"])

    buffer_seconds = len(state["buffer"]) / float(state["sample_rate"])
    if buffer_seconds < float(chunk_seconds):
        message = _live_status_message(state, f"Recording {state['recording_seconds']:.1f}s; buffer {buffer_seconds:.1f}s.")
        return _live_outputs(state, doc_state, _job_status("Live recording", message))

    document = TranscriptDocument.from_dict(doc_state)
    partial = _transcribe_live_buffer(state, document, model_name, language_label, device, compute_type, glossary, rolling_context)
    document.segments.extend(partial.segments)
    state["last_chunk_count"] = len(partial.segments)
    _record_live_commit(state, "stream", len(partial.segments), buffer_seconds)
    state["buffer"] = np.array([], dtype=np.float32)
    state["offset_seconds"] += buffer_seconds
    _sync_live_metadata(document, state)
    _auto_save_live(document, autosave_every, state)
    _sync_live_metadata(document, state)
    return _live_outputs(state, document.to_dict(), _job_status("Live committed", _live_status_message(state, f"Added {len(partial.segments)} live segments.")))


def _flush_live(
    audio_state: dict[str, Any] | None,
    doc_state: dict[str, Any] | None,
    model_name: str,
    language_label: str,
    device: str,
    compute_type: str,
    glossary: str,
    rolling_context: bool,
    autosave_every: float | None,
) -> tuple[dict[str, Any] | None, dict[str, Any], pd.DataFrame, str, str]:
    if not audio_state or len(audio_state.get("buffer", [])) == 0:
        return _live_outputs(audio_state, doc_state, _job_status("Live", "No live buffer to commit."))

    document = TranscriptDocument.from_dict(doc_state)
    buffer_seconds = len(audio_state["buffer"]) / float(audio_state["sample_rate"])
    partial = _transcribe_live_buffer(audio_state, document, model_name, language_label, device, compute_type, glossary, rolling_context)
    document.segments.extend(partial.segments)
    audio_state["last_chunk_count"] = len(partial.segments)
    _record_live_commit(audio_state, "manual_flush", len(partial.segments), buffer_seconds)
    audio_state["buffer"] = np.array([], dtype=np.float32)
    audio_state["offset_seconds"] += buffer_seconds
    _sync_live_metadata(document, audio_state)
    _auto_save_live(document, autosave_every, audio_state)
    _sync_live_metadata(document, audio_state)
    return _live_outputs(audio_state, document.to_dict(), _job_status("Live committed", _live_status_message(audio_state, f"Flushed {len(partial.segments)} segments.")))


def _diarize_live(
    audio_state: dict[str, Any] | None,
    doc_state: dict[str, Any] | None,
    hf_token: str,
    diarization_model: str,
    device: str,
    min_speakers: float | None,
    max_speakers: float | None,
) -> tuple[dict[str, Any], pd.DataFrame, str, str]:
    if not audio_state or len(audio_state.get("full_audio", [])) == 0:
        raise gr.Error("No live recording is available.")

    document = TranscriptDocument.from_dict(doc_state)
    if not document.segments:
        raise gr.Error("Commit live transcript segments first.")

    output_path = OUTPUT_DIR / f"live_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
    samples_16k = resample_audio(audio_state["full_audio"], int(audio_state["sample_rate"]), TARGET_SAMPLE_RATE)
    wav_path = write_wav(output_path, samples_16k, TARGET_SAMPLE_RATE)
    started = datetime.now()
    started_perf = perf_counter()
    diarizer = SpeakerDiarizer(_diarization_config(hf_token, diarization_model, device, min_speakers, max_speakers))
    document = diarizer.apply(wav_path, document)
    document.audio_path = wav_path
    _record_run_metadata(
        document,
        "live_diarization",
        started,
        started_perf,
        model=diarization_model or DEFAULT_DIARIZATION_MODEL,
        requested_device=device,
        speaker_count=len(document.speaker_labels()),
        segment_count=len(document.segments),
    )
    _sync_live_metadata(document, audio_state)
    return _document_outputs(document, _job_status("Live diarized", f"Speaker labels assigned to {len(document.speaker_labels())} speakers."))


def _reset_live() -> tuple[None, bool, Any, dict[str, Any], pd.DataFrame, str, str]:
    document = TranscriptDocument()
    return None, False, gr.update(value="Pause"), document.to_dict(), _to_dataframe(document), "", _job_status("Live reset", "Ready for a new live session.")


def _toggle_live_pause(paused: bool) -> tuple[bool, Any, str]:
    next_paused = not bool(paused)
    label = "Resume" if next_paused else "Pause"
    status = "Live paused" if next_paused else "Live resumed"
    message = (
        "Microphone chunks will not be transcribed while paused."
        if next_paused
        else "Literal live transcription is active again."
    )
    return next_paused, gr.update(value=label), _job_status(status, message)


def _toggle_settings_panel(opened: bool) -> tuple[bool, Any]:
    next_open = not bool(opened)
    label = "Close settings" if next_open else "Settings"
    return next_open, gr.update(value=label)


def _clear_settings_search(_selection: str | None) -> Any:
    return gr.update(value=None)


def _discard_last_live_chunk(
    audio_state: dict[str, Any] | None,
    doc_state: dict[str, Any] | None,
) -> tuple[dict[str, Any] | None, dict[str, Any], pd.DataFrame, str, str]:
    document = TranscriptDocument.from_dict(doc_state)
    count = int((audio_state or {}).get("last_chunk_count") or 0)
    if count <= 0:
        return _live_outputs(audio_state, document.to_dict(), _job_status("Live", "No committed live chunk to discard."))
    document.segments = document.segments[:-count]
    if audio_state:
        audio_state["last_chunk_count"] = 0
        _record_live_commit(audio_state, "discard", -count, 0.0)
        _sync_live_metadata(document, audio_state)
    return _live_outputs(audio_state, document.to_dict(), _job_status("Live", _live_status_message(audio_state, f"Discarded {count} segments.")))


def _apply_edits(rows: Any, doc_state: dict[str, Any] | None) -> tuple[dict[str, Any], pd.DataFrame, str, str]:
    document = _document_from_rows(rows, doc_state)
    return _document_outputs(document, _job_status("Corrections", f"Applied {len(document.segments)} edited rows."))


def _apply_names(
    rows: Any,
    doc_state: dict[str, Any] | None,
    mapping_rows: Any,
) -> tuple[dict[str, Any], pd.DataFrame, str, str]:
    document = _document_from_rows(rows, doc_state)
    mapping = _mapping_from_table(mapping_rows)
    document.rename_speakers(mapping)
    return _document_outputs(document, _job_status("Speakers", f"Applied {len(mapping)} speaker mappings."))


def _export(rows: Any, doc_state: dict[str, Any] | None) -> tuple[list[str], list[str], str]:
    document = _document_from_rows(rows, doc_state)
    if not document.segments:
        raise gr.Error("No transcript is available to export.")
    paths = document.write_exports(OUTPUT_DIR, stem=f"transcript_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    return paths, paths, _job_status("Exported", f"Wrote {len(paths)} files to `{OUTPUT_DIR}`.")


def _summarize(
    rows: Any,
    doc_state: dict[str, Any] | None,
    provider: str,
    model: str,
    endpoint: str,
    api_key: str,
    template: str,
    custom_prompt: str,
    glossary: str,
    artifact_language: str,
    llm_outputs: dict[str, Any] | None,
) -> tuple[str, str, dict[str, Any], str, str]:
    document = _document_from_rows(rows, doc_state)
    if not document.segments:
        raise gr.Error("No transcript is available to summarize.")

    config = _summary_config(provider, model, endpoint, api_key, template, custom_prompt, artifact_language)
    task = TEMPLATE_TASKS.get(config.template, "meeting_summary")
    summary = run_llm_task(document, config, task=task, glossary_text=glossary)
    provenance = llm_provenance(document, config, task, glossary)
    next_outputs = _store_llm_output(llm_outputs, task, summary, provenance)
    status = _job_status("LLM output", f"Created `{task}` with `{provider}:{config.model}`. Hash `{provenance['transcript_hash'][:12]}`.")
    return summary, summary, next_outputs, _llm_provenance_markdown(next_outputs), status


def _follow_up_email(
    rows: Any,
    doc_state: dict[str, Any] | None,
    provider: str,
    model: str,
    endpoint: str,
    api_key: str,
    glossary: str,
    artifact_language: str,
    llm_outputs: dict[str, Any] | None,
) -> tuple[str, str, dict[str, Any], str, str]:
    document = _document_from_rows(rows, doc_state)
    if not document.segments:
        raise gr.Error("No transcript is available for a follow-up email.")

    config = _summary_config(provider, model, endpoint, api_key, "Follow-up email", "", artifact_language)
    text = run_llm_task(document, config, task="follow_up_email", glossary_text=glossary)
    provenance = llm_provenance(document, config, "follow_up_email", glossary)
    next_outputs = _store_llm_output(llm_outputs, "follow_up_email", text, provenance)
    status = _job_status("Follow-up email", f"Created with `{provider}:{config.model}`. Hash `{provenance['transcript_hash'][:12]}`.")
    return text, text, next_outputs, _llm_provenance_markdown(next_outputs), status


def _run_correction_pass(
    rows: Any,
    doc_state: dict[str, Any] | None,
    provider: str,
    model: str,
    endpoint: str,
    api_key: str,
    glossary: str,
    artifact_language: str,
    llm_outputs: dict[str, Any] | None,
) -> tuple[dict[str, Any], pd.DataFrame, str, str, str, dict[str, Any], str, str]:
    document = _document_from_rows(rows, doc_state)
    if not document.segments:
        raise gr.Error("No transcript is available for correction.")

    config = _summary_config(provider, model, endpoint, api_key, "Correction pass", "", artifact_language)
    try:
        result = correct_transcript(document, config, glossary)
    except Exception as exc:
        raise gr.Error(f"Correction pass failed without applying changes: {exc}") from exc

    audit_text = _correction_audit_markdown(result.changes, result.review_flags)
    next_outputs = _store_llm_output(llm_outputs, "correction_pass", audit_text, result.provenance)
    next_outputs["correction_pass"]["changes"] = [change.__dict__ for change in result.changes]
    next_outputs["correction_pass"]["review_flags"] = [flag.__dict__ for flag in result.review_flags]
    status = _job_status(
        "Correction pass",
        f"Applied {len(result.changes)} segment corrections and flagged {len(result.review_flags)} review items with `{provider}:{config.model}`.",
    )
    return document.to_dict(), _to_dataframe(document), document.to_markdown(), audit_text, audit_text, next_outputs, _llm_provenance_markdown(next_outputs), status


def _corrected_mode_selected(transcription_mode: str) -> bool:
    return str(transcription_mode or "").strip().casefold() == "corrected"


def _run_file_correction(
    raw_document: TranscriptDocument,
    provider: str,
    model: str,
    endpoint: str,
    api_key: str,
    glossary: str,
    artifact_language: str,
) -> tuple[TranscriptDocument, str]:
    _ensure_local_correction_provider(provider, endpoint)
    corrected_document = TranscriptDocument.from_dict(raw_document.to_dict())
    config = _summary_config(provider, model, endpoint, api_key, "Correction pass", "", artifact_language)
    result = correct_transcript(corrected_document, config, glossary)
    paths = _write_raw_corrected_variants(raw_document, corrected_document)
    corrected_document.metadata["transcript_mode"] = "corrected"
    corrected_document.metadata["transcript_variants"] = paths
    corrected_document.metadata.setdefault("llm_outputs", {})["correction_pass"] = {
        "text": _correction_audit_markdown(result.changes, result.review_flags),
        "provenance": result.provenance,
        "changes": [change.__dict__ for change in result.changes],
        "review_flags": [flag.__dict__ for flag in result.review_flags],
    }
    return (
        corrected_document,
        f"Applied `{len(result.changes)}` corrections and flagged `{len(result.review_flags)}` items. "
        f"Stored raw and corrected variants in `{OUTPUT_DIR}`.",
    )


def _ensure_local_correction_provider(provider: str, endpoint: str) -> None:
    if not ACTIVE_POLICY.automatic_correction_local_only:
        return
    normalized = (provider or "").strip()
    if normalized in {"llama.cpp", "ollama"}:
        return
    if normalized == "openai-compatible" and _is_local_endpoint(endpoint or os.getenv("OPENAI_BASE_URL", "http://localhost:1234/v1")):
        return
    raise gr.Error("Automatic file correction is local-only. Use llama.cpp, Ollama, or a localhost OpenAI-compatible endpoint.")


def _is_local_endpoint(endpoint: str) -> bool:
    parsed = urlparse((endpoint or "").strip())
    host = (parsed.hostname or "").casefold()
    return host in {"localhost", "127.0.0.1", "::1"} or host.startswith("127.")


def _write_raw_corrected_variants(raw_document: TranscriptDocument, corrected_document: TranscriptDocument) -> dict[str, Any]:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    raw_doc = TranscriptDocument.from_dict(raw_document.to_dict())
    corrected_doc = TranscriptDocument.from_dict(corrected_document.to_dict())
    raw_doc.metadata["transcript_mode"] = "raw"
    corrected_doc.metadata["transcript_mode"] = "corrected"

    raw_json = OUTPUT_DIR / f"raw_transcript_{stamp}.json"
    raw_md = OUTPUT_DIR / f"raw_transcript_{stamp}.md"
    corrected_json = OUTPUT_DIR / f"corrected_transcript_{stamp}.json"
    corrected_md = OUTPUT_DIR / f"corrected_transcript_{stamp}.md"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    raw_json_text = json.dumps(raw_doc.to_dict(), ensure_ascii=False, indent=2)
    corrected_json_text = json.dumps(corrected_doc.to_dict(), ensure_ascii=False, indent=2)
    raw_json.write_text(raw_json_text, encoding="utf-8")
    raw_md.write_text(raw_doc.to_markdown(), encoding="utf-8")
    corrected_json.write_text(corrected_json_text, encoding="utf-8")
    corrected_md.write_text(corrected_doc.to_markdown(), encoding="utf-8")
    return {
        "raw": {
            "json": str(raw_json.resolve()),
            "markdown": str(raw_md.resolve()),
            "sha256": hashlib.sha256(raw_json_text.encode("utf-8")).hexdigest(),
        },
        "corrected": {
            "json": str(corrected_json.resolve()),
            "markdown": str(corrected_md.resolve()),
            "sha256": hashlib.sha256(corrected_json_text.encode("utf-8")).hexdigest(),
        },
    }


def _transcribe_live_buffer(
    state: dict[str, Any],
    document: TranscriptDocument,
    model_name: str,
    language_label: str,
    device: str,
    compute_type: str,
    glossary: str,
    rolling_context: bool,
) -> TranscriptDocument:
    transcriber = WhisperTranscriber(_asr_config(model_name, language_label, device, compute_type))
    prompt_text = glossary
    if rolling_context and document.segments:
        recent = " ".join(segment.text for segment in document.segments[-4:])
        prompt_text = f"{glossary}\nRecent context: {recent}".strip()
    partial = transcriber.transcribe_audio_array(
        state["buffer"],
        int(state["sample_rate"]),
        prompt_text,
        offset_seconds=float(state["offset_seconds"]),
    )
    for segment in partial.segments:
        segment.speaker = segment.speaker or "LIVE"
    return partial


def _preflight_audio(
    audio_path: str | None,
    model_name: str,
    device: str,
    compute_type: str,
    chunk_enabled: bool,
    chunk_seconds: float,
) -> tuple[str, str]:
    if not audio_path:
        return "", _job_status("Idle", "Choose an audio file.")
    metadata = audio_metadata(audio_path)
    diagnostics = collect_diagnostics(model_name, device, compute_type, cpu_threads=ACTIVE_POLICY.runtime.cpu_threads)
    duration = _format_optional_duration(metadata.duration_seconds)
    estimated_mode = "chunked" if chunk_enabled else "single pass"
    if metadata.duration_seconds and not chunk_enabled and metadata.duration_seconds > 1800:
        estimated_mode += " (chunking recommended)"
    markdown = (
        "### Audio Preflight\n"
        f"- Duration: `{duration}`\n"
        f"- Sample rate: `{metadata.sample_rate or 'unknown'}`\n"
        f"- Channels: `{metadata.channels or 'unknown'}`\n"
        f"- Format: `{metadata.format_name or 'unknown'}`\n"
        f"- Size: `{_format_bytes(metadata.size_bytes)}`\n"
        f"- Processing: `{estimated_mode}`"
        + (f" with `{int(chunk_seconds)}` second chunks" if chunk_enabled else "")
        + f"\n- Expected runtime: `{diagnostics.expected_runtime}`"
    )
    return markdown, _job_status("Preflight", f"Audio ready. Expected runtime `{diagnostics.expected_runtime}`.")


def _diagnostics(model_name: str, device: str, compute_type: str, diarization_model: str) -> tuple[str, str, str]:
    diagnostics = collect_diagnostics(model_name, device, compute_type, diarization_model, cpu_threads=ACTIVE_POLICY.runtime.cpu_threads)
    return (
        _runtime_cards_html(diagnostics),
        diagnostics_markdown(diagnostics),
        _job_status("Diagnostics", f"Expected runtime `{diagnostics.expected_runtime}`."),
    )


def _readiness(
    model_name: str,
    device: str,
    compute_type: str,
    diarization_model: str,
    provider: str,
    summary_model: str,
    endpoint: str,
    llama_cpp_dir: str,
    llama_cpp_model_source: str,
) -> tuple[str, str, str]:
    readiness = collect_readiness(
        model_name,
        device,
        compute_type,
        diarization_model,
        provider,
        summary_model,
        endpoint,
        llama_cpp_dir,
        llama_cpp_model_source,
        cpu_threads=ACTIVE_POLICY.runtime.cpu_threads,
        disabled_features=ACTIVE_POLICY.disabled_features(),
    )
    attention_count = sum(1 for item in readiness.items if not item.ok)
    return (
        _runtime_cards_html(readiness.diagnostics),
        readiness_markdown(readiness),
        _job_status("Readiness", f"{attention_count} checks need attention. Recommended `{readiness.recommended_profile}`."),
    )


def _summary_health(provider: str, model: str, endpoint: str, api_key: str) -> str:
    ok, message = provider_health(_summary_config(provider, model, endpoint, api_key, "Meeting summary", ""))
    return _job_status("Summary health" if ok else "Summary unavailable", message)


def _save_session(
    rows: Any,
    doc_state: dict[str, Any] | None,
    model_name: str,
    language_label: str,
    device: str,
    compute_type: str,
    glossary: str,
    speaker_rows: Any,
    summary: str,
    exports: list[str] | None,
    llm_outputs: dict[str, Any] | None,
) -> tuple[str, str]:
    document = _document_from_rows(rows, doc_state)
    if not document.segments:
        raise gr.Error("No transcript is available to save.")
    path = save_session(
        OUTPUT_DIR,
        document,
        settings={
            "model": model_name,
            "language": language_label,
            "device": device,
            "compute_type": compute_type,
        },
        glossary=glossary,
        speaker_mapping=_mapping_from_table(speaker_rows),
        summary=summary or "",
        exports=exports or [],
        llm_outputs=llm_outputs or {},
    )
    return path, _job_status("Session saved", f"`{path}`")


def _load_session(path: str | None) -> tuple[dict[str, Any], pd.DataFrame, str, str, pd.DataFrame, str, str, dict[str, Any], str, str]:
    if not path:
        raise gr.Error("Choose a session JSON file first.")
    payload = load_session(path)
    document = payload["document"]
    glossary = str(payload.get("glossary") or "")
    mapping = payload.get("speaker_mapping") or {}
    summary = str(payload.get("summary") or "")
    llm_outputs = payload.get("llm_outputs") or document.metadata.get("llm_outputs") or {}
    return (
        document.to_dict(),
        _to_dataframe(document),
        document.to_markdown(),
        glossary,
        _mapping_to_dataframe(mapping),
        summary,
        summary,
        llm_outputs,
        _llm_provenance_markdown(llm_outputs),
        _job_status("Session loaded", f"Loaded {len(document.segments)} segments."),
    )


def _filter_transcript(rows: Any, query: str, speaker: str, start: float | None, end: float | None) -> tuple[pd.DataFrame, str]:
    document = TranscriptDocument.from_rows(rows)
    query_fold = (query or "").casefold()
    speaker_fold = (speaker or "").casefold()
    filtered = []
    for segment in document.segments:
        if query_fold and query_fold not in segment.text.casefold():
            continue
        if speaker_fold and speaker_fold not in (segment.speaker or "").casefold():
            continue
        if start is not None and segment.end < float(start):
            continue
        if end is not None and segment.start > float(end):
            continue
        filtered.append(segment)
    out = TranscriptDocument(segments=filtered)
    return _to_dataframe(out), _job_status("Filter", f"Showing {len(filtered)} rows.")


def _clear_filter(doc_state: dict[str, Any] | None) -> tuple[pd.DataFrame, str]:
    document = TranscriptDocument.from_dict(doc_state)
    return _to_dataframe(document), _job_status("Filter cleared", f"Showing {len(document.segments)} rows.")


def _merge_next(rows: Any, doc_state: dict[str, Any] | None, row_number: float | None) -> tuple[dict[str, Any], pd.DataFrame, str, str]:
    document = _document_from_rows(rows, doc_state)
    index = max(0, int(row_number or 1) - 1)
    if index >= len(document.segments) - 1:
        raise gr.Error("Choose a row that has a following row.")
    current = document.segments[index]
    nxt = document.segments.pop(index + 1)
    current.end = max(current.end, nxt.end)
    current.text = f"{current.text.rstrip()} {nxt.text.lstrip()}".strip()
    current.words.extend(nxt.words)
    current.confidence = _mean_optional([current.confidence, nxt.confidence])
    return _document_outputs(document, _job_status("Merged", f"Merged row {index + 1} with row {index + 2}."))


def _split_row(rows: Any, doc_state: dict[str, Any] | None, row_number: float | None, split_time: float | None) -> tuple[dict[str, Any], pd.DataFrame, str, str]:
    document = _document_from_rows(rows, doc_state)
    index = max(0, int(row_number or 1) - 1)
    if index >= len(document.segments):
        raise gr.Error("Choose an existing row.")
    segment = document.segments[index]
    pivot = float(split_time) if split_time is not None else (segment.start + segment.end) / 2
    if pivot <= segment.start or pivot >= segment.end:
        raise gr.Error("Split time must be inside the segment.")
    words = segment.text.split()
    midpoint = max(1, len(words) // 2)
    left_text = " ".join(words[:midpoint]) or segment.text
    right_text = " ".join(words[midpoint:]) or segment.text
    segment.end = pivot
    segment.text = left_text
    document.segments.insert(
        index + 1,
        type(segment)(start=pivot, end=max(pivot, segment.end), speaker=segment.speaker, text=right_text, confidence=segment.confidence),
    )
    document.segments[index + 1].end = max(pivot, _to_end_from_rows(rows, index))
    return _document_outputs(document, _job_status("Split", f"Split row {index + 1}."))


def _speaker_table(rows: Any, doc_state: dict[str, Any] | None) -> tuple[pd.DataFrame, str]:
    document = _document_from_rows(rows, doc_state)
    df = pd.DataFrame([[speaker, ""] for speaker in document.speaker_labels()], columns=["speaker", "name"])
    return df, _job_status("Speakers", f"Found {len(df.index)} speaker labels.")


def _copy_transcript(rows: Any, doc_state: dict[str, Any] | None) -> tuple[str, str]:
    document = _document_from_rows(rows, doc_state)
    return document.plain_text(), _job_status("Transcript text", "Plain text ready to copy.")


def _copy_summary(summary: str) -> tuple[str, str]:
    return summary or "", _job_status("Summary text", "Summary ready to copy.")


def _store_llm_output(
    llm_outputs: dict[str, Any] | None,
    task: str,
    text: str,
    provenance: dict[str, str],
) -> dict[str, Any]:
    next_outputs = dict(llm_outputs or {})
    next_outputs[task] = {"text": text, "provenance": provenance}
    return next_outputs


def _llm_provenance_markdown(llm_outputs: dict[str, Any] | None) -> str:
    if not llm_outputs:
        return "No LLM outputs yet."
    lines = ["### LLM Provenance", "", "| Task | Provider | Model | Language | Created | Transcript |", "|---|---|---|---|---|---|"]
    for task, payload in sorted((llm_outputs or {}).items()):
        provenance = (payload or {}).get("provenance") or payload or {}
        lines.append(
            "| "
            f"{_escape_table(task)} | "
            f"{_escape_table(provenance.get('provider', ''))} | "
            f"{_escape_table(provenance.get('model', ''))} | "
            f"{_escape_table(provenance.get('artifact_language_name', provenance.get('artifact_language', '')))} | "
            f"{_escape_table(provenance.get('created_at', ''))} | "
            f"`{str(provenance.get('transcript_hash', ''))[:12]}` |"
        )
    return "\n".join(lines)


def _correction_audit_markdown(changes: list[Any], review_flags: list[Any] | None = None) -> str:
    review_flags = review_flags or []
    if not changes and not review_flags:
        return "No transcript corrections or review flags were applied."

    lines: list[str] = []
    if changes:
        lines.extend(["### Applied Corrections", "", "| Segment | Speaker | Original | Corrected | Reason |", "|---:|---|---|---|---|"])
    for change in changes:
        lines.append(
            "| "
            f"{int(change.segment_index) + 1} | "
            f"{_escape_table(change.speaker)} | "
            f"{_escape_table(change.original_text)} | "
            f"{_escape_table(change.corrected_text)} | "
            f"{_escape_table(change.reason or 'ASR/typo correction')} |"
        )
    if review_flags:
        if lines:
            lines.append("")
        lines.extend(["### Review Flags", "", "| Segment | Speaker | Text | Reason |", "|---:|---|---|---|"])
        for flag in review_flags:
            lines.append(
                "| "
                f"{int(flag.segment_index) + 1} | "
                f"{_escape_table(flag.speaker)} | "
                f"{_escape_table(flag.text)} | "
                f"{_escape_table(flag.reason or 'Needs human review')} |"
            )
    return "\n".join(lines)


def _escape_table(value: str) -> str:
    return str(value).replace("\n", " ").replace("|", "\\|")


def _export_llm_artifacts(
    rows: Any,
    doc_state: dict[str, Any] | None,
    llm_outputs: dict[str, Any] | None,
) -> tuple[list[str], str]:
    document = _document_from_rows(rows, doc_state)
    outputs = dict(llm_outputs or {})
    if not outputs and not document.segments:
        raise gr.Error("No LLM outputs or transcript are available to export.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    paths: list[Path] = []

    bundle_md = OUTPUT_DIR / f"llm_artifacts_{stamp}.md"
    bundle_json = OUTPUT_DIR / f"llm_artifacts_{stamp}.json"
    bundle_md.write_text(_llm_bundle_markdown(outputs), encoding="utf-8")
    bundle_json.write_text(json.dumps({"llm_outputs": outputs, "document": document.to_dict()}, ensure_ascii=False, indent=2), encoding="utf-8")
    paths.extend([bundle_md, bundle_json])

    if document.segments:
        corrected = OUTPUT_DIR / f"corrected_transcript_{stamp}.md"
        corrected.write_text(document.to_markdown(), encoding="utf-8")
        paths.append(corrected)

    for task, payload in sorted(outputs.items()):
        text = str((payload or {}).get("text") or "")
        if not text.strip():
            continue
        task_path = OUTPUT_DIR / f"{_safe_stem(task)}_{stamp}.md"
        task_path.write_text(text, encoding="utf-8")
        paths.append(task_path)

    resolved = [str(path.resolve()) for path in paths]
    return resolved, _job_status("LLM artifacts", f"Wrote {len(resolved)} files to `{OUTPUT_DIR}`.")


def _llm_bundle_markdown(outputs: dict[str, Any]) -> str:
    if not outputs:
        return "# LLM Artifacts\n\nNo LLM outputs were available."
    lines = ["# LLM Artifacts", ""]
    for task, payload in sorted(outputs.items()):
        lines.extend([f"## {task}", ""])
        text = str((payload or {}).get("text") or "")
        if text.strip():
            lines.extend([text, ""])
        changes = (payload or {}).get("changes") or []
        if changes:
            lines.extend(["### Correction Audit", ""])
            for change in changes:
                lines.append(
                    f"- Segment {int(change.get('segment_index', 0)) + 1}: "
                    f"{change.get('original_text', '')} -> {change.get('corrected_text', '')}"
                )
            lines.append("")
    return "\n".join(lines).strip() + "\n"


def _safe_stem(value: str) -> str:
    clean = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value.lower())
    return clean.strip("_") or "llm_output"


def _open_outputs_folder() -> str:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    try:
        os.startfile(str(OUTPUT_DIR))
        return _job_status("Outputs", f"Opened `{OUTPUT_DIR}`.")
    except Exception as exc:
        return _job_status("Outputs", f"Could not open folder: {exc}")


def _load_glossary(path: str | None) -> tuple[str, str]:
    if not path:
        raise gr.Error("Choose a glossary file first.")
    text = Path(path).read_text(encoding="utf-8-sig")
    return text, _job_status("Glossary", f"Loaded `{Path(path).name}`.")


def _export_glossary(text: str) -> tuple[str, str]:
    target = OUTPUT_DIR / f"glossary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    target.write_text(text or "", encoding="utf-8")
    return str(target.resolve()), _job_status("Glossary", f"Exported `{target.name}`.")


def _cancel_file_job() -> str:
    FILE_CANCEL_EVENT.set()
    return _job_status("Cancelling", "Cancellation requested. The current segment or chunk may finish first.")


def _apply_model_preset(preset: str) -> tuple[Any, Any, Any]:
    model, device, compute = MODEL_PRESETS.get(preset, MODEL_PRESETS["Balanced"])
    return gr.update(value=model), gr.update(value=device), gr.update(value=compute)


def _apply_workflow_profile(profile: str) -> tuple[Any, Any, Any, Any, Any, Any, str]:
    selected = WORKFLOW_PROFILES.get(profile, WORKFLOW_PROFILES["Balanced file"])
    model, device, compute = selected["asr"]
    provider, llm_model, endpoint = selected["llm"]
    return (
        gr.update(value=model),
        gr.update(value=device),
        gr.update(value=compute),
        gr.update(value=provider),
        gr.update(value=llm_model),
        gr.update(value=endpoint),
        _job_status("Profile applied", f"`{profile}`: {selected['description']}"),
    )


def _apply_summary_provider_defaults(provider: str, model: str, endpoint: str) -> tuple[Any, Any]:
    if provider == "llama.cpp":
        model_value = (model or "").strip()
        if not model_value or model_value == "llama3.1":
            model_value = os.getenv("LLAMA_CPP_MODEL", DEFAULT_LLAMA_CPP_MODEL)
        endpoint_value = (endpoint or "").strip() or os.getenv("LLAMA_CPP_BASE_URL", default_llama_cpp_endpoint())
        return gr.update(value=model_value), gr.update(value=endpoint_value)

    if provider == "openai-compatible" and not (endpoint or "").strip():
        return gr.update(), gr.update(value=os.getenv("OPENAI_BASE_URL", "http://localhost:1234/v1"))

    return gr.update(), gr.update()


def _start_llama_cpp(
    model: str,
    llama_cpp_dir: str,
    model_source: str,
    port: float | None,
    gpu_layers: str,
    context_size: float | None,
    chat_template: str,
) -> tuple[Any, Any, Any, str]:
    if not (model_source or "").strip():
        raise gr.Error("Set a GGUF model path or Hugging Face GGUF repo before starting llama.cpp.")

    alias = _llama_cpp_model_alias(model, model_source)
    config = LlamaCppServerConfig(
        llama_cpp_dir=(llama_cpp_dir or "").strip() or None,
        model_source=model_source.strip(),
        model_alias=alias,
        port=_positive_int(port, DEFAULT_LLAMA_CPP_PORT),
        gpu_layers=(gpu_layers or DEFAULT_LLAMA_CPP_GPU_LAYERS).strip(),
        context_size=_positive_int(context_size, DEFAULT_LLAMA_CPP_CONTEXT_SIZE),
        chat_template=(chat_template or DEFAULT_LLAMA_CPP_CHAT_TEMPLATE).strip(),
    )
    try:
        message, log_path = start_llama_cpp_server(config, OUTPUT_DIR)
    except Exception as exc:
        raise gr.Error(str(exc)) from exc

    endpoint = default_llama_cpp_endpoint(config.port, config.host)
    status = _job_status("llama.cpp", f"{message} Log `{log_path}`.")
    return gr.update(value="llama.cpp"), gr.update(value=alias), gr.update(value=endpoint), status


def _stop_llama_cpp() -> str:
    return _job_status("llama.cpp", stop_llama_cpp_server())


def _llama_cpp_model_alias(model: str, model_source: str) -> str:
    current = (model or "").strip()
    if current and current != "llama3.1":
        return current

    configured = os.getenv("LLAMA_CPP_MODEL", "").strip()
    if configured:
        return configured

    source = (model_source or "").strip().replace("\\", "/").rstrip("/")
    source_name = source.split("/")[-1].split(":")[0]
    if source_name:
        return source_name.removesuffix(".gguf").removesuffix("-GGUF")
    return DEFAULT_LLAMA_CPP_MODEL


def _asr_config(model_name: str, language_label: str, device: str, compute_type: str) -> ASRConfig:
    runtime = ACTIVE_POLICY.runtime
    return ASRConfig(
        model_size=model_name,
        language=LANGUAGES.get(language_label),
        device=device,
        compute_type=compute_type,
        beam_size=runtime.beam_size,
        cpu_threads=runtime.cpu_threads,
        num_workers=runtime.num_workers,
    )


def _diarization_config(hf_token: str, diarization_model: str, device: str, min_speakers: float | None, max_speakers: float | None) -> DiarizationConfig:
    token = (hf_token or os.getenv("HUGGINGFACE_TOKEN") or os.getenv("HF_TOKEN") or "").strip() or None
    return DiarizationConfig(
        model_name_or_path=(diarization_model or DEFAULT_DIARIZATION_MODEL).strip(),
        hf_token=token,
        device=device,
        min_speakers=_optional_int(min_speakers),
        max_speakers=_optional_int(max_speakers),
    )


def _summary_config(
    provider: str,
    model: str,
    endpoint: str,
    api_key: str,
    template: str,
    custom_prompt: str,
    artifact_language: str = "Dutch",
) -> SummaryConfig:
    if provider == "openai" and not api_key:
        api_key = os.getenv("OPENAI_API_KEY", "")
    if provider == "openai" and not model:
        model = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
    if provider == "llama.cpp":
        if not endpoint:
            endpoint = os.getenv("LLAMA_CPP_BASE_URL", default_llama_cpp_endpoint())
        if not model or model == "llama3.1":
            model = os.getenv("LLAMA_CPP_MODEL", DEFAULT_LLAMA_CPP_MODEL)
    if provider == "openai-compatible" and not endpoint:
        endpoint = os.getenv("OPENAI_BASE_URL", "http://localhost:1234/v1")
    return SummaryConfig(
        provider=provider,
        model=model.strip() or "llama3.1",
        endpoint=endpoint.strip() or None,
        api_key=api_key.strip() or None,
        template=template or "Meeting summary",
        custom_prompt=custom_prompt.strip() or None,
        artifact_language=ARTIFACT_LANGUAGES.get(artifact_language, "nl"),
    )


def _positive_int(value: float | None, default: int) -> int:
    try:
        parsed = int(value) if value is not None else default
    except (TypeError, ValueError):
        return default
    return max(1, parsed)


def _document_from_rows(rows: Any, doc_state: dict[str, Any] | None) -> TranscriptDocument:
    existing = TranscriptDocument.from_dict(doc_state)
    document = TranscriptDocument.from_rows(rows, language=existing.language, audio_path=existing.audio_path, metadata=existing.metadata)
    document.created_at = existing.created_at
    return document


def _document_outputs(document: TranscriptDocument, status: str) -> tuple[dict[str, Any], pd.DataFrame, str, str]:
    return document.to_dict(), _to_dataframe(document), document.to_markdown(), status


def _empty_outputs(status: str) -> tuple[dict[str, Any], pd.DataFrame, str, str]:
    document = TranscriptDocument()
    return document.to_dict(), _to_dataframe(document), "", status


def _live_outputs(audio_state: dict[str, Any] | None, doc_state: dict[str, Any] | None, status: str) -> tuple[dict[str, Any] | None, dict[str, Any], pd.DataFrame, str, str]:
    document = TranscriptDocument.from_dict(doc_state)
    return audio_state, document.to_dict(), _to_dataframe(document), document.to_markdown(), status


def _to_dataframe(document: TranscriptDocument) -> pd.DataFrame:
    return pd.DataFrame(document.to_rows(), columns=TRANSCRIPT_COLUMNS)


def _ensure_live_state(state: dict[str, Any] | None, sample_rate: int) -> dict[str, Any]:
    if state is None:
        return {
            "sample_rate": int(sample_rate),
            "buffer": np.array([], dtype=np.float32),
            "full_audio": np.array([], dtype=np.float32),
            "offset_seconds": 0.0,
            "last_chunk_count": 0,
            "committed_segments": 0,
            "recording_seconds": 0.0,
            "last_autosave_segments": 0,
            "last_safe_session_path": "",
            "last_safe_saved_at": "",
            "commit_history": [],
        }
    if int(state["sample_rate"]) != int(sample_rate):
        state["buffer"] = resample_audio(state["buffer"], int(state["sample_rate"]), int(sample_rate))
        state["full_audio"] = resample_audio(state["full_audio"], int(state["sample_rate"]), int(sample_rate))
        state["sample_rate"] = int(sample_rate)
    state.setdefault("last_safe_session_path", "")
    state.setdefault("last_safe_saved_at", "")
    state.setdefault("commit_history", [])
    return state


def _append_audio(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    if len(left) == 0:
        return right.astype(np.float32, copy=False)
    return np.concatenate([left, right.astype(np.float32, copy=False)])


def _auto_save_live(document: TranscriptDocument, every: float | None, state: dict[str, Any]) -> None:
    interval = _optional_int(every)
    if not interval:
        return
    committed = int(state.get("committed_segments", 0))
    last = int(state.get("last_autosave_segments", 0))
    if committed >= interval and committed - last >= interval:
        path = save_session(OUTPUT_DIR, document, name="live_autosave")
        state["last_autosave_segments"] = committed
        state["last_safe_session_path"] = path
        state["last_safe_saved_at"] = datetime.now().isoformat(timespec="seconds")


def _record_live_commit(state: dict[str, Any], source: str, segment_count: int, buffer_seconds: float) -> None:
    committed = max(0, int(state.get("committed_segments", 0)) + int(segment_count))
    state["committed_segments"] = committed
    history = list(state.get("commit_history") or [])
    history.append(
        {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "source": source,
            "segment_count": int(segment_count),
            "buffer_seconds": round(float(buffer_seconds or 0.0), 3),
            "recording_seconds": round(float(state.get("recording_seconds") or 0.0), 3),
            "total_committed_segments": committed,
        }
    )
    state["commit_history"] = history[-50:]


def _sync_live_metadata(document: TranscriptDocument, state: dict[str, Any]) -> None:
    document.metadata["live_status"] = {
        "recording_seconds": round(float(state.get("recording_seconds") or 0.0), 3),
        "committed_segments": int(state.get("committed_segments", 0)),
        "last_safe_session_path": state.get("last_safe_session_path", ""),
        "last_safe_saved_at": state.get("last_safe_saved_at", ""),
        "commit_history": list(state.get("commit_history") or []),
    }


def _live_status_message(state: dict[str, Any] | None, lead: str) -> str:
    state = state or {}
    committed = int(state.get("committed_segments", 0) or 0)
    recording = float(state.get("recording_seconds", 0.0) or 0.0)
    safe_path = str(state.get("last_safe_session_path") or "")
    safe_at = str(state.get("last_safe_saved_at") or "")
    safe = f"`{safe_path}` at `{safe_at}`" if safe_path else "not autosaved yet"
    return f"{lead}\n\nCommitted segments: `{committed}`. Recording: `{recording:.1f}s`. Last safe save: {safe}."


def _record_run_metadata(
    document: TranscriptDocument,
    kind: str,
    started_at: datetime,
    started_perf: float,
    **fields: Any,
) -> None:
    entry = {
        "kind": kind,
        "started_at": started_at.isoformat(timespec="seconds"),
        "finished_at": datetime.now().isoformat(timespec="seconds"),
        "wall_seconds": round(perf_counter() - started_perf, 3),
    }
    entry.update({key: value for key, value in fields.items() if value not in (None, "")})
    document.metadata.setdefault("run_history", []).append(entry)


def _mapping_from_table(rows: Any) -> dict[str, str]:
    if rows is None:
        return {}
    if hasattr(rows, "values"):
        data = rows.values.tolist()
    elif isinstance(rows, dict):
        data = rows.get("data") or []
    else:
        data = rows
    mapping: dict[str, str] = {}
    for row in data:
        if len(row) >= 2 and str(row[0] or "").strip() and str(row[1] or "").strip():
            mapping[str(row[0]).strip()] = str(row[1]).strip()
    return mapping


def _mapping_to_dataframe(mapping: Any) -> pd.DataFrame:
    if isinstance(mapping, dict):
        rows = [[key, value] for key, value in mapping.items()]
    else:
        rows = mapping or []
    return pd.DataFrame(rows, columns=["speaker", "name"])


def _should_emit_progress(completed_seconds: float, total_seconds: float | None, last_progress_bucket: int, last_progress_seconds: float) -> tuple[bool, int, float]:
    if total_seconds and total_seconds > 0:
        percent = min(99, int((completed_seconds / total_seconds) * 100))
        bucket = percent // PROGRESS_BUCKET_PERCENT
        if bucket > last_progress_bucket:
            return True, bucket, completed_seconds
        return False, last_progress_bucket, last_progress_seconds
    if completed_seconds - last_progress_seconds >= PROGRESS_BUCKET_SECONDS:
        return True, last_progress_bucket, completed_seconds
    return False, last_progress_bucket, last_progress_seconds


def _progress_status(completed_seconds: float, total_seconds: float | None, device: str, compute_type: str, started: datetime | None = None) -> str:
    if total_seconds and total_seconds > 0:
        percent = min(99, int((completed_seconds / total_seconds) * 100))
        message = f"{format_timestamp(completed_seconds, '.')} / {format_timestamp(total_seconds, '.')} on `{device}/{compute_type}`"
        return _job_status("Transcribing", message, percent=percent, elapsed=started)
    return _job_status("Transcribing", f"Processed {format_timestamp(completed_seconds, '.')} on `{device}/{compute_type}`", elapsed=started)


def _job_status(phase: str, message: str, *, percent: int | None = None, elapsed: datetime | None = None) -> str:
    progress = f"\n\n`{_progress_bar(percent)}` **{percent}%**" if percent is not None else ""
    elapsed_text = f"\n\nElapsed: `{(datetime.now() - elapsed).total_seconds():.1f}s`" if elapsed else ""
    return f"### {phase}\n{message}{progress}{elapsed_text}"


def _progress_bar(percent: int) -> str:
    bounded = max(0, min(100, int(percent)))
    filled = round((bounded / 100) * PROGRESS_BAR_WIDTH)
    return f"[{'#' * filled}{'-' * (PROGRESS_BAR_WIDTH - filled)}]"


def _optional_int(value: float | None) -> int | None:
    if value is None:
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _optional_float_value(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _mean_optional(values: list[float | None]) -> float | None:
    clean = [float(value) for value in values if value is not None]
    return sum(clean) / len(clean) if clean else None


def _to_end_from_rows(rows: Any, index: int) -> float:
    normalized = TranscriptDocument.from_rows(rows).segments
    return normalized[index].end if index < len(normalized) else 0.0


def _format_optional_duration(value: float | None) -> str:
    return format_timestamp(value, ".") if value is not None else "unknown"


def _format_bytes(value: int | None) -> str:
    if value is None:
        return "unknown"
    size = float(value)
    for unit in ["B", "KB", "MB", "GB"]:
        if size < 1024 or unit == "GB":
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} GB"


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


def main() -> None:
    app = build_app()
    host = os.getenv("TRANSCRIPT_LOK_HOST") or os.getenv("TRANSCRIPT_LOK_SERVER_NAME") or "127.0.0.1"
    port = _env_int("TRANSCRIPT_LOK_PORT", 7860)
    root_path = os.getenv("TRANSCRIPT_LOK_ROOT_PATH") or None
    max_threads = _env_int("TRANSCRIPT_LOK_QUEUE_MAX_THREADS", ACTIVE_POLICY.runtime.queue_max_threads)
    app.queue().launch(
        server_name=host,
        server_port=port,
        root_path=root_path,
        max_threads=max_threads,
        theme=gr.themes.Soft(),
    )


if __name__ == "__main__":
    main()
