# Research Notes

This project uses a local-first stack:

- `faster-whisper` for speech recognition. It runs Whisper through CTranslate2, supports GPU/CPU inference, word timestamps, VAD filtering, multilingual Whisper models, and prompting/hotword support.
- `pyannote.audio` Community-1 for speaker diarization. It can run locally after the model is downloaded and provides speaker turns that are assigned back onto Whisper segments.
- Gradio for the local interface. It supports file upload and microphone streaming in one lightweight Python app.
- Ollama, `llama.cpp`/Gemma, OpenAI, or OpenAI-compatible chat endpoints for correction passes and meeting artifacts. The transcript remains local unless the user selects a cloud endpoint and provides credentials.

Important sources checked during setup:

- https://github.com/SYSTRAN/faster-whisper
- https://github.com/m-bain/whisperX
- https://github.com/pyannote/pyannote-audio
- https://github.com/snakers4/silero-vad
- https://www.gradio.app/guides/real-time-speech-recognition/
- https://platform.openai.com/docs/api-reference/chat

## Notes

`pyannote/speaker-diarization-community-1` requires accepting the Hugging Face model terms and providing a Hugging Face token for the first download. After download, inference runs on the local machine. A local model directory can be used instead of the Hugging Face model id.
