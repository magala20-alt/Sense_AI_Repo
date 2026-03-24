# Sense AI Repo

Sense AI is a desktop, real-time American Sign Language (ASL) translation system that supports two communication directions:

1. Speaker to Signer: text or speech to ASL gloss and avatar signing.
2. Signer to Speaker: webcam frames to translated text using hand and facial model signals.

The project combines a Tkinter frontend, a Python backend service layer, WebSocket streaming, and MindSpore model integration.

## What This System Does

Sense AI is designed to bridge conversations between Deaf and hearing users in a shared session.

- Speaker workflow:
  - Accepts typed or spoken input.
  - Converts English text into ASL-style gloss.
  - Sends output toward SiGML/CWASA avatar signing.
- Signer workflow:
  - Captures webcam frames in real time.
  - Extracts hand pose and facial expression signals.
  - Streams model outputs through backend translation endpoints.
- Shared backend:
  - Handles auth, translation requests, and persistence.
  - Exposes WebSocket message contracts for real-time interaction.

## Core Features

- Dual UI modes: Signer screen and Speaker screen.
- Real-time frame processing pipeline (target around 30 FPS on suitable hardware).
- Model manager abstraction for:
  - Text to ASL gloss translation.
  - Hand keypoint detection.
  - Facial expression detection.
- WebSocket server for low-latency translation requests.
- Conversation and user data persistence through backend database services.
- Setup and integration verification scripts.

## High-Level Architecture

```text
Tkinter Frontend (sense_ai/)
	|- Speaker Screen (text/speech -> gloss -> avatar)
	|- Signer Screen  (camera -> hand/facial outputs -> translation)
	|
	v
Services Layer
	|- model_loader.py (ASL, hand, facial models)
	|- camera_processor.py (frame pipeline + buffering)
	|
	v
Backend WebSocket Server (sense_ai/Backend/websocket_server.py)
	|- auth
	|- translate_text
	|- translate_frame
	|- translate_multimodal
	|
	v
Database + ASL service logic
```

## Repository Layout

Top-level folders and files you will use most:

- `sense_ai/`: desktop app, screens, services, config, backend package.
- `sense_ai/Backend/`: WebSocket server, data layer, DB manager, service logic.
- `gloss_mindspore_models/`: ASL translator  vocab.
- `facial_recognition_model/`: Facial recognition model.
- `hand_recognition_model/`: hand pose model asset.
- `MODEL_INTEGRATION_GUIDE.md`: detailed model integration reference.
- `IMPLEMENTATION_SUMMARY.md`: integration status and architecture summary.
- `setup_verify.py`: checks files/dependencies/camera/MindSpore presence.
- `test_integration.py`: component-level integration tests.

## Prerequisites

- Python 3.8+
- Webcam (for signer workflow)
- OS support for your MindSpore build
- Optional but recommended:
  - CWASA/SiGML player if using avatar output
  - Microphone if using speech-to-text input

## Required Model Files

Ensure these files exist before full integration testing:

```text
gloss_mindspore_models/final_models (.mindir)/asl_translator.mindir
gloss_mindspore_models/final_models/vocab.json
gloss_mindspore_models/facial_recognition_model/FacialExpressionModel.mindir
hand_recognition_model/hand_recognition_model.mindir
```

## Installation

From repository root:

```powershell
cd sense_ai
pip install -r requirements.txt
```

If MindSpore is not already available in your environment, install the build appropriate for your OS and Python version, then verify:

```powershell
python -c "import mindspore; print(mindspore.__version__)"
```

## Quick Start

Start backend server (terminal 1):

```powershell
cd sense_ai\Backend
python websocket_server.py
```

Start desktop app (terminal 2):

```powershell
cd sense_ai
python main.py
```

Default runtime endpoints and ports:

- Backend WebSocket: `ws://localhost:8765`
- SiGML/CWASA host: `127.0.0.1`
- SiGML/CWASA port: `4569`

These defaults are configured in `sense_ai/config.py`.

## Validation and Health Checks

Run setup verification from repository root:

```powershell
python setup_verify.py
```

Run integration tests:

```powershell
python test_integration.py
```

Current tests cover:

- Model manager initialization and sample translation.
- Camera processor startup and frame loop.
- WebSocket server initialization.
- Speaker pipeline smoke path.
- Signer pipeline smoke path.

## Runtime Workflows

### Speaker Pipeline

1. User enters text (or uses microphone input).
2. Text goes to model manager translation.
3. Gloss output with confidence is returned.
4. Gloss is sent to avatar signer service.
5. Event is stored in conversation history/backend.

### Signer Pipeline

1. Camera frames are captured continuously.
2. Hand and facial detections run per frame.
3. Frame outputs are buffered and streamed to backend.
4. Backend returns translated text + confidence.
5. UI updates live translation and history.

## WebSocket Message Types

The backend server accepts these message types:

- `ping`
- `auth`
- `translate_text`
- `translate_frame`
- `translate_multimodal`

Example request (`translate_text`):

```json
{
  "type": "translate_text",
  "text": "Hello, how are you?",
  "user_id": 1,
  "session_token": "optional"
}
```

Example response (`translation_result`):

```json
{
  "type": "translation_result",
  "ok": true,
  "translation": {
    "translated_text": "HELLO HOW YOU",
    "confidence_score": 0.85,
    "model_used": "asl_translator"
  }
}
```

## Configuration

Update runtime defaults in `sense_ai/config.py`:

- `BACKEND_WS_URL`
- `SIGML_HOST`
- `SIGML_PORT`
- `CAMERA_INDEX`

The setup script can also generate a `.env` template under `sense_ai/.env`.

## Important Notes About Model Runtime

- The code supports MindSpore model loading paths for `.mindir` and `.ckpt` assets.
- In environments where full model runtime is unavailable, parts of the pipeline can fall back to deterministic placeholder behavior.
- For production-grade inference throughput and accuracy, ensure compatible MindSpore runtime/serving configuration and validated model artifacts.

## Troubleshooting

- Backend connection fails:
  - Confirm server is running on port `8765`.
  - Check `BACKEND_WS_URL` in config.
- Camera unavailable:
  - Try different camera indices (`0`, `1`, `2`).
- Avatar not responding:
  - Confirm CWASA/SiGML endpoint is reachable at configured host/port.
- MindSpore import errors:
  - Install OS/Python-compatible MindSpore build and re-run verification.

## Additional Documentation

- `MODEL_INTEGRATION_GUIDE.md`: full model integration guide.
- `IMPLEMENTATION_SUMMARY.md`: implementation status, API flow, and performance notes.
- `sense_ai/README.md`: detailed frontend-specific structure and behaviors.

## Project Status

This repository includes integrated UI, backend services, model connectors, and test/verification scripts. It is suitable for development, demos, and iterative hardening toward production.
