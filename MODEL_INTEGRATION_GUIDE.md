# Sense AI - Complete Model Integration Guide

## 📋 Overview

This guide covers the full integration of MindSpore models (ASL Translator, Hand Recognition, and Facial Recognition) into your Sense AI application for both **Signer View** (camera → text) and **Speaker View** (text → ASL gloss).

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      UI Layer (Tkinter)                     │
├─────────────────────────────────────────────────────────────┤
│  Speaker Screen              │         Signer Screen        │
│  - Text Input → Translation  │    - Camera Capture          │
│  - Speech Recognition        │    - Real-time Frame Proc.   │
│  - SIGML Avatar              │    - Model Inference         │
└─────────────────────────────────────────────────────────────┘
                                ↓
┌─────────────────────────────────────────────────────────────┐
│                 Services Layer (model_loader)               │
├─────────────────────────────────────────────────────────────┤
│  - ASLTranslatorModel                                       │
│  - HandRecognitionModel                                     │
│  - FacialRecognitionModel                                   │
│  - ModelManager (centralized access)                        │
└─────────────────────────────────────────────────────────────┘
                                ↓
┌─────────────────────────────────────────────────────────────┐
│            Processing Layer (camera_processor)              │
├─────────────────────────────────────────────────────────────┤
│  - CameraFrameProcessor (real-time capture + inference)     │
│  - FrameBuffer (gesture accumulation)                       │
└─────────────────────────────────────────────────────────────┘
                                ↓
┌─────────────────────────────────────────────────────────────┐
│        Backend WebSocket Server (websocket_server)          │
├─────────────────────────────────────────────────────────────┤
│  - Text Translation (speaker → gloss)                       │
│  - Frame Translation (signer → text)                        │
│  - Multimodal Fusion (hand + facial + grammar)              │
│  - Database persistence                                     │
└─────────────────────────────────────────────────────────────┘
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
cd e:\PycharmProjects\Sense_AI_Repo\sense_ai
pip install -r requirements.txt
```

**Key Dependencies:**

- `mindspore` - Model inference engine
- `opencv-python` - Video frame processing
- `websockets` - Real-time communication
- `speechrecognition` - Voice-to-text conversion
- `pyaudio` - Microphone input

### 2. Verify Model Files

Ensure these model files exist:

```
gloss_mindspore_models/
├── final_models (.mindir)/
│   ├── asl_translator.mindir      ← ASL to gloss translation
│   └── model.ckpt                 ← Checkpoint
├── final_models/
│   └── vocab.json                 ← Vocabulary mapping
├── facial_recognition_model/
│   └── FacialExpressionModel.mindir ← Emotion/grammar detection
└── hand_recognition_model/
    └── hand_recognition_model.mindir ← Hand pose detection
```

### 3. Start Backend Server

```bash
cd sense_ai\Backend
python websocket_server.py
```

Expected output:

```
INFO:root:Starting WebSocket server on ws://0.0.0.0:8765
INFO:root:Server running on ws://0.0.0.0:8765
```

### 4. Run UI Application

```bash
cd sense_ai
python main.py
```

## 📱 Feature Breakdown

### Speaker View (Text → ASL Gloss)

**Flow:**

1. User types text in input field
2. Clicks "Sign" button or presses Enter
3. Text is sent to `ModelManager.translate_text_to_gloss()`
4. Returns ASL gloss representation
5. Gloss is sent to SIGML avatar for animation
6. Translation is saved to database

**Key Code:**

- [speaker_screen.py](speaker_screen.py) - UI implementation
- [model_loader.py](#asltranlatormodel) - `ASLTranslatorModel.translate()`

**Example:**

```
Input:  "Yes, I am coming."
Output: "YES I COME"  (ASL gloss)
```

### Signer View (Camera → Text)

**Flow:**

1. Camera starts capturing frames at 30 FPS
2. `CameraFrameProcessor` runs hand + facial models on each frame
3. Frame results are buffered (30 frame window)
4. Every 3 frames (~100ms), detector outputs sent to server
5. Server runs multimodal fusion (attention mechanism)
6. Returns text translation and saves to database

**Key Code:**

- [signer_screen.py](signer_screen.py) - UI implementation
- [camera_processor.py](camera_processor.py) - Frame processing
- [websocket_server.py](../../Backend/websocket_server.py) - Server

**Detection Models:**

- **Hand Recognition**: 21 keypoints per hand
- **Facial Expression**: 7 emotions (neutral, happy, sad, angry, surprised, disgusted, fearful)
- **Attention Fusion**: Combines both streams with learned weights

## 🔧 API Reference

### ModelManager

```python
from services.model_loader import get_model_manager

manager = get_model_manager()

# Text to ASL gloss
result = manager.translate_text_to_gloss("Hello, how are you?")
# Returns: {
#     "gloss": "HELLO HOW YOU",
#     "confidence": 0.85,
#     "model": "asl_translator"
# }

# Detect hand pose in frame
hand_result = manager.detect_hand_pose(frame)
# Returns: {
#     "detected": True,
#     "keypoints": [[x1, y1], [x2, y2], ...],  # 21 points
#     "confidence": 0.9
# }

# Detect facial expression in frame
facial_result = manager.detect_facial_expression(frame)
# Returns: {
#     "detected": True,
#     "expression": "happy",
#     "confidence": 0.87,
#     "all_emotions": {"neutral": 0.1, "happy": 0.87, ...}
# }
```

### CameraFrameProcessor

```python
from services.camera_processor import CameraFrameProcessor

processor = CameraFrameProcessor(camera_index=0, fps=30)
processor.on_frame_processed = my_callback  # Optional
processor.start()

# Callback receives ProcessedFrame:
def my_callback(processed_frame):
    frame = processed_frame.frame  # numpy array
    hand_data = processed_frame.hand_data
    facial_data = processed_frame.facial_data
    frame_id = processed_frame.frame_id

processor.stop()
```

### WebSocket Server Messages

**Speaker View (Text Translation):**

```json
{
  "type": "translate_text",
  "user_id": 1,
  "session_token": "...",
  "text": "Hello, how are you?"
}
```

Response:

```json
{
  "type": "translation_result",
  "ok": true,
  "translation": {
    "translated_text": "HELLO HOW YOU",
    "confidence_score": 0.85,
    "id": 123
  }
}
```

**Signer View (Frame Translation):**

```json
{
    "type": "translate_frame",
    "hand_data": {
        "detected": true,
        "keypoints": [[x1, y1], ...],
        "confidence": 0.9
    },
    "facial_data": {
        "detected": true,
        "expression": "happy",
        "confidence": 0.87
    },
    "frame_id": 42,
    "user_id": 1
}
```

Response:

```json
{
  "type": "translation_result",
  "ok": true,
  "translation": {
    "translated_text": "[Sign detected] Expression: happy",
    "confidence_score": 0.9,
    "id": 124
  }
}
```

## 🎛️ Configuration

### [config.py](../config.py)

```python
BACKEND_WS_URL = "ws://localhost:8765"  # WebSocket server
SIGML_HOST = "127.0.0.1"                # CWASA avatar host
SIGML_PORT = 4569                       # CWASA avatar port
CAMERA_INDEX = 0                        # Webcam device index
```

### Model Paths

Edit `ModelManager._initialize_models()` if models are in different location:

```python
asl_model_path = "path/to/asl_translator.mindir"
asl_vocab_path = "path/to/vocab.json"
hand_model_path = "path/to/hand_recognition_model.mindir"
facial_model_path = "path/to/FacialExpressionModel.mindir"
```

## 📊 Performance Optimization

### Frame Processing Speed

- **Current FPS Target**: 30 FPS
- **Typical Processing**: ~33ms per frame
  - Hand detection: ~10-15ms
  - Facial detection: ~8-10ms
  - WebSocket send: ~5ms

### Optimization Tips

1. **Reduce Frame Resolution**: Edit `CameraFrameProcessor`

   ```python
   self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)  # From 640
   self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240) # From 480
   ```

2. **Lower FPS**: In `signer_screen.py`

   ```python
   self.translation_update_frequency = 20  # Reduce from 10
   ```

3. **Use GPU**: In `model_loader.py`

   ```python
   context.set_context(device_target="GPU")  # From CPU
   ```

4. **Model Quantization**: Use `.mindir` (compiled) instead of `.ckpt`

## 🐛 Troubleshooting

### Camera Not Detected

**Error:** `Camera unavailable on index 0`

**Solution:**

```python
# Check available cameras
import cv2
for i in range(5):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        print(f"Camera {i} available")
        cap.release()

# Update config.py
CAMERA_INDEX = 1  # or your working camera index
```

### Model Loading Failed

**Error:** `ASL model not found at ...`

**Solution:**

1. Verify model files exist in correct path
2. Check path spelling and capitalization
3. Use absolute paths instead of relative paths

```python
from pathlib import Path
model_path = Path(__file__).parent / "models" / "asl_translator.mindir"
```

### WebSocket Connection Failed

**Error:** `Connection refused`

**Solution:**

1. Ensure backend server is running: `python websocket_server.py`
2. Check port 8765 is not in use:
   ```bash
   netstat -ano | findstr :8765  # Windows
   lsof -i :8765  # Linux/Mac
   ```
3. Update `config.py` BACKEND_WS_URL if needed

### Low Confidence Scores

**Cause:** Model hasn't seen enough training data or poor lighting

**Solutions:**

- Improve lighting conditions
- Increase gesture size (move closer to camera)
- Adjust confidence threshold in code
- Retrain models on your dataset

## 📈 Next Steps

### Avatar Animation API Integration

**Currently**: Uses local SIGML avatar

**To Connect Remote Avatar API:**

1. Update `services/sigml_sender.py`:

```python
import requests

def speak_to_avatar(gloss: str, callback):
    response = requests.post(
        "https://your-avatar-api.com/sign",
        json={"gloss": gloss},
        timeout=5
    )
    animation_url = response.json()["video_url"]
    # Display video in UI
```

### Real-time Streaming

For continuous translation without discrete frames:

```python
# In signer_screen.py
self.translation_update_frequency = 1  # Every frame
```

### Grammar/Punctuation Detection

Enhance `FacialRecognitionModel` to detect:

- Questions (eyebrow raise)
- Negation (head shake)
- Emphasis (mouth open)

## 📚 References

- [MindSpore Documentation](https://www.mindspore.cn/)
- [ASL Gloss Standards](https://www.lifeprint.com/)
- [SIGML Specification](https://www.vcom3d.com/)
- [OpenCV Video Capture](https://docs.opencv.org/master/d8/dfe/classcv_1_1VideoCapture.html)

## 🤝 Support

For issues or questions, check:

1. [Troubleshooting](#-troubleshooting) section above
2. Model-specific issues in `Backend/services/asl_service.py`
3. WebSocket issues in `Backend/websocket_server.py`
4. UI issues in `screens/speaker_screen.py` and `screens/signer_screen.py`
