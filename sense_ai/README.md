# Sense.AI Frontend — Python Tkinter Application

A real-time ASL (American Sign Language) translation desktop application with dual user interfaces for Deaf signers and hearing speakers.

## Project Structure

```
sense_ai/
├── main.py                      # Entry point
├── theme.py                     # Design system (colors, fonts)
├── config.py                    # Configuration
├── app_state.py                 # Global application state
│
├── screens/
│   ├── welcome_screen.py        # Welcome/role selection
│   ├── login_screen.py          # User login
│   ├── signup_screen.py         # User registration
│   ├── session_screen.py        # Session ID entry
│   ├── signer_screen.py         # Deaf signer main interface (camera + translation)
│   └── speaker_screen.py        # Hearing speaker main interface (text input + avatar)
│
├── components/
│   ├── window_bar.py            # Top title bar with colored dots
│   ├── status_bar.py            # Bottom status strip with pulsing indicator
│   ├── sidebar.py               # Left icon sidebar
│   ├── grammar_tag.py           # Grammar type badge
│   ├── detection_chips.py       # Tier confidence percentage chips
│   ├── face_avatar.py           # Canvas-based face avatar drawing
│   ├── tier_confidence.py       # Three-tier confidence progress bars
│   └── conv_history.py          # Scrollable conversation history
│
├── services/
│   ├── websocket_client.py      # WebSocket connection to ASL backend
│   └── sigml_sender.py          # TCP socket communication with CWASA avatar
│
├── assets/                      # (For future image/media files)
└── requirements.txt             # Python dependencies
```

## Installation & Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

Or manually:

```bash
pip install opencv-python pillow websockets speechrecognition
```

### 2. Run the Application

```bash
python main.py
```

The app will launch in a 480×720 window with the Welcome screen displayed.

## Features

### 6 Screens

1. **Welcome Screen**
   - Role selection: Signer (Deaf) vs Speaker (Hearing)
   - Quick access buttons for Login/Sign Up
   - Pulsing status indicator

2. **Login Screen**
   - Email and password authentication
   - Link to Sign Up page

3. **Sign Up Screen**
   - Email, password, and confirm password fields
   - Account creation workflow

4. **Session Screen**
   - Conversation ID entry
   - Quick navigation to Signer or Speaker mode

5. **Signer Screen** (Deaf user interface)
   - Live camera feed with overlay dots for facial detection
   - Real-time ASL translation text output
   - Grammar type badge (YES_NO_QUESTION, NEGATION, WH_QUESTION, STATEMENT)
   - Confidence tier chips (Physical, Grammar, Semantic %)
   - Scrollable conversation history
   - Left sidebar for navigation
   - Status bar with pulsing live indicator
   - WebSocket connection to Python ASL backend (ws://localhost:8765)
   - _Demo mode_: If backend unavailable, generates mock translations every 3 seconds

6. **Speaker Screen** (Hearing user interface)
   - Text input field for typing
   - "Sign" button to send text to avatar
   - "Tap to Speak" button for voice-to-ASL (speech recognition)
   - Canvas-based animated face avatar showing signing
   - Avatar status indicator
   - Scrollable conversation history
   - Left sidebar with gold accent (hearing community color)
   - TCP socket communication with CWASA SiGML Player (port 4569)

## Design System

### Color Palette

- **Navy** `#1a2744` — window chrome, sidebars, headings
- **Teal** `#1d7a8a` — primary buttons, Deaf community brand
- **Gold** `#e8a020` — Speaker/hearing accent color
- **Cream** `#fdf8f2` — warm, accessible backgrounds
- **White** `#ffffff` — card/panel surfaces
- **Teal BG** `#e8f6f8` — soft section backgrounds

### Typography

- **Heading**: Helvetica 18 bold
- **Body**: Helvetica 12 regular
- **Labels**: Helvetica 9 bold
- **Status**: Courier 9 (monospace)

## Components

All 8 reusable UI components:

- `WindowBar` — Top bar with colored dots and title
- `StatusBar` — Bottom strip with pulsing dot status
- `Sidebar` — Left navigation + logout
- `GrammarTag` — Purple-bordered grammar type label
- `DetectionChips` — Three colored confidence indicators
- `FaceAvatar` — Canvas-drawn face for avatar display
- `TierConfidence` — Three progress bars for confidence levels
- `ConversationHistory` — Scrollable message log

## Services

### WebSocket Client (`services/websocket_client.py`)

Connects to Python ASL backend at `ws://localhost:8765`.

```json
Incoming message format:
{
  "translation": "Are you coming?",
  "grammar_type": "YES_NO_QUESTION",
  "tier_scores": {
    "physical": 88,
    "grammar": 75,
    "semantic": 69
  }
}
```

**Demo mode**: If backend unavailable (after 5s retries), auto-generates mock translations in an infinite loop.

### SiGML Sender (`services/sigml_sender.py`)

Sends SiGML XML commands to CWASA SiGML Player via TCP socket (127.0.0.1:4569).

- `send_sigml_text(sigml_xml)` — Send SiGML XML string
- `send_sigml_file(filepath)` — Load and send .sigml file
- `sign_gloss(gloss)` — Sign a single gloss
- `speak_to_avatar(text)` — Convert English text to glosses and sign (runs in background thread)

## Threading & UI Safety

- **All UI updates from background threads use `self.after(0, callback)`** to marshal changes to the main thread
- Camera capture runs in daemon thread (OpenCV VideoCapture)
- WebSocket connection runs in background async loop
- Speech recognition and avatar signing run in background threads
- All threads are daemon threads so they auto-terminate when app closes

## Configuration

Edit `config.py` to change:

```python
BACKEND_WS_URL = "ws://localhost:8765"   # ASL backend
SIGML_HOST = "127.0.0.1"                 # CWASA SiGML Player
SIGML_PORT = 4569                        # CWASA SiGML Port
CAMERA_INDEX = 0                         # Webcam device index
```

## Application State

Global `AppState` object shared across all screens tracks:

- `user_role` — "signer" or "speaker"
- `session_id` — conversation session ID
- `conv_history` — list of messages with metadata
- `is_connected` — WebSocket status
- `current_translation` — last translation text
- `current_grammar` — last grammar type
- `tier_scores` — confidence percentages

## Navigation Flow

```
Welcome
  ├─ Login→ → Session → Signer/Speaker
  ├─ Create Account → Session → Signer/Speaker
  ├─ I use ASL (Signer card) → Session → Signer
  └─ I Speak / Hear (Speaker card) → Session → Speaker

Signer [Out button] → Welcome
Speaker [Out button] → Welcome
```

## Key Implementation Details

### Camera Feed (Signer Screen)

1. OpenCV VideoCapture in background thread
2. Resize frames to 430×160 pixels
3. Convert BGR → RGB → PIL Image → PhotoImage
4. Update tk.Label image via `self.after()`
5. Overlay Canvas with detection dots (4 small colored ovals)
6. ~30fps update rate

### Translation Display

- WebSocket message updates grammar tag and text label
- Tier confidence chips update inline
- Message added to scrollable ConversationHistory with alternating row colors

### Avatar Signing (Speaker Screen)

- Canvas-drawn face with hair, eyes, nose, mouth, shoulders
- Simple gloss-based signing (text splitting demo)
- Status indicator updates: "Signing..." → "Ready"

## Keyboard Shortcuts

- **Signer Screen**: None (click buttons)
- **Speaker Screen**:
  - `<Return>` in text field = Sign button
  - `<Button-1>` on "Tap to Speak" = Voice input

## Troubleshooting

**App won't start**

- Check `requirements.txt` installed: `pip install -r requirements.txt`
- Verify Python 3.8+ with tkinter: `python -m tkinter`

**Camera not displaying**

- Check camera index in `config.py` (try 0, 1, 2, etc.)
- Ensure OpenCV installed: `pip install opencv-python`

**WebSocket connection fails**

- Check backend running on `ws://localhost:8765`
- App auto-switches to demo mode after 5s timeout
- Check network connectivity

**SiGML Avatar not responding**

- Check CWASA player running on 127.0.0.1:4569
- Verify SiGML XML syntax
- Check firewall blocking port 4569

## Future Enhancements

- Real database backend for user authentication
- Actual speech-to-text integration (speech_recognition placeholder)
- Grammar-constrained attention detection display
- SiGML file library and gloss dictionary
- Conversation export (JSON/PDF)
- Dark mode theme toggle
- Mobile responsive layout
- Multi-language support

---

**Built with**: Python, Tkinter, OpenCV, Pillow, Websockets, MediaPipe (backend only)
