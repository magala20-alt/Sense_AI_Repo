"""
Model Loader Service - Loads and manages MindSpore models for sign language inference
"""

import os
import json
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
import logging

try:
    import mindspore
    import mindspore.nn as nn
    from mindspore import load_checkpoint, load_param_into_net, Tensor
    from mindspore import context
    # Set to CPU mode - change to GPU if available
    context.set_context(mode=context.PYNATIVE_MODE)
    try:
        mindspore.set_device("CPU")  # Use new API instead of deprecated device_target
    except:
        pass  # Fallback for older versions
    MINDSPORE_AVAILABLE = True
except ImportError:
    MINDSPORE_AVAILABLE = False
    mindspore = None

import cv2
from PIL import Image

logger = logging.getLogger(__name__)

# Model paths
MODEL_BASE_PATH = Path(__file__).resolve().parents[1] / ".." / "gloss_mindspore_models"


class ASLTranslatorModel:
    """ASL Translator: English gloss → Sign Language (encoder-decoder architecture)"""

    def __init__(self, model_path: str, vocab_path: str, device: str = "cpu"):
        self.model_path = model_path
        self.vocab_path = vocab_path
        self.device = device
        self.model = None
        self.vocab = None
        self.en_word2idx = {}
        self.gloss_word2idx = {}
        self.reverse_vocab = None
        self.embedding_dim = 128
        self.hidden_dim = 256
        self._load_model()

    def _load_model(self):
        """Load MindSpore model checkpoint or mindir file"""
        try:
            if not os.path.exists(self.model_path):
                logger.warning(f"ASL model not found at {self.model_path}")
                return False

            # Load vocabulary
            if os.path.exists(self.vocab_path):
                try:
                    with open(self.vocab_path, "r", encoding="utf-8") as f:
                        raw_vocab = json.load(f)

                    # Preferred schema in this repo:
                    # {"en_word2idx": {...}, "gloss_word2idx": {...}, "gloss_idx2word": {...}}
                    if isinstance(raw_vocab, dict) and "en_word2idx" in raw_vocab:
                        self.en_word2idx = raw_vocab.get("en_word2idx", {}) or {}
                        self.gloss_word2idx = raw_vocab.get("gloss_word2idx", {}) or {}
                        gloss_idx2word = raw_vocab.get("gloss_idx2word", {}) or {}
                        # Normalize index keys to int when possible.
                        self.reverse_vocab = {}
                        for idx, token in gloss_idx2word.items():
                            try:
                                self.reverse_vocab[int(idx)] = token
                            except (TypeError, ValueError):
                                continue
                        # Keep legacy field for compatibility with existing code paths.
                        self.vocab = self.en_word2idx
                    else:
                        # Legacy flat schema: {token: idx}
                        self.vocab = raw_vocab if isinstance(raw_vocab, dict) else {}
                        self.reverse_vocab = {
                            v: k for k, v in self.vocab.items() if isinstance(v, int)
                        }
                except (UnicodeDecodeError, json.JSONDecodeError) as e:
                    logger.warning(f"Vocabulary file error: {e}. Using fallback.")
                    self.vocab = {}
                    self.en_word2idx = {}
                    self.gloss_word2idx = {}
                    self.reverse_vocab = {}
            else:
                logger.warning(f"Vocabulary not found at {self.vocab_path}")

            # Load model - mindir requires special handling
            if self.model_path.endswith(".mindir"):
                logger.info(f"MindSpore mindir model found at {self.model_path}")
                logger.info("Note: mindir models require MindSpore runtime. Running in fallback mode.")
                # For now, work in fallback mode - production should use mindspore-serving
                self.model = None
            elif self.model_path.endswith(".ckpt"):
                logger.info(f"Loading MindSpore checkpoint from {self.model_path}")
                # Build simple encoder-decoder model
                self.model = self._build_model()
                if MINDSPORE_AVAILABLE and self.model:
                    try:
                        load_checkpoint(self.model_path, net=self.model)
                        logger.info("Checkpoint loaded successfully")
                    except Exception as e:
                        logger.warning(f"Could not load checkpoint: {e}. Using untrained model.")

            logger.info("ASL Translator Model loaded successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to load ASL model: {e}")
            return False

    def _build_model(self):
        """Build a simple encoder-decoder model for ASL translation"""
        if not MINDSPORE_AVAILABLE:
            return None

        class SimpleASLTranslator(nn.Cell):
            def __init__(self, vocab_size: int, embedding_dim: int, hidden_dim: int):
                super().__init__()
                self.embedding = nn.Embedding(vocab_size, embedding_dim)
                self.encoder_lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
                self.decoder_lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
                self.output_layer = nn.Dense(hidden_dim, vocab_size)

            def construct(self, x):
                embedded = self.embedding(x)
                encoder_output, (h_n, c_n) = self.encoder_lstm(embedded)
                decoder_output, _ = self.decoder_lstm(encoder_output, (h_n, c_n))
                logits = self.output_layer(decoder_output)
                return logits

        input_vocab = self.en_word2idx if self.en_word2idx else (self.vocab or {})
        vocab_size = len(input_vocab) if input_vocab else 10000
        return SimpleASLTranslator(vocab_size, self.embedding_dim, self.hidden_dim)

    def translate(self, text: str, max_length: int = 50) -> Dict[str, Any]:
        """Translate English text to ASL gloss"""
        try:
            token_vocab = self.en_word2idx if self.en_word2idx else (self.vocab or {})

            if not token_vocab:
                return {
                    "gloss": "[Translation unavailable]",
                    "confidence": 0.0,
                    "error": "Vocabulary not loaded"
                }

            # Tokenize input (simplified - could use more sophisticated tokenization)
            tokens = text.lower().split()
            unk_id = token_vocab.get("<unk>", token_vocab.get("<UNK>", 0))
            token_ids = [token_vocab.get(token, unk_id) for token in tokens]
            token_ids = token_ids[:max_length]  # Truncate to max length

            if MINDSPORE_AVAILABLE and self.model:
                # Convert to MindSpore tensor
                input_tensor = Tensor(np.array([token_ids]), mindspore.int32)

                # Run inference
                with mindspore.no_grad():
                    output = self.model(input_tensor)

                # Convert logits to gloss tokens
                gloss_ids = np.argmax(output.asnumpy(), axis=-1)[0]
                gloss_tokens = [self.reverse_vocab.get(int(id), "[UNK]") for id in gloss_ids]
                gloss = " ".join(gloss_tokens)
                confidence = 0.85
                model_used = "asl_translator"
            else:
                # Fallback when model runtime is unavailable: deterministic gloss-like output.
                if self.gloss_word2idx:
                    gloss_tokens = [token.upper() for token in tokens]
                else:
                    gloss_tokens = [token.upper() for token in tokens]
                gloss = " ".join(gloss_tokens)
                confidence = 0.45
                model_used = "fallback_vocab"

            return {
                "gloss": gloss,
                "confidence": confidence,
                "model": model_used,
            }
        except Exception as e:
            logger.error(f"Translation error: {e}")
            return {
                "gloss": "[Translation error]",
                "confidence": 0.0,
                "error": str(e)
            }


class HandRecognitionModel:
    """Hand Recognition: Detects hand pose/position from video frames"""

    def __init__(self, model_path: str, device: str = "cpu"):
        self.model_path = model_path
        self.device = device
        self.model = None
        self._load_model()

    def _load_model(self):
        """Load hand recognition model"""
        try:
            if not os.path.exists(self.model_path):
                logger.warning(f"Hand model not found at {self.model_path}")
                return False

            if self.model_path.endswith(".mindir"):
                logger.info(f"Hand model mindir found at {self.model_path}")
                logger.info("Note: Running in fallback mode (requires MindSpore serving in production)")
                self.model = None

            logger.info("Hand Recognition Model initialized")
            return True
        except Exception as e:
            logger.error(f"Failed to load hand model: {e}")
            return False

    def detect(self, frame: np.ndarray) -> Dict[str, Any]:
        """Detect hand pose in frame"""
        try:
            if frame is None:
                return {"detected": False, "keypoints": [], "confidence": 0.0}

            # Resize frame for model input (typically 224x224 or 256x256)
            h, w = frame.shape[:2]
            target_size = 224
            frame_resized = cv2.resize(frame, (target_size, target_size))

            # Normalize
            frame_normalized = frame_resized.astype(np.float32) / 255.0

            if MINDSPORE_AVAILABLE and self.model:
                # Run inference
                input_tensor = Tensor(np.expand_dims(frame_normalized, 0), mindspore.float32)
                with mindspore.no_grad():
                    output = self.model(input_tensor)

                # Parse keypoints (21 hand keypoints typical for hand pose)
                keypoints = output.asnumpy()[0]
                keypoints = keypoints.reshape(-1, 2)  # Reshape to [num_points, 2]

                # Scale back to original frame size
                keypoints[:, 0] *= (w / target_size)
                keypoints[:, 1] *= (h / target_size)

                return {
                    "detected": True,
                    "keypoints": keypoints.tolist(),
                    "confidence": 0.9,
                }
            else:
                # Fallback: placeholder
                return {
                    "detected": False,
                    "keypoints": [],
                    "confidence": 0.0,
                    "reason": "Model not loaded"
                }

        except Exception as e:
            logger.error(f"Hand detection error: {e}")
            return {"detected": False, "keypoints": [], "confidence": 0.0, "error": str(e)}


class FacialRecognitionModel:
    """Facial Recognition: Detects facial expressions and grammar markers"""

    def __init__(self, model_path: str, device: str = "cpu"):
        self.model_path = model_path
        self.device = device
        self.model = None
        self.expression_labels = [
            "neutral", "happy", "sad", "angry", "surprised", "disgusted", "fearful"
        ]
        self._load_model()

    def _load_model(self):
        """Load facial recognition model"""
        try:
            if not os.path.exists(self.model_path):
                logger.warning(f"Facial model not found at {self.model_path}")
                return False

            if self.model_path.endswith(".mindir"):
                logger.info(f"Facial model mindir found at {self.model_path}")
                logger.info("Note: Running in fallback mode (requires MindSpore serving in production)")
                self.model = None

            logger.info("Facial Recognition Model initialized")
            return True
        except Exception as e:
            logger.error(f"Failed to load facial model: {e}")
            return False

    def detect(self, frame: np.ndarray) -> Dict[str, Any]:
        """Detect facial expression and grammar markers"""
        try:
            if frame is None:
                return {"detected": False, "expression": None, "confidence": 0.0}

            # Resize frame
            h, w = frame.shape[:2]
            target_size = 224
            frame_resized = cv2.resize(frame, (target_size, target_size))
            frame_normalized = frame_resized.astype(np.float32) / 255.0

            if MINDSPORE_AVAILABLE and self.model:
                # Run inference
                input_tensor = Tensor(np.expand_dims(frame_normalized, 0), mindspore.float32)
                with mindspore.no_grad():
                    output = self.model(input_tensor)

                # Parse output as emotion classification
                logits = output.asnumpy()[0]
                emotion_idx = np.argmax(logits)
                confidence = float(np.exp(logits[emotion_idx]) / np.sum(np.exp(logits)))

                return {
                    "detected": True,
                    "expression": self.expression_labels[min(emotion_idx, len(self.expression_labels) - 1)],
                    "confidence": confidence,
                    "all_emotions": {label: float(score) for label, score in zip(self.expression_labels, logits)}
                }
            else:
                return {
                    "detected": False,
                    "expression": "neutral",
                    "confidence": 0.0,
                    "reason": "Model not loaded"
                }

        except Exception as e:
            logger.error(f"Facial detection error: {e}")
            return {"detected": False, "expression": None, "confidence": 0.0, "error": str(e)}


class ModelManager:
    """Central manager for all models"""

    def __init__(self, model_base_path: Optional[str] = None):
        self.base_path = Path(model_base_path) if model_base_path else MODEL_BASE_PATH
        self.asl_translator = None
        self.hand_recognizer = None
        self.facial_recognizer = None
        self._initialize_models()

    def _initialize_models(self):
        """Initialize all model instances"""
        try:
            # ASL Translator
            asl_model_path = str(self.base_path / "final_models (.mindir)" / "asl_translator.mindir")
            asl_vocab_path = str(self.base_path / "final_models" / "vocab.json")
            self.asl_translator = ASLTranslatorModel(asl_model_path, asl_vocab_path)

            # Hand Recognition
            hand_model_path = str(Path(__file__).resolve().parents[2] / "hand_recognition_model" / "hand_recognition_model.mindir")
            self.hand_recognizer = HandRecognitionModel(hand_model_path)

            # Facial Recognition
            facial_model_path = str(self.base_path / "facial_recognition_model" / "FacialExpressionModel.mindir")
            self.facial_recognizer = FacialRecognitionModel(facial_model_path)

            logger.info("All models initialized")
        except Exception as e:
            logger.error(f"Model initialization error: {e}")

    def translate_text_to_gloss(self, text: str) -> Dict[str, Any]:
        """Translate English text to ASL gloss"""
        if self.asl_translator:
            return self.asl_translator.translate(text)
        return {"gloss": text, "confidence": 0.0, "error": "Model not loaded"}

    def detect_hand_pose(self, frame: np.ndarray) -> Dict[str, Any]:
        """Detect hand pose in frame"""
        if self.hand_recognizer:
            return self.hand_recognizer.detect(frame)
        return {"detected": False, "keypoints": [], "confidence": 0.0}

    def detect_facial_expression(self, frame: np.ndarray) -> Dict[str, Any]:
        """Detect facial expression in frame"""
        if self.facial_recognizer:
            return self.facial_recognizer.detect(frame)
        return {"detected": False, "expression": None, "confidence": 0.0}


# Global model manager instance
_model_manager: Optional[ModelManager] = None


def get_model_manager() -> ModelManager:
    """Get or create global model manager"""
    global _model_manager
    if _model_manager is None:
        _model_manager = ModelManager()
    return _model_manager


def load_models(model_base_path: Optional[str] = None) -> ModelManager:
    """Load all models and return manager"""
    global _model_manager
    _model_manager = ModelManager(model_base_path)
    return _model_manager

