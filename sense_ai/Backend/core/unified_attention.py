import asyncio
import queue
import threading
import time
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional

import numpy as np


@dataclass
class ModelOutput:
    """Standardized output format from all models."""

    text: str
    confidence: float
    timestamp: float
    raw_data: Dict[str, Any]
    model_type: str


class AttentionState(Enum):
    QUESTION = "question"
    STATEMENT = "statement"
    EMPHASIS = "emphasis"
    NEGATION = "negation"
    CONDITIONAL = "conditional"
    TOPICALIZATION = "topic"
    UNCERTAIN = "uncertain"


class SynchronizedAttentionMechanism:
    def __init__(self, sync_window_ms: int = 100):
        """
        Unified attention mechanism with synchronization.

        Args:
            sync_window_ms: Time window to wait for all models (milliseconds)
        """
        self.sync_window = sync_window_ms / 1000

        self.face_queue = queue.Queue()
        self.hand_queue = queue.Queue()
        self.grammar_queue = queue.Queue()

        self.sync_lock = threading.Lock()
        self.sync_event = threading.Event()

        self.synchronized_outputs = deque(maxlen=100)

        self.attention_weights = {
            "facial": 0.33,
            "hand": 0.33,
            "grammar": 0.34,
        }

        self.context_memory = deque(maxlen=30)
        self.temporal_weights = deque(maxlen=10)

        self.model_latencies = {
            "facial": [],
            "hand": [],
            "grammar": [],
        }

        self.running = True
        self.sync_thread = threading.Thread(target=self._synchronize_outputs, daemon=True)
        self.sync_thread.start()

    def feed_face_output(self, output: ModelOutput):
        """Feed facial recognition model output."""
        self.face_queue.put(output)
        self._track_latency("facial", output.timestamp)

    def feed_hand_output(self, output: ModelOutput):
        """Feed hand recognition model output."""
        self.hand_queue.put(output)
        self._track_latency("hand", output.timestamp)

    def feed_grammar_output(self, output: ModelOutput):
        """Feed grammar model output."""
        self.grammar_queue.put(output)
        self._track_latency("grammar", output.timestamp)

    def _track_latency(self, model_type: str, timestamp: float):
        """Track model latency for synchronization tuning."""
        latency = time.time() - timestamp
        self.model_latencies[model_type].append(latency)
        if len(self.model_latencies[model_type]) > 100:
            self.model_latencies[model_type].pop(0)

    def _synchronize_outputs(self):
        """Synchronize outputs from all three models based on timestamps."""
        while self.running:
            try:
                face_out = self._get_latest_output(self.face_queue, timeout=0.05)
                hand_out = self._get_latest_output(self.hand_queue, timeout=0.05)
                grammar_out = self._get_latest_output(self.grammar_queue, timeout=0.05)

                if face_out and hand_out and grammar_out:
                    timestamps = [face_out.timestamp, hand_out.timestamp, grammar_out.timestamp]
                    max_ts = max(timestamps)
                    min_ts = min(timestamps)

                    if max_ts - min_ts <= self.sync_window:
                        synchronized = self._process_synchronized(face_out, hand_out, grammar_out)
                        self.synchronized_outputs.append(synchronized)
                        self.sync_event.set()
                    else:
                        self._realign_outputs(face_out, hand_out, grammar_out)

                time.sleep(0.01)
            except Exception as e:
                print(f"Synchronization error: {e}")

    def _get_latest_output(self, output_queue, timeout):
        """Get the most recent output from a model queue."""
        _ = timeout
        latest = None
        try:
            while True:
                latest = output_queue.get_nowait()
        except queue.Empty:
            pass
        return latest

    def _realign_outputs(self, face_out: ModelOutput, hand_out: ModelOutput, grammar_out: ModelOutput):
        """Realign outputs when they fall out of the sync window."""
        timestamps = {
            "facial": face_out.timestamp,
            "hand": hand_out.timestamp,
            "grammar": grammar_out.timestamp,
        }
        oldest_model = min(timestamps, key=timestamps.get)

        if oldest_model == "facial":
            self._clear_queue(self.face_queue)
        elif oldest_model == "hand":
            self._clear_queue(self.hand_queue)
        else:
            self._clear_queue(self.grammar_queue)

    def _clear_queue(self, q):
        """Clear all items from a queue."""
        try:
            while True:
                q.get_nowait()
        except queue.Empty:
            pass

    def _process_synchronized(self, face_out: ModelOutput, hand_out: ModelOutput, grammar_out: ModelOutput):
        """Core attention mechanism for synchronized model outputs."""
        facial_features = self._extract_facial_features(face_out)
        hand_features = self._extract_hand_features(hand_out)
        grammar_features = self._extract_grammar_features(grammar_out)

        dynamic_weights = self._compute_dynamic_weights(facial_features, hand_features, grammar_features)
        lead_model, lead_conf = self._determine_lead_model(dynamic_weights)
        _ = lead_conf

        unified = self._generate_unified_output(face_out, hand_out, grammar_out, dynamic_weights, lead_model)
        self._update_context(unified, dynamic_weights, lead_model)
        return unified

    def _extract_facial_features(self, face_out: ModelOutput):
        """Extract relevant features from facial model output."""
        return {
            "expression": face_out.raw_data.get("expression", "neutral"),
            "intensity": face_out.raw_data.get("intensity", 0),
            "eyebrow_position": face_out.raw_data.get("eyebrow_position", 0),
            "head_tilt": face_out.raw_data.get("head_tilt", 0),
            "mouth_movement": face_out.raw_data.get("mouth_movement", 0),
            "confidence": face_out.confidence,
            "timestamp": face_out.timestamp,
        }

    def _extract_hand_features(self, hand_out: ModelOutput):
        """Extract relevant features from hand model output."""
        return {
            "sign": hand_out.text,
            "complexity": hand_out.raw_data.get("complexity", 0),
            "movement_speed": hand_out.raw_data.get("movement_speed", 0),
            "handshape": hand_out.raw_data.get("handshape", ""),
            "location": hand_out.raw_data.get("location", ""),
            "confidence": hand_out.confidence,
            "timestamp": hand_out.timestamp,
        }

    def _extract_grammar_features(self, grammar_out: ModelOutput):
        """Extract relevant features from grammar model output."""
        return {
            "suggestion": grammar_out.text,
            "error_detected": grammar_out.raw_data.get("error_detected", False),
            "error_type": grammar_out.raw_data.get("error_type", "none"),
            "confidence": grammar_out.confidence,
            "timestamp": grammar_out.timestamp,
        }

    def _compute_dynamic_weights(self, facial, hand, grammar):
        """Compute attention weights from facial, hand, and grammar features."""
        weights = {"facial": 0.33, "hand": 0.33, "grammar": 0.34}

        if facial["intensity"] > 0.7:
            weights["facial"] += 0.3
            weights["hand"] -= 0.15
            weights["grammar"] -= 0.15

        if hand["complexity"] > 0.8:
            weights["hand"] += 0.25
            weights["facial"] -= 0.125
            weights["grammar"] -= 0.125

        if grammar["error_detected"]:
            weights["grammar"] += 0.35
            weights["facial"] -= 0.175
            weights["hand"] -= 0.175

        if facial["confidence"] < 0.5:
            weights["facial"] *= 0.5
            weights["hand"] += 0.25
            weights["grammar"] += 0.25

        if hand["confidence"] < 0.5:
            weights["hand"] *= 0.5
            weights["facial"] += 0.25
            weights["grammar"] += 0.25

        if grammar["confidence"] < 0.5:
            weights["grammar"] *= 0.5
            weights["facial"] += 0.25
            weights["hand"] += 0.25

        total = sum(weights.values())
        for key in weights:
            weights[key] = max(0.0, weights[key] / total)

        if self.temporal_weights:
            prev_weights = self.temporal_weights[-1]
            smoothing_factor = 0.7
            for key in weights:
                weights[key] = smoothing_factor * weights[key] + (1 - smoothing_factor) * prev_weights[key]

        total = sum(weights.values())
        for key in weights:
            weights[key] = weights[key] / total

        self.temporal_weights.append(weights.copy())
        return weights

    def _determine_lead_model(self, weights):
        """Determine which model leads interpretation."""
        lead = max(weights, key=weights.get)
        lead_confidence = weights[lead]

        second_best = sorted(weights.values())[-2]
        if lead_confidence - second_best < 0.05:
            if self.context_memory:
                prev_lead = self.context_memory[-1].get("lead_model", "hand")
                return prev_lead, 0.6
        return lead, lead_confidence

    def _generate_unified_output(self, face_out, hand_out, grammar_out, weights, lead_model):
        """Generate final unified output using attention-weighted fusion."""
        face_text = face_out.text
        hand_text = hand_out.text
        grammar_text = grammar_out.text

        if lead_model == "facial":
            final_text, fusion_strategy = self._facial_led_fusion(face_text, hand_text, grammar_text, face_out, hand_out)
        elif lead_model == "hand":
            final_text, fusion_strategy = self._hand_led_fusion(face_text, hand_text, grammar_text, hand_out, grammar_out)
        else:
            final_text, fusion_strategy = self._grammar_led_fusion(hand_text, grammar_text, hand_out, grammar_out)

        confidence = (
            weights["facial"] * face_out.confidence
            + weights["hand"] * hand_out.confidence
            + weights["grammar"] * grammar_out.confidence
        )

        return {
            "text": final_text,
            "confidence": confidence,
            "lead_model": lead_model,
            "attention_weights": weights,
            "fusion_strategy": fusion_strategy,
            "timestamp": time.time(),
            "raw_inputs": {
                "facial": face_text,
                "hand": hand_text,
                "grammar": grammar_text,
            },
        }

    def _facial_led_fusion(self, face_text, hand_text, grammar_text, face_out, hand_out):
        """Fusion strategy when facial expression leads."""
        _ = face_text, grammar_text, hand_out
        expression = face_out.raw_data.get("expression", "neutral")

        if expression == "question":
            return f"{hand_text}?", "facial_question"
        if expression == "negation":
            return f"NOT {hand_text}", "facial_negation"
        if expression == "emphasis":
            return f"VERY {hand_text}!", "facial_emphasis"
        if expression == "surprise":
            return f"Wow! {hand_text}", "facial_surprise"
        if expression == "confusion":
            return f"{hand_text}??", "facial_confusion"
        return hand_text, "facial_neutral"

    def _hand_led_fusion(self, face_text, hand_text, grammar_text, hand_out, grammar_out):
        """Fusion strategy when hand sign leads."""
        _ = face_text, grammar_text
        sign = hand_text
        if grammar_out.confidence > 0.7:
            corrected = grammar_out.text
            if corrected != sign:
                return corrected, "hand_grammar_corrected"
            return sign, "hand_primary"
        return sign, "hand_primary_low_grammar"

    def _grammar_led_fusion(self, hand_text, grammar_text, hand_out, grammar_out):
        """Fusion strategy when grammar model leads."""
        if grammar_out.confidence > hand_out.confidence:
            return grammar_text, "grammar_correction"

        error_type = grammar_out.raw_data.get("error_type", "none")
        if error_type == "word_order":
            return grammar_text, "grammar_word_order"
        if error_type == "missing_word":
            return grammar_text, "grammar_completion"
        return hand_text, "grammar_hand_fallback"

    def _update_context(self, unified, weights, lead_model):
        """Update context memory with current output."""
        self.context_memory.append(
            {
                "timestamp": time.time(),
                "output": unified,
                "weights": weights,
                "lead_model": lead_model,
            }
        )

    def get_synchronized_output(self, timeout: float = 0.1):
        """Get latest synchronized output, or None if unavailable."""
        _ = timeout
        if self.synchronized_outputs:
            return self.synchronized_outputs[-1]
        return None

    def stop(self):
        """Stop the synchronization thread."""
        self.running = False
        if self.sync_thread.is_alive():
            self.sync_thread.join(timeout=1.0)


class UnifiedAttentionMechanism(SynchronizedAttentionMechanism):
    """
    Backward-compatible wrapper used by ASLService.translate_multimodal.
    """

    def compute_attention(
        self,
        face_output: Dict[str, Any],
        hand_output: Dict[str, Any],
        grammar_state: Dict[str, Any],
    ) -> Dict[str, Any]:
        now = time.time()

        face_model_output = ModelOutput(
            text=face_output.get("expression", ""),
            confidence=float(face_output.get("confidence", 0.0)),
            timestamp=float(face_output.get("timestamp", now)),
            raw_data=face_output,
            model_type="facial",
        )
        hand_model_output = ModelOutput(
            text=hand_output.get("sign", ""),
            confidence=float(hand_output.get("confidence", 0.0)),
            timestamp=float(hand_output.get("timestamp", now)),
            raw_data=hand_output,
            model_type="hand",
        )
        grammar_model_output = ModelOutput(
            text=grammar_state.get("suggestion", grammar_state.get("corrected_text", "")),
            confidence=float(grammar_state.get("confidence", 0.0)),
            timestamp=float(grammar_state.get("timestamp", now)),
            raw_data=grammar_state,
            model_type="grammar",
        )

        result = self._process_synchronized(face_model_output, hand_model_output, grammar_model_output)
        result["context_used"] = len(self.context_memory)
        return result


__all__ = [
    "AttentionState",
    "ModelOutput",
    "SynchronizedAttentionMechanism",
    "UnifiedAttentionMechanism",
]
