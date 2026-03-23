"""
Test script for model integration
Tests each component independently
"""

import sys
from pathlib import Path
import time
import json

# Add sense_ai to path
sense_ai_path = Path(__file__).parent / "sense_ai"
sys.path.insert(0, str(sense_ai_path))

def test_model_manager():
    """Test model manager initialization and inference"""
    print("\n" + "="*60)
    print("TEST 1: Model Manager")
    print("="*60)
    
    try:
        from sense_ai.services.model_loader import get_model_manager
        
        print("Initializing ModelManager...")
        manager = get_model_manager()
        
        # Test text translation
        print("\nTest: Text to ASL Gloss Translation")
        test_texts = [
            "Hello",
            "How are you?",
            "Nice to meet you"
        ]
        
        for text in test_texts:
            result = manager.translate_text_to_gloss(text)
            print(f"  Input:  '{text}'")
            print(f"  Output: '{result.get('gloss', 'ERROR')}'")
            print(f"  Confidence: {result.get('confidence', 0):.2%}")
            print()
        
        print("✓ Test passed")
        return True
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_camera_processor():
    """Test camera frame processor"""
    print("\n" + "="*60)
    print("TEST 2: Camera Frame Processor")
    print("="*60)
    
    try:
        from sense_ai.services.camera_processor import CameraFrameProcessor
        import cv2
        
        print("Initializing CameraFrameProcessor...")
        processor = CameraFrameProcessor(camera_index=0, fps=30)
        
        if not processor.start():
            print("✗ Failed to start camera")
            return False
        
        print("Camera started. Capturing 5 frames...")
        
        for i in range(5):
            time.sleep(1)
            frame = processor.get_latest_frame()
            if frame:
                hand_detected = frame.hand_data.get("detected", False)
                facial_detected = frame.facial_data.get("detected", False)
                print(f"  Frame {frame.frame_id}: Hand={'✓' if hand_detected else '✗'}, "
                      f"Facial={'✓' if facial_detected else '✗'}")
            else:
                print(f"  Frame {i}: No data yet")
        
        processor.stop()
        print("✓ Test passed")
        return True
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_websocket_server():
    """Test WebSocket server startup"""
    print("\n" + "="*60)
    print("TEST 3: WebSocket Server")
    print("="*60)
    
    try:
        import asyncio
        sys.path.insert(0, str(sense_ai_path / "Backend"))
        from sense_ai.Backend.websocket_server import WebSocketServer
        
        print("Creating WebSocket server instance...")
        server = WebSocketServer(host="localhost", port=8765)
        
        print("Initializing server...")
        # Don't actually start the server, just verify initialization
        result = asyncio.run(server.initialize())
        
        if result:
            print("✓ Server initialized successfully")
            return True
        else:
            print("✗ Server initialization failed")
            return False
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_speaker_screen():
    """Test speaker screen integration"""
    print("\n" + "="*60)
    print("TEST 4: Speaker Screen Integration")
    print("="*60)
    
    try:
        from sense_ai.services.model_loader import get_model_manager
        
        print("Testing speaker screen (text → gloss) pipeline...")
        manager = get_model_manager()
        
        # Simulate speaker screen workflow
        test_phrase = "I love sign language"
        print(f"Input text: '{test_phrase}'")
        
        result = manager.translate_text_to_gloss(test_phrase)
        
        if result.get("gloss"):
            print(f"Translated gloss: '{result['gloss']}'")
            print(f"Confidence: {result.get('confidence', 0):.2%}")
            print("✓ Test passed")
            return True
        else:
            print("✗ Translation failed")
            return False
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_signer_screen():
    """Test signer screen integration"""
    print("\n" + "="*60)
    print("TEST 5: Signer Screen Integration")
    print("="*60)
    
    try:
        from sense_ai.services.camera_processor import CameraFrameProcessor, FrameBuffer
        
        print("Testing signer screen (camera → text) pipeline...")
        
        # Initialize components
        processor = CameraFrameProcessor(camera_index=0, fps=30)
        frame_buffer = FrameBuffer(max_frames=30)
        
        if not processor.start():
            print("✗ Failed to start camera")
            return False
        
        print("Capturing 10 frames...")
        for i in range(10):
            time.sleep(0.5)
            frame = processor.get_latest_frame()
            if frame:
                frame_buffer.add_frame(frame)
                hand = frame.hand_data.get("detected", False)
                face = frame.facial_data.get("detected", False)
                print(f"  Frame {i}: Hand detected={hand}, Face detected={face}")
        
        # Check trajectory
        trajectory = frame_buffer.get_hand_trajectory()
        if trajectory:
            print(f"✓ Hand trajectory detected ({len(trajectory)} points)")
        
        expression_seq = frame_buffer.get_expression_sequence()
        if expression_seq:
            print(f"✓ Expression sequence detected ({len(expression_seq)} frames)")
        
        processor.stop()
        print("✓ Test passed")
        return True
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("="*60)
    print("🧪 SENSE AI - MODEL INTEGRATION TESTS")
    print("="*60)
    
    tests = [
        ("Model Manager", test_model_manager),
        ("Camera Processor", test_camera_processor),
        ("WebSocket Server", test_websocket_server),
        ("Speaker Screen", test_speaker_screen),
        ("Signer Screen", test_signer_screen),
    ]
    
    results = {}
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"\n✗ Unexpected error in {test_name}: {e}")
            results[test_name] = False
    
    # Print summary
    print("\n" + "="*60)
    print("📊 TEST SUMMARY")
    print("="*60)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {status}: {test_name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    print("="*60)
    
    return 0 if passed == total else 1

if __name__ == "__main__":
    sys.exit(main())
