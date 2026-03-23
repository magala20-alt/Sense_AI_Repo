"""
Setup and verification script for Sense AI model integration
Run this to verify all components are properly configured
"""

import os
import sys
from pathlib import Path
import json

def check_directory_structure():
    """Verify required directories and files exist"""
    print("🔍 Checking directory structure...")
    
    base_path = Path(__file__).resolve().parent
    required_paths = [
        "gloss_mindspore_models/final_models (.mindir)/asl_translator.mindir",
        "gloss_mindspore_models/final_models/vocab.json",
        "gloss_mindspore_models/facial_recognition_model/FacialExpressionModel.mindir",
        "hand_recognition_model/hand_recognition_model.mindir",
        "sense_ai/services/model_loader.py",
        "sense_ai/services/camera_processor.py",
        "sense_ai/Backend/websocket_server.py",
        "sense_ai/screens/speaker_screen.py",
        "sense_ai/screens/signer_screen.py",
    ]
    
    all_exist = True
    for path in required_paths:
        full_path = base_path / path
        exists = "✓" if full_path.exists() else "✗"
        status = "EXISTS" if full_path.exists() else "MISSING"
        print(f"  {exists} {path}: {status}")
        if not full_path.exists():
            all_exist = False
    
    return all_exist

def check_python_dependencies():
    """Verify required Python packages are installed"""
    print("\n📦 Checking Python dependencies...")
    
    required_packages = {
        "cv2": "opencv-python",
        "PIL": "pillow",
        "websockets": "websockets",
        "speech_recognition": "speechrecognition",
        "numpy": "numpy",
    }
    
    missing = []
    for import_name, package_name in required_packages.items():
        try:
            __import__(import_name)
            print(f"  ✓ {package_name}: INSTALLED")
        except ImportError:
            print(f"  ✗ {package_name}: MISSING")
            missing.append(package_name)
    
    return len(missing) == 0, missing

def check_mindspore():
    """Check MindSpore installation"""
    print("\n🧠 Checking MindSpore...")
    
    try:
        import mindspore
        print(f"  ✓ MindSpore: INSTALLED (version {mindspore.__version__})")
        
        # Check device availability
        try:
            from mindspore import context
            context.set_context(mode=context.PYNATIVE_MODE)
            print(f"  ✓ MindSpore context: INITIALIZED")
            return True
        except Exception as e:
            print(f"  ✗ MindSpore context error: {e}")
            return False
    except ImportError:
        print(f"  ✗ MindSpore: NOT INSTALLED")
        print("    Install with: pip install mindspore")
        return False

def check_camera():
    """Test camera access"""
    print("\n📷 Checking camera...")
    
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            ret, frame = cap.read()
            cap.release()
            if ret:
                print(f"  ✓ Camera: ACCESSIBLE")
                return True
        print(f"  ✗ Camera: NOT ACCESSIBLE or returns no frames")
        return False
    except Exception as e:
        print(f"  ✗ Camera check failed: {e}")
        return False

def test_model_loading():
    """Test loading models"""
    print("\n🤖 Testing model loading...")
    
    try:
        sys.path.insert(0, str(Path(__file__).parent / "sense_ai"))
        from .sense_ai.Backend.services import get_model_manager
        
        print("  Loading models...")
        manager = get_model_manager()
        
        # Test text translation
        result = manager.translate_text_to_gloss("Hello")
        if result and "gloss" in result:
            print(f"  ✓ ASL Translation: {result['gloss']}")
        else:
            print(f"  ✗ ASL Translation failed: {result}")
            return False
        
        print("  ✓ Models loaded successfully")
        return True
    except Exception as e:
        print(f"  ✗ Model loading error: {e}")
        return False

def create_env_file():
    """Create .env file with configuration"""
    print("\n⚙️  Creating configuration file...")
    
    env_content = """# Sense AI Configuration

# Backend WebSocket Server
BACKEND_WS_URL=ws://localhost:8765
BACKEND_HOST=0.0.0.0
BACKEND_PORT=8765

# Avatar Configuration
SIGML_HOST=127.0.0.1
SIGML_PORT=4569

# Camera Configuration
CAMERA_INDEX=0
CAMERA_WIDTH=640
CAMERA_HEIGHT=480
CAMERA_FPS=30

# Model Configuration
MODEL_BATCH_SIZE=1
MODEL_DEVICE=CPU
MODEL_CONFIDENCE_THRESHOLD=0.5

# UI Configuration
UI_THEME=dark
UI_LANGUAGE=en
"""
    
    env_path = Path(__file__).parent / "sense_ai" / ".env"
    env_path.write_text(env_content)
    print(f"  ✓ Created {env_path}")

def print_next_steps():
    """Print next steps"""
    print("\n" + "="*60)
    print("📋 NEXT STEPS")
    print("="*60)
    
    print("\n1. Install missing dependencies (if any):")
    print("   cd sense_ai")
    print("   pip install -r requirements.txt")
    
    print("\n2. Start the backend WebSocket server:")
    print("   cd sense_ai/Backend")
    print("   python websocket_server.py")
    
    print("\n3. In another terminal, start the UI:")
    print("   cd sense_ai")
    print("   python main.py")
    
    print("\n4. Test the integration:")
    print("   - Speaker View: Type text and click 'Sign'")
    print("   - Signer View: Move hands in front of camera")
    
    print("\n📚 For more details, see: MODEL_INTEGRATION_GUIDE.md")
    print("="*60)

def main():
    """Run all checks"""
    print("="*60)
    print("🚀 SENSE AI MODEL INTEGRATION SETUP")
    print("="*60)
    
    checks = {
        "Directory Structure": check_directory_structure(),
        "Python Dependencies": check_python_dependencies()[0],
        "MindSpore": check_mindspore(),
        "Camera": check_camera(),
    }
    
    # Optional model loading test
    try:
        checks["Model Loading"] = test_model_loading()
    except Exception as e:
        print(f"⚠️  Model loading test skipped: {e}")
    
    # Create env file
    try:
        create_env_file()
    except Exception as e:
        print(f"⚠️  Could not create .env file: {e}")
    
    # Print summary
    print("\n" + "="*60)
    print("✅ VERIFICATION SUMMARY")
    print("="*60)
    
    all_passed = True
    for check_name, result in checks.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {status}: {check_name}")
        if not result:
            all_passed = False
    
    if all_passed:
        print("\n🎉 All checks passed! Ready to use Sense AI.")
    else:
        print("\n⚠️  Some checks failed. Please fix the issues above.")
    
    print_next_steps()
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())
