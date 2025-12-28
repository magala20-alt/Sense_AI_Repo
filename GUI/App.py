import tkinter as tk
from tkinter import ttk, filedialog
import cv2
from PIL import Image, ImageTk, ImageDraw
import threading
import numpy as np
#import pyttsx3  # For text-to-speech
#import speech_recognition as sr  # For speech recognition

class ASLTranslatorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Sense.AI - ASL Translator")
        self.root.geometry("1600x900")
        #self.root.configure(bg='#1e293b')
        
        # Initialize models (placeholder for now)
        self.baseline_model = None  # Will be: ASLTranslator("models/baseline/model.ckpt")
        self.enhanced_model = None  # Will be: ASLTranslator("models/enhanced/model.ckpt")
        self.feature_extractor = None  # Will be: FeatureExtractor()
        
        # State variables
        self.is_recording = False
        self.video_source = None
        self.cap = None
        
        self.setup_ui()

     
    def setup_ui(self):
        # Get screen dimensions for gradient
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        
        # Create gradient background
        gradient_img = self.create_horizontal_gradient(
            screen_width, 
            screen_height,
            "#7F0BB5",  # Dark blue-gray (left)
            "#29298C",  # Purple (middle) 
            "#1e293b"   # Dark blue-gray (right)
        )
        
        # Create background canvas for gradient
        self.bg_canvas = tk.Canvas(self.root, highlightthickness=0)
        self.bg_canvas.pack(fill='both', expand=True)
        
        # Display gradient
        self.gradient_photo = ImageTk.PhotoImage(gradient_img)
        self.bg_canvas.create_image(0, 0, image=self.gradient_photo, anchor='nw')
        
        # Create scrollable canvas ON TOP of gradient
        self.canvas = tk.Canvas(self.bg_canvas, bg='#29298C', highlightthickness=0)
        scrollbar = tk.Scrollbar(self.root, orient="vertical", command=self.canvas.yview)
        
        # Scrollable frame - match the base color
        self.scrollable_frame = tk.Frame(self.canvas, bg='#29298C')
        
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )
        
        # Create window with proper width
        self.canvas_window = self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        
        # Bind canvas width changes
        def configure_canvas_window(event):
            self.canvas.itemconfig(self.canvas_window, width=event.width)
        
        self.canvas.bind('<Configure>', configure_canvas_window)
        self.canvas.configure(yscrollcommand=scrollbar.set)
        
        # Pack scrollbar and canvas
        scrollbar.pack(side="right", fill="y")
        self.canvas.pack(side="left", fill="both", expand=True)
        
        # Mouse wheel scrolling
        def _on_mousewheel(event):
            self.canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        
        self.canvas.bind_all("<MouseWheel>", _on_mousewheel)
        self.canvas.bind_all("<Button-4>", lambda e: self.canvas.yview_scroll(-1, "units"))
        self.canvas.bind_all("<Button-5>", lambda e: self.canvas.yview_scroll(1, "units"))
        
        # Main container - use matching background
        main_frame = tk.Frame(self.scrollable_frame, bg='#29298C')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Left side - Video and controls
        left_frame = tk.Frame(main_frame, bg='#29298C')
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        # Controls frame
        controls_container = tk.Frame(left_frame, bg='#29298C')
        controls_container.pack(fill=tk.X)
        controls_frame = tk.Frame(controls_container, bg='#29298C', relief=tk.RAISED, bd=2)
        controls_frame.pack(fill=tk.X, pady=10)
        
        # Buttons
        btn_frame = tk.Frame(controls_frame, bg='#29298C')
        btn_frame.pack(pady=15)
        
       
        self.start_btn = tk.Button(
            btn_frame,
            text="▶ Start Recognition",
            command=self.start_recording,
            bg='#8b5cf6',
            fg='white',
            font=('Arial', 12, 'bold'),
            padx=20,
            pady=10,
            relief=tk.FLAT,
            cursor='hand2'
        )
        self.start_btn.pack(side=tk.LEFT, padx=5)
        
        self.stop_btn = tk.Button(
            btn_frame,
            text="■ Stop",
            command=self.stop_recording,
            bg='#ef4444',
            fg='white',
            font=('Arial', 12, 'bold'),
            padx=20,
            pady=10,
            relief=tk.FLAT,
            cursor='hand2',
            state=tk.DISABLED
        )
        self.stop_btn.pack(side=tk.LEFT, padx=5)
        
        upload_btn = tk.Button(
            btn_frame,
            text="📁 Upload Video",
            command=self.upload_video,
            bg='#475569',
            fg='white',
            font=('Arial', 12, 'bold'),
            padx=20,
            pady=10,
            relief=tk.FLAT,
            cursor='hand2'
        )
        upload_btn.pack(side=tk.LEFT, padx=5)
         
         # Comparison checkbox
        self.compare_var = tk.BooleanVar()
        compare_check = tk.Checkbutton(
            btn_frame,
            text="Compare Models",
            variable=self.compare_var,
            bg='#29298C',
            fg='white',
            font=('Arial', 11),
            selectcolor='#8b5cf6',
            activebackground='#334155',
            activeforeground='white'
        )
        compare_check.pack(side=tk.LEFT, padx=20)
        
        # Microphone button for speech input
        self.mic_btn = tk.Button(
            btn_frame,
            text="🎤 Voice Input",
            command=self.toggle_microphone,
            bg='#2196F3',
            fg='white',
            font=('Arial', 12, 'bold'),
            padx=20,
            pady=10,
            relief=tk.FLAT,
            cursor='hand2'
        )
        self.mic_btn.pack(side=tk.LEFT, padx=5)
        
        # Status indicator for listening
        self.mic_status = tk.Label(
            btn_frame,
            text="off",
            bg='#334155',
            fg='#4CAF50',
            font=('Arial', 10, 'bold')
        )
        self.mic_status.pack(side=tk.LEFT, padx=10)

         # Translation display
        translation_frame = tk.LabelFrame(
            left_frame,
            text="Translation",
            bg='#29298C',
            fg='white',
            font=('Arial', 14, 'bold'),
            relief=tk.RAISED,
            bd=2
        )
        translation_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        self.translation_label = tk.Label(
            translation_frame,
            text="Ready to translate...",
            font=('Arial', 16, 'bold'),
            fg='#a78bfa',
            bg='#29298C',
            pady=30
        )
        self.translation_label.pack()
         
        # Video display
        video_container = tk.Frame(left_frame, bg='#000000', width=800, height=500)
        video_container.pack(pady=10)
        video_container.pack_propagate(False)
    
        self.video_frame = tk.Label(video_container, bg='#000000', width=800, height=500)
        self.video_frame.pack()

        # Comparison display
        self.comparison_frame = tk.Frame(translation_frame, bg='#334155')
        
        comp_left = tk.Frame(self.comparison_frame, bg='#29298C', relief=tk.RAISED, bd=1)
        comp_left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=10)
        
        tk.Label(comp_left, text="Baseline (Hands Only)", bg='#475569', fg='#94a3b8', font=('Arial', 10)).pack(pady=5)
        self.baseline_result = tk.Label(comp_left, text="--", bg='#475569', fg='white', font=('Arial', 16, 'bold'))
        self.baseline_result.pack(pady=10)
        tk.Label(comp_left, text="Accuracy: 72%", bg='#475569', fg='#94a3b8', font=('Arial', 9)).pack()
        
        comp_right = tk.Frame(self.comparison_frame, bg='#7c3aed', relief=tk.RAISED, bd=1)
        comp_right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=10)
        
        tk.Label(comp_right, text="Enhanced (+ Facial)", bg='#7c3aed', fg='#ddd6fe', font=('Arial', 10)).pack(pady=5)
        self.enhanced_result = tk.Label(comp_right, text="--", bg='#7c3aed', fg='white', font=('Arial', 16, 'bold'))
        self.enhanced_result.pack(pady=10)
        tk.Label(comp_right, text="Accuracy: 87%", bg='#7c3aed', fg='#ddd6fe', font=('Arial', 9)).pack()
        # Speak button - converts translation to speech
        self.speak_btn = tk.Button(
            translation_frame,
            text="🔊 Speak Translation",
            command=self.speak_translation,
            bg="#10A524",
            fg="#FFFFFF",
            font=('Arial', 11, 'bold'),
            padx=10,
            pady=5,
            relief=tk.FLAT,
            cursor='hand2',
            state=tk.DISABLED  # Enabled when there's text to speak
        )
        self.speak_btn.pack(side=tk.RIGHT, padx=5)
        self.confidence_label = tk.Label(
            translation_frame,
            text="Confidence: --",
            font=('Arial', 14),
            fg="#ffffff",
            bg='#29298C',
            pady=10
        )
        self.confidence_label.pack()
        # Right side - Facial features
        right_frame = tk.Frame(main_frame, bg='#29298C', width=450)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, padx=(10, 0))
        right_frame.pack_propagate(False)
        
        # Facial features panel
        features_frame = tk.LabelFrame(
            right_frame,
            text="Detected Facial Features",
            bg='#29298C',
            fg='white',
            font=('Arial', 12, 'bold'),
            relief=tk.RAISED,
            bd=2
        )
        features_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        # Eyebrows
        eyebrow_frame = tk.Frame(features_frame, bg='#475569', relief=tk.RAISED, bd=1)
        eyebrow_frame.pack(fill=tk.X, padx=10, pady=10)
        
        tk.Label(eyebrow_frame, text="Eyebrows", bg='#475569', fg='#94a3b8', font=('Arial', 10)).pack(anchor='w', padx=10, pady=(5, 0))
        self.eyebrow_label = tk.Label(eyebrow_frame, text="NEUTRAL", bg='#475569', fg='#60a5fa', font=('Arial', 16, 'bold'))
        self.eyebrow_label.pack(anchor='w', padx=10, pady=5)
        self.eyebrow_desc = tk.Label(eyebrow_frame, text="→ Statement", bg='#475569', fg='#64748b', font=('Arial', 9))
        self.eyebrow_desc.pack(anchor='w', padx=10, pady=(0, 5))
        
        # Eyes
        eyes_frame = tk.Frame(features_frame, bg='#475569', relief=tk.RAISED, bd=1)
        eyes_frame.pack(fill=tk.X, padx=10, pady=10)
        
        tk.Label(eyes_frame, text="Eye Aperture", bg='#475569', fg='#94a3b8', font=('Arial', 10)).pack(anchor='w', padx=10, pady=(5, 0))
        self.eyes_label = tk.Label(eyes_frame, text="NORMAL", bg='#475569', fg='#94a3b8', font=('Arial', 16, 'bold'))
        self.eyes_label.pack(anchor='w', padx=10, pady=5)
        self.eyes_desc = tk.Label(eyes_frame, text="→ Neutral state", bg='#475569', fg='#64748b', font=('Arial', 9))
        self.eyes_desc.pack(anchor='w', padx=10, pady=(0, 5))
        
        # Mouth
        mouth_frame = tk.Frame(features_frame, bg='#475569', relief=tk.RAISED, bd=1)
        mouth_frame.pack(fill=tk.X, padx=10, pady=10)
        
        tk.Label(mouth_frame, text="Mouth Shape", bg='#475569', fg='#94a3b8', font=('Arial', 10)).pack(anchor='w', padx=10, pady=(5, 0))
        self.mouth_label = tk.Label(mouth_frame, text="NEUTRAL", bg='#475569', fg='#94a3b8', font=('Arial', 16, 'bold'))
        self.mouth_label.pack(anchor='w', padx=10, pady=5)
        self.mouth_desc = tk.Label(mouth_frame, text="→ Normal articulation", bg='#475569', fg='#64748b', font=('Arial', 9))
        self.mouth_desc.pack(anchor='w', padx=10, pady=(0, 5))
        
        # Statistics panel
        stats_frame = tk.LabelFrame(
            right_frame,
            text="Performance Metrics",
            bg='#334155',
            fg='white',
            font=('Arial', 12, 'bold'),
            relief=tk.RAISED,
            bd=2
        )
        stats_frame.pack(fill=tk.X, pady=(0, 10))
        
        stats_data = [
            ("Processing Speed:", "30 FPS"),
            ("Signs Recognized:", "50"),
            ("Model Accuracy:", "87%"),
            ("Improvement:", "+15%")
        ]
        
        for label, value in stats_data:
            row = tk.Frame(stats_frame, bg='#334155')
            row.pack(fill=tk.X, padx=10, pady=5)
            tk.Label(row, text=label, bg='#334155', fg='#94a3b8', font=('Arial', 10)).pack(side=tk.LEFT)
            tk.Label(row, text=value, bg='#334155', fg='white', font=('Arial', 10, 'bold')).pack(side=tk.RIGHT)
            
    def start_recording(self):
        self.is_recording = True
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        
        # Start webcam
        self.cap = cv2.VideoCapture(0)
        self.update_frame()
    
    def stop_recording(self):
        self.is_recording = False
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        
        if self.cap:
            self.cap.release()
            # Clear the video frame
            self.video_frame.configure(image='')
    
    def update_frame(self):
        if self.is_recording and self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                # Process frame (placeholder - will call your model)
                self.process_frame(frame)
                
                # Convert to display format
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (800, 600))
                img = Image.fromarray(frame)
                imgtk = ImageTk.PhotoImage(image=img)
                self.video_frame.imgtk = imgtk
                self.video_frame.configure(image=imgtk)
            
            # Schedule next frame
            self.root.after(33, self.update_frame)  # ~30 FPS
    def speak_translation(self):
        """Convert the current translation text to speech"""
        pass

    def toggle_microphone(self):
        """Toggle microphone listening for speech input"""
        pass

    def process_frame(self, frame):
        """Process the frame to extract features and perform recognition"""
        
        # TODO: Replace with actual model inference
        # For now, simulate detection for testing
        
        # Simulated facial features (replace with actual extraction)
        import random
        simulated_features = {
            'eyebrows': random.choice(['RAISED', 'NEUTRAL', 'FURROWED']),
            'eyes': random.choice(['WIDE', 'NORMAL', 'SQUINTED']),
            'mouth': random.choice(['OPEN', 'NEUTRAL', 'CLOSED'])
        }
        
        # Update facial display
        self.update_facial_display(simulated_features)
        
        # Simulated translation (replace with actual model output)
        if random.random() > 0.7:  # Update occasionally
            signs = ['HELLO', 'THANK YOU', 'PLEASE', 'YES', 'NO']
            predicted_sign = random.choice(signs)
            confidence = random.randint(70, 95)
            
            if self.compare_var.get():
                # Show comparison
                self.show_comparison(predicted_sign, random.choice(signs))
            else:
                # Show single result
                self.translation_label.config(text=predicted_sign)
                self.confidence_label.config(text=f"Confidence: {confidence}%")

    def update_facial_display(self, features):
        """Update the facial features display based on extracted features"""
        
        # Color mapping
        colors = {
            'RAISED': '#60a5fa',
            'NEUTRAL': '#94a3b8',
            'FURROWED': '#fb923c',
            'WIDE': '#a78bfa',
            'NORMAL': '#94a3b8',
            'SQUINTED': '#fbbf24',
            'OPEN': '#34d399',
            'CLOSED': '#f87171'
        }
        
        # Description mapping
        eyebrow_desc = {
            'RAISED': '→ Question/Topic marker',
            'NEUTRAL': '→ Statement',
            'FURROWED': '→ WH-question/Negation'
        }
        
        eyes_desc = {
            'WIDE': '→ Emphasis/Surprise',
            'NORMAL': '→ Neutral state',
            'SQUINTED': '→ Intensity marker'
        }
        
        mouth_desc = {
            'OPEN': '→ Mouth morpheme detected',
            'NEUTRAL': '→ Normal articulation',
            'CLOSED': '→ Size modifier'
        }
        
        # Update eyebrows
        eyebrow_state = features.get('eyebrows', 'NEUTRAL')
        self.eyebrow_label.config(text=eyebrow_state, fg=colors.get(eyebrow_state, '#94a3b8'))
        self.eyebrow_desc.config(text=eyebrow_desc.get(eyebrow_state, ''))
        
        # Update eyes
        eye_state = features.get('eyes', 'NORMAL')
        self.eyes_label.config(text=eye_state, fg=colors.get(eye_state, '#94a3b8'))
        self.eyes_desc.config(text=eyes_desc.get(eye_state, ''))
        
        # Update mouth
        mouth_state = features.get('mouth', 'NEUTRAL')
        self.mouth_label.config(text=mouth_state, fg=colors.get(mouth_state, '#94a3b8'))
        self.mouth_desc.config(text=mouth_desc.get(mouth_state, ''))

    def show_comparison(self, enhanced_result, baseline_result):
        """Show comparison between baseline and enhanced models"""
        self.comparison_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        self.translation_label.pack_forget()
        self.confidence_label.pack_forget()
        
        self.baseline_result.config(text=baseline_result)
        self.enhanced_result.config(text=enhanced_result)

    def upload_video(self):
        filename = filedialog.askopenfilename(
            title="Select ASL Video",
            filetypes=[("Video files", "*.mp4 *.avi *.mov"), ("All files", "*.*")]
        )
        if filename:
            # Process uploaded video
            self.process_video_file(filename)
    
    def process_video_file(self, filepath):
        """Process a video file"""
        # Stop any ongoing recording
        if self.is_recording:
            self.stop_recording()
        
        # Open video file
        self.cap = cv2.VideoCapture(filepath)
        if not self.cap.isOpened():
            print(f"Error: Could not open video file {filepath}")
            return
        
        self.is_recording = True
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        
        # Start processing frames
        self.update_frame()

    # other utilities as needed
    def hex_to_rgb(self, hex_color):
        """Convert hex color to RGB tuple"""
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

    def create_horizontal_gradient(self, width, height, color1, color2, color3):
        """
        Create a horizontal gradient with 3 colors
        Transitions: color1 -> color2 -> color3
        """
        img = Image.new('RGB', (width, height))
        draw = img.load()
        
        # Convert hex to RGB
        rgb1 = self.hex_to_rgb(color1)
        rgb2 = self.hex_to_rgb(color2)
        rgb3 = self.hex_to_rgb(color3)

        # Split into two segments
        mid_point = width // 2
        
        # First half: color1 to color2
        for x in range(mid_point):
            ratio = x / mid_point
            r = int(rgb1[0] * (1 - ratio) + rgb2[0] * ratio)
            g = int(rgb1[1] * (1 - ratio) + rgb2[1] * ratio)
            b = int(rgb1[2] * (1 - ratio) + rgb2[2] * ratio)
            
            for y in range(height):
                draw[x, y] = (r, g, b)
        
        # Second half: color2 to color3
        for x in range(mid_point, width):
            ratio = (x - mid_point) / (width - mid_point)
            r = int(rgb2[0] * (1 - ratio) + rgb3[0] * ratio)
            g = int(rgb2[1] * (1 - ratio) + rgb3[1] * ratio)
            b = int(rgb2[2] * (1 - ratio) + rgb3[2] * ratio)
            
            for y in range(height):
                draw[x, y] = (r, g, b)
        
        return img
if __name__ == "__main__":
    root = tk.Tk()
    app = ASLTranslatorApp(root) 
    root.mainloop()