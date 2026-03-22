# ============================================
# SIMPLE TKINTER UI FOR ASL TRANSLATION
# ============================================

import tkinter as tk
from tkinter import scrolledtext
from asl_translator import ASLTranslator

class ASLTranslatorUI:
    def __init__(self, root, translator):
        self.root = root
        self.translator = translator
        
        root.title("English to ASL Gloss Translator")
        root.geometry("600x400")
        root.configure(bg='#f0f0f0')
        
        # Input label
        tk.Label(root, text="Enter English Sentence:", 
                 font=('Arial', 12), bg='#f0f0f0').pack(pady=10)
        
        # Input text area
        self.input_text = scrolledtext.ScrolledText(
            root, height=5, font=('Arial', 12), wrap=tk.WORD
        )
        self.input_text.pack(padx=20, pady=5, fill=tk.BOTH, expand=True)
        
        # Translate button
        tk.Button(root, text="Translate to ASL", 
                  command=self.translate,
                  font=('Arial', 12), bg='#4CAF50', fg='white',
                  padx=20, pady=5).pack(pady=10)
        
        # Output label
        tk.Label(root, text="ASL Gloss Translation:", 
                 font=('Arial', 12), bg='#f0f0f0').pack(pady=5)
        
        # Output text area
        self.output_text = scrolledtext.ScrolledText(
            root, height=5, font=('Arial', 12), wrap=tk.WORD, bg='#e8f4f5'
        )
        self.output_text.pack(padx=20, pady=5, fill=tk.BOTH, expand=True)
        
        # Status bar
        self.status = tk.Label(root, text="Ready", bd=1, relief=tk.SUNKEN, anchor=tk.W)
        self.status.pack(side=tk.BOTTOM, fill=tk.X)
    
    def translate(self):
        """Perform translation"""
        english = self.input_text.get("1.0", tk.END).strip()
        if not english:
            self.status.config(text="Please enter a sentence")
            return
        
        self.status.config(text="Translating...")
        self.root.update()
        
        try:
            asl_gloss = self.translator.translate_sentence(english)
            self.output_text.delete("1.0", tk.END)
            self.output_text.insert("1.0", asl_gloss)
            self.status.config(text="Translation complete")
        except Exception as e:
            self.status.config(text=f"Error: {e}")

# Run the app
if __name__ == "__main__":
    # Initialize translator (update path to your model folder)
    translator = ASLTranslator(model_dir="./asl_translation_model")
    
    root = tk.Tk()
    app = ASLTranslatorUI(root, translator)
    root.mainloop()