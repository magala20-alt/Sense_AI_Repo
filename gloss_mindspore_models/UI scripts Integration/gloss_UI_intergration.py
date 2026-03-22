# ============================================
# ASL TRANSLATION INFERENCE MODULE
# Use this in your Tkinter UI
# ============================================

import json
import numpy as np
import mindspore as ms
from mindspore import nn, ops, Tensor
from mindspore import context
import os

class ASLTranslator:
    """English-to-ASL Gloss Translator"""
    
    def __init__(self, model_dir="."):
        """
        Initialize the translator
        Args:
            model_dir: Directory containing model.ckpt and vocab.json
        """
        # Set to CPU mode (works on any computer)
        context.set_context(mode=context.PYNATIVE_MODE, device_target="CPU")
        
        # Load vocabulary
        vocab_path = os.path.join(model_dir, "vocab.json")
        with open(vocab_path, 'r') as f:
            vocab = json.load(f)
        
        self.en_word2idx = vocab['en_word2idx']
        self.gloss_word2idx = vocab['gloss_word2idx']
        
        # Build reverse mapping for ASL gloss
        self.idx_to_gloss = {v: k for k, v in self.gloss_word2idx.items()}
        
        # Define model architecture
        class SimpleModel(nn.Cell):
            def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes):
                super().__init__()
                self.embedding = nn.Embedding(vocab_size, embed_dim)
                self.rnn = nn.GRU(embed_dim, hidden_dim, batch_first=True)
                self.fc = nn.Dense(hidden_dim, num_classes)
                
            def construct(self, x):
                emb = self.embedding(x)
                _, hidden = self.rnn(emb)
                return self.fc(hidden[-1])
        
        # Model parameters (must match training)
        self.EMBED_DIM = 128
        self.HIDDEN_DIM = 256
        self.VOCAB_SIZE = len(self.gloss_word2idx)
        self.ENC_VOCAB_SIZE = len(self.en_word2idx)
        
        # Initialize model
        self.model = SimpleModel(
            self.ENC_VOCAB_SIZE, 
            self.EMBED_DIM, 
            self.HIDDEN_DIM, 
            self.VOCAB_SIZE
        )
        
        # Load trained weights
        model_path = os.path.join(model_dir, "model.ckpt")
        param_dict = ms.load_checkpoint(model_path)
        ms.load_param_into_net(self.model, param_dict)
        self.model.set_train(False)
        
        print(f"✅ Model loaded! Vocab size: {len(self.en_word2idx)} English words")
    
    def translate_word(self, word):
        """Translate a single English word to ASL gloss"""
        word_lower = word.lower()
        
        if word_lower not in self.en_word2idx:
            return f"[{word}]"
        
        src = Tensor(np.array([[self.en_word2idx[word_lower]]], dtype=np.int32))
        output = self.model(src)
        pred_idx = output.argmax(1).asnumpy()[0]
        
        # Find ASL gloss
        for gloss_word, idx in self.gloss_word2idx.items():
            if idx == pred_idx:
                return gloss_word
        
        return f"[{pred_idx}]"
    
    def translate_sentence(self, sentence):
        """Translate full English sentence to ASL gloss"""
        words = sentence.lower().split()
        gloss_words = []
        
        # Skip punctuation and special tokens
        skip_words = ['.', ',', ';', ':', '?', '!', '<PAD>', '<SOS>', '<EOS>', '<UNK>']
        
        for w in words:
            if w in self.en_word2idx:
                gloss = self.translate_word(w)
                if gloss not in skip_words:
                    gloss_words.append(gloss)
            else:
                gloss_words.append(f"[{w}]")
        
        return ' '.join(gloss_words)
    
    def translate_batch(self, sentences):
        """Translate multiple sentences"""
        return [self.translate_sentence(s) for s in sentences]

# ============================================
# EXAMPLE USAGE
# ============================================

if __name__ == "__main__":
    # Initialize translator
    translator = ASLTranslator(model_dir=".")  # or specify path
    
    # Test translations
    test_sentences = [
        "I love you",
        "We need help",
        "Family is important",
        "I go to the store"
    ]
    
    print("\n🎯 Translations:")
    print("="*50)
    for sent in test_sentences:
        result = translator.translate_sentence(sent)
        print(f"EN: {sent}")
        print(f"ASL: {result}")
        print()