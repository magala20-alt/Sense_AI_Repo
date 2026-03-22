# ASL Translation Model - Instructions

## Files Included
- model.ckpt - Trained model weights (best model from epoch 6)
- vocab.json - English and ASL gloss vocabularies
- model_info.json - Model architecture parameters
- training_history.json - Training loss history

## Requirements
pip install mindspore numpy

## Quick Test
```python
from asl_translator import ASLTranslator

translator = ASLTranslator(".")
print(translator.translate_sentence("I love you"))
# Output: X-I LOVE X-YOU