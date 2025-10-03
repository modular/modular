# Custom embedding architecture example

This repository demonstrates how to create your own custom embedding architecture
for the MAX platform. It provides a complete example implementation of BERT and
RoBERTa architectures that can be used as templates for building your own
embedding models.

## Component overview

### BERT module (`bert/`)

- Implements the complete BERT architecture for embeddings generation
- Supports both `BertForMaskedLM` and `BertModel` architectures
- Handles sentence-transformers models
- Custom tokenizer to ensure EOS token availability

### RoBERTa module (`roberta/`)

- Extends BERT implementation with RoBERTa-specific adaptations
- Supports DistilRoBERTa variants
- Inherits most functionality from BERT with custom weight mappings

### Common components (`common/`)

- **attention.py**: Reusable multi-head self-attention implementation
- **embeddings.py**: Word, position, and token type embeddings
- **feedforward.py**: MLP layers with residual connections
- **transformer.py**: Complete transformer encoder stack
- **base_model.py**: Base class for encoder models
- **weight_adapters.py**: Pattern-based weight transformation system

### Architecture flow

1. Input tokens → Embedding layer (word + position + token type)
2. Embedded tokens → Transformer encoder (multiple layers)
3. Encoder output → Optional pooling layer
4. Final output → Dense embeddings vector

## Usage with Pixi

This project uses Pixi for environment management and task automation.

### Setup

1. Install Pixi if you haven't already:

   ```bash
   curl -fsSL https://pixi.sh/install.sh | bash
   ```

2. Install the project environment:

   ```bash
   pixi install
   ```

### Running encode tasks

Basic encoding with default prompt:

```bash
pixi run encode
```

Encode custom text:

```bash
pixi run encode "Your text to embed here"
```

Use the large model for more accurate embeddings:

```bash
pixi run encode-large "Your text here"
```
