# 🚀 Scalable GPT-like LLM Training Framework 🧠

Welcome to our cutting-edge framework for training Generative Pre-trained Transformer (GPT)-like Large Language Models (LLMs) from scratch! ✨ This isn't just another LLM script; it's a powerhouse built with **scalability** at its core. We leverage the latest PyTorch magic to train massive models on vast datasets, efficiently and effectively.

Our model is a classic decoder-only Transformer, but the training pipeline is packed with modern features like custom SentencePiece tokenization, lightning-fast mixed-precision training, and the incredible Fully Sharded Data Parallel (FSDP) for conquering multi-GPU setups.

---

## 📊 Performance Snapshot

*   **HellaSwag Benchmark:** 🎯 Our model achieved an accuracy of **25%** on the HellaSwag common-sense NLI benchmark. Given HellaSwag is a 4-choice task, 25% is the baseline for random guessing. This result provides a transparent starting point for this configuration and dataset.
*   **Training & Validation Loss:** 📉 See our model learn! The plot below shows the training and validation loss curves, illustrating its progress over time.

    ![Training and Validation Loss](Screenshot 2025-05-22 153211.png])
    *(Heads up! You'll need to replace `[link_to_your_loss_plot_image.png]` with the actual path/URL to your awesome loss plot image!)*

---

## ✨ Features Packed In

*   🧱 **Custom GPT-like Model:** Core Transformer blocks (Multi-Head Attention, LayerNorm, GELU, FeedForward) built from the ground up.
*   🗣️ **SentencePiece Tokenization:** Smooth integration with `sentencepiece` for your custom vocabularies.
*   🚚 **Efficient Data Handling:** Custom `Dataset` and `DataLoader` designed for speed and memory efficiency.
*   🚀 **Supercharged Scalable Training Loop:**
    *   🔗 **Fully Sharded Data Parallel (FSDP):** Distribute model parameters, gradients, and optimizer states across GPUs. Train models bigger than a single GPU!
    *   ⚡ **Mixed Precision Training (AMP):** Faster training and less memory usage with FP16/BF16, managed by `torch.amp.autocast` and `GradScaler`.
    *   ⚙️ **`torch.compile`:** Squeeze out extra performance with JIT compilation.
    *   ➕ **Gradient Accumulation:** Simulate larger batch sizes without the memory headache.
    *   🔥 **TF32 Support:** Even more speed on compatible NVIDIA GPUs.
*   📈 **Smart Learning Rate Scheduling:** Linear warmup followed by a smooth cosine decay.
*   💾 **Checkpointing:** Never lose your progress! Saves model and optimizer states.
*   📝 **Evaluation & Sampling:** Built-in functions to check loss and generate text samples.

---

## 🏗️ Model Architecture Unveiled

Our `GPTModel` is a decoder-only Transformer, the workhorse behind many successful LLMs. Here’s a peek under the hood:

1.  **Token Embeddings (`nn.Embedding`):** 🔤 Turns words (tokens) into numbers (vectors).
2.  **Positional Embeddings (`nn.Embedding`):** 📍 Gives the model a sense of word order.
3.  **Transformer Blocks (`TransformerBlock`):** 🧠 The brain of the operation, stacked multiple times (`n_layers`). Each block contains:
    *   **Layer Normalization (`LayerNorm`):** Keeps training stable (pre-norm style).
    *   **Multi-Head Self-Attention (`MultiHeadAttention`):** 🧐 Lets the model weigh which words are important for understanding other words. It's causal, meaning it only looks at past words.
    *   **Residual Connections & Dropout:** Helps with learning and prevents overfitting.
    *   **Feed-Forward Network (`FeedForward`):** 💡 More number-crunching with a GELU activation.
4.  **Final Layer Normalization (`LayerNorm`):** One last normalization pass.
5.  **Output Linear Layer (`nn.Linear`):** 🔮 Predicts the next word!

*(Optional: Consider adding a simple GIF or static image here illustrating a Transformer block or data flow if you have one!)*
`[Optional: Insert a simple diagram of a Transformer block here]`

---

## 🚀 Our Secret Sauce: Focus on Scalability!

Training big models needs big power. Here’s how we make it happen:

1.  **🔗 Fully Sharded Data Parallel (FSDP):**
    *   **The Gist:** Instead of copying the whole model to every GPU (like DDP), FSDP smartly breaks up (shards) the model, its gradients, and optimizer states across all GPUs.
    *   **The Win:** Dramatically less memory needed per GPU, letting you train truly ENORMOUS models.
    *   **In Action:** We use `torch.distributed.fsdp.FullyShardedDataParallel` with auto-wrapping and a `MixedPrecision` policy.
    *   **Bonus:** CPU offloading is an option for even bigger models (though we keep it on GPU for speed here).

    `[Optional: Insert a GIF here comparing DDP memory vs FSDP memory, or a "data sharding" animation]`

2.  **⚡ Mixed Precision Training (AMP):**
    *   **The Gist:** Uses faster, smaller number types (like FP16) for many calculations, with `GradScaler` to keep things numerically stable.
    *   **The Win:** Speeds up training and slashes memory use on GPUs with Tensor Cores.
    *   **In Action:** `autocast` and `GradScaler` work hand-in-hand with FSDP's `MixedPrecision` policy.

3.  **⚙️ `torch.compile()`:**
    *   **The Gist:** PyTorch's JIT compiler turns your Python model code into a super-optimized version.
    *   **The Win:** Can give a nice speed boost by fusing operations and reducing overhead.

4.  **➕ Gradient Accumulation:**
    *   **The Gist:** Pretend you have a bigger batch size by doing several forward/backward passes before updating weights.
    *   **The Win:** Better training stability with large effective batch sizes, without needing tons of GPU RAM.

5.  ** eficiente Data Handling for Huge Datasets:**
    *   **The Gist:** We process data chunk by chunk (file by file), so you don't need to load everything into RAM at once.
    *   **The Win:** Train on datasets that are way bigger than your system's memory. Pinned memory and non-blocking transfers help speed things up.

6.  **🔥 TF32 Precision on NVIDIA Ampere+ GPUs:**
    *   **The Gist:** A sneaky way to get faster matrix math with almost no precision loss.
    *   **The Win:** A free speed boost on compatible hardware!

These features work together to create a training setup that's ready for serious LLM action!

---

##📋 Prerequisites

*   Python 3.8+
*   PyTorch (>=1.13, **ideally 2.0+** for `torch.compile` & latest FSDP goodies)
*   SentencePiece (`sentencepiece`)
*   Matplotlib (for those nice loss plots 📈)
*   CUDA-enabled GPU(s) (essential for GPU training!)
*   A pre-trained SentencePiece model file (e.g., `test_bpe.model`)
*   Your dataset: `.txt` files, ready to go!

You can install the Python gang using:
```bash
pip install torch torchvision torchaudio sentencepiece matplotlib
