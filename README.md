# 🚀 Text-to-Code Generation with TensorFlow, Hugging Face & MBPP

Transforming natural language descriptions into executable Python code using the powerful **CodeT5** model. This project implements a robust, scalable solution for **text-to-code generation** using **TensorFlow**, **Hugging Face Transformers**, and the **Mostly Basic Python Programming (MBPP)** dataset.

## 📌 Project Overview

This project focuses on fine-tuning **CodeT5**, a transformer-based model specialized for code generation tasks, to convert natural language prompts into Python code. It leverages modern deep-learning techniques such as **mixed precision training**, **distributed strategies**, and **custom TensorFlow training loops**.

### ✨ Key Features
- **Advanced Code Generation**: Converts natural language prompts into executable Python code.
- **Custom Training Loops**: Fine-tunes the CodeT5 model using a customized TensorFlow training pipeline within a Jupyter Notebook.
- **Mixed Precision & XLA Optimization**: Accelerates training and reduces memory usage using hardware-specific optimizations.
- **Multi-Device Strategy**: Supports single-GPU, multi-GPU, and TPU training with TensorFlow's `tf.distribute` API.
- **Dynamic Inference**: Perform inference on real-time inputs with flexible decoding methods (e.g., Top-p sampling).

## 📊 Model & Dataset Details

### 1. **CodeT5 Model**
- Based on T5's encoder-decoder architecture.
- Pre-trained on **CodeSearchNet** across multiple programming languages.
- Fine-tuned for Python code generation tasks.

### 2. **MBPP Dataset**
- Contains **1,000** Python problems for evaluating code generation models.
- Includes natural language descriptions, Python code solutions, and automated test cases.
- This project focuses on the verified subset of **426 problems** for higher accuracy.

## 🛠️ Implementation Breakdown

### 1. **Setup & Configuration**
- Ensure proper environment setup with TensorFlow and Hugging Face libraries.
- Supports mixed precision and XLA optimization for faster execution.

```bash
pip install tensorflow transformers datasets
```

### 2. **Data Processing**
- Download and preprocess the MBPP dataset using Hugging Face's `datasets` library.
- Prepare input-output pairs for model training within the notebook.

### 3. **Model Fine-Tuning**
- Implements a custom training loop with TensorFlow's `GradientTape` for precise control.
- Supports **warm-up learning rate** and **AdamW** optimizer.

```python
from transformers import TFT5ForConditionalGeneration

model = TFT5ForConditionalGeneration.from_pretrained('Salesforce/codet5-base')
```

### 4. **Optimization Strategies**
- **Mixed Precision**: Uses `float16` for performance boost.
- **XLA (Accelerated Linear Algebra)**: Enables faster graph execution.

### 5. **Custom Training Loop**
- Implemented entirely within the Jupyter Notebook using TensorFlow's `tf.function`.
- Distributed across multiple GPUs using `tf.distribute.MirroredStrategy()`.
- Includes detailed logging and checkpoint management.

```python
with strategy.scope():
    model = create_model()
    optimizer = create_optimizer()
```

### 6. **Inference & Code Generation**
- Supports both dataset-based and custom input inference directly from the notebook.

```python
def predict_from_text(text):
    query = "Generate Python: " + text
    generated_code = model.generate(input_ids, max_length=256)
    print(tokenizer.decode(generated_code[0]))
```

## 📈 Performance Insights

- **Training Efficiency**: Leveraged mixed precision and distributed training to improve speed and reduce memory overhead.
- **Evaluation**: Achieved low validation loss and high generation accuracy.

## 📤 How to Run

1. Clone the repository and set up the environment:

```bash
git clone https://github.com/Shreyash-Gaur/TensorFlow_Python_Code_Generation.git
cd TensorFlow_Python_Code_Generation
pip install -r requirements.txt
```

2. Open the Jupyter Notebook and execute the cells step-by-step:

```bash
jupyter notebook Python_Code_Generation.ipynb
```

3. Perform inference using the provided methods:

```python
predict_from_text(args, "Write a function to concatenate two dictionary"); print()
```

## 📚 Insights & Takeaways

- **Custom TensorFlow Loops**: Provides granular control over training and debugging.
- **Task Prefixing**: Improved model generalization by adding task-specific prompts.
- **Scalability**: Compatible with multi-GPU and TPU environments.

## 📌 Future Scope

- Enhance generation quality with larger CodeT5 models.
- Explore alternative code benchmarks (e.g., HumanEval, APPS).
- Integrate with real-world AI coding assistants.

## 🧑‍💻 Contributing

Contributions are welcome! Feel free to fork the repository and submit pull requests.

## 📄 License

This project is licensed under the **MIT License**.

---

🌟 **If you found this project useful, give it a star!**

