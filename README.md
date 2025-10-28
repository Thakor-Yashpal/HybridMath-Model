# HybridMath Model

![Python Version](https://img.shields.io/badge/python-3.10+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)
![Transformers](https://img.shields.io/badge/🤗%20Transformers-FFD21E?style=flat&logo=huggingface&logoColor=black)
![Gradio](https://img.shields.io/badge/Gradio-FF7C00?style=flat&logo=gradio&logoColor=white)

A high-speed, accurate math word problem solver. This project is built on a "hybrid" architecture that leverages a fine-tuned **T5 model** to translate natural language into a formal math expression, and then uses the **SymPy** library to solve that expression with 100% accuracy.

## 🚀 Core Concept

The model avoids "hallucinated" math errors (e.g., `2+2=5`) common in pure LLMs by separating the two key tasks:
1.  **Translation (AI):** A fine-tuned T5 model translates a word problem into a machine-readable equation.
2.  **Calculation (Symbolic Logic):** The `SymPy` library executes that equation perfectly.

**Flow:**
`"What is 5 times 10?"` → `[T5 Translator Model]` → `"5 * 10"` → `[SymPy Solver]` → `50`

## ✨ Features

* **High-Speed Inference:** Designed to provide answers in under 2 seconds (on GPU).
* **100% Accurate Calculation:** By offloading the final calculation to `SymPy`, the model is guaranteed to be accurate.
* **Natural Language Translation:** A fine-tuned `t5-small` model translates text word problems into solvable math expressions.
* **Interactive Web UI:** A simple and clean front-end built with `Gradio`.

## 📂 Project Structure

This project is organized to separate model training from the application logic.

```
 HybridMath Model/
│
├── 📜 train.py
│
├── 🚀 app.py
│
├── 🧠 core_logic.py
│
├── 📦 saved_model/
│   ├── pytorch_model.bin
│   ├── config.json
│   ├── tokenizer_config.json
│   └── spiece.model
│
└── 📝 requirements.txt
```

## 🛠️ Installation & Setup

1.  **Clone the Repository:**
    ```bash
    git clone [https://github.com/Thakor-Yashpal/HybridMath-Model](https://github.com/Thakor-Yashpal/HybridMath-Model.git)
    cd HybridMath-Model
    ```

2.  **Create a Virtual Environment** (Recommended):
    ```bash
    # For macOS/Linux
    python3 -m venv venv
    source venv/bin/activate

    # For Windows
    python -m venv venv
    .\venv\Scripts\activate
    ```

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## 🏃 How to Run

There are two main steps: training the model, then running the app.

### 1. Train Your Model (Run Once)

First, you must run the training script to fine-tune the `t5-small` model on our math dataset. This will create and save your custom model to the `/saved_model` directory.

```bash
python train.py
```

(This will take a few minutes, especially on a CPU. Using a GPU is recommended.)

2. Run the Web App
Once the model is trained, you can start the interactive web application.

Bash
```
python app.py
```

This will start a local Gradio server. Open your web browser and go to the URL shown in your terminal (usually http://127.0.0.1:7860).

### 💻 Technology Stack

<li>Model: PyTorch</li>

<li>AI Framework: Hugging Face transformers (for T5)</li>

<li>Calculator: SymPy (for symbolic math)</li>

<li>Web UI: Gradio</li>

<li>Training: Hugging Face datasets</li>


#### 📜 License
This project is licensed under the MIT License.
