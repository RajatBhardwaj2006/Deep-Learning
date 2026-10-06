# Text Summarizer

A deep learning based text summarization application built using a fine-tuned T5 Transformer model, Hugging Face Transformers, PyTorch, and FastAPI.

## Features

- Text summarization using a T5 Transformer model
- Fine-tuned model loaded from `Saved_Model`
- FastAPI backend
- HTML/CSS/JavaScript frontend
- Automatic device selection for CUDA, MPS, or CPU
- Input text cleaning
- Interactive FastAPI documentation

## Tech Stack

- Python
- PyTorch
- Hugging Face Transformers
- T5
- FastAPI
- Uvicorn
- Pydantic
- Pandas
- Jinja2
- HTML
- CSS
- JavaScript

## Project Structure

```text
Text Summarizer/
├── Saved_Model/
├── CSV/
├── results/
├── app.py
├── index.html
├── Text_summarizer.ipynb
└── README.md
```

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/RajatBhardwaj2006/Deep-Learning.git
cd Deep-Learning
cd "Text Summarizer"
```

### 2. Create a virtual environment

Windows:

```powershell
python -m venv venv
venv\Scripts\activate
```

Linux/macOS:

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install fastapi uvicorn transformers torch pandas jinja2 sentencepiece
```

## Run the Application

From the `Text Summarizer` directory:

```bash
python -m uvicorn app:app --reload
```

Open:

```text
http://127.0.0.1:8000
```

Do not open `index.html` directly with Live Server. The frontend is served through FastAPI.

## API

### POST `/summarize/`

Request:

```json
{
    "dialogue": "Artificial intelligence is transforming many industries..."
}
```

Response:

```json
{
    "summary": "AI systems are becoming more capable..."
}
```

## FastAPI Documentation

After starting the application:

```text
http://127.0.0.1:8000/docs
```

## How It Works

```text
User enters text
       ↓
index.html
       ↓
POST /summarize/
       ↓
FastAPI
       ↓
Input cleaning
       ↓
T5 Tokenizer
       ↓
T5 Transformer
       ↓
Generated summary
       ↓
FastAPI JSON response
       ↓
Summary displayed in browser
```

## Model

The application loads the trained T5 model and tokenizer from:

```text
Saved_Model/
```

The model is loaded with:

```python
T5model = T5ForConditionalGeneration.from_pretrained("./Saved_Model")
tokenizer = T5Tokenizer.from_pretrained("./Saved_Model")
```

The application automatically selects:

- CUDA when an NVIDIA GPU is available
- MPS on supported Apple hardware
- CPU otherwise

## Input Processing

Before summarization, the application:

- Removes line breaks
- Removes extra whitespace
- Removes HTML tags
- Converts text to lowercase
- Removes leading and trailing whitespace

## Troubleshooting

### `pip` is not recognized

Use:

```bash
python -m pip install fastapi
```

instead of:

```bash
pip install fastapi
```

### Missing Python package

For example:

```text
ModuleNotFoundError: No module named 'transformers'
```

Run:

```bash
python -m pip install transformers
```

Or reinstall all dependencies:

```bash
python -m pip install fastapi uvicorn transformers torch pandas jinja2 sentencepiece
```

### Port 8000 is already in use

Run:

```bash
python -m uvicorn app:app --reload --port 8001
```

Then open:

```text
http://127.0.0.1:8001
```

### Model not found

Make sure `Saved_Model` exists inside the `Text Summarizer` directory:

```text
Text Summarizer/
├── app.py
├── index.html
└── Saved_Model/
```

## Future Improvements

- Adjustable summary length
- PDF/DOCX upload
- Larger document support
- Summary history
- Downloadable summaries
- Improved summarization quality
- Docker deployment
- Cloud deployment
- API authentication
- Production monitoring

## Author

**Rajat Bhardwaj**

GitHub: https://github.com/RajatBhardwaj2006

## Project

**Text Summarizer**

Built using Python, PyTorch, Hugging Face Transformers, T5, FastAPI, Uvicorn, HTML, CSS, and JavaScript.
