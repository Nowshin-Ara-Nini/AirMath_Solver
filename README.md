# Air Math Solver

CPU-friendly webcam app for drawing math expressions in the air with your finger, recognizing simple arithmetic locally, optionally using Mathpix OCR for advanced expressions, and evaluating results with SymPy.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Local OCR works without cloud credentials for simple digits and operators. For advanced OCR, copy `.env.example` to `.env` and add your Mathpix credentials:

```env
MATHPIX_APP_ID=your_app_id_here
MATHPIX_APP_KEY=your_app_key_here
MATHPIX_API_URL=https://api.mathpix.com/v3/text
MATHPIX_TIMEOUT=15
```

Mathpix API keys are created in the Mathpix console. The app uses the image OCR endpoint with `app_id` and `app_key` headers and multipart image upload.

## Run

```powershell
python main.py
```

## Controls

- Raise your index finger to draw.
- Press `R` or hover over `ERASER` to toggle eraser mode; then use your index finger to erase.
- Hover your fingertip over toolbar buttons to click: `CLEAR`, `UNDO`, `ERASER`, `EVALUATE`, `QUIT`.
- Keyboard fallbacks: `c` clear, `z` undo, `d` debug overlay, `e` or space evaluate, `r` eraser, `q` quit.

## CPU Notes

- Webcam resolution is fixed at `640x480`.
- MediaPipe Hands runs with `model_complexity=0` and one hand.
- No TensorFlow, PyTorch, transformer, or heavy OCR inference is used.
- OCR happens only when you press `EVALUATE`, so the live webcam loop stays light.

## OCR Architecture

`ocr_engine.py` defines `BaseOCREngine` and `OCRResult`. The app uses a hybrid engine: Mathpix runs when configured, and a local OpenCV recognizer handles simple handwritten digits/operators when Mathpix is missing or unavailable.

If Mathpix credentials are missing, the app keeps running with local simple OCR instead of treating OCR as fully unavailable.

## Supported Examples

- `1+2`
- `5*8-3`
- `sqrt(25)`
- `x^2+5x+6`
- `sin(x**2)+3*cos(x)*2`
- `ln(x**2+5*x)`
- `tan(x)+log(x)`

Saved OCR snapshots are written to `temp/`.
