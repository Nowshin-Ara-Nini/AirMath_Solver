# Air Math Solver

CPU-friendly webcam app for drawing math expressions in the air with your finger, recognizing them with Mathpix OCR, and evaluating them with SymPy.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and add your Mathpix credentials:

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

- Raise only your index finger to draw.
- Raise index + middle finger to erase, or hover over `ERASER`.
- Hover your fingertip over toolbar buttons to click: `CLEAR`, `ERASER`, `EVALUATE`, `QUIT`.
- Keyboard fallbacks: `c` clear, `z` undo, `e` or space evaluate, `r` eraser, `q` quit.

## CPU Notes

- Webcam resolution is fixed at `640x480`.
- MediaPipe Hands runs with `model_complexity=0` and one hand.
- No TensorFlow, PyTorch, transformer, or local OCR inference is used.
- OCR happens only when you press `EVALUATE`, so the live webcam loop stays light.

## OCR Architecture

`ocr_engine.py` defines `BaseOCREngine` and `OCRResult`. The rest of the app only depends on that interface, so a future OCR backend such as pix2tex, TrOCR, or a custom CNN can replace `MathpixOCREngine` without changing drawing, parsing, or evaluation code.

If Mathpix credentials are missing or the API request fails, the error is shown inside the OpenCV window and the app keeps running.

## Supported Examples

- `1+2`
- `5*8-3`
- `sqrt(25)`
- `x^2+5x+6`
- `sin(x**2)+3*cos(x)*2`
- `ln(x**2+5*x)`
- `tan(x)+log(x)`

Saved OCR snapshots are written to `temp/`.
