# VocalEdge AI

Python prototype for recording and analysing speech audio and turning the results into a report.

## What the code does

- Captures microphone audio (PyAudio) and transcribes it with the SpeechRecognition library
- Extracts audio features with librosa, SciPy and NumPy
- Plots results with Matplotlib and generates PDF reports with ReportLab (output goes to `reports/`)
- Includes a small web chatbot page in `chatbot/index.html`

## Project layout

- `main.py`: audio capture, analysis and report generation
- `map.py`, `sign.py`: supporting scripts
- `chatbot/`: chatbot front end
- `reports/`, `resources/`: generated reports and assets
- `requirements.txt`: pinned dependencies

## Run locally

```bash
pip install -r requirements.txt
python main.py
```

PyAudio needs the PortAudio system library; install it first if the PyAudio build fails.

## Status

Early prototype from April 2025. For my production-oriented voice work see [bank-assist](https://github.com/Kushalsharma0702/bank-assist) and [financial_voice_chatbot](https://github.com/Kushalsharma0702/financial_voice_chatbot).

Licensed under the terms in [LICENSE](LICENSE).
