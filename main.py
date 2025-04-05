import pyaudio
import numpy as np
import librosa
import time
import os
import speech_recognition as sr
from scipy.signal import find_peaks
from scipy.ndimage import median_filter
import soundfile as sf
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from datetime import datetime
import matplotlib.pyplot as plt

# Audio stream config
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 22050
RECORD_SECONDS = 20

# NLP Use-Case Detection
USE_CASE_KEYWORDS = {
    "interview": ["interview", "introduction", "self intro", "tell me about yourself"],
    "singing": ["song", "singing", "practice singing"],
    "public_speaking": ["speech", "presentation", "talk"]
}

USE_CASE_SUGGESTIONS = {
    "interview": [
        "Practice using a timer to simulate interview pressure.",
        "Use STAR format (Situation, Task, Action, Result) in responses.",
        "Keep answers concise and confident."
    ],
    "singing": [
        "Avoid dairy or cold items like ice cream before singing.",
        "Warm up your voice with humming or lip trills.",
        "Stay hydrated and avoid yelling before sessions."
    ],
    "public_speaking": [
        "Practice in front of a mirror or record yourself.",
        "Work on intonation and pace to maintain engagement.",
        "Use pauses strategically for emphasis."
    ]
}

def check_initial_silence(y, sr, threshold=0.01, duration_sec=5):
    check_samples = int(sr * duration_sec)
    energy = np.mean(np.abs(y[:check_samples]))
    if energy < threshold:
        print("⚠️ You remained silent in the first few seconds. Try starting promptly.")

def check_audio_presence(y):
    if np.max(np.abs(y)) < 0.005:
        print("❌ No voice detected in the recording. Please try again.")
        return False
    return True

def detect_use_case_from_text(transcribed_text):
    transcribed_text = transcribed_text.lower()
    for use_case, keywords in USE_CASE_KEYWORDS.items():
        for keyword in keywords:
            if keyword in transcribed_text:
                return use_case
    return None

def get_custom_suggestions(use_case):
    return USE_CASE_SUGGESTIONS.get(use_case, [])

def analyze_voice(y, rate):
    y = y / (np.max(np.abs(y)) + 1e-5)
    check_initial_silence(y, rate)
    if not check_audio_presence(y):
        return ("No Voice", 0, ["Please speak clearly and close to the mic."], 0, 0, 0, 0, 0, 0)

    pitches, magnitudes = librosa.piptrack(y=y, sr=rate, fmin=80, fmax=600)
    pitch_values = []
    for t in range(magnitudes.shape[1]):
        index = np.argmax(magnitudes[:, t])
        pitch = pitches[index, t]
        if pitch > 50:
            pitch_values.append(pitch)

    if len(pitch_values) > 5:
        pitch_values = median_filter(pitch_values, size=5)
    pitch_std = np.std(pitch_values)
    pitch_mean = np.mean(pitch_values)

    energy = librosa.feature.rms(y=y)[0]
    energy_mean = np.mean(energy)
    energy_std = np.std(energy)

    non_silent = librosa.effects.split(y, top_db=30)
    total_silence_duration = 0
    pause_count = 0
    for i in range(1, len(non_silent)):
        gap = (non_silent[i][0] - non_silent[i - 1][1]) / rate
        if gap > 0.25:
            pause_count += 1
            total_silence_duration += gap

    smoothed = np.convolve(np.abs(y), np.ones(1000)/1000, mode='valid')
    peaks, _ = find_peaks(smoothed, height=0.01, distance=1000)
    filler_count = len(peaks) // 30

    print("\n[DEBUG INFO]")
    print(f"Pitch Mean: {pitch_mean:.1f} Hz, STD: {pitch_std:.2f}")
    print(f"Energy Mean: {energy_mean:.5f}, STD: {energy_std:.5f}")
    print(f"Pauses: {pause_count}, Total Silence: {total_silence_duration:.2f}s")
    print(f"Filler Count: {filler_count}")

    normalized_pitch_std = np.clip(pitch_std / 50.0, 0, 1)
    normalized_energy = np.clip(energy_mean * 50, 0, 1)
    normalized_fillers = np.clip(filler_count / 10, 0, 1)
    normalized_pauses = np.clip(pause_count / 5, 0, 1)

    confidence_score = 100
    confidence_score -= normalized_pitch_std * 20
    confidence_score -= (1 - normalized_energy) * 25
    confidence_score -= normalized_fillers * 30
    confidence_score -= normalized_pauses * 25
    confidence_score = max(confidence_score, 0)

    if confidence_score >= 75:
        confidence_level = "Confident"
    elif confidence_score >= 50:
        confidence_level = "Moderate"
    else:
        confidence_level = "Needs Improvement"

    suggestions = []
    if pitch_std < 20:
        suggestions.append("Increase pitch variation to sound more engaging.")
    if energy_mean < 0.02:
        suggestions.append("Speak with more volume and energy.")
    if filler_count >= 3:
        suggestions.append("Practice reducing filler words like 'um' and 'uh'.")
    if pause_count >= 3:
        suggestions.append("Minimize long pauses for smoother delivery.")

    try:
        r = sr.Recognizer()
        with sr.AudioFile("temp.wav") as source:
            audio = r.record(source)
        transcribed_text = r.recognize_google(audio)
        use_case = detect_use_case_from_text(transcribed_text)
        suggestions += get_custom_suggestions(use_case)
    except Exception:
        pass

    return (confidence_level, confidence_score, suggestions, 
            pitch_mean, pitch_std, energy_mean, energy_std, pause_count, filler_count)

def process_audio_file(file_path):
    y, sr = librosa.load(file_path, sr=RATE)
    sf.write("temp.wav", y, sr)
    return analyze_voice(y, sr)

def record_and_process():
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

    print("\n🎙️ Speak now... (Recording for 20 seconds)")
    frames = []
    for _ in range(int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK, exception_on_overflow=False)
        frames.append(data)

    stream.stop_stream()
    stream.close()
    p.terminate()

    print("🔍 Processing your voice...")
    audio_data = b''.join(frames)
    y = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32)
    sf.write("temp.wav", y, RATE)
    return analyze_voice(y, RATE)

def print_report(level, score, suggestions, pitch_mean, pitch_std, energy_mean, energy_std, pause_count, filler_count):
    report = "\n🧠 Voice Health Report\n"
    report += f"Confidence Level: {level} ({score:.1f}%)\n"
    report += f"Pitch Mean: {pitch_mean:.1f} Hz, Pitch STD: {pitch_std:.2f}\n"
    report += f"Energy Mean: {energy_mean:.5f}, Energy STD: {energy_std:.5f}\n"
    report += f"Pauses Detected: {pause_count}, Fillers Estimated: {filler_count}\n"
    if suggestions:
        report += "\nSuggestions to Improve:\n"
        for s in suggestions:
            report += f"- {s}\n"
    else:
        report += "✅ Your voice sounds confident and fluent!\n"
    print(report)
    return report

def save_report(report_str, confidence_score, pitch_mean, energy_mean, pause_count, filler_count):
    save_choice = input("Do you want to save the report as PDF? (y/n): ")
    if save_choice.lower() == 'y':
        reports_dir = os.path.join(os.getcwd(), "reports")
        os.makedirs(reports_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"vocaledge-ai_report_{timestamp}.pdf"
        filepath = os.path.join(reports_dir, filename)

        try:
            fig, ax = plt.subplots()
            categories = ['Confidence', 'Pitch', 'Energy', 'Pauses', 'Fillers']
            values = [confidence_score, pitch_mean, energy_mean * 1000, pause_count, filler_count]
            ax.bar(categories, values, color=['green', 'blue', 'orange', 'purple', 'red'])
            ax.set_title('Voice Feature Overview')
            plt.tight_layout()
            graph_path = os.path.join(reports_dir, "graph.png")
            plt.savefig(graph_path)
            plt.close()

            c = canvas.Canvas(filepath, pagesize=A4)
            width, height = A4
            x_margin, y_margin = 50, 800

            c.setFont("Helvetica-Bold", 18)
            c.drawString(x_margin, y_margin, "🧠 VocalEdge AI - Voice Health Report")
            c.setFont("Helvetica", 12)
            y_margin -= 30

            for line in report_str.split('\n'):
                c.drawString(x_margin, y_margin, line)
                y_margin -= 15
                if y_margin < 100:
                    c.showPage()
                    y_margin = 800

            c.drawImage(graph_path, 100, 300, width=400, preserveAspectRatio=True)
            c.save()

            print(f"✅ Report saved as {filename} in 'reports/' folder.")
        except Exception as e:
            print(f"❌ Failed to save PDF report: {e}")
    else:
        print("Report not saved.")

def main():
    while True:
        print("\nChoose input method:")
        print("1. Live voice recording")
        print("2. Upload an audio file")
        print("3. Exit")
        choice = input("Enter choice (1, 2 or 3): ")

        if choice == '1':
            result = record_and_process()
        elif choice == '2':
            file_path = input("Enter path to audio file (.wav or .mp3): ")
            if not os.path.exists(file_path):
                print("❌ File not found.")
                continue
            result = process_audio_file(file_path)
        elif choice == '3':
            print("👋 Exiting... Stay vocal!")
            break
        else:
            print("❌ Invalid choice.")
            continue

        report_str = print_report(*result)
        save_report(report_str, result[1], result[3], result[5], result[7], result[8])
        print("\n🔁 Analysis complete. Returning to menu...")

if __name__ == '__main__':
    main()
