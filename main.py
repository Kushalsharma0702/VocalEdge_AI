import pyaudio
import numpy as np
import librosa
import time
import os
from scipy.signal import find_peaks
from scipy.ndimage import median_filter

# Audio stream config
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 22050
RECORD_SECONDS = 5  # Recording duration

def analyze_voice(y, rate):
    y = y / (np.max(np.abs(y)) + 1e-5)

    # Pitch estimation (limit to human voice range)
    pitches, magnitudes = librosa.piptrack(y=y, sr=rate, fmin=80, fmax=600)
    pitch_values = []
    for t in range(magnitudes.shape[1]):
        index = np.argmax(magnitudes[:, t])
        pitch = pitches[index, t]
        if pitch > 50:
            pitch_values.append(pitch)
    # Smoothing to reduce random spikes
    if len(pitch_values) > 5:
        pitch_values = median_filter(pitch_values, size=5)
    pitch_std = np.std(pitch_values) if len(pitch_values) > 0 else 0
    pitch_mean = np.mean(pitch_values) if len(pitch_values) > 0 else 0

    # Energy calculation
    energy = librosa.feature.rms(y=y)[0]
    energy_mean = np.mean(energy)
    energy_std = np.std(energy)

    # Silence detection (pauses)
    non_silent = librosa.effects.split(y, top_db=30)
    total_silence_duration = 0
    pause_count = 0
    for i in range(1, len(non_silent)):
        gap = (non_silent[i][0] - non_silent[i - 1][1]) / rate
        if gap > 0.25:
            pause_count += 1
            total_silence_duration += gap

    # Filler approximation using smoothed peaks
    smoothed = np.convolve(np.abs(y), np.ones(1000)/1000, mode='valid')
    peaks, _ = find_peaks(smoothed, height=0.01, distance=1000)
    filler_count = len(peaks) // 30

    print("\n[DEBUG INFO]")
    print(f"Pitch Mean: {pitch_mean:.1f} Hz, STD: {pitch_std:.2f}")
    print(f"Energy Mean: {energy_mean:.5f}, STD: {energy_std:.5f}")
    print(f"Pauses: {pause_count}, Total Silence: {total_silence_duration:.2f}s")
    print(f"Filler Count: {filler_count}")

    # --- Enhanced Scoring (Normalization) ---
    normalized_pitch_std = np.clip(pitch_std / 50.0, 0, 1)       # Good if near 50 Hz variability
    normalized_energy = np.clip(energy_mean * 50, 0, 1)            # Adjust factor based on typical energy
    normalized_fillers = np.clip(filler_count / 10, 0, 1)
    normalized_pauses = np.clip(pause_count / 5, 0, 1)

    # Dynamic confidence score calculation
    confidence_score = 100
    confidence_score -= normalized_pitch_std * 20       # Less variation deducts points
    confidence_score -= (1 - normalized_energy) * 25      # Lower energy deducts more points
    confidence_score -= normalized_fillers * 30           # More fillers = lower score
    confidence_score -= normalized_pauses * 25            # More pauses = lower score

    confidence_score = max(confidence_score, 0)

    # Map score to descriptive level
    if confidence_score >= 75:
        confidence_level = "Confident"
    elif confidence_score >= 50:
        confidence_level = "Moderate"
    else:
        confidence_level = "Needs Improvement"

    # --- Enhanced Suggestions ---
    suggestions = []
    if pitch_std < 20:
        suggestions.append("Increase pitch variation to sound more engaging.")
    if energy_mean < 0.02:
        suggestions.append("Speak with more volume and energy.")
    if filler_count >= 3:
        suggestions.append("Practice reducing filler words like 'um' and 'uh'.")
    if pause_count >= 3:
        suggestions.append("Minimize long pauses for smoother delivery.")

    return (confidence_level, confidence_score, suggestions, 
            pitch_mean, pitch_std, energy_mean, energy_std, pause_count, filler_count)

def process_audio_file(file_path):
    y, sr = librosa.load(file_path, sr=RATE)
    return analyze_voice(y, sr)

def record_and_process():
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    print("\n🎙️ Speak now... (Recording for 5 seconds)")
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

def save_report(report_str):
    save_choice = input("Do you want to save the report? (y/n): ")
    if save_choice.lower() == 'y':
        file_name = input("Enter the filename to save (e.g., report.txt): ")
        try:
            with open(file_name, "w", encoding="utf-8") as f:
                f.write(report_str)
            print(f"✅ Report saved as {file_name}")
        except Exception as e:
            print(f"❌ Failed to save report: {e}")
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
        save_report(report_str)
        print("\n🔁 Analysis complete. Returning to menu...")

if __name__ == '__main__':
    main()
