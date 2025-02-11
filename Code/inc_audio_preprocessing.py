import os
import librosa
import numpy as np
import pandas as pd
import librosa.display
import matplotlib.pyplot as plt
import scipy.signal as signal
import scipy.io.wavfile as wav

def noiseFilter(soundPath):
    output_root = "./Dataset/Filtered_Audios/"
    os.makedirs(output_root, exist_ok=True)
    
    for subdir, _, files in os.walk(soundPath):
        subdir_name = os.path.basename(subdir)
        specific_output_root = os.path.join("./Dataset/Filtered_Audios/", subdir_name)
        os.makedirs(specific_output_root, exist_ok=True)
        
        for file in files:
            if file.endswith(".wav"):
                input_path = os.path.join(subdir, file)
                output_path = os.path.join(specific_output_root, file)
                
                sample_rate, data = wav.read(input_path)
                if len(data.shape) > 1:
                    data = np.mean(data, axis=1)
                
                filtered_audio = butterworth_filter(data, fs=sample_rate, order=4)
                wav.write(output_path, sample_rate, filtered_audio.astype(np.int16))
    
    return output_root

def butterworth_filter(data, lowcut=20, highcut=400, fs=4000, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = signal.butter(order, [low, high], btype='band')
    return signal.filtfilt(b, a, data)

def makeSpectrograms(filtered_dataPath, incremental_dataPath):
    spectrogram_output_root = "./Dataset/Spectrograms/"
    os.makedirs(spectrogram_output_root, exist_ok=True)

    # Step 1: Collect base names from Incremental dataset
    incremental_filenames = set()
    for subdir, _, files in os.walk(incremental_dataPath):
        for file in files:
            if file.endswith(".wav"):
                incremental_filenames.add(file)  # Store only the base name

    # Step 2: Generate spectrograms only for matching files
    for subdir, _, files in os.walk(filtered_dataPath):
        subdir_name = os.path.basename(subdir)
        specific_spectrogram_root = os.path.join(spectrogram_output_root, subdir_name)
        os.makedirs(specific_spectrogram_root, exist_ok=True)

        for file in files:
            if file.endswith(".wav") and file in incremental_filenames:  # Process only Incremental files
                input_path = os.path.join(subdir, file)
                output_image_path = os.path.join(specific_spectrogram_root, file.replace(".wav", ".png"))

                generate_spectrogram(input_path, output_image_path)

    return True

def generate_spectrogram(audio_path, output_path):
    y, sr = librosa.load(audio_path, sr=None)
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)

    plt.figure(figsize=(8, 4))
    librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log')
    plt.colorbar(format="%+2.0f dB")
    plt.title("Spectrogram")
    plt.tight_layout()
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def extractMFCC(filtered_dataPath, incremental_dataPath, n_mfcc=13):
    data = []

    # Step 1: Collect base names from Incremental dataset
    incremental_filenames = set()
    for subdir, _, files in os.walk(incremental_dataPath):
        for file in files:
            if file.endswith(".wav"):
                incremental_filenames.add(file)  # Store only the base name

    # Step 2: Extract MFCCs only for Incremental files
    for subdir, _, files in os.walk(filtered_dataPath):
        subdir_name = os.path.basename(subdir)
        for file in files:
            if file.endswith(".wav") and file in incremental_filenames:  # Process only Incremental files
                file_path = os.path.join(subdir, file)
                spectrogram_path = os.path.join("./Dataset/Spectrograms", subdir_name, file.replace(".wav", ".png"))

                mfcc_features = extract_mfcc(file_path, n_mfcc)
                data.append([file, subdir_name, spectrogram_path] + list(mfcc_features))

    # Convert to DataFrame and save
    csv_filename = os.path.join("./Dataset", "inc_audio_mfcc_features_with_labels.csv")
    df = pd.DataFrame(data, columns=["Filename", "Label", "Spectrogram_Path"] + [f"MFCC_{i}" for i in range(1, n_mfcc + 1)])
    df.to_csv(csv_filename, index=False)

    return df

def extract_mfcc(audio_path, n_mfcc=13):
    y, sr = librosa.load(audio_path, sr=None)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return np.mean(mfccs, axis=1)  # Compute mean MFCC across time frames


def SoundConversionPipeline(soundPath):
    filtered_dataPath = noiseFilter(soundPath)
    confirm_spec = makeSpectrograms(filtered_dataPath, soundPath)
    if confirm_spec:
        mfcc_data = extractMFCC(filtered_dataPath, soundPath)
    return mfcc_data

def main():
    soundPath = "./Dataset/Incremental_Audio_1/"
    mfcc_data = SoundConversionPipeline(soundPath)
    # print(mfcc_data.head())

main()