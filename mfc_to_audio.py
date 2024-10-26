import pandas as pd
import numpy as np
import librosa
import soundfile as sf
import noisereduce as nr
from scipy.signal import butter, lfilter

def invert_MFCC_to_audio(mfcc_coefficients, sr=44100, n_iter=32):
    """
    Invert MFCC coefficients back to audio signal.
    
    Parameters:
    mfcc_coefficients (numpy.ndarray): The MFCC coefficients.
    sr (int): The sampling rate of the audio.
    n_iter (int): Number of iterations for the Griffin-Lim algorithm.
    
    Returns:
    numpy.ndarray: The reconstructed audio signal.
    """
    # Check for non-finite values and replace them with zeros
    mfcc_coefficients = np.nan_to_num(mfcc_coefficients)
    
    # Invert MFCC to Mel spectrogram
    mel_spectrogram = librosa.feature.inverse.mfcc_to_mel(mfcc_coefficients)
    
    # Check for non-finite values in the Mel spectrogram and replace them with zeros
    mel_spectrogram = np.nan_to_num(mel_spectrogram)
    
    # Invert Mel spectrogram to audio using Griffin-Lim algorithm with fewer iterations
    audio_signal = librosa.feature.inverse.mel_to_audio(mel_spectrogram, sr=sr, n_iter=n_iter)
    
    # Check for non-finite values in the audio signal and replace them with zeros
    audio_signal = np.nan_to_num(audio_signal)
    
    return audio_signal

def normalize_audio(audio_signal):
    """
    Normalize the audio signal to have a consistent volume level.
    
    Parameters:
    audio_signal (numpy.ndarray): The audio signal.
    
    Returns:
    numpy.ndarray: The normalized audio signal.
    """
    return librosa.util.normalize(audio_signal)

def butter_highpass(cutoff, fs, order=5):
    """
    Create a high-pass filter.
    
    Parameters:
    cutoff (float): The cutoff frequency of the filter.
    fs (int): The sampling rate of the audio.
    order (int): The order of the filter.
    
    Returns:
    tuple: The filter coefficients.
    """
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

def highpass_filter(audio_signal, cutoff=100, fs=44100, order=5):
    """
    Apply a high-pass filter to the audio signal.
    
    Parameters:
    audio_signal (numpy.ndarray): The audio signal.
    cutoff (float): The cutoff frequency of the filter.
    fs (int): The sampling rate of the audio.
    order (int): The order of the filter.
    
    Returns:
    numpy.ndarray: The filtered audio signal.
    """
    b, a = butter_highpass(cutoff, fs, order=order)
    y = lfilter(b, a, audio_signal)
    return y

# Read the CSV file into a pandas DataFrame
data = pd.read_csv('11-MFCC.csv', header=None)

# Convert the single row DataFrame to a numpy array
mfccs = data.values.flatten()

# Reshape the MFCC coefficients to the correct 2D array format
# Assuming the original MFCCs had 20 coefficients per frame
n_mfcc = 20
mfccs = mfccs.reshape((n_mfcc, -1))

# Invert MFCC to audio
audio_signal = invert_MFCC_to_audio(mfccs, sr=44100, n_iter=128)

# Apply noise reduction
audio_signal = nr.reduce_noise(y=audio_signal, sr=44100)

# Normalize the audio signal
audio_signal = normalize_audio(audio_signal)

# Apply a high-pass filter to remove low-frequency noise
audio_signal = highpass_filter(audio_signal, cutoff=100, fs=44100, order=5)

# Store the audio signal to a file
sf.write('11_refined.wav', audio_signal, 44100)