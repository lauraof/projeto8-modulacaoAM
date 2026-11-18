import sounddevice as sd
from scipy.io import wavfile
import numpy as np

def calculo_FFT(dados, freqAmostra):
    qtd_amostras = len(dados)
    yf = np.abs(np.fft.rfft(dados)) / qtd_amostras
    xf = np.fft.rfftfreq(qtd_amostras, d=1/freqAmostra)
    return xf, yf

def main():
    fs = 44100
    print(f"Gravando 5s")
    gravacao = sd.rec(5*fs, samplerate=fs, channels=1)
    sd.wait()
    wavfile.write("mensagem.wav", fs, (gravacao * 32767).astype(np.int16))
    print(f"Gravação concluída!")

    tempo = np.linspace(0, 5, 5*fs)
    portadora = np.sin(2*np.pi*4000*tempo)
    

if __name__ == '__main__':
    main()