import numpy as np
import sounddevice as sd 
import matplotlib.pyplot as plt

def fase_fi(bit): 
    if bit == '0': 
        return 0
    return np.pi

def calculo_FFT(dados, freqAmostra):
    qtd_amostras = len(dados)
    yf = np.abs(np.fft.rfft(dados)) / qtd_amostras
    xf = np.fft.rfftfreq(qtd_amostras, d=1/freqAmostra)
    return xf, yf

def main():
    frase = input("Digite uma frase: ")
    string_mensagem = [string_mensagem + format(ord(caractere), '08b') for caractere in frase] 
    tempo = np.linspace(0, len(string_mensagem)/100, len(string_mensagem))
    modulado = np.sin(2*np.pi*4000*tempo + fase_fi(string_mensagem))
    fs = 44100
    print("Sinal começou a tocar")
    sd.play(modulado, fs)
    sd.wait()
    print("Sinal acabou de tocar")
    xf, yf = calculo_FFT(modulado, fs)
    plt.plot(xf, yf)
    plt.title("Gráfico de FFT da frase emitida")
    plt.xlabel("Frequências (Hz)")
    plt.ylabel("Magnitude")
    plt.show()        