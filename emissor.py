import numpy as np
import sounddevice as sd 
import matplotlib.pyplot as plt

PREAMBULO = '0' * 20
MARCADOR = '10110100'

def calculo_FFT(dados, freqAmostra):
    qtd_amostras = len(dados)
    yf = np.abs(np.fft.rfft(dados)) / qtd_amostras
    xf = np.fft.rfftfreq(qtd_amostras, d=1/freqAmostra)
    return xf, yf

def main():
    fs = 44100
    n_amostras = fs // 100
    frase = input("Digite uma frase: ")
    tamanho_frase = format(len(frase), '08b')
    string_mensagem = PREAMBULO + MARCADOR + tamanho_frase
    for caractere in frase: 
        string_mensagem += format(ord(caractere), '08b') 
    tempo = np.linspace(0, len(string_mensagem)/100, len(string_mensagem)*n_amostras, endpoint=False)
    fases = np.zeros(len(string_mensagem)*n_amostras)
    for i in range(len(string_mensagem)): 
        if string_mensagem[i] == '1': 
            fases[i*n_amostras : (i+1)*n_amostras] = np.pi
    modulado = np.sin(2*np.pi*4000*tempo + fases)
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

if __name__ == '__main__': 
    main()