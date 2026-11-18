import numpy as np
import sounddevice as sd
from scipy.io import wavfile
import matplotlib.pyplot as plt

PREAMBULO = '0' * 20        # 0,2 s de portadora pura, fase constante
MARCADOR = '10110100'       # combinado com o emissor


def calculo_FFT(dados, freqAmostra):
    qtd_amostras = len(dados)
    yf = np.abs(np.fft.rfft(dados)) / qtd_amostras
    xf = np.fft.rfftfreq(qtd_amostras, d=1/freqAmostra)
    return xf, yf


def main():
    fs = 44100
    n_amostras = fs // 100

    duracao = 6
    print(f"Gravando {duracao}s")
    gravacao = sd.rec(duracao*fs, samplerate=fs, channels=1)
    sd.wait()
    print("Gravação concluída!")

    # sd.rec devolve uma matriz de 1 coluna; precisamos de um vetor simples,
    # senão as multiplicações por portadora dão broadcast errado
    gravacao = gravacao[:, 0]
    wavfile.write("mensagem.wav", fs, (gravacao * 32767).astype(np.int16))

    # ==============================================================
    # PASSO 1 - achar grosseiramente onde o sinal começa, pela energia
    # onde há silêncio as amostras são quase zero -> soma baixa
    # onde há sinal elas oscilam forte -> soma alta
    # o abs é essencial, senão positivos cancelam negativos
    # ==============================================================
    energias = []
    posicoes = []
    for ini in range(0, len(gravacao) - n_amostras, n_amostras // 4):
        energias.append(np.sum(np.abs(gravacao[ini : ini + n_amostras])))
        posicoes.append(ini)

    limiar = (max(energias) + min(energias)) / 2
    n0 = 0
    for k in range(len(energias)):
        if energias[k] > limiar:
            n0 = posicoes[k]
            break

    print(f"Passo 1: energia de silêncio {min(energias):.1f}, "
          f"de sinal {max(energias):.1f}, limiar {limiar:.1f}")
    print(f"Passo 1: o sinal começa perto da amostra {n0}")

    # ==============================================================
    # PASSO 2 - achar a amostra exata, testando offsets contra o MARCADOR
    #
    # Por que procurar entre n0 e n0+441: a janela de energia começa a
    # pegar sinal uma janela ANTES do início real, então o limiar dispara
    # cedo. O início verdadeiro está um pouco depois de n0, dentro de 1 bit.
    #
    # Critério: NÃO contamos quantos bits do marcador bateram, porque vários
    # offsets errados acertam todos e empatam. Somamos as próprias somas com
    # o sinal esperado de cada bit (+1 para bit 0, -1 para bit 1). Essa soma
    # é máxima no alinhamento certo e cai suave ao redor, sem empatar.
    # ==============================================================
    esperado = []
    for bit in MARCADOR:
        if bit == '0':
            esperado.append(1)      # bit 0 -> a soma deve dar positiva
        else:
            esperado.append(-1)     # bit 1 -> a soma deve dar negativa

    melhor_d = n0
    melhor_nota = None
    for d in range(n0, n0 + n_amostras):
        inicio_marcador = d + len(PREAMBULO)*n_amostras
        if inicio_marcador + len(MARCADOR)*n_amostras > len(gravacao):
            break

        # defasagem que o canal acrescentou, medida no preâmbulo
        trecho = gravacao[d : inicio_marcador]
        tempo = np.arange(len(trecho)) / fs
        soma_sen = np.sum(trecho * np.sin(2*np.pi*4000*tempo))
        soma_cos = np.sum(trecho * np.cos(2*np.pi*4000*tempo))
        theta = np.arctan2(soma_cos, soma_sen)

        # quanto esse offset "casa" com o marcador
        tempo = np.arange(n_amostras) / fs
        portadora = np.sin(2*np.pi*4000*tempo + theta)
        nota = 0.0
        for i in range(len(MARCADOR)):
            a = inicio_marcador + i*n_amostras
            nota += esperado[i] * np.sum(gravacao[a : a + n_amostras] * portadora)

        if melhor_nota is None or nota > melhor_nota:
            melhor_nota = nota
            melhor_d = d

    print(f"Passo 2: o sinal começa exatamente na amostra {melhor_d} "
          f"(ajuste de {melhor_d - n0:+d} amostras)")

    # ==============================================================
    # PASSO 3 - medir a defasagem no início certo
    #
    # O preâmbulo é portadora pura, então o que chega é A*sin(w*t + theta).
    # De  sin(x + theta) = sin(x)*cos(theta) + cos(x)*sin(theta):
    #   somar trecho*sin(x) -> proporcional a cos(theta)
    #   somar trecho*cos(x) -> proporcional a sin(theta)
    # e o arctan2 devolve o ângulo.
    # ==============================================================
    inicio_marcador = melhor_d + len(PREAMBULO)*n_amostras
    trecho = gravacao[melhor_d : inicio_marcador]
    tempo = np.arange(len(trecho)) / fs
    soma_sen = np.sum(trecho * np.sin(2*np.pi*4000*tempo))
    soma_cos = np.sum(trecho * np.cos(2*np.pi*4000*tempo))
    theta = np.arctan2(soma_cos, soma_sen)
    print(f"Passo 3: defasagem do canal = {theta:.3f} rad")

    # ==============================================================
    # DEMODULAÇÃO - detector por produto
    #
    # bit 0 ->  sin * sin =  sin^2 -> sempre >= 0 -> soma POSITIVA
    # bit 1 -> -sin * sin = -sin^2 -> sempre <= 0 -> soma NEGATIVA
    # então o bit é só o sinal da soma
    # ==============================================================
    tempo = np.arange(n_amostras) / fs
    portadora = np.sin(2*np.pi*4000*tempo + theta)

    # lê tudo o que couber a partir do marcador, e depois recorta
    quantos_cabem = (len(gravacao) - inicio_marcador) // n_amostras
    bits = ""
    for i in range(quantos_cabem):
        a = inicio_marcador + i*n_amostras
        if np.sum(gravacao[a : a + n_amostras] * portadora) > 0:
            bits += '0'
        else:
            bits += '1'

    marcador_lido = bits[0:8]
    if marcador_lido == MARCADOR:
        print(f"Marcador lido: {marcador_lido} (confere)")
    else:
        print(f"Marcador lido: {marcador_lido}, esperado {MARCADOR} "
              f"-> sincronismo falhou")

    qtd_caracteres = int(bits[8:16], 2)
    print(f"Tamanho da mensagem: {qtd_caracteres} caracteres")

    bits_mensagem = bits[16 : 16 + qtd_caracteres*8]
    frase = ""
    for i in range(0, len(bits_mensagem) - 7, 8):
        frase += chr(int(bits_mensagem[i:i+8], 2))

    print()
    print(f"MENSAGEM RECEBIDA: {frase}")
    print()

    # ==============================================================
    # FFT do sinal recebido e banda ocupada
    # ==============================================================
    fim = inicio_marcador + (16 + qtd_caracteres*8)*n_amostras
    sinal = gravacao[melhor_d : fim]

    banda = 2 * 100      # 2 x taxa de bits
    print(f"Taxa de transmissão: 100 bits/s")
    print(f"Banda necessária: {banda} Hz "
          f"(de {4000 - banda//2} Hz a {4000 + banda//2} Hz)")

    xf, yf = calculo_FFT(sinal, fs)
    plt.plot(xf, yf)
    plt.title("Gráfico de FFT da frase recebida")
    plt.xlabel("Frequências (Hz)")
    plt.ylabel("Magnitude")
    plt.show()


if __name__ == '__main__':
    main()