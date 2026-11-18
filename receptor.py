import numpy as np
import sounddevice as sd
from scipy.io import wavfile
import matplotlib.pyplot as plt

PREAMBULO = '0' * 20        # 0,2 s de portadora pura, dá tempo do som estabilizar
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

    # sd.rec devolve matriz de 1 coluna; precisamos de vetor simples, senão
    # as multiplicações por portadora dão broadcast errado
    gravacao = gravacao[:, 0]
    wavfile.write("mensagem.wav", fs, (gravacao * 32767).astype(np.int16))

    # a portadora local de 1 bit. Como 4000/100 = 40 é inteiro, cabem 40
    # ciclos completos em cada bit, então essa mesma portadora serve para
    # todos os bits: não precisa de vetor de tempo gigante.
    tempo = np.arange(n_amostras) / fs
    seno = np.sin(2*np.pi*4000*tempo)
    cosseno = np.cos(2*np.pi*4000*tempo)

    # o cabeçalho é a parte que já conhecemos: preâmbulo + marcador = 28 bits.
    # procuramos ele inteiro, e não só o marcador de 8 bits: a chance de 8
    # bits da mensagem casarem por acaso é 1 em 256, e como a busca testa
    # milhares de posições, aparecem falsos positivos toda hora. Com 28 bits
    # a chance cai para 1 em 2^28, ou seja, nunca.
    CABECALHO = PREAMBULO + MARCADOR

    # +1 onde esperamos bit 0 (soma positiva), -1 onde esperamos bit 1
    esperado = []
    for bit in CABECALHO:
        if bit == '0':
            esperado.append(1)
        else:
            esperado.append(-1)

    # ==============================================================
    # SINCRONISMO - procurar o CABEÇALHO na gravação inteira
    #
    # Para cada posição candidata, multiplicamos cada bit do cabeçalho pela
    # portadora e somamos, com o sinal esperado daquele bit. Fazemos isso
    # duas vezes: com seno (soma 'a') e com cosseno (soma 'b').
    #
    # A nota é sqrt(a^2 + b^2). Essa nota NÃO depende da fase com que o som
    # chegou, então não precisamos descobrir a defasagem antes de procurar.
    # Ela é máxima na posição certa e cai ao redor.
    # ==============================================================
    def nota_do_offset(d):
        a = 0.0
        b = 0.0
        energia = 0.0
        for i in range(len(CABECALHO)):
            trecho = gravacao[d + i*n_amostras : d + (i+1)*n_amostras]
            a += esperado[i] * np.sum(trecho * seno)
            b += esperado[i] * np.sum(trecho * cosseno)
            energia += np.sum(trecho * trecho)
        # dividir pela energia do trecho deixa a nota independente do volume.
        # sem isso, se o microfone aumentar o ganho ao longo da gravacao, a
        # nota premia a parte mais alta em vez da parte que casa com o cabecalho
        return a, b, np.sqrt(energia) + 1e-12

    ultimo = len(gravacao) - len(CABECALHO)*n_amostras

    # busca grossa: anda de 50 em 50 amostras pela gravação toda.
    # dá para pular de 50 em 50 porque a nota varia suavemente (é por isso
    # que usamos sqrt(a^2+b^2): se dependesse da fase, ela oscilaria a cada
    # 11 amostras e a busca grossa passaria direto pelo pico)
    melhor_d = 0
    melhor_nota = -1.0
    for d in range(0, ultimo, 50):
        a, b, norma = nota_do_offset(d)
        nota = np.sqrt(a*a + b*b) / norma
        if nota > melhor_nota:
            melhor_nota = nota
            melhor_d = d
    print(f"Busca grossa: cabeçalho perto da amostra {melhor_d}")

    # busca fina: de 1 em 1 em volta do melhor da busca grossa
    inicio_fino = max(0, melhor_d - 60)
    fim_fino = min(ultimo, melhor_d + 60)
    melhor_nota = -1.0
    melhor_a = 0.0
    melhor_b = 0.0
    for d in range(inicio_fino, fim_fino):
        a, b, norma = nota_do_offset(d)
        nota = np.sqrt(a*a + b*b) / norma
        if nota > melhor_nota:
            melhor_nota = nota
            melhor_d = d
            melhor_a = a
            melhor_b = b
    print(f"Busca fina: a mensagem começa na amostra {melhor_d}")

    # a mesma conta que deu a nota já dá a defasagem do canal
    theta = np.arctan2(melhor_b, melhor_a)
    print(f"Defasagem do canal: {theta:.3f} rad")

    # ==============================================================
    # DEMODULAÇÃO - detector por produto
    #
    # bit 0 ->  sin * sin =  sin^2 -> sempre >= 0 -> soma POSITIVA
    # bit 1 -> -sin * sin = -sin^2 -> sempre <= 0 -> soma NEGATIVA
    # então o bit é só o sinal da soma
    # ==============================================================
    portadora = np.sin(2*np.pi*4000*tempo + theta)

    quantos_cabem = (len(gravacao) - melhor_d) // n_amostras
    bits = ""
    for i in range(quantos_cabem):
        a = melhor_d + i*n_amostras
        if np.sum(gravacao[a : a + n_amostras] * portadora) > 0:
            bits += '0'
        else:
            bits += '1'

    # melhor_d aponta para o começo do quadro, então:
    #   bits[0:20]  = preâmbulo
    #   bits[20:28] = marcador
    #   bits[28:36] = tamanho
    #   bits[36:]   = mensagem
    marcador_lido = bits[20:28]
    if marcador_lido == MARCADOR:
        print(f"Marcador lido: {marcador_lido} (confere)")
    else:
        print(f"Marcador lido: {marcador_lido}, esperado {MARCADOR} "
              f"-> sincronismo falhou")

    qtd_caracteres = int(bits[28:36], 2)
    print(f"Tamanho da mensagem: {qtd_caracteres} caracteres")

    bits_mensagem = bits[36 : 36 + qtd_caracteres*8]
    frase = ""
    for i in range(0, len(bits_mensagem) - 7, 8):
        frase += chr(int(bits_mensagem[i:i+8], 2))

    print()
    print(f"MENSAGEM RECEBIDA: {frase}")
    print()

    # ==============================================================
    # FFT do sinal recebido e banda ocupada
    # ==============================================================
    fim_sinal = min(len(gravacao),
                    melhor_d + (36 + qtd_caracteres*8)*n_amostras)
    sinal = gravacao[melhor_d : fim_sinal]

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