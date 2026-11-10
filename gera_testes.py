import numpy as np
from scipy.io import wavfile
from scipy import signal

fs = 44100
duracao = 5
t = np.linspace(0, duracao, fs * duracao, endpoint=False)


def envelope(x, ataque=0.01, queda=0.01):
    """Fade in/out curto para nao estourar click no inicio e no fim."""
    n = len(x)
    na = int(ataque * fs)
    nq = int(queda * fs)
    env = np.ones(n)
    env[:na] = np.linspace(0, 1, na)
    env[-nq:] = np.linspace(1, 0, nq)
    return x * env


def normaliza(x, pico=0.85):
    return pico * x / np.max(np.abs(x))


def salva(nome, x):
    x = normaliza(envelope(x))
    wavfile.write(nome, fs, (x * 32767).astype(np.int16))
    return x


# ---------------------------------------------------------------
# AUDIO 1 - arpejo de tons puros (sobe em degraus)
# 5 notas de 1 s: 220, 330, 440, 550, 660 Hz
# ---------------------------------------------------------------
notas = [220, 330, 440, 550, 660]
a1 = np.zeros_like(t)
dur_nota = duracao / len(notas)
for i, f in enumerate(notas):
    ini = int(i * dur_nota * fs)
    fim = int((i + 1) * dur_nota * fs)
    tn = t[ini:fim] - t[ini]
    nota = np.sin(2 * np.pi * f * tn)
    # envelope proprio da nota, para separar os degraus
    env = np.minimum(1, np.linspace(0, 1, len(tn)) * 20)
    env *= np.minimum(1, np.linspace(1, 0, len(tn)) * 20)
    a1[ini:fim] = nota * env

# ---------------------------------------------------------------
# AUDIO 2 - varredura (chirp) linear de 200 Hz a 1400 Hz
# ---------------------------------------------------------------
a2 = signal.chirp(t, f0=200, f1=1400, t1=duracao, method='linear')

# ---------------------------------------------------------------
# AUDIO 3 - tom rico: fundamental 150 Hz + harmonicos ate 1350 Hz,
# com tremolo de 4 Hz (amplitude oscilando)
# ---------------------------------------------------------------
a3 = np.zeros_like(t)
for n in range(1, 10):                      # 150, 300, ..., 1350 Hz
    a3 += (1.0 / n) * np.sin(2 * np.pi * 150 * n * t)
tremolo = 0.65 + 0.35 * np.sin(2 * np.pi * 4 * t)
a3 *= tremolo

x1 = salva('teste1.wav', a1)
x2 = salva('teste2.wav', a2)
x3 = salva('teste3.wav', a3)

# ---------------------------------------------------------------
# verificacao: nenhuma energia relevante acima de 1500 Hz
# ---------------------------------------------------------------
for nome, x in [('teste1', x1), ('teste2', x2), ('teste3', x3)]:
    yf = np.abs(np.fft.rfft(x)) / len(x)
    xf = np.fft.rfftfreq(len(x), d=1 / fs)
    total = np.sum(yf ** 2)
    acima = np.sum(yf[xf > 1500] ** 2)
    pico = xf[np.argmax(yf)]
    print(f"{nome}: pico em {pico:7.1f} Hz | energia acima de 1500 Hz = "
          f"{100 * acima / total:.4f}%")
