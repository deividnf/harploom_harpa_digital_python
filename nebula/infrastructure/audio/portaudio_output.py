"""Síntese de corda dedilhada em callback contínuo de baixa latência."""

import numpy as np
import sounddevice as sd

from nebula.domain.harploom import Pluck


class AudioEngine:
    """Saída mono duplicada em estéreo; novos ataques substituem a nota anterior."""

    def __init__(self, sample_rate: int = 48_000, blocksize: int = 256) -> None:
        self.sample_rate, self.blocksize = sample_rate, blocksize
        self._pluck = Pluck()
        self._event_id = 0
        self._age = 0
        self._phase = 0.0
        self._stream: sd.OutputStream | None = None

    def set_pluck(self, pluck: Pluck) -> None:
        self._pluck = pluck

    def _callback(self, outdata, frames, time_info, status) -> None:
        # Este callback tem prazo de tempo real: sem câmera, disco, logs ou esperas.
        pluck = self._pluck
        if pluck.event_id != self._event_id:
            # Um novo identificador reinicia o envelope para evitar sobrepor notas antigas.
            self._event_id = pluck.event_id
            self._age = 0
        if pluck.frequency <= 0 or pluck.event_id == 0:
            outdata.fill(0)
            return
        sample = self._age + np.arange(frames, dtype=np.float32)
        seconds = sample / self.sample_rate
        # Envelope exponencial simples aproxima o decaimento de uma corda dedilhada.
        decay = np.exp(-seconds / 1.15)
        attack = np.minimum(1.0, sample / 96.0)
        # A abertura da mão direita controla uma modulação leve do pitch.
        vibrato = 1.0 + (0.0001 + 0.0015 * pluck.modulation) * np.sin(2 * np.pi * 5.2 * seconds)
        phase = self._phase + np.cumsum((2 * np.pi * pluck.frequency / self.sample_rate) * vibrato)
        signal = pluck.amplitude * attack * decay * np.sin(phase)
        outdata[:, 0] = signal
        if outdata.shape[1] > 1:
            outdata[:, 1] = signal
        if frames:
            self._phase = float(phase[-1] % (2 * np.pi))
        self._age += frames

    def start(self) -> None:
        self._stream = sd.OutputStream(samplerate=self.sample_rate, blocksize=self.blocksize,
                                       channels=2, dtype="float32", latency="low",
                                       callback=self._callback)
        self._stream.start()

    def close(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
