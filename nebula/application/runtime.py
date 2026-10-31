"""Composição da captura e síntese do HarpLoom."""

from time import sleep

from nebula.infrastructure.audio.portaudio_output import AudioEngine
from nebula.infrastructure.vision.mediapipe_tracker import HarpLoomVision


def main() -> None:
    """Inicia visão e áudio; a thread principal só publica o evento mais recente."""
    vision = HarpLoomVision()
    audio = AudioEngine()
    try:
        vision.start()
        audio.start()
        while vision.error is None and not vision.stopped:
            audio.set_pluck(vision.latest().pluck)
            sleep(.004)
    except KeyboardInterrupt:
        pass
    finally:
        vision.close()
        audio.close()
    if vision.error:
        print(vision.error)
