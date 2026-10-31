"""Gramática musical do HarpLoom, sem dependência de câmera ou áudio."""

from dataclasses import dataclass
from math import pow
from time import monotonic


@dataclass(frozen=True, slots=True)
class HandPose:
    """Landmarks normalizados da imagem; os nomes Left/Right vêm do MediaPipe."""

    side: str
    landmarks: tuple[tuple[float, float], ...]


@dataclass(frozen=True, slots=True)
class Pluck:
    """Um ataque sonoro imutável, publicado para o callback de áudio."""

    frequency: float = 0.0
    amplitude: float = 0.0
    modulation: float = 0.0
    event_id: int = 0


@dataclass(frozen=True, slots=True)
class PerformanceFrame:
    """Estado visual mais recente e o último ataque reconhecido."""

    pitch_level: int = 3
    right_spread: float | None = None
    crossed_string: int | None = None
    pluck: Pluck = Pluck()


class HarpLoomController:
    """Converte poses em ataques e níveis de pitch sem depender de hardware."""

    STRING_X = tuple(.16 + i * .097 for i in range(8))
    SCALE = (0, 2, 4, 5, 7, 9, 11, 12)  # Dó maior inicial, uma oitava

    def __init__(self, tonic_midi: int = 60) -> None:
        self.tonic_midi = tonic_midi
        self._previous_x: dict[tuple[str, int], float] = {}
        self._event_id = 0
        self._last_pluck = Pluck()
        self._pitch_level = 3
        self._level_candidate: int | None = None
        self._level_candidate_since = 0.0
        self._last_any_crossing = 0.0

    @staticmethod
    def _right_hand_spread(landmarks: tuple[tuple[float, float], ...]) -> float:
        # Abertura entre polegar e mínimo, normalizada pelo comprimento da palma.
        thumb, pinky = landmarks[4], landmarks[20]
        wrist, middle_base = landmarks[0], landmarks[9]
        palm = max(.04, ((middle_base[0] - wrist[0]) ** 2 + (middle_base[1] - wrist[1]) ** 2) ** .5)
        spread = (((thumb[0] - pinky[0]) ** 2 + (thumb[1] - pinky[1]) ** 2) ** .5) / (palm * 2.2)
        return min(1.0, max(0.0, spread))

    def update(self, hands: tuple[HandPose, ...], now: float | None = None) -> PerformanceFrame:
        """Atualiza controles e reconhece no máximo um ataque por quadro.

        O nível de pitch é selecionado por permanência na barra lateral direita;
        cruzamentos das cordas exigem indicador estendido e em movimento.
        """
        now = monotonic() if now is None else now
        left = next((hand for hand in hands if hand.side == "Left"), None)
        right = next((hand for hand in hands if hand.side == "Right"), None)
        if left is not None:
            tip_x, tip_y = left.landmarks[8]
            # A barra permanece à direita; o indicador esquerdo escolhe um nível memorizado.
            if .875 <= tip_x <= .965 and .12 <= tip_y <= .90:
                # A barra tem cinco zonas; um dwell curto impede seleção acidental ao passar.
                candidate = 5 - min(4, int((tip_y - .12) / .156))
                if candidate != self._level_candidate:
                    self._level_candidate = candidate
                    self._level_candidate_since = now
                elif now - self._level_candidate_since >= .28:
                    self._pitch_level = candidate
            else:
                self._level_candidate = None
        spread = self._right_hand_spread(right.landmarks) if right else None
        crossed: int | None = None

        if right is not None:
            fingertip, pip, wrist, mcp = right.landmarks[8], right.landmarks[6], right.landmarks[0], right.landmarks[5]
            palm = max(.04, ((mcp[0] - wrist[0]) ** 2 + (mcp[1] - wrist[1]) ** 2) ** .5)
            tip_distance = ((fingertip[0] - wrist[0]) ** 2 + (fingertip[1] - wrist[1]) ** 2) ** .5
            pip_distance = ((pip[0] - wrist[0]) ** 2 + (pip[1] - wrist[1]) ** 2) ** .5
            x, y = fingertip
            key = (right.side, 8)
            previous = self._previous_x.get(key)
            previous_time = self._previous_x.get((right.side, -1), now)
            self._previous_x[key] = x
            self._previous_x[(right.side, -1)] = now
            speed = abs(x - previous) / max(.001, now - previous_time) if previous is not None else 0.0
            # Exigir extensão do indicador evita que uma mão aberta raspe várias cordas.
            index_extended = tip_distance > pip_distance + .12 * palm
            if previous is not None and index_extended and .12 <= y <= .90 and speed >= .18 and now - self._last_any_crossing > .045:
                for index, line_x in enumerate(self.STRING_X):
                    # Uma margem pequena mantém rejeição de tremor sem exigir uma varredura brusca.
                    if (previous < line_x - .008 and x >= line_x + .008) or (previous > line_x + .008 and x <= line_x - .008):
                        crossed = index
                        self._last_any_crossing = now
                        break
        else:
            self._previous_x = {key: value for key, value in self._previous_x.items() if key[0] != "Right"}

        if crossed is not None and spread is not None:
            # Cada nível desloca uma oitava; a corda escolhe o grau dentro dela.
            register = (self._pitch_level - 1) * 12
            midi = self.tonic_midi + self.SCALE[crossed] + register
            self._event_id += 1
            self._last_pluck = Pluck(
                440.0 * pow(2.0, (midi - 69) / 12.0),
                .12 + .12 * spread,
                spread,
                self._event_id,
            )
        return PerformanceFrame(self._pitch_level, spread, crossed, self._last_pluck)
