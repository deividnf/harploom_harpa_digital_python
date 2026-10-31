"""Captura, landmarks e visualização do instrumento HarpLoom."""

from pathlib import Path
from threading import Event, Lock, Thread
from time import monotonic

import cv2
import mediapipe as mp

from nebula.domain.harploom import HandPose, HarpLoomController, PerformanceFrame


class HarpLoomVision:
    """Adaptador de webcam/MediaPipe e visualização da performance.

    Captura e inferência ficam nesta thread; a aplicação lê apenas o último
    PerformanceFrame disponível, sem acumular quadros antigos.
    """

    MODEL = Path(__file__).resolve().parents[3] / "models" / "hand_landmarker.task"
    BONES = ((0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
             (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15),
             (15, 16), (13, 17), (17, 18), (18, 19), (19, 20), (0, 17))

    def __init__(self, camera_index: int = 1, width: int = 640, height: int = 480) -> None:
        self.camera_index, self.width, self.height = camera_index, width, height
        self._stop = Event()
        self._lock = Lock()
        self._performance = PerformanceFrame()
        self._thread: Thread | None = None
        self.error: str | None = None
        self.stopped = False

    def start(self) -> None:
        self._thread = Thread(target=self._run, name="harploom-vision", daemon=True)
        self._thread.start()

    def latest(self) -> PerformanceFrame:
        with self._lock:
            return self._performance

    def _draw(self, frame, hands: tuple[HandPose, ...], state: PerformanceFrame) -> None:
        h, w = frame.shape[:2]
        top, bottom = int(h * .12), int(h * .90)
        rail_x1, rail_x2 = int(w * .875), int(w * .965)
        cv2.rectangle(frame, (rail_x1, top), (rail_x2, bottom), (35, 43, 54), -1)
        cv2.putText(frame, "NIVEL", (rail_x1, top - 9), cv2.FONT_HERSHEY_SIMPLEX, .4, (235, 240, 245), 1)
        level_h = (bottom - top) / 5
        for level in range(1, 6):
            y1 = int(bottom - level * level_h)
            y2 = int(bottom - (level - 1) * level_h)
            color = (70, 190, 150) if level == state.pitch_level else (100, 115, 130)
            cv2.rectangle(frame, (rail_x1 + 3, y1 + 2), (rail_x2 - 3, y2 - 2), color, -1)
            cv2.putText(frame, str(level), (rail_x1 + 14, y2 - 6), cv2.FONT_HERSHEY_SIMPLEX, .5, (245, 245, 245), 1)
        for i, x_norm in enumerate(HarpLoomController.STRING_X):
            x = int(x_norm * w)
            color = (190, 225, 235) if state.crossed_string == i else (120, 170, 190)
            thickness = 3 if state.crossed_string == i else 1
            cv2.line(frame, (x, top), (x, bottom), color, thickness, cv2.LINE_AA)
            cv2.putText(frame, "CDEFGABC"[i], (x - 7, top - 9), cv2.FONT_HERSHEY_SIMPLEX,
                        .5, (235, 240, 245), 1, cv2.LINE_AA)

        for hand in hands:
            color = (70, 230, 180) if hand.side == "Left" else (80, 180, 255)
            for a, b in self.BONES:
                p1, p2 = hand.landmarks[a], hand.landmarks[b]
                cv2.line(frame, (int(p1[0] * w), int(p1[1] * h)),
                         (int(p2[0] * w), int(p2[1] * h)), color, 2, cv2.LINE_AA)
            for index in (4, 8, 12, 16, 20):
                point = hand.landmarks[index]
                cv2.circle(frame, (int(point[0] * w), int(point[1] * h)), 5, color, -1)
            wrist = hand.landmarks[0]
            cv2.putText(frame, hand.side.upper(), (int(wrist[0] * w) + 8, int(wrist[1] * h)),
                        cv2.FONT_HERSHEY_SIMPLEX, .48, color, 1, cv2.LINE_AA)

        cv2.rectangle(frame, (0, 0), (w, 35), (20, 25, 34), -1)
        cv2.putText(frame, "HARPLOOM  |  cruze as cordas com a mao direita",
                    (10, 24), cv2.FONT_HERSHEY_SIMPLEX, .55, (245, 245, 245), 1, cv2.LINE_AA)
        cv2.rectangle(frame, (0, h - 43), (w, h), (20, 25, 34), -1)
        left_text = f"NIVEL DE PITCH {state.pitch_level} | indicador esquerdo seleciona a barra lateral"
        right_text = (f"abertura direita: {state.right_spread:.0%} | cruze cordas com o indicador" if state.right_spread is not None
                      else "cruze as cordas com o indicador direito")
        cv2.putText(frame, left_text, (9, h - 25), cv2.FONT_HERSHEY_SIMPLEX, .38, (70, 230, 180), 1, cv2.LINE_AA)
        cv2.putText(frame, right_text, (9, h - 7), cv2.FONT_HERSHEY_SIMPLEX, .38, (80, 180, 255), 1, cv2.LINE_AA)

    def _run(self) -> None:
        if not self.MODEL.is_file():
            self.error = f"Modelo de mão ausente: {self.MODEL}"
            self.stopped = True
            return
        # DirectShow funciona bem com webcams comuns no Windows e permite escolher o índice.
        capture = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not capture.isOpened():
            self.error = "Não foi possível abrir a câmera."
            self.stopped = True
            return
        controller = HarpLoomController()
        try:
            vision = mp.tasks.vision
            options = vision.HandLandmarkerOptions(
                base_options=mp.tasks.BaseOptions(model_asset_buffer=self.MODEL.read_bytes()),
                running_mode=vision.RunningMode.VIDEO,
                num_hands=2,
                min_hand_detection_confidence=.55,
                min_hand_presence_confidence=.55,
                min_tracking_confidence=.55,
            )
            with vision.HandLandmarker.create_from_options(options) as landmarker:
                while not self._stop.is_set():
                    ok, frame = capture.read()
                    if not ok:
                        continue
                    # Espelhar a imagem deixa o movimento visual intuitivo, como num espelho.
                    frame = cv2.flip(frame, 1)
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                    result = landmarker.detect_for_video(image, int(monotonic() * 1000))
                    poses: list[HandPose] = []
                    for i, landmarks in enumerate(result.hand_landmarks or []):
                        side = "Unknown"
                        if result.handedness and i < len(result.handedness):
                            side = result.handedness[i][0].category_name
                        poses.append(HandPose(side, tuple((float(p.x), float(p.y)) for p in landmarks)))
                    hands = tuple(poses)
                    state = controller.update(hands)
                    with self._lock:
                        self._performance = state
                    self._draw(frame, hands, state)
                    cv2.imshow("HarpLoom", frame)
                    cv2.waitKey(1)
                    if cv2.getWindowProperty("HarpLoom", cv2.WND_PROP_VISIBLE) < 1:
                        self._stop.set()
        except Exception as exc:
            self.error = f"Falha na visão: {exc}"
        finally:
            self.stopped = True
            capture.release()
            cv2.destroyAllWindows()

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
