# HarpLoom

**Uma harpa digital tocada pelo movimento.** O HarpLoom transforma cruzamentos das mãos em notas: o indicador direito dedilha cordas verticais no espaço, enquanto o indicador esquerdo seleciona uma faixa de altura musical numa barra lateral. Abertura da mão direita acrescenta uma modulação sutil.

O projeto está em fase de protótipo. A prioridade atual é tornar o gesto previsível e confortável antes de ampliar os timbres, escalas e recursos de composição.

## Como tocar

1. Inicie o programa e posicione-se de modo que as duas mãos apareçam na câmera.
2. Para mudar a faixa de pitch, leve o indicador esquerdo até a barra **NÍVEL** à direita e mantenha-o sobre um dos cinco níveis por um instante. A seleção fica memorizada; a mão direita continua dedicada às cordas.
3. Com a mão direita, estenda o indicador e faça uma varredura lateral cruzando uma das cordas verticais. Cada corda corresponde a um grau de Dó maior.
4. Abra ou feche a mão direita para alterar a modulação da nota.

Só o indicador direito aciona as cordas. O cruzamento precisa acontecer dentro da área musical, com velocidade suficiente e com o dedo estendido. Esses filtros reduzem disparos causados por tremores e movimentos casuais.

**Encerrar:** feche a janela do HarpLoom ou pressione `Ctrl+C` no terminal.

## Requisitos

- Windows 10 ou 11
- Python 3.11–3.13
- Webcam
- Saída de áudio e driver compatível com PortAudio

O protótipo usa MediaPipe Tasks para localizar as mãos, OpenCV para câmera e visualização, NumPy para síntese e `sounddevice`/PortAudio para áudio. O arquivo do modelo está em `models/hand_landmarker.task` e é carregado em memória para acomodar caminhos do Windows com caracteres acentuados.

## Instalação

No PowerShell, na pasta do projeto:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python main.py
```

Se o comando `py -3.12` não estiver disponível, use a versão de Python instalada para criar o ambiente virtual. O índice da câmera padrão é `1`; se a janela não abrir a webcam desejada, altere `camera_index` no construtor de `HarpLoomVision` em `nebula/infrastructure/vision/mediapipe_tracker.py` (valores comuns são `0` e `1`).

## Arquitetura

```text
main.py
└── application/runtime.py
    ├── infrastructure/vision/mediapipe_tracker.py
    │   └── câmera, landmarks, gesto e desenho das cordas
    ├── domain/harploom.py
    │   └── níveis de pitch, cruzamentos e eventos musicais
    └── infrastructure/audio/portaudio_output.py
        └── callback de áudio e envelope da nota
```

O rastreamento roda em uma thread própria e publica somente o estado mais recente. O áudio roda em um callback contínuo, separado da câmera. Assim, quadros atrasados não formam uma fila que aumente a latência musical. O callback não acessa arquivos nem espera a visão computacional.

## Mapeamento atual

| Gesto | Resposta |
| --- | --- |
| Indicador esquerdo mantido na barra vertical à direita | Seleciona e memoriza um de cinco níveis de pitch |
| Indicador direito estendido cruza uma corda | Aciona uma nota da escala de Dó maior |
| Mão direita mais aberta ou mais fechada | Altera levemente a modulação do pitch |

O modelo de áudio atual usa uma onda senoidal com decaimento curto. A modulação não é ainda um eco ou filtro. As oito cordas correspondem a Dó, Ré, Mi, Fá, Sol, Lá, Si e Dó. Os cinco níveis transpõem essas notas em oitavas.

## Ajustes de resposta

Os valores iniciais de detecção e conforto ficam em `nebula/domain/harploom.py`:

- `STRING_X`: espaçamento e posição horizontal das cordas.
- `0.28`: tempo de permanência do indicador esquerdo para confirmar um nível.
- `0.008`: margem de cruzamento para reduzir oscilações na borda da corda.
- `0.18`: velocidade horizontal mínima do indicador direito, em coordenadas normalizadas por segundo.
- `0.045`: intervalo mínimo entre ataques.

São pontos de partida para experimentação, não valores universais. Câmera, iluminação, distância e conforto de cada pessoa podem pedir calibração.

## Privacidade

O vídeo é processado localmente e não é gravado nem enviado pela aplicação. A câmera só é acessada durante a execução.

## Próximos passos

- Calibração guiada de câmera e área confortável.
- Testes de conforto e confiabilidade dos cruzamentos.
- Mais escalas, tonalidades e opções de afinação.
- Timbres, ressonância e efeitos com controle gestual.
- Cenas musicais e configurações salvas.

## Modelo de visão

O modelo `hand_landmarker.task` é da família MediaPipe Hand Landmarker. Consulte a [documentação oficial do MediaPipe](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker) para detalhes de uso e licença do modelo.
