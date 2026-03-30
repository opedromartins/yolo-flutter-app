# YOLO Flutter App — Benchmark de Modelos no Android

App Flutter para testar e benchmarkar modelos YOLO customizados em dispositivos Android físicos. Permite comparar o desempenho de modelos exportados em diferentes formatos (float32, float16, int8) diretamente no celular.

---

## Funcionalidades

- **Camera Inference** — inferência em tempo real pela câmera com FPS ao vivo
- **Single Image** — inferência em uma imagem por vez da galeria
- **Benchmark** — processa múltiplas imagens, mede tempo de inferência e exporta resultados em CSV
- **Inferência em lote (dataset)** — a partir da tela da câmera (ícone de galeria no canto superior direito): pasta ou imagens, escolha de modelo, exportação de JSON e CSV de tempos; no PC, métricas de deteção (P/R, mAP) com `example/scripts/compute_metrics.py`

---

## Pré-requisitos

- Windows 10+, macOS 12+ ou Linux (Ubuntu 22.04 ou similar)
- Celular Android com **modo desenvolvedor** ativado
- Cabo USB com suporte a transferência de dados
- Modelos exportados no formato `.tflite` com **batch size = 1**

> **Importante:** modelos exportados com batch size diferente de 1 causam falha no delegate GPU. O app roda na CPU por padrão para garantir compatibilidade.

---

## 1. Ativar depuração USB no celular

1. Vá em **Configurações → Sobre o telefone**
2. Toque **7 vezes** em **Número da versão** para ativar o modo desenvolvedor
3. Vá em **Configurações → Opções do desenvolvedor**
4. Ative **Depuração USB**
5. Conecte o cabo USB ao computador
6. No celular, confirme o popup **"Permitir depuração USB"** → toque em **Permitir**

> **Dica:** quando aparecer a opção "Uso do USB", selecione **Transferência de arquivos (MTP)**.

> **Após reiniciar o celular**, o popup de autorização aparece novamente — confirme novamente.

---

## 2. Instalar o Java 17

O Gradle exige Java 17. Sem ele o build falha com `Cannot find a Java installation`.

**macOS:**
```bash
brew install --cask temurin@17
```
Adicione ao `~/.zshrc`:
```bash
export JAVA_HOME=$(/usr/libexec/java_home -v 17)
source ~/.zshrc
```

**Linux:**
```bash
sudo apt install openjdk-17-jdk
```

**Windows:** Baixe o instalador do [Adoptium Temurin 17](https://adoptium.net/temurin/releases/?version=17).

Verifique:
```bash
java -version
# openjdk version "17.x.x"
```

---

## 3. Instalar o Flutter

**macOS:**
```bash
brew install --cask flutter
```

**Linux:**
```bash
sudo snap install flutter --classic
```

**Windows:**
```powershell
winget install --id Google.Flutter
```

Verifique (também baixa o SDK, ~1.4 GB):
```bash
flutter doctor
```

---

## 4. Instalar o Android Studio e o Android SDK

Baixe em [developer.android.com/studio](https://developer.android.com/studio) e instale para o seu sistema. Na primeira abertura o Android SDK é instalado automaticamente.

Em seguida, dentro do Android Studio:
1. **More Actions → SDK Manager → SDK Tools**
2. Marque **Android SDK Command-line Tools (latest)** → Apply

Aceite as licenças:
```bash
flutter doctor --android-licenses
```

Resultado esperado:
```
[✓] Flutter
[✓] Android toolchain
[✓] Connected devices
```

> Erros de Xcode/CocoaPods podem ser ignorados se o objetivo é somente Android.

---

## 5. Adicionar os modelos ao projeto

Coloque os arquivos `.tflite` em:

```
example/android/app/src/main/assets/
```

**macOS / Linux:**
```bash
cp /caminho/para/model_float32.tflite android/app/src/main/assets/
cp /caminho/para/model_float16.tflite android/app/src/main/assets/
cp /caminho/para/model_int8.tflite    android/app/src/main/assets/
```

**Windows (PowerShell):**
```powershell
Copy-Item "C:\caminho\para\model_float32.tflite" "android\app\src\main\assets\"
Copy-Item "C:\caminho\para\model_float16.tflite" "android\app\src\main\assets\"
Copy-Item "C:\caminho\para\model_int8.tflite"    "android\app\src\main\assets\"
```

---

## 6. Registrar os modelos no código

Abra `lib/models/models.dart` e adicione uma entrada no enum `ModelType` para cada modelo:

```dart
enum ModelType {
  detect('yolo11n', YOLOTask.detect),
  // ... outros modelos padrão ...
  customFloat32('model_float32', YOLOTask.detect),
  customFloat16('model_float16', YOLOTask.detect),
  customInt8('model_int8', YOLOTask.detect);

  final String modelName;
  final YOLOTask task;
  const ModelType(this.modelName, this.task);
}
```

> O `modelName` deve ser o nome do arquivo **sem a extensão** `.tflite`.

---

## 7. Rodar o app no celular

```bash
flutter pub get
flutter devices    # confirma que o celular aparece
flutter run
```

Na primeira compilação o Gradle baixa dependências — pode levar 5 a 15 minutos. As seguintes são incrementais e bem mais rápidas.

---

## 8. Transferir o dataset para o celular

O benchmark usa o **Construction Site Safety Image Dataset** do Kaggle:

> [kaggle.com/datasets/snehilsanyal/construction-site-safety-image-dataset-roboflow](https://www.kaggle.com/datasets/snehilsanyal/construction-site-safety-image-dataset-roboflow)

Estrutura após extrair:
```
css-data/
├── train/images/    ← não usar para benchmark
├── valid/images/    ← 114 imagens
└── test/images/     ← 82 imagens  ← recomendado
```

Use `test/images/` ou `valid/images/` — o modelo não foi treinado nelas, garantindo avaliação justa. Use o **mesmo conjunto para todos os modelos**.

**Transferir via adb:**
```bash
adb push /caminho/css-data/test/images /sdcard/Pictures/css-dataset/test
```

> No macOS, o `adb` fica em `~/Library/Android/sdk/platform-tools/adb` se não estiver no PATH.

Após transferir, **abra a Galeria no celular** para indexar as imagens. Se não aparecerem, reinicie o celular.

---

## 9. Usar a tela de Benchmark

1. Abra o app → toque em **Benchmark**
2. Selecione o modelo no dropdown (`model_float32`, `model_float16` ou `model_int8`)
3. Toque em **Selecionar da galeria** e escolha as imagens (até 100 por vez)
4. Toque em **Rodar Benchmark**
5. Aguarde — a barra de progresso indica o andamento

### Métricas exibidas

| Métrica | Descrição |
|--------|-----------|
| Imagens processadas | Total de imagens processadas com sucesso |
| Tempo médio | Tempo médio de inferência por imagem (ms) |
| Tempo mínimo / máximo | Intervalo de tempo registrado (ms) |
| FPS equivalente | 1000 / tempo_médio — velocidade de processamento |
| Total de detecções | Objetos detectados em todas as imagens |
| Média detecções/imagem | Média de objetos por imagem |

### Exportar resultados

- **Salvar CSV** — salva no armazenamento interno do app
- **Compartilhar** — abre o share sheet do Android (email, Drive, etc.)

Para recuperar o CSV via terminal:
```bash
adb pull /data/user/0/com.ultralytics.yolo_example/files/benchmark_model_float32_<timestamp>.csv .
```

Formato do CSV:
```
image,inference_time_ms,detections
img001.jpg,45.2,3
img002.jpg,38.7,1
...
SUMMARY
model,images,avg_ms,min_ms,max_ms,fps,total_detections,avg_detections
model_float32,82,41.9,35.1,58.3,23.8,190,2.3
```

---

## 10. Inferência em lote, CSV de tempo e métricas no PC (`compute_metrics.py`)

Fluxo: **inferir no app (lote)** → **exportar JSON (e opcionalmente CSV de tempos)** → **calcular métricas de qualidade no computador** com o script Python. O CSV do app mede **desempenho** (latência, FPS); precisão, recall, F1 e mAP vêm do script no PC.

### 10.1. Onde fica no app

1. Tela inicial → **Camera Inference**
2. Ícone de **galeria / pastas** (canto superior direito) — abre a inferência em lote no dataset
3. Escolha o **modelo** no dropdown (ex.: `model_float16`, `exp_yolo26m_epi_negative_float32`)
4. **Selecionar pasta** (Android: gestor de arquivos) ou **Selecionar imagens da galeria**

**Pasta recomendada:** preferir uma pasta que contenha **imagens e labels** juntas (ex.: `valid/` com `images/` e `labels/`), para o app associar `.txt` pelo **nome do arquivo sem extensão**. Se só selecionar `images/`, o JSON ainda terá predições; no PC, o `compute_metrics.py` só encontra GT se o stem do `file` no JSON coincidir com um `.txt` em `--labels`.

### 10.2. Exportar resultados do lote

| Ação | Conteúdo |
|------|----------|
| **JSON completo** | `results`, tempos, `summary`, `benchmark_summary` (e campos extras se houver labels no lote) |
| **JSON só detecções** | `model` + `results` com `file`, `detections`, `inference_time_ms`, `boxes` (ou `error`) |
| **CSV (cartão Benchmark)** | **Salvar CSV** / **CSV**: uma linha por imagem (`image`, `inference_time_ms`, `detections`); bloco `SUMMARY` com média/mín/máx ms, FPS, detecções, falhas; nome do arquivo tipo `batch_<modelo>_<timestamp>.csv`. Pode haver colunas vazias ou preenchidas para precisão/recall/etc. se labels foram carregadas no app |

Use **compartilhar** para enviar os arquivos ao PC (Download, Drive, USB).

### 10.3. Rodar `compute_metrics.py` no PC

Coloque o JSON exportado e aponte a pasta das labels YOLO (ex.: `.../valid/labels`).

**PowerShell (Windows):** não use `^` para quebrar linhas (isso é do `cmd`). Use uma linha só ou **backtick** `` ` `` no fim de cada linha:

```powershell
cd caminho\para\yolo-flutter-app\example\scripts

python compute_metrics.py `
  --inference "D:\caminho\para\resultado.json" `
  --labels "D:\caminho\para\dataset\css-data\valid\labels"
```

Exemplo em uma linha:

```powershell
python compute_metrics.py --inference "D:\resultado.json" --labels "D:\dataset\valid\labels"
```

| Argumento | Descrição |
|-----------|-----------|
| `--inference` / `-i` | Caminho do JSON exportado pelo app |
| `--labels` / `-l` | Pasta com os `.txt` YOLO |
| `--iou` | IoU para TP (padrão `0.5`) |
| `--conf` | Confiança mínima (padrão `0.001`) |
| `--classes` / `-c` | Opcional — só se o mapa de classes for diferente do padrão do script |
| `--output` / `-o` | Salvar métricas em JSON |
| `--debug` | Amostras predição vs GT |

O script já traz um **mapa de classes padrão** (dataset EPI / construction). `--classes` só é necessário se seus IDs/nomes divergirem.

O script usa o **stem** do nome no JSON (`file`) e procura o mesmo stem nos `.txt` (ex.: `foto.jpg` → `foto.txt`). Se TP = 0 e FP > 0, confira nomes, classes e teste `--iou 0.3` e `--debug`.

### 10.4. Resumo do fluxo

1. (Opcional) Copiar dataset com imagens + labels para o celular ou usar `adb push`
2. App → inferência em lote → pasta com imagens (idealmente `valid` com `images` + `labels`)
3. Exportar **JSON completo** e, se quiser relatório de tempo, o **CSV** do cartão Benchmark
4. No PC: `python compute_metrics.py --inference <json> --labels <pasta_labels>`
5. Opcional: `--output metricas.json`

---

## Solução de problemas

| Problema | Solução |
|---------|---------|
| `flutter devices` não lista o celular | Verifique o cabo, reative depuração USB, confirme popup no celular |
| `Cannot find a Java installation` | Instale Java 17 e configure `JAVA_HOME` (ver seção 2) |
| `Unable to locate Android SDK` | Abra o Android Studio e instale o SDK pela primeira vez |
| `cmdline-tools component is missing` | No SDK Manager, instale **Android SDK Command-line Tools** |
| `Device is not authorized` | Confirme o popup de autorização no celular |
| `Model not found` | Verifique se o `.tflite` está em `assets/` e o `modelName` no enum está sem extensão |
| Métricas zeradas / erro de delegate GPU | Modelos com batch size > 1 — re-exporte com `batch_size=1` ou garanta `useGpu: false` no controller |
| `flutter.jar` not found | Rode `flutter clean && flutter precache --android && flutter run` |
| Flutter não encontrado no PATH (macOS) | Adicione `export PATH="$PATH:$HOME/flutter/bin"` ao `~/.zshrc` |
| Imagens não aparecem na galeria | Reinicie o celular para forçar a indexação |
| `snap` travado em 0B/s (Linux) | Cancele com `Ctrl+C` e baixe manualmente pelo site oficial |

---

## Estrutura do projeto

```
example/
├── android/app/src/main/assets/  ← modelos .tflite aqui
├── scripts/
│   └── compute_metrics.py        ← métricas P/R/mAP a partir do JSON do app
└── lib/
    ├── main.dart                 ← entry point (HomeScreen)
    ├── models/
    │   ├── models.dart           ← registrar novos modelos aqui
    │   └── benchmark_result.dart ← data classes do benchmark
    ├── services/
    │   └── model_manager.dart    ← carregamento de modelos
    ├── utils/
    │   └── detection_eval_metrics.dart  ← métricas on-device (opcional no lote)
    └── presentation/
        ├── controllers/
        │   ├── camera_inference_controller.dart
        │   └── benchmark_controller.dart  ← lógica do benchmark
        └── screens/
            ├── home_screen.dart            ← tela inicial
            ├── camera_inference_screen.dart  ← ícone galeria → inferência em lote
            ├── single_image_screen.dart
            ├── benchmark_screen.dart       ← tela de benchmark
            └── batch_inference_screen.dart ← inferência em lote + JSON/CSV
```
