## Orphan

---

### Описание

Проект представляет функционал генерации изображений по текстовому описанию (Text-to-Image). 
Для генерации используется локально запущенная модель `Stable Diffusion` (версий 1.5, 2.1).

**UPD: Важные уточнения:** 

Проект пока сырой, сделан на коленке за вечер (но полностью работоспособен). Даже его название намекает на это.
Возможно в дальнейшем приведу код в порядок, и оберну апишкой/ботом/чем-нибудь еще.

### Глоссарий:

- Инференс — это процесс работы нейросети (обученной) на конечном устройстве, или её логический вывод (конечный результат обработки данных).
  С точки зрения разработчика инференс это третий этап жизненного цикла искусственной нейронной сети (после её обучения и развёртывания).
- Stable Diffusion - это открытая диффузионная модель, предназначенная для генерации изображений на основе текстовых описаний.

--- 

### Локальный запуск

#### Добавление обученных весов модели к проекту

Для получения весов модели, необходимо выполнить три простых шага:

1) Найти репозиторий `Stable diffusion`, например на `Hugging Face`. Хотя и на других ресурсах есть куча обученных весов, на любой вкус.
   Если лень искать, то вот:
   - [version 1.5](https://huggingface.co/lmz/rust-stable-diffusion-v1-5/tree/main/weights)
   - [version 2.1](https://huggingface.co/lmz/rust-stable-diffusion-v2-1/tree/main/weights).

2) Скачать из репозитория файлы.

   - Для v1.5 это: `clip.safetensors`, `vae.safetensors`, `pytorch_model.safetensors`, `bpe_simple_vocab_16e6.txt`.
   - Для v2.1 это: `clip_v2.1.safetensors`, `vae_v2.1.safetensors`, `unet_v2.1.safetensors`, `bpe_simple_vocab_16e6.txt`.

3) Привести их названия к виду `clip.safetensors`, `vae.safetensors`, `unet.safetensors`, `bpe_simple_vocab_16e6.txt`.
   Положить все четыре файла в директорию `model` проекта, не забыв проверить пути к ним в `config.yaml`.

#### Конфигурация параметров генерации

Перед запуском, необходимо выставить нужные параметры для генерации изображений. 
Это можно сделать, перейдя в файл `config.yaml` (в корне проекта). 

#### Запуск

После выставления параметров, находясь в корне проекта, выполните нижеуказанную команду. Это запустит инференс модели, и генерацию изображений.
```Shell
cargo run --release
```

После окончания генерации, вы можете найти результаты в директории `assets` (если не переназначили в `config.yaml`).

### P.S.

Ставить звезды такому проекту не стоит!