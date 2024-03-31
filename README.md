# LLM_USAGE_SERVICE
### Инструкция установки с докером
* В папке проекта создать папку models и положить туда модели, указанные в конфиге
* Выполнить docker compose в корневой папке проекта

### Инструкция установки без докера
* Установить библиотеки из req.txt
* Настроить систему как https://github.com/abetlen/llama-cpp-python/discussions/871
* Если есть GPU: python -m pip install llama-cpp-python --prefer-binary --no-cache-dir --extra-index-url=https://jllllll.github.io/llama-cpp-python-cuBLAS-wheels/AVX2/cu122
* Установить библиотеки из req.txt
* Сконфигурировать конфиг файл пути до ваших моделей
* Сконфигурировать url до векторной базы QDrant (Если ее нет, контейнер можно поднять с помощью docker compose в папке QDrant)

## Модели
* LLM - https://huggingface.co/IlyaGusev/saiga_mistral_7b_gguf Подгрузить интересную вам из версий (мы использовали q4)
* Embed - https://huggingface.co/ai-forever/sbert_large_mt_nlu_ru Написать просто название ai-forever/sbert_large_mt_nlu_ru. Если не подгружается автоматически, скачать все файлы из версий, положить в папку, написать путь до папки в конфиг
* Папка с моделями (по сути зеркало): https://drive.google.com/drive/folders/1vDAn4aZe8XSDWkCulV2ni_7GIhX7CllQ?usp=sharing

## Общение с ИИ
* Если запускать из докера или файл telegram_bot, то он с ботом можно пообщаться в https://t.me/OhMyLittleFriendBot поддерживает одновременно один контекст для всех пользователей, иногда сообщения могут пропадать из-за блокировки основного потока моделью (отдельная задача борьбы с этим)
* Если запускать totalRag, то можно пообщаться с моделью в консольном режиме