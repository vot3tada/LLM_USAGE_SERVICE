# LLM_USAGE_SERVICE
### Инструкция в случае проблем с докером и для работы с видеокартой
* Установить библиотеки из req.txt
* Настроить систему как https://github.com/abetlen/llama-cpp-python/discussions/871
* python -m pip install llama-cpp-python --prefer-binary --no-cache-dir --extra-index-url=https://jllllll.github.io/llama-cpp-python-cuBLAS-wheels/AVX2/cu122

### Если запускать на процессоре
* Установить библиотеки из req.txt

## Модели
* LLM - https://huggingface.co/IlyaGusev/saiga_mistral_7b_gguf Подгрузить интересную вам из версий (мы использовали q4)
* Embed - https://huggingface.co/ai-forever/sbert_large_mt_nlu_ru Написать просто название ai-forever/sbert_large_mt_nlu_ru. Если не подгружается автоматически, скачать все файлы из версий, положить в папку, написать путь до папки в конфиг
* Папка с моделями (по сути зеркало): https://drive.google.com/drive/folders/1vDAn4aZe8XSDWkCulV2ni_7GIhX7CllQ?usp=sharing
## Общение с ИИ
* Если запускать из докера или файл telegram_bot, то он с ботом можно пообщаться в https://t.me/OhMyLittleFriendBot поддерживает одновременно один контекст для всех пользователей, иногда сообщения могут пропадать из-за блокировки основного потока моделью (отдельная задача борьбы с этим)
* Если запускать totalRag, то можно пообщаться с моделью в консольном режиме