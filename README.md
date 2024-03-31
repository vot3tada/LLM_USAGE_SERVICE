# LLM_USAGE_SERVICE
### Инструкция в случае проблем с докером и для работы с видеокартой
* Установить библиотеки из req.txt
* Установить как в пункте Prerequisites https://github.com/abetlen/llama-cpp-python/discussions/871
* python -m pip install llama-cpp-python --prefer-binary --no-cache-dir --extra-index-url=https://jllllll.github.io/llama-cpp-python-cuBLAS-wheels/AVX2/cu122

### Если запускать на процессоре
* Установить библиотеки из req.txt
*  pip install llama-cpp-python

## Модели
* LLM - https://huggingface.co/IlyaGusev/saiga_mistral_7b_gguf Подгрузить интересную вам из версий (мы использовали q4)
* Embed - https://huggingface.co/ai-forever/sbert_large_mt_nlu_ru Написать просто название ai-forever/sbert_large_mt_nlu_ru. Если не подгружается автоматически, скачать все файлы из версий, положить в папку, написать путь до папки в конфиг