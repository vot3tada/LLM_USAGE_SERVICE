FROM nvidia/cuda:12.2.0-runtime-ubuntu20.04
RUN apt-get update && \
    apt-get install -y python3-pip python3-dev && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir "app"

WORKDIR /app
ADD ./req.txt /app/req.txt
RUN pip install llama-cpp-python --upgrade --force-reinstall --prefer-binary --no-cache-dir --extra-index-url=https://jllllll.github.io/llama-cpp-python-cuBLAS-wheels/AVX2/cu122
RUN pip install -r req.txt

ADD telegram_bot.py /app/telegram_bot.py
ADD totalRag.py /app/totalRag.py
ADD config.ini /app/config.ini

CMD ["python3", "telegram_bot.py"]
