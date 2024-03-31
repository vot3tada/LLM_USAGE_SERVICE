FROM nvidia/cuda:12.2.0-runtime-ubuntu20.04
RUN apt-get update && \
    apt-get install -y python3-pip python3-dev && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir "app"

COPY req.txt /app/req.txt
COPY telegram_bot.py app/telegram_bot.py
COPY totalRag.py app/totalReg.py
WORKDIR /app

RUN pip install --no-cache-dir -r req.txt

CMD ["python3", "-m", "pip", "install", "llama-cpp-python", "--prefer-binary", "--no-cache-dir", "--extra-index-url=https://jllllll.github.io/llama-cpp-python-cuBLAS-wheels/AVX2/cu122"]
CMD ["python3", "telegram_bot.py"]
