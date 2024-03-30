import llama_cpp
import torch
import time
import sys
import asyncio

import fire
from llama_cpp import Llama

from transformers import AutoModelForCausalLM, AutoTokenizer
from aiogram import Bot, Dispatcher, types
from aiogram.enums import ParseMode
from aiogram.filters import CommandStart
from aiogram.types import Message
from aiogram.utils.markdown import hbold
import logging

SYSTEM_PROMPT = "You're name in Lenin"
# SYSTEM_PROMPT = """db.get("<SQL QUERY>", <PARAMS>)
# replace <SQL QUERY> with sql query with question marks where need pass the param, replace <PARAMS> with nedeed params separeted by commas.
# db.get is a js function in my code
# Examples:
# db.get("SELECT reg_date FROM service WHERE service_number = ?", row.service_number)
# db.get("SELECT gudata_created, gudata_modified FROM service WHERE service_number = ?", row.ed_num)
# db.get("SELECT gudata_created, gudata_modified FROM service WHERE service_number = ? AND type = ?", row.service_number, row.type || gu_incidents.date)
# Show only code in one row.
# """
SYSTEM_TOKEN = 1788
USER_TOKEN = 1404
BOT_TOKEN = 9225
LINEBREAK_TOKEN = 13

ROLE_TOKENS = {
    "user": USER_TOKEN,
    "bot": BOT_TOKEN,
    "system": SYSTEM_TOKEN
}

model_path = 'model-q4_K.gguf'
n_ctx = 2000
top_k = 30
top_p = 0.9
temperature = 0.2
repeat_penalty = 1.1

model = Llama(
    model_path=model_path,
    n_ctx=n_ctx,
    n_parts=1,
    n_gpu_layers=35,
    verbose=True,
    main_gpu=0
)


def get_message_tokens(model, role, content):
    message_tokens = model.tokenize(content.encode("utf-8"))
    message_tokens.insert(1, ROLE_TOKENS[role])
    message_tokens.insert(2, LINEBREAK_TOKEN)
    message_tokens.append(model.token_eos())
    return message_tokens


def get_system_tokens(model):
    system_message = {
        "role": "system",
        "content": SYSTEM_PROMPT
    }
    return get_message_tokens(model, **system_message)


# torch.cuda.empty_cache()
#
# torch.cuda.memory_summary(device=0, abbreviated=False)


# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
# model
print(f'videocard: {torch.cuda.is_available()}')


async def chat_with_model(input_text):
    system_tokens = get_system_tokens(model)
    tokens = system_tokens
    model.eval(tokens)

    message_tokens = get_message_tokens(model=model, role="user", content=input_text)
    role_tokens = [model.token_bos(), BOT_TOKEN, LINEBREAK_TOKEN]
    tokens += message_tokens + role_tokens
    generator = model.generate(
        tokens,
        top_k=top_k,
        top_p=top_p,
        temp=temperature,
        repeat_penalty=repeat_penalty
    )
    response = ''
    for token in generator:
        token_str = model.detokenize([token]).decode("utf-8", errors="ignore")
        tokens.append(token)
        if token == model.token_eos():
            break
        response += token_str
    return response


token = '6993126964:AAFs-Q6X_tDMqqRM8sE0CGlRhVhBHNXG7lc'
dp = Dispatcher()


@dp.message(CommandStart())
async def start(message: types.Message):
    await message.answer("Hello! I'm chatbot! Talk with me!")


@dp.message()
async def echo(message: types.Message):
    print(f'Question: {message.text}')
    response = await chat_with_model(message.text)
    print(f'Answer: {response}')
    await message.answer(response)


async def main() -> None:
    # Initialize Bot instance with a default parse mode which will be passed to all API calls
    bot = Bot(token, parse_mode=ParseMode.HTML)
    # And the run events dispatching
    await dp.start_polling(bot, skip_updates=True)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    asyncio.run(main())
