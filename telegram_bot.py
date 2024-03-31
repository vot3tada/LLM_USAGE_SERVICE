import asyncio
from aiogram import Bot, Dispatcher, types
from aiogram.enums import ParseMode
from aiogram.filters import CommandStart, Command
import totalRag as ll
import configparser

config = configparser.ConfigParser()
config.read('./config.ini', encoding='utf-8')

token = config['BOT']['TOKEN']
dp = Dispatcher()


@dp.message(CommandStart())
async def start(message: types.Message):
    await message.answer("Hello! I'm chatbot! Talk with me!")


@dp.message(Command('reset'))
async def reset(message: types.Message):
    llm.reset()
    await message.answer('Bot reset!')


@dp.message()
async def echo(message: types.Message):
    print(f'Question: {message.text}')
    response = llm.send_quest(message.text)
    print(f'Answer: {response}')
    await message.answer(str(response))


async def main() -> None:
    bot = Bot(token, parse_mode=ParseMode.HTML)
    await dp.start_polling(bot, skip_updates=True)


if __name__ == "__main__":
    asyncio.run(main())
