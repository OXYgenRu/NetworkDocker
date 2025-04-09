import pprint

import requests
from telegram import Update, ReplyKeyboardMarkup, ReplyKeyboardRemove, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ConversationHandler, CallbackContext, \
    CallbackQueryHandler

# Определяем этапы диалога
REQUEST_TYPE, REQUEST_URL, REQUEST_BODY, FILE_FLAG, REQUEST_TRANSMIT, NEXT_STEP = range(6)

# Клавиатура для выбора пола
reply_keyboard = [["GET", "POST", "PUT"]]
reply_keyboard_1 = [["File", "Message"]]
markup = ReplyKeyboardMarkup(reply_keyboard, one_time_keyboard=True)
markup1 = ReplyKeyboardMarkup(reply_keyboard_1, one_time_keyboard=True)


async def start(update: Update, context: CallbackContext) -> int:
    await update.message.reply_text("Select request type", reply_markup=markup)
    return REQUEST_TYPE


async def request_tp(update: Update, context: CallbackContext) -> int:
    context.user_data["request_type"] = update.message.text
    await update.message.reply_text("Type url", reply_markup=ReplyKeyboardRemove())
    return REQUEST_URL


async def request_url(update: Update, context: CallbackContext) -> int:
    context.user_data["url"] = update.message.text
    # await update.message.reply_text(
    #     f"Спасибо! Анкета заполнена.\nПол: {context.user_data['gender']}\nВозраст: {context.user_data['age']}")
    await update.message.reply_text("Type requst body")
    return REQUEST_BODY


async def request_body(update: Update, context: CallbackContext) -> int:
    context.user_data["body"] = update.message.text
    await update.message.reply_text("Type as file?", reply_markup=markup1)
    return FILE_FLAG


async def file_flag(update: Update, context: CallbackContext) -> int:
    context.user_data["file"] = update.message.text
    await update.message.reply_text(
        f"{context.user_data['request_type']} , {context.user_data['url']} , {context.user_data['body']} ,"
        f" {context.user_data['file']}")
    return await request_transmit(update, context)


async def request_transmit(update: Update, context: CallbackContext) -> int:
    keyboard = [
        [InlineKeyboardButton("Transmit", callback_data="transmit")],
        [InlineKeyboardButton("Cancel", callback_data="cancel")]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    print("adsfs")
    await update.message.reply_text("Transmit request?", reply_markup=reply_markup)
    return NEXT_STEP


async def next_step(update: Update, context: CallbackContext) -> int:
    query = update.callback_query
    await query.answer()

    if query.data == "transmit":
        await query.edit_message_text("Try to transmit request")
        return await request_answer(update, context)
    elif query.data == "cancel":
        await query.edit_message_text("Request canceled")
        return ConversationHandler.END


async def request_answer(update: Update, context: CallbackContext) -> int:
    print("go request")
    try:
        if context.user_data["request_type"] == "POST":
            response = requests.post(context.user_data["url"], json=eval(context.user_data["body"]))
        elif context.user_data["request_type"] == "GET":
            response = requests.get(context.user_data["url"], json=eval(context.user_data["body"]))
        else:
            response = requests.put(context.user_data["url"], json=eval(context.user_data["body"]))

        message_text = pprint.pformat(response.json(), width=80, indent=2)

    except Exception as e:
        message_text = f"Error: {e}"
    if len(message_text) > 4096 or context.user_data["file"] == "File":
        if context.user_data["file"] == "File":
            with open("local/response.txt", "w") as file:
                file.write(message_text)
            if update.callback_query:
                await update.callback_query.message.reply_text("Response is too large, here is the file:",
                                                               reply_markup=ReplyKeyboardRemove())
                await update.callback_query.message.reply_document(open("local/response.txt", "rb"))
            else:
                await update.message.reply_text("Response is too large, here is the file:",
                                                reply_markup=ReplyKeyboardRemove())
                await update.message.reply_document(open("local/response.txt", "rb"))
        else:
            part_size = 4096
            for i in range(0, len(message_text), part_size):
                part = message_text[i:i + part_size]
                if update.callback_query:
                    await update.callback_query.message.reply_text(part, reply_markup=ReplyKeyboardRemove())
                else:
                    await update.message.reply_text(part, reply_markup=ReplyKeyboardRemove())
    else:
        if update.callback_query:
            await update.callback_query.message.reply_text(message_text, reply_markup=ReplyKeyboardRemove())
        elif update.message:
            await update.message.reply_text(message_text, reply_markup=ReplyKeyboardRemove())

    return ConversationHandler.END


async def cancel(update: Update, context: CallbackContext) -> int:
    await update.message.reply_text("Request canceled", reply_markup=ReplyKeyboardRemove())
    return ConversationHandler.END


def main():
    app = Application.builder().token().build()

    conv_handler = ConversationHandler(
        entry_points=[CommandHandler("start", start)],
        states={
            REQUEST_TYPE: [MessageHandler(filters.TEXT & ~filters.COMMAND, request_tp)],
            REQUEST_URL: [MessageHandler(filters.TEXT & ~filters.COMMAND, request_url)],
            REQUEST_BODY: [MessageHandler(filters.TEXT & ~filters.COMMAND, request_body)],
            FILE_FLAG: [MessageHandler(filters.TEXT & ~filters.COMMAND, file_flag)],
            REQUEST_TRANSMIT: [CallbackQueryHandler(request_transmit)],
            NEXT_STEP: [CallbackQueryHandler(next_step)],
        },
        fallbacks=[CommandHandler("cancel", cancel)],
    )

    app.add_handler(conv_handler)

    print("Бот запущен...")
    app.run_polling()


if __name__ == "__main__":
    main()
