from __future__ import annotations

import itertools
import json
import logging
import os
import base64
import asyncio  # ✅ THÊM DÒNG NÀY
from goose3 import Goose

import telegram
from telegram import Message, MessageEntity, Update, ChatMember, constants
from telegram.ext import CallbackContext, ContextTypes

from usage_tracker import UsageTracker
from openai import AsyncOpenAI
from playwright.async_api import async_playwright  # ✅ Thêm cho Playwright support
from bs4 import BeautifulSoup

def get_openai_client():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("❌ Thiếu biến môi trường OPENAI_API_KEY.")
    return AsyncOpenAI(api_key=api_key)

def extract_text_from_url(url: str) -> str:
    try:
        with Goose() as g:
            article = g.extract(url=url)
            return article.cleaned_text
    except Exception as e:
        return f"❌ Lỗi khi trích xuất nội dung: {e}"

# ✅ Hàm fallback dùng Playwright nếu Goose fail
async def fetch_page_with_playwright(url: str) -> str:
    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True, args=[
                "--no-sandbox",
                "--disable-setuid-sandbox",
                "--disable-dev-shm-usage",
                "--disable-gpu",
                "--no-zygote",
                "--disable-software-rasterizer"
            ])
            context = await browser.new_context(
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118.0.5993.90 Safari/537.36"
            )
            page = await context.new_page()
            await page.goto(url, timeout=20000)
            await page.wait_for_timeout(3000)
            html = await page.content()
            await browser.close()
            return html
    except Exception as e:
        return f"❌ Lỗi Playwright: {e}"

def extract_text_from_html(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")

    # Loại bỏ script, style
    for tag in soup(["script", "style"]):
        tag.decompose()

    return soup.get_text(separator="\n")  # Trả về văn bản sạch, có xuống dòng

async def summarize_url(url: str, update: Update = None, context: CallbackContext = None) -> str:
    """
    Tóm tắt nội dung của một URL nếu bot được mention hoặc được reply trong nhóm.
    """
    if update and context:
        if update.effective_chat.type in [constants.ChatType.GROUP, constants.ChatType.SUPERGROUP]:
            bot_username = context.bot.username.lower()
            message_text_lower = (update.message.text or "").lower()
            is_mentioned = f"@{bot_username}" in message_text_lower
            is_reply_to_bot = (
                update.message.reply_to_message
                and update.message.reply_to_message.from_user.id == context.bot.id
            )
            if not is_mentioned and not is_reply_to_bot:
                return None

    content = extract_text_from_url(url)
    if not content or len(content.strip()) < 100:
        content = await fetch_page_with_playwright(url)  # fallback nếu Goose fail
        if not content or len(content.strip()) < 100:
            return "E chưa tóm tắt được nội dung. Cho e xin link rõ ràng hơn ạ."
    
        # Giới hạn nội dung ~6000 ký tự để tránh vượt context limit
        trimmed_content = content[:6000]
    
        prompt = (
            "Em là 1 biên tập viên tin tức ngọt ngào. "
            "Nhiệm vụ là **tóm tắt nội dung sau như đang kể lại cho bạn thân nghe**. "
            "Hãy trình bày theo đúng format dưới đây:\n\n"
            "• Mỗi ý bắt đầu bằng: ❖ \n"
            "• Kết thúc mỗi ý bằng: <br/>\n"
            "• Cuối cùng chốt bằng 1 câu thân mật như: Em gửi nha sếp 😌 hoặc Vậy nha anh yêu 🫶\n\n"
            "Ví dụ:\n"
            "❖ Dự án này thuộc tuyến cao tốc Bắc Nam.<br/>\n"
            "❖ Chiều dài 30 km, đi qua Hà Tĩnh.<br/>\n"
            "❖ Tổng đầu tư 7.664 tỷ đồng.<br/>\n"
            "❖ Đang triển khai thi công, dự kiến hoàn thành cuối năm 2025.<br/>\n"
            "Vậy nha anh yêu 🫶\n\n"
            f"Dưới đây là nội dung cần tóm tắt:\n\n"

        f"{trimmed_content}"

            )
    try:
        client = get_openai_client()
        response = await client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.8,
            max_tokens=500,
        )
        message = response.choices[0].message
        if message and message.content:
            return message.content.strip()
        return "❌ Không lấy được nội dung từ OpenAI."
    except Exception as e:
        return f"❌ Lỗi khi gọi OpenAI: {e}"


# (The rest of the file remains unchanged.)



# (The rest of the file remains unchanged.)


# (The rest of the file remains unchanged.)


# (The rest of the file remains un


# (The rest of the file remains unchanged.)from __future__ import annotations

import itertools
import json
import logging
import os
import base64
from goose3 import Goose

import telegram
from telegram import Message, MessageEntity, Update, ChatMember, constants
from telegram.ext import CallbackContext, ContextTypes

from usage_tracker import UsageTracker
from openai import AsyncOpenAI


def get_openai_client():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("❌ Thiếu biến môi trường OPENAI_API_KEY.")
    return AsyncOpenAI(api_key=api_key)


def extract_text_from_url(url: str) -> str:
    try:
        with Goose() as g:
            article = g.extract(url=url)
            return article.cleaned_text
    except Exception as e:
        return f"❌ Lỗi khi trích xuất nội dung: {e}"


async def summarize_url(url: str, update: Update = None, context: CallbackContext = None) -> str:
    """
    Tóm tắt nội dung của một URL nếu bot được mention hoặc được reply trong nhóm.
    """
    if update and context:
        if update.effective_chat.type in [constants.ChatType.GROUP, constants.ChatType.SUPERGROUP]:
            bot_username = context.bot.username.lower()
            message_text_lower = (update.message.text or "").lower()
            is_mentioned = f"@{bot_username}" in message_text_lower
            is_reply_to_bot = (
                update.message.reply_to_message
                and update.message.reply_to_message.from_user.id == context.bot.id
            )
            if not is_mentioned and not is_reply_to_bot:
                return None

    content = extract_text_from_url(url)
    if not content or len(content.strip()) < 100:
        html = await fetch_page_with_playwright(url)
        if not html or len(html.strip()) < 100:
            return "❌ Không lấy được nội dung từ trang. Gửi link rõ hơn giúp em!"

        # 🧼 Parse HTML thành text sạch
        content = extract_text_from_html(html)

    # 🧹 Loại bỏ dòng đầu nếu là "Đây là mã HTML..."
    lines = content.strip().split("\n")
    if lines and lines[0].lower().startswith("đây là mã html"):
        content = "\n".join(lines[1:])

        # Giới hạn nội dung ~6000 ký tự để tránh vượt context limit
    trimmed_content = content[:6000]
    
    prompt = (
        "Tóm tắt nội dung sau bằng tiếng Việt. Trình bày ngắn gọn, mỗi ý trên một dòng rõ ràng."
        " Tránh viết đoạn văn dài.\n\n"
        f"{trimmed_content}"
    )

    try:
        client = get_openai_client()
        response = await client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
            max_tokens=500,
        )
        message = response.choices[0].message
        if message and message.content:
            return message.content.strip()
        return "❌ Không lấy được nội dung từ OpenAI."
    except Exception as e:
        return f"❌ Lỗi khi gọi OpenAI: {e}"


# (The rest of the file remains unchanged.)


def message_text(message: Message) -> str:
    """
    Returns the text of a message, excluding any bot commands.
    """
    message_txt = message.text
    if message_txt is None:
        return ''

    for _, text in sorted(message.parse_entities([MessageEntity.BOT_COMMAND]).items(),
                          key=(lambda item: item[0].offset)):
        message_txt = message_txt.replace(text, '').strip()

    return message_txt if len(message_txt) > 0 else ''


async def is_user_in_group(update: Update, context: CallbackContext, user_id: int) -> bool:
    """
    Checks if user_id is a member of the group
    """
    try:
        chat_member = await context.bot.get_chat_member(update.message.chat_id, user_id)
        return chat_member.status in [ChatMember.OWNER, ChatMember.ADMINISTRATOR, ChatMember.MEMBER]
    except telegram.error.BadRequest as e:
        if str(e) == "User not found":
            return False
        else:
            raise e
    except Exception as e:
        raise e


def get_thread_id(update: Update) -> int | None:
    """
    Gets the message thread id for the update, if any
    """
    if update.effective_message and update.effective_message.is_topic_message:
        return update.effective_message.message_thread_id
    return None


def get_stream_cutoff_values(update: Update, content: str) -> int:
    """
    Gets the stream cutoff values for the message length
    """
    if is_group_chat(update):
        # group chats have stricter flood limits
        return 180 if len(content) > 1000 else 120 if len(content) > 200 \
            else 90 if len(content) > 50 else 50
    return 90 if len(content) > 1000 else 45 if len(content) > 200 \
        else 25 if len(content) > 50 else 15


def is_group_chat(update: Update) -> bool:
    """
    Checks if the message was sent from a group chat
    """
    if not update.effective_chat:
        return False
    return update.effective_chat.type in [
        constants.ChatType.GROUP,
        constants.ChatType.SUPERGROUP
    ]


def split_into_chunks(text: str, chunk_size: int = 4096) -> list[str]:
    """
    Splits a string into chunks of a given size.
    """
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]


async def wrap_with_indicator(update: Update, context: CallbackContext, coroutine,
                              chat_action: constants.ChatAction = "", is_inline=False):
    """
    Wraps a coroutine while repeatedly sending a chat action to the user.
    """
    task = context.application.create_task(coroutine(), update=update)
    while not task.done():
        if not is_inline:
            context.application.create_task(
                update.effective_chat.send_action(chat_action, message_thread_id=get_thread_id(update))
            )
        try:
            await asyncio.wait_for(asyncio.shield(task), 4.5)
        except asyncio.TimeoutError:
            pass


async def edit_message_with_retry(context: ContextTypes.DEFAULT_TYPE, chat_id: int | None,
                                  message_id: str, text: str, markdown: bool = True, is_inline: bool = False):
    """
    Edit a message with retry logic in case of failure (e.g. broken markdown)
    :param context: The context to use
    :param chat_id: The chat id to edit the message in
    :param message_id: The message id to edit
    :param text: The text to edit the message with
    :param markdown: Whether to use markdown parse mode
    :param is_inline: Whether the message to edit is an inline message
    :return: None
    """
    try:
        await context.bot.edit_message_text(
            chat_id=chat_id,
            message_id=int(message_id) if not is_inline else None,
            inline_message_id=message_id if is_inline else None,
            text=text,
            parse_mode=constants.ParseMode.MARKDOWN if markdown else None,
        )
    except telegram.error.BadRequest as e:
        if str(e).startswith("Message is not modified"):
            return
        try:
            await context.bot.edit_message_text(
                chat_id=chat_id,
                message_id=int(message_id) if not is_inline else None,
                inline_message_id=message_id if is_inline else None,
                text=text,
            )
        except Exception as e:
            logging.warning(f'Failed to edit message: {str(e)}')
            raise e

    except Exception as e:
        logging.warning(str(e))
        raise e


async def error_handler(_: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Handles errors in the telegram-python-bot library.
    """
    logging.error(f'Exception while handling an update: {context.error}')


async def is_allowed(config, update: Update, context: CallbackContext, is_inline=False) -> bool:
    """
    Checks if the user is allowed to use the bot.
    """
    if config['allowed_user_ids'] == '*':
        return True

    user_id = update.inline_query.from_user.id if is_inline else update.message.from_user.id
    if is_admin(config, user_id):
        return True
    name = update.inline_query.from_user.name if is_inline else update.message.from_user.name
    allowed_user_ids = config['allowed_user_ids'].split(',')
    # Check if user is allowed
    if str(user_id) in allowed_user_ids:
        return True
    # Check if it's a group a chat with at least one authorized member
    if not is_inline and is_group_chat(update):
        admin_user_ids = config['admin_user_ids'].split(',')
        for user in itertools.chain(allowed_user_ids, admin_user_ids):
            if not user.strip():
                continue
            if await is_user_in_group(update, context, user):
                logging.info(f'{user} is a member. Allowing group chat message...')
                return True
        logging.info(f'Group chat messages from user {name} '
                     f'(id: {user_id}) are not allowed')
    return False


def is_admin(config, user_id: int, log_no_admin=False) -> bool:
    """
    Checks if the user is the admin of the bot.
    The first user in the user list is the admin.
    """
    if config['admin_user_ids'] == '-':
        if log_no_admin:
            logging.info('No admin user defined.')
        return False

    admin_user_ids = config['admin_user_ids'].split(',')

    # Check if user is in the admin user list
    if str(user_id) in admin_user_ids:
        return True

    return False


def get_user_budget(config, user_id) -> float | None:
    """
    Get the user's budget based on their user ID and the bot configuration.
    :param config: The bot configuration object
    :param user_id: User id
    :return: The user's budget as a float, or None if the user is not found in the allowed user list
    """

    # no budget restrictions for admins and '*'-budget lists
    if is_admin(config, user_id) or config['user_budgets'] == '*':
        return float('inf')

    user_budgets = config['user_budgets'].split(',')
    if config['allowed_user_ids'] == '*':
        # same budget for all users, use value in first position of budget list
        if len(user_budgets) > 1:
            logging.warning('multiple values for budgets set with unrestricted user list '
                            'only the first value is used as budget for everyone.')
        return float(user_budgets[0])

    allowed_user_ids = config['allowed_user_ids'].split(',')
    if str(user_id) in allowed_user_ids:
        user_index = allowed_user_ids.index(str(user_id))
        if len(user_budgets) <= user_index:
            logging.warning(f'No budget set for user id: {user_id}. Budget list shorter than user list.')
            return 0.0
        return float(user_budgets[user_index])
    return None


def get_remaining_budget(config, usage, update: Update, is_inline=False) -> float:
    """
    Calculate the remaining budget for a user based on their current usage.
    :param config: The bot configuration object
    :param usage: The usage tracker object
    :param update: Telegram update object
    :param is_inline: Boolean flag for inline queries
    :return: The remaining budget for the user as a float
    """
    # Mapping of budget period to cost period
    budget_cost_map = {
        "monthly": "cost_month",
        "daily": "cost_today",
        "all-time": "cost_all_time"
    }

    user_id = update.inline_query.from_user.id if is_inline else update.message.from_user.id
    name = update.inline_query.from_user.name if is_inline else update.message.from_user.name
    if user_id not in usage:
        usage[user_id] = UsageTracker(user_id, name)

    # Get budget for users
    user_budget = get_user_budget(config, user_id)
    budget_period = config['budget_period']
    if user_budget is not None:
        cost = usage[user_id].get_current_cost()[budget_cost_map[budget_period]]
        return user_budget - cost

    # Get budget for guests
    if 'guests' not in usage:
        usage['guests'] = UsageTracker('guests', 'all guest users in group chats')
    cost = usage['guests'].get_current_cost()[budget_cost_map[budget_period]]
    return config['guest_budget'] - cost


def is_within_budget(config, usage, update: Update, is_inline=False) -> bool:
    """
    Checks if the user reached their usage limit.
    Initializes UsageTracker for user and guest when needed.
    :param config: The bot configuration object
    :param usage: The usage tracker object
    :param update: Telegram update object
    :param is_inline: Boolean flag for inline queries
    :return: Boolean indicating if the user has a positive budget
    """
    user_id = update.inline_query.from_user.id if is_inline else update.message.from_user.id
    name = update.inline_query.from_user.name if is_inline else update.message.from_user.name
    if user_id not in usage:
        usage[user_id] = UsageTracker(user_id, name)
    remaining_budget = get_remaining_budget(config, usage, update, is_inline=is_inline)
    return remaining_budget > 0


def add_chat_request_to_usage_tracker(usage, config, user_id, used_tokens):
    """
    Add chat request to usage tracker
    :param usage: The usage tracker object
    :param config: The bot configuration object
    :param user_id: The user id
    :param used_tokens: The number of tokens used
    """
    try:
        if int(used_tokens) == 0:
            logging.warning('No tokens used. Not adding chat request to usage tracker.')
            return
        # add chat request to users usage tracker
        usage[user_id].add_chat_tokens(used_tokens, config['token_price'])
        # add guest chat request to guest usage tracker
        allowed_user_ids = config['allowed_user_ids'].split(',')
        if str(user_id) not in allowed_user_ids and 'guests' in usage:
            usage["guests"].add_chat_tokens(used_tokens, config['token_price'])
    except Exception as e:
        logging.warning(f'Failed to add tokens to usage_logs: {str(e)}')
        pass


def get_reply_to_message_id(config, update: Update):
    """
    Returns the message id of the message to reply to
    :param config: Bot configuration object
    :param update: Telegram update object
    :return: Message id of the message to reply to, or None if quoting is disabled
    """
    if config['enable_quoting'] or is_group_chat(update):
        return update.message.message_id
    return None


def is_direct_result(response: any) -> bool:
    """
    Checks if the dict contains a direct result that can be sent directly to the user
    :param response: The response value
    :return: Boolean indicating if the result is a direct result
    """
    if type(response) is not dict:
        try:
            json_response = json.loads(response)
            return json_response.get('direct_result', False)
        except:
            return False
    else:
        return response.get('direct_result', False)


async def handle_direct_result(config, update: Update, response: any):
    """
    Handles a direct result from a plugin
    """
    if type(response) is not dict:
        response = json.loads(response)

    result = response['direct_result']
    kind = result['kind']
    format = result['format']
    value = result['value']

    common_args = {
        'message_thread_id': get_thread_id(update),
        'reply_to_message_id': get_reply_to_message_id(config, update),
    }

    if kind == 'photo':
        if format == 'url':
            await update.effective_message.reply_photo(**common_args, photo=value)
        elif format == 'path':
            await update.effective_message.reply_photo(**common_args, photo=open(value, 'rb'))
    elif kind == 'gif' or kind == 'file':
        if format == 'url':
            await update.effective_message.reply_document(**common_args, document=value)
        if format == 'path':
            await update.effective_message.reply_document(**common_args, document=open(value, 'rb'))
    elif kind == 'dice':
        await update.effective_message.reply_dice(**common_args, emoji=value)

    if format == 'path':
        cleanup_intermediate_files(response)


def cleanup_intermediate_files(response: any):
    """
    Deletes intermediate files created by plugins
    """
    if type(response) is not dict:
        response = json.loads(response)

    result = response['direct_result']
    format = result['format']
    value = result['value']

    if format == 'path':
        if os.path.exists(value):
            os.remove(value)


# Function to encode the image
def encode_image(fileobj):
    image = base64.b64encode(fileobj.getvalue()).decode('utf-8')
    return f'data:image/jpeg;base64,{image}'


def decode_image(imgbase64):
    image = imgbase64[len('data:image/jpeg;base64,'):]
    return base64.b64decode(image)
