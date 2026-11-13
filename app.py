import os
import requests
from flask import Flask, request, jsonify

app = Flask(__name__)

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4.1-mini")
OPENAI_URL = "https://api.openai.com/v1/responses"


def alice_reply(text, session_state=None):
    body = {
        "version": "1.0",
        "response": {"text": str(text)[:1024], "end_session": False}
    }
    if session_state is not None:
        body["session_state"] = session_state
    return jsonify(body), 200


def extract_output_text(payload):
    direct = payload.get("output_text")
    if isinstance(direct, str) and direct.strip():
        return direct.strip()
    parts = []
    for item in payload.get("output", []):
        if item.get("type") == "message" and item.get("role") == "assistant":
            for content in item.get("content", []):
                if content.get("type") == "output_text" and content.get("text"):
                    parts.append(content["text"])
    return "\n".join(parts).strip()


@app.get("/")
def health():
    return "Alice ChatGPT webhook is running.", 200


@app.post("/alice")
def alice_webhook():
    if not OPENAI_API_KEY:
        return alice_reply("Сервер не настроен: не задан API-ключ OpenAI.")

    data = request.get_json(silent=True) or {}
    req = data.get("request") or {}
    command = (req.get("command") or "").strip()

    if req.get("type") != "SimpleUtterance" or not command:
        return alice_reply("Привет! Я ChatGPT. Задай мне вопрос.")

    state = (data.get("state") or {}).get("session") or {}
    previous_id = state.get("previous_response_id")

    payload = {
        "model": OPENAI_MODEL,
        "instructions": (
            "Ты голосовой помощник ChatGPT внутри навыка Яндекс Алисы. "
            "Отвечай по-русски, дружелюбно, точно и кратко. "
            "Не используй таблицы и Markdown: ответ будет озвучен."
        ),
        "input": command,
        "max_output_tokens": 100
    }
    if previous_id:
        payload["previous_response_id"] = previous_id

    try:
        response = requests.post(
            OPENAI_URL,
            headers={
                "Authorization": f"Bearer {OPENAI_API_KEY}",
                "Content-Type": "application/json"
            },
            json=payload,
            timeout=(0.8, 3.7)
        )
        response.raise_for_status()
        result = response.json()
        answer = extract_output_text(result) or "Не удалось получить текстовый ответ."
        new_id = result.get("id")
        new_state = {"previous_response_id": new_id} if new_id else None
        return alice_reply(answer, new_state)

    except requests.Timeout:
        app.logger.warning("OpenAI API request timed out")
        return alice_reply("Ответ готовится слишком долго. Попробуй задать вопрос ещё раз.")
    except requests.RequestException as exc:
        app.logger.exception("OpenAI API request failed: %s", exc)
        return alice_reply("Не удалось связаться с нейросетью. Проверь настройки сервера.")
    except (ValueError, TypeError):
        app.logger.exception("Invalid OpenAI API response")
        return alice_reply("Получен некорректный ответ от нейросети.")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "10000")))
