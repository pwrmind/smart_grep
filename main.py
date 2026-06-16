import sys
import datetime
import re
from pathlib import Path
import ollama

MODEL_NAME = "gemma4:e4b-it-q4_K_M"

# ---------- Лемматизация (pymorphy2 или аварийный стеммер) ----------
try:
    import pymorphy2
    try:
        import pymorphy2_dicts_ru
        dict_path = Path(pymorphy2_dicts_ru.__path__[0]) / 'data'
        morph = pymorphy2.MorphAnalyzer(path=str(dict_path))
    except ImportError:
        morph = pymorphy2.MorphAnalyzer()
    USE_PYMORPHY = True
except Exception:
    USE_PYMORPHY = False
    def get_word_stem(word: str) -> str:
        word = "".join(c for c in word if c.isalpha()).lower().strip()
        if len(word) <= 3:
            return word
        rv_re = re.compile(r'(ами|ями|ов|ев|ей|ия|ие|ию|я|а|е|и|о|у|ы|ь|ия|ьях|ых|их|ого|ому|ыми|ными|ческий|ческая|ческое|ских|ского|скому|скими|ской)$')
        stem = rv_re.sub('', word)
        if len(stem) > 5:
            stem = re.sub(r'(ани|ени|ств|ни|тель|ова|ива)$', '', stem)
        return stem if len(stem) >= 3 else word

STOP_LEMMAS = {
    'я', 'ты', 'он', 'она', 'оно', 'мы', 'вы', 'они',
    'что', 'как', 'где', 'когда', 'почему', 'зачем',
    'этот', 'тот', 'весь', 'наш', 'ваш', 'свой',
    'мочь', 'рассказать', 'лекция', 'который'
}

def lemmatize_word(word: str) -> str | None:
    word = word.strip().lower()
    if not word or len(word) < 3:
        return None
    if re.search(r'\d', word):
        return None
    if USE_PYMORPHY:
        try:
            normal = morph.parse(word)[0].normal_form
        except Exception:
            return None
    else:
        normal = get_word_stem(word)
        if not normal or len(normal) < 3:
            return None
    if normal in STOP_LEMMAS:
        return None
    return normal

def lemmatize_text(text: str) -> list[str]:
    words = re.findall(r'[а-яёa-z]+', text, re.IGNORECASE)
    lemmas = []
    for w in words:
        l = lemmatize_word(w)
        if l:
            lemmas.append(l)
    return lemmas

# ---------- Ключевые слова через модель ----------
def get_search_keywords(user_query: str) -> list[str]:
    prompt = f"""Проанализируй вопрос пользователя. Выдели из него от 2 до 4 главных одиночных слов для поиска информации (существительные или глаголы в начальной форме).
Игнорируй предлоги и общие слова (что, как, где, расскажи, лекция).
Выведи ТОЛЬКО эти слова через запятую. Больше ничего не пиши.

Вопрос: "{user_query}"
Пример вывода: выгорание, профилактика, психолог"""

    try:
        response = ollama.generate(model=MODEL_NAME, prompt=prompt, options={"temperature": 0.1})
        raw = response['response'].strip()
        for tag in ("</thought>", "</thinking>"):
            if tag in raw:
                raw = raw.split(tag)[-1].strip()
        raw_words = re.split(r'[,\s.\-+]+', raw)
        lemmas = []
        for w in raw_words:
            l = lemmatize_word(w)
            if l:
                lemmas.append(l)
        return lemmas[:4]
    except Exception:
        return lemmatize_text(user_query)[:4]

# ---------- Поиск на лету (без индексов) ----------
def search_files(kb_path: Path, query_lemmas: list[str], top_n=10) -> str:
    if not query_lemmas:
        return ""

    found = []

    for file_path in kb_path.rglob('*'):                # рекурсивный обход всех папок
        if file_path.suffix.lower() not in ['.md', '.txt']:
            continue
        if not file_path.is_file() or ".chat_history" in file_path.parts:
            continue
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
        except Exception:
            continue

        # Заголовок файла (первые 3 строки)
        file_header = "".join(lines[:3]).strip() if len(lines) >= 3 else ""

        for idx, line in enumerate(lines):
            line_lemmas = lemmatize_text(line)
            if not line_lemmas:
                continue

            matched = set()
            for q_lemma in query_lemmas:
                if re.search(r'\b' + re.escape(q_lemma) + r'\b', ' '.join(line_lemmas)):
                    matched.add(q_lemma)
            if not matched:
                continue

            score = len(matched)
            # Расширенное окно: ±10 строк
            start = max(0, idx - 10)
            end = min(len(lines), idx + 11)
            snippet = "".join(lines[start:end]).strip()

            full_context = f"Файл: {file_path.name}\nЗаголовок: {file_header}\n---\n{snippet}"
            found.append((score, full_context))

    found.sort(key=lambda x: x[0], reverse=True)
    top_frags = [f"Фрагмент {i+1}:\n{text}\n---" for i, (_, text) in enumerate(found[:top_n])]
    return "\n\n".join(top_frags)

# ---------- История ----------
def save_chat_history(kb_path: Path, user_query: str, ai_response: str):
    hist_dir = kb_path / ".chat_history"
    hist_dir.mkdir(exist_ok=True)
    date_str = datetime.datetime.now().strftime("%Y-%m-%d")
    hist_file = hist_dir / f"chat_{date_str}.md"
    timestamp = datetime.datetime.now().strftime("%H:%M:%S")
    with open(hist_file, 'a', encoding='utf-8') as f:
        f.write(f"### Диалог [{timestamp}]\n**Пользователь:** {user_query}\n\n**Ассистент:** {ai_response}\n\n" + "="*40 + "\n\n")

# ---------- Главный цикл ----------
def main():
    if len(sys.argv) < 2:
        print("Использование: uv run main.py 'C:\\путь\\к\\базе'")
        sys.exit(1)

    kb_path = Path(sys.argv[1]).resolve()
    if not kb_path.exists():
        print(f"Ошибка: Путь {kb_path} не существует.")
        sys.exit(1)

    print(f"🤖 База знаний: {kb_path}")
    if USE_PYMORPHY:
        print("📚 Лемматизация: pymorphy2")
    else:
        print("📚 Лемматизация: встроенный стеммер")
    print("Задайте вопрос (exit для выхода):\n")

    while True:
        try:
            q = input("❓ Вы: ").strip()
            if not q:
                continue
            if q.lower() in ('exit', 'quit', 'выход'):
                break

            print("🔍 Извлекаю ключевые слова...")
            lemmas = get_search_keywords(q)
            print(f"📂 Ключевые леммы: {lemmas}")

            print("📄 Поиск по файлам...")
            ctx_block = search_files(kb_path, lemmas, top_n=10)    # до 10 фрагментов
            if not ctx_block:
                print("⚠ Ничего не найдено.")
                ctx_block = "Информация в локальных файлах по этим ключевым словам отсутствует."

            print("🧠 Формирую ответ...")
            # Улучшенный промпт для развёрнутого ответа
            prompt = f"""Ты — ИИ-ассистент. У тебя есть фрагменты из личных заметок пользователя.
Проанализируй эти фрагменты и дай **развёрнутый, структурированный ответ** на вопрос.
Опирайся на факты из контекста. Если информации мало, отметь это, но всё равно попробуй дать максимально осмысленный ответ на основе того, что есть.
Не пересказывай контекст дословно, а объясни суть.

КОНТЕКСТ:
{ctx_block}

ВОПРОС:
{q}

Твой развёрнутый ответ:"""
            resp = ollama.generate(model=MODEL_NAME, prompt=prompt, options={"temperature": 0.2})
            answer = resp['response']
            print(f"\n🤖 Ответ:\n{answer}\n")
            save_chat_history(kb_path, q, answer)

        except KeyboardInterrupt:
            print("\nДо свидания!")
            break
        except Exception as e:
            print(f"Ошибка: {e}")

if __name__ == "__main__":
    main()