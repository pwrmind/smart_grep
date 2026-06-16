import os
import sys
import datetime
import re
from pathlib import Path
import ollama

MODEL_NAME = "gemma4:e4b-it-q4_K_M"

def get_word_stem(word: str) -> str:
    """Очищает слово и оставляет его основу (удаляет окончания/суффиксы)"""
    word = "".join(c for c in word if c.isalpha()).lower().strip()
    if len(word) <= 3:
        return word
    
    # Регулярное выражение для типичных русских окончаний
    rv_re = re.compile(r'(ами|ями|ов|ев|ей|ия|ие|ию|я|а|е|и|о|у|ы|ь|ия|ьях|ых|их|ого|ому|ыми|ными|ческий|ческая|ческое|ских|ского|скому|скими|ской)$')
    stem = rv_re.sub('', word)
    
    if len(stem) > 5:
        stem = re.sub(r'(ани|ени|ств|ни|тель|ова|ива)$', '', stem)
        
    return stem if len(stem) >= 3 else word

def get_search_keywords(user_query: str) -> list:
    """Просим Gemma выделить важные слова. Никакого JSON, только текст через запятую."""
    prompt = f"""Проанализируй вопрос пользователя. Выдели из него от 2 до 4 главных одиночных слов для поиска информации (существительные или глаголы в начальной форме).
Игнорируй предлоги и общие слова (что, как, где, расскажи, лекция).
Выведи ТОЛЬКО эти слова через запятую. Больше ничего не пиши.

Вопрос: "{user_query}"
Пример вывода: выгорание, профилактика, психолог"""
    
    try:
        response = ollama.generate(model=MODEL_NAME, prompt=prompt, options={"temperature": 0.1})
        raw_text = response['response'].strip()
        
        # Защита от включенного режима рассуждения (Thinking/Thought) у Gemma
        if "</thought>" in raw_text:
            raw_text = raw_text.split("</thought>")[-1].strip()
        elif "</thinking>" in raw_text:
            raw_text = raw_text.split("</thinking>")[-1].strip()

        # Разбиваем строку по запятым, пробелам или дефисам
        words = re.split(r'[,\s\n\.\-\+]+', raw_text)
        
        stems = set()
        for w in words:
            stem = get_word_stem(w)
            if len(stem) >= 3:
                stems.add(stem)
                
        return list(stems)
    except Exception as e:
        # Если Ollama упала, делаем тупой, но надежный разбор самой строки пользователя
        return list({get_word_stem(w) for w in user_query.split() if len(w) > 3})

def search_files(kb_path: Path, stems: list) -> str:
    """Ищет пересечения основ слов (Умный grep)"""
    if not stems:
        return ""
        
    found_segments = []
    
    for file_path in kb_path.rglob('*'):
        if file_path.suffix.lower() in ['.md', '.txt'] and file_path.is_file():
            # Пропускаем служебные файлы нашей истории
            if ".chat_history" in file_path.parts:
                continue
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                
                for idx, line in enumerate(lines):
                    line_lower = line.lower()
                    
                    # Считаем сколько основ слов встретилось в строке
                    matches_count = sum(1 for stem in stems if stem in line_lower)
                    
                    # Если ищем 1-2 слова — достаточно 1 совпадения. 
                    # Если ищем 3+ слов — нужно минимум 2 совпадения в одной строке, чтобы не спамить.
                    required = 1 if len(stems) <= 2 else 2
                    
                    if matches_count >= required:
                        start = max(0, idx - 2)
                        end = min(len(lines), idx + 3)
                        context = "".join(lines[start:end]).strip()
                        
                        # Сохраняем кортеж: (кол-во совпадений, текст фрагмента)
                        found_segments.append((matches_count, f"Файл: {file_path.name} (Строка {idx+1}):\n{context}\n---"))
            except Exception:
                continue

    # Сортируем по количеству совпадений (релевантности) от большего к меньшему
    found_segments.sort(key=lambda x: x[0], reverse=True)
    
    # Возвращаем только текст топ-15 результатов
    return "\n\n".join([item[1] for item in found_segments[:15]])

def save_chat_history(kb_path: Path, user_query: str, ai_response: str):
    """Сохраняем логи диалога"""
    history_dir = kb_path / ".chat_history"
    history_dir.mkdir(exist_ok=True)
    date_str = datetime.datetime.now().strftime("%Y-%m-%d")
    history_file = history_dir / f"chat_{date_str}.md"
    
    timestamp = datetime.datetime.now().strftime("%H:%M:%S")
    with open(history_file, 'a', encoding='utf-8') as f:
        f.write(f"### Диалог [{timestamp}]\n**Пользователь:** {user_query}\n\n**Ассистент:** {ai_response}\n\n" + "="*40 + "\n\n")

def main():
    if len(sys.argv) < 2:
        print("Использование: uv run main.py 'C:\\путь\\к\\базе'")
        sys.exit(1)
        
    kb_path = Path(sys.argv[1]).resolve()
    if not kb_path.exists():
        print(f"Ошибка: Путь {kb_path} не существует.")
        sys.exit(1)

    print(f"🤖 База знаний подключена: {kb_path}")
    print("Задайте ваш вопрос (для выхода введите 'exit'):\n")

    while True:
        try:
            user_query = input("❓ Вы: ").strip()
            if not user_query:
                continue
            if user_query.lower() in ['exit', 'quit', 'выход']:
                break

            print("🔍 Извлекаю корни ключевых слов...")
            stems = get_search_keywords(user_query)
            print(f"📂 Ищу в файлах по основам: {stems}...")
            
            context = search_files(kb_path, stems)
            
            if not context:
                print("⚠ В локальных файлах ничего не найдено.")
                context = "Информация в локальных файлах базы знаний по этим ключевым словам отсутствует."

            print("🧠 Формирую ответ...")
            
            system_prompt = f"Ты полезный ИИ-ассистент. Ответь на вопрос пользователя, опираясь исключительно на предоставленный контекст из его личных файлов. Если информации нет, так и скажи.\n\nКОНТЕКСТ ИЗ ФАЙЛОВ:\n{context}"

            response = ollama.generate(
                model=MODEL_NAME,
                prompt=f"Вопрос пользователя: {user_query}",
                system=system_prompt,
                options={"temperature": 0.2}
            )
            
            ai_response = response['response']
            print(f"\n🤖 Ответ:\n{ai_response}\n")
            save_chat_history(kb_path, user_query, ai_response)

        except KeyboardInterrupt:
            print("\nСессия прервана.")
            break
        except Exception as e:
            print(f"Произошла ошибка в цикле: {e}")

if __name__ == "__main__":
    main()
