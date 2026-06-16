import os
import sys
import datetime
import json
from pathlib import Path
import ollama

MODEL_NAME = "gemma4:e4b-it-q4_K_M"

def get_search_keywords(user_query: str) -> list:
    """Шаг 1: Просим Gemma 4 выделить 2-3 главных ключевых слова для grep"""
    prompt = f"""Преврати этот вопрос пользователя в 2-3 отдельных ключевых слова или короткие фразы на русском языке для текстового поиска (grep). 
Выведи ТОЛЬКО JSON массив строк. Никакого лишнего текста или форматирования.
Вопрос: "{user_query}"
Пример вывода: ["ошибка 403", "токен", "авторизация"]"""
    
    try:
        response = ollama.generate(model=MODEL_NAME, prompt=prompt, options={"temperature": 0.1})
        # Очищаем ответ от возможных markdown-тегов ```json
        text = response['response'].strip().replace("```json", "").replace("```", "").strip()
        keywords = json.loads(text)
        return [str(k).lower() for k in keywords if k]
    except Exception:
        # Фолбэк на случай, если модель нарушила формат JSON
        return [w.lower() for w in user_query.split() if len(w) > 3][:3]

def search_files(kb_path: Path, keywords: list) -> str:
    """Шаг 2: Обычный классический поиск по файлам (Простой аналог grep)"""
    found_segments = []
    
    # Рекурсивно обходим .md и .txt файлы
    for file_path in kb_path.rglob('*'):
        if file_path.suffix.lower() in ['.md', '.txt'] and file_path.is_file():
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                
                # Ищем совпадения построчно (как grep -n)
                for idx, line in enumerate(lines):
                    line_lower = line.lower()
                    # Если хотя бы одно ключевое слово совпало
                    if any(kw in line_lower for kw in keywords):
                        # Берем контекст: строку до, текущую и строку после
                        start = max(0, idx - 1)
                        end = min(len(lines), idx + 2)
                        context = "".join(lines[start:end]).strip()
                        
                        found_segments.append(f"Файл: {file_path.name} (Строка {idx+1}):\n{context}\n---")
            except Exception:
                continue # Игнорируем проблемы с кодировкой отдельных файлов

    # Ограничиваем общий объем контекста, чтобы не перегрузить модель
    return "\n\n".join(found_segments[:15])

def save_chat_history(kb_path: Path, user_query: str, ai_response: str):
    """Шаг 4: Сохраняем диалог прямо в базу знаний (Обеспечиваем авто-память)"""
    history_dir = kb_path / ".chat_history"
    history_dir.mkdir(exist_ok=True)
    
    date_str = datetime.datetime.now().strftime("%Y-%m-%d")
    history_file = history_dir / f"chat_{date_str}.md"
    
    timestamp = datetime.datetime.now().strftime("%H:%M:%S")
    with open(history_file, 'a', encoding='utf-8') as f:
        f.write(f"### Диалог [{timestamp}]\n")
        f.write(f"**Пользователь:** {user_query}\n\n")
        f.write(f"**Ассистент:** {ai_response}\n\n")
        f.write("="*40 + "\n\n")

def main():
    if len(sys.argv) < 2:
        print("Использование: python smart_grep.py /путь/к/базе/знаний")
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
                print("Пока!")
                break

            print("🔍 Извлекаю ключевые слова...")
            keywords = get_search_keywords(user_query)
            print(f"📂 Ищу в файлах по маске: {keywords}...")
            
            context = search_files(kb_path, keywords)
            
            if not context:
                print("⚠ В файлах ничего не найдено по ключевым словам. Задаю вопрос напрямую модели...")
                context = "Информация в локальных файлах базы знаний не найдена."

            print("🧠 Формирую ответ на основе найденного...")
            
            system_prompt = f"""Ты полезный ИИ-ассистент. Ответь на вопрос пользователя, опираясь на предоставленный контекст, который был найден в его файлах базы знаний. Если в контексте есть история прошлых диалогов, учитывай её как память.
            
КОНТЕКСТ ИЗ ФАЙЛОВ:
{context}"""

            # Шаг 3: Генерация финального ответа
            response = ollama.generate(
                model=MODEL_NAME,
                prompt=f"Вопрос пользователя: {user_query}",
                system=system_prompt,
                options={"temperature": 0.3}
            )
            
            ai_response = response['response']
            print(f"\n🤖 Ответ:\n{ai_response}\n")
            
            # Сохраняем диалог в ту же папку (в скрытую директорию .chat_history)
            save_chat_history(kb_path, user_query, ai_response)

        except KeyboardInterrupt:
            print("\nСессия прервана.")
            break

if __name__ == "__main__":
    main()
