import aiohttp
import asyncio
import json
import hashlib
from bs4 import BeautifulSoup
from urllib.parse import urlparse
from tqdm import tqdm
import csv
import os

# =====================================================
# ⚙️ Настройки
# =====================================================
BASE_URL = "https://udmurt.media"
START_URL = BASE_URL + "/ud/themes/"
OUTPUT_FILE = "../data/raw/udmurt_media/udmurt_media.jsonl"

HEADERS = {
    "User-Agent": "Mozilla/5.0"
}

PER_RUBRIC_TARGET = 286
CONCURRENT_REQUESTS = 30
MAX_RETRIES = 3

hash_storage = {}
global_seen_urls = set()  # Глобальный набор ВСЕХ собранных URL

# =====================================================
# 🔧 Utils
# =====================================================
def get_domain(url):
    return urlparse(url).netloc

def fix_url(url):
    url = url.rstrip('/')
    if url.startswith('/'):
        return BASE_URL + url + '/'
    elif url.startswith('http'):
        return url + '/'
    else:
        return BASE_URL + '/' + url + '/'

async def fetch(session, url):
    for _ in range(MAX_RETRIES):
        try:
            async with session.get(url, timeout=10) as resp:
                if resp.status == 200:
                    return await resp.text()
        except:
            await asyncio.sleep(1)
    return None

def extract(html, url):
    soup = BeautifulSoup(html, "html.parser")

    title = soup.find("h1")
    title = title.get_text(strip=True) if title else None

    category = soup.select_one(".badge")
    category = category.get_text(strip=True) if category else None

    paragraphs = []

    main_content = soup.find("main")
    if main_content:
        for p in main_content.find_all("p"):
            txt = p.get_text(strip=True)
            if txt and len(txt) > 20:
                paragraphs.append(txt)

    if not paragraphs:
        for p in soup.select(".page p"):
            txt = p.get_text(strip=True)
            if txt and len(txt) > 20:
                paragraphs.append(txt)

    if not paragraphs and main_content:
        for script in main_content(["script", "style", "svg"]):
            script.decompose()
        text = main_content.get_text(separator="\n", strip=True)
        lines = []
        for line in text.split("\n"):
            line = line.strip()
            if line and len(line) > 20 and not line.startswith("http") and "window." not in line:
                lines.append(line)
        if lines:
            paragraphs = lines

    content = "\n\n".join(paragraphs).strip()

    if not content:
        return None

    hash_hex = hashlib.sha256(content.encode()).hexdigest()

    return {
        "url": url,
        "title": title,
        "content": content,
        "category": category,
        "site": get_domain(url),
        "hash": hash_hex
    }

def write(data):
    with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(data, ensure_ascii=False) + "\n")

# =====================================================
# 📂 Рубрики
# =====================================================
def get_rubrics():
    import requests
    html = requests.get(START_URL, headers=HEADERS).text
    soup = BeautifulSoup(html, "html.parser")

    rubrics = []
    for a in soup.select("a.menu__link"):
        href = a.get("href")
        if href and "/ud/rubrics/" in href:
            full = fix_url(href)
            if full not in rubrics:
                rubrics.append(full)

    return rubrics

# =====================================================
# 📊 Сбор ссылок (с ГЛОБАЛЬНОЙ фильтрацией дублей)
# =====================================================
async def collect_urls_batch(session, rubric, needed):
    """
    Собирает URL, проверяя уникальность ГЛОБАЛЬНО (по всем рубрикам)
    """
    global global_seen_urls
    
    urls = []
    page = 1

    while len(urls) < needed:
        url = f"{rubric}?PAGEN_1={page}" if page > 1 else rubric
        html = await fetch(session, url)

        if not html:
            break

        soup = BeautifulSoup(html, "html.parser")
        blocks = soup.select("div.news")

        if not blocks:
            break

        for b in blocks:
            if len(urls) >= needed:
                break
            link = b.get("data-url")
            if link:
                full = fix_url(link)
                # Проверяем ГЛОБАЛЬНО - не собирали ли этот URL где-то еще
                if full not in global_seen_urls:
                    global_seen_urls.add(full)
                    urls.append(full)

        # Если на странице не нашли новых URL - выходим
        if len(urls) >= needed:
            break

        page += 1
        await asyncio.sleep(0.05)

    return urls

async def collect_all_urls(rubrics, target):
    global global_seen_urls
    global_seen_urls = set()  # Очищаем перед сбором
    
    all_urls = {}

    async with aiohttp.ClientSession(headers=HEADERS) as session:
        tasks = [collect_urls_batch(session, r, target) for r in rubrics]
        results = await asyncio.gather(*tasks)

        for rubric, urls in zip(rubrics, results):
            all_urls[rubric] = urls
            name = rubric.split('/')[-2]
            if len(urls) >= target:
                print(f"  {name:<20} → {len(urls)}/{target}")
            else:
                print(f"  {name:<20} → {len(urls)}/{target}")

    return all_urls

# =====================================================
# 📥 Скачивание
# =====================================================
async def scrape_rubric(session, rubric, urls, target):
    rubric_name = rubric.split('/')[-2]
    unique = 0

    pbar = tqdm(total=target,
                desc=f"{rubric_name:<15}",
                leave=False,
                bar_format='{desc}: {n}/{total} | уник:{postfix}')

    for url in urls:
        if unique >= target:
            break

        html = await fetch(session, url)
        if not html:
            continue

        data = extract(html, url)
        if not data:
            continue

        # Проверка на дубль контента
        if data["hash"] in hash_storage:
            continue

        hash_storage[data["hash"]] = url
        write(data)
        unique += 1
        pbar.update(1)
        pbar.set_postfix({"уник": unique})

    pbar.close()

    return {
        "rubric": rubric_name,
        "target": target,
        "unique": unique
    }

async def scrape_all(rubric_urls, target):
    global hash_storage
    hash_storage = {}

    async with aiohttp.ClientSession(headers=HEADERS) as session:
        tasks = [scrape_rubric(session, r, urls, target) for r, urls in rubric_urls.items()]
        results = await asyncio.gather(*tasks)

    print("\n" + "="*55)
    print("РЕЗУЛЬТАТЫ")
    print("="*55)
    print(f"{'Рубрика':<22} {'Цель':<6} {'Уник.':<7}")
    print("-"*55)

    total_unique = 0
    for s in sorted(results, key=lambda x: x['unique'], reverse=True):
        icon = "OK " if s['unique'] >= s['target'] else "ER "
        print(f"{icon} {s['rubric']:<20} {s['target']:<6} {s['unique']:<7}")
        total_unique += s['unique']

    print("-"*55)
    print(f"{'ВСЕГО':<22} {target * len(results):<6} {total_unique:<7}")
    print("="*55)
    print(f"\nУникальных статей: {total_unique}")
    print(f"Файл: {OUTPUT_FILE}")

# =====================================================
# 📄 Конвертация JSONL → CSV
# =====================================================
def jsonl_to_csv(input_file, output_file):
    if not os.path.exists(input_file):
        print(f"Файл {input_file} не найден!")
        return

    records = []
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    data = json.loads(line)
                    records.append({
                        "title": data.get("title", ""),
                        "content": data.get("content", ""),
                        "category": data.get("category", "")
                    })
                except:
                    continue

    if not records:
        print("Нет данных!")
        return

    with open(output_file, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["title", "content", "category"])
        writer.writeheader()
        writer.writerows(records)

    print(f"CSV: {output_file} ({len(records)} записей)")
    
if __name__ == "__main__":
    print("UDMURT.MEDIA PARSER\n")

    rubrics = get_rubrics()
    print(f"Рубрик: {len(rubrics)}")
    print(f"Цель: {PER_RUBRIC_TARGET} статей с рубрики\n")

    print("Сбор ссылок...")
    rubric_urls = asyncio.run(collect_all_urls(rubrics, PER_RUBRIC_TARGET))

    print("\nСкачивание...\n")
    asyncio.run(scrape_all(rubric_urls, PER_RUBRIC_TARGET))

    print("\nКонвертация в CSV...")
    jsonl_to_csv(OUTPUT_FILE, "../data/raw/udmurt_media/udmurt_media.csv")
    
    print("\nГотово!")