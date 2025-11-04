import requests
import os
from datetime import datetime, timedelta

API_KEY = os.getenv("NEWSDATA_API_KEY")
BASE_URL = "https://newsdata.io/api/1/news"

def fetch_all_news(qintitle: str):
    all_news = []
    qintitle_lower = qintitle.lower()
    next_page = None

    today = datetime.utcnow().date()
    yesterday = today - timedelta(days=1)

    while True:
        params = {
            "apikey": API_KEY,
            "qInTitle": qintitle,
            "language": "en",
        }

        if next_page:
            params["page"] = next_page

        response = requests.get(BASE_URL, params=params, timeout=15)
        response.raise_for_status()
        data = response.json()

        results = data.get("results", [])
        if not results:
            break

        for item in results:
            pub_date = item.get("pubDate")
            link = item.get("link")
            title = item.get("title", "")

            if not pub_date or not link or qintitle_lower not in title.lower():
                continue

            date_only = pub_date.split(" ")[0]
            try:
                date_obj = datetime.strptime(date_only, "%Y-%m-%d").date()
            except ValueError:
                continue

            if yesterday <= date_obj <= today:
                all_news.append([date_obj.strftime("%Y-%m-%d"), link])

        next_page = data.get("nextPage")
        if not next_page:
            break

    return all_news


if __name__ == "__main__":
    q = input("Enter company or keyword in title: ").strip()
    print(f"\nFetching latest news for '{q}'...\n")

    try:
        news = fetch_all_news(q)
        if news:
            print(f"Found {len(news)} results:\n")
            for result in news:
                print(result, "\n")
        else:
            print("No results found for yesterday–today.")
    except requests.exceptions.RequestException as e:
        print(f"Network error: {e}")
