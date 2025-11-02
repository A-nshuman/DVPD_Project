import google.generativeai as genai
import json
import os
import time

api_key = os.environ.get("GEMINI_STOCK_KEY")
if not api_key:
    raise ValueError("GEMINI_STOCK_KEY environment variable not set.")

genai.configure(api_key=api_key)

list_of_urls = [
    "https://www.variety.com/2024/digital/news/netflix-q4-2023-earnings-subscribers-1235882046/",
    "https://www.forbes.com/sites/greatspeculations/2024/05/17/netflix-stock-looks-fully-valued/",
    "https://www.hollywoodreporter.com/business/business-news/netflix-new-movies-adam-sandler-1235910644/"
]

prompt_template = """
Analyze the news article at the following URL: {}

Your task is to rate its potential impact on the Netflix (NFLX) stock price on a scale from -10 to 10.

Use these definitions:
* -10: Very bad news. Will severely harm the company's profitability.
* 0: Irrelevant. Has no impact on company health.
* 10: Very good news. Will bring much better profitability.

Return your answer in a strict JSON format with two keys:
1. "score": (an integer between -10 and 10)
2. "reasoning": (a brief explanation for your score)
"""

model = genai.GenerativeModel("gemini-2.5-pro")

generation_config = {
    "response_mime_type": "application/json",
    "temperature": 0.2,
}

print("--- Starting Netflix News Sentiment Analysis ---")

for url in list_of_urls:
    print(f"\nAnalyzing: {url}")
    final_prompt = prompt_template.format(url)

    try:
        response = model.generate_content(
            final_prompt,
            generation_config=generation_config
        )

        data = json.loads(response.text)
        print(f"  Score: {data.get('score')}")
        print(f"  Reasoning: {data.get('reasoning')}")

    except Exception as e:
        print(f"  Error analyzing URL: {e}")

    time.sleep(1)

print("\n--- Analysis Complete ---")