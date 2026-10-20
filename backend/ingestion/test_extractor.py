
from extractor import extract_page

url = "https://www.fnx.co.il/car-insurance/comprehensive/"

page = extract_page(url)

print("URL:", page["url"])
print("Title:", page["title"])
print("Content length:", len(page["content"]))

print("\nFirst 500 characters:")
print(page["content"][:500])
