from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.firefox.options import Options
import time
import csv

options = Options()
options.headless = False
driver = webdriver.Firefox(options=options)

# ihtiyaca göre düzenledim buraları değiştirerek bazı yerleri birçok kere veri çektim
driver.get("https://www.trthaber.com/haber/dunya/4000.sayfa.html")
time.sleep(3)

def accept_cookies():
    try:
        buttons = driver.find_elements(By.TAG_NAME, "button")
        for btn in buttons:
            if "kabul" in btn.text.lower() or "onayla" in btn.text.lower() or "tamam" in btn.text.lower():
                btn.click()
                print("✅ Çerez kabul edildi.")
                break
    except:
        pass

accept_cookies()


for i in range(300):
    driver.find_element(By.TAG_NAME, "body").send_keys(Keys.END)
    time.sleep(0.5)

titles = []
elements = driver.find_elements(By.CLASS_NAME, "site-url")

for idx, el in enumerate(elements):
    try:
       
        try:
            summary_el = el.find_element(By.CLASS_NAME, "hbbiText")
            summary_text = summary_el.text.strip()
        except:
            summary_text = ""

    
        full_text = el.text.strip()
        clean_text = full_text.replace(summary_text, "").strip()

        if len(clean_text) > 15:
            titles.append({
                "category": "dunya",
                "title": clean_text
            })
            print(f"[{idx+1}] ✅ {clean_text}")
    except Exception as e:
        print(f"❌ Hata: {e}")


with open("dunya.csv", "w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["category", "title"])
    writer.writeheader()
    writer.writerows(titles)

print(f"✅ {len(titles)} başlık başarıyla çekildi.")
driver.quit()

