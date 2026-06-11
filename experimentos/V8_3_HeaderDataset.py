import os
import json
import random
import pathlib
from playwright.sync_api import sync_playwright

OUT_DIR = pathlib.Path("dataset_headers")
IMG_DIR = OUT_DIR / "images"
CODE_DIR = OUT_DIR / "code"
META_DIR = OUT_DIR / "metadata"

for d in [IMG_DIR, CODE_DIR, META_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# 3 Famílias de Headers para testar a Separação de Rotas
FAMILIES = {
    "A_Simples": [
        '<header style="{style}"><h1>{logo}</h1></header>'
    ],
    "B_Navegacao": [
        '<header style="{style}"><h1>{logo}</h1><nav><span>{link1}</span> &nbsp; <span>{link2}</span></nav></header>'
    ],
    "C_Completo": [
        '<header style="{style}"><h1>{logo}</h1><nav><span>{link1}</span> &nbsp; <span>{link2}</span></nav><button style="{btn_style}">{btn_text}</button></header>'
    ]
}

LOGOS = ["Logo", "Brand", "MySite", "App", "System"]
LINKS = ["Home", "About", "Services", "Contact", "Dashboard", "Profile"]
BTNS = ["Login", "Sign Up", "Enter", "Get Started"]

def random_color():
    return f"#{random.randint(0, 0xFFFFFF):06x}"

def generate_style(family):
    bg_color = random_color()
    color = "#ffffff" if random.choice([True, False]) else "#000000"
    padding = f"{random.randint(10, 20)}px {random.randint(20, 40)}px"
    
    base = f"width: 100%; box-sizing: border-box; display: flex; justify-content: space-between; align-items: center; padding: {padding}; background-color: {bg_color}; color: {color}; font-family: sans-serif;"
    return base

def generate_btn_style():
    bg = random_color()
    return f"padding: 8px 16px; border-radius: 4px; border: none; background-color: {bg}; color: #fff; cursor: pointer; font-weight: bold;"

def main():
    print("Gerando Dataset V8.3: 1000 Headers (3 Famílias Funcionais)...")
    samples_per_family = 334 # ~1000 total
    idx = 0
    
    with sync_playwright() as p:
        browser = p.chromium.launch()
        # Viewport estreito e largo, padrão de cabeçalho web
        page = browser.new_page(viewport={"width": 1024, "height": 300}) 
        
        for family_name, templates in FAMILIES.items():
            for _ in range(samples_per_family):
                idx += 1
                idx_str = f"{idx:06d}"
                template = random.choice(templates)
                
                logo = random.choice(LOGOS)
                l1, l2 = random.sample(LINKS, 2)
                btn = random.choice(BTNS)
                
                style = generate_style(family_name)
                btn_style = generate_btn_style()
                
                # A string html_body é o nosso "Target" (ignoramos head e body)
                html_body = template.format(style=style, logo=logo, link1=l1, link2=l2, btn_style=btn_style, btn_text=btn)
                
                html_content = f"""
                <!DOCTYPE html>
                <html>
                <head><meta charset="utf-8"></head>
                <body style="margin: 0; padding: 0;">
                    {html_body}
                </body>
                </html>
                """
                
                html_path = CODE_DIR / f"{idx_str}.html"
                with open(html_path, "w", encoding="utf-8") as f:
                    f.write(html_content)
                
                meta = {
                    "id": idx_str,
                    "family": family_name,
                    "html_body": html_body
                }
                with open(META_DIR / f"{idx_str}.json", "w", encoding="utf-8") as f:
                    json.dump(meta, f, indent=2)
                
                page.goto(f"file://{html_path.resolve()}")
                
                # Isolar apenas a tag <header> para não pegar branco sobrando da tela
                element = page.locator("header")
                if element.count() > 0:
                    element.first.screenshot(path=str(IMG_DIR / f"{idx_str}.png"))
                else:
                    page.screenshot(path=str(IMG_DIR / f"{idx_str}.png"))
                    
                if idx % 100 == 0:
                    print(f"[{idx_str}] Gerado -> {family_name}")

        browser.close()
    print(f"\nDataset V8_3 gerado com sucesso! Total: {idx} headers salvos em ./dataset_headers/")

if __name__ == "__main__":
    main()
