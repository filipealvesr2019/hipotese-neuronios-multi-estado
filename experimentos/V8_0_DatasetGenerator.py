import os
import json
import random
import pathlib
from playwright.sync_api import sync_playwright

OUT_DIR = pathlib.Path("dataset")
IMG_DIR = OUT_DIR / "images"
CODE_DIR = OUT_DIR / "code"
META_DIR = OUT_DIR / "metadata"

for d in [IMG_DIR, CODE_DIR, META_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Templates Base
COMPONENTS = {
    "button": [
        '<button style="{style}">{text}</button>'
    ],
    "input": [
        '<input type="text" placeholder="{text}" style="{style}" />'
    ],
    "card": [
        '<div style="{style}"><h3>{text}</h3><p>Descrição do card genérico gerado dinamicamente para teste de UI.</p></div>'
    ],
    "navbar": [
        '<nav style="{style}"><span><b>Logo</b></span> <span>{text}</span></nav>'
    ]
}

TEXTS = {
    "button": ["Submit", "Click Me", "Add to Cart", "Login", "Register", "Save Changes"],
    "input": ["Enter your name...", "Search...", "Email address...", "Password...", "Type here..."],
    "card": ["Product Name", "User Profile", "Analytics Summary", "Settings", "Dashboard Info"],
    "navbar": ["Home | About", "Dashboard | Logout", "Menu | Contact", "Products | Services"]
}

def random_color():
    return f"#{random.randint(0, 0xFFFFFF):06x}"

def generate_style(comp_type):
    padding = f"{random.randint(8, 24)}px {random.randint(16, 32)}px"
    border_radius = f"{random.randint(0, 24)}px"
    bg_color = random_color()
    color = "#ffffff" if random.choice([True, False]) else "#000000"
    font_size = f"{random.randint(12, 24)}px"
    
    base = f"padding: {padding}; border-radius: {border_radius}; background-color: {bg_color}; color: {color}; font-size: {font_size}; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; border: 1px solid #ccc; box-sizing: border-box;"
    
    if comp_type == "card":
        base += " width: 300px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); text-align: left;"
    elif comp_type == "navbar":
        base += " width: 100%; display: flex; justify-content: space-between; align-items: center;"
        
    return base

def main():
    print("Gerando Dataset Sintético de UI (V8_0)...")
    
    samples_per_component = 25 # Começando pequeno: 25 variações para 4 componentes = 100 samples
    
    idx = 0
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 800, "height": 600})
        
        for comp_type, templates in COMPONENTS.items():
            for _ in range(samples_per_component):
                idx += 1
                idx_str = f"{idx:06d}"
                
                template = random.choice(templates)
                text = random.choice(TEXTS[comp_type])
                style = generate_style(comp_type)
                
                # O body usa flex para centralizar, facilitando a captura do elemento isolado
                html_content = f"""
                <!DOCTYPE html>
                <html>
                <head><meta charset="utf-8"></head>
                <body style="display: flex; justify-content: center; align-items: center; height: 100vh; margin: 0; background-color: #f8f9fa;">
                    {template.format(style=style, text=text)}
                </body>
                </html>
                """
                
                # Salvar HTML
                html_path = CODE_DIR / f"{idx_str}.html"
                with open(html_path, "w", encoding="utf-8") as f:
                    f.write(html_content)
                
                # Salvar Metadata
                meta = {
                    "id": idx_str,
                    "component_type": comp_type,
                    "text": text,
                    "style": style
                }
                with open(META_DIR / f"{idx_str}.json", "w", encoding="utf-8") as f:
                    json.dump(meta, f, indent=2)
                
                # Capturar Screenshot
                page.goto(f"file://{html_path.resolve()}")
                
                # Captura apenas a bounding box do elemento gerado, evitando brancos gigantes
                element = page.locator("body > *")
                if element.count() > 0:
                    element.first.screenshot(path=str(IMG_DIR / f"{idx_str}.png"))
                else:
                    page.screenshot(path=str(IMG_DIR / f"{idx_str}.png"))
                    
                print(f"[{idx_str}] Gerado -> {comp_type}")

        browser.close()
    
    print(f"\nDataset fundacional V8 gerado com sucesso! Total: {idx} amostras salvas em ./dataset/")

if __name__ == "__main__":
    main()
