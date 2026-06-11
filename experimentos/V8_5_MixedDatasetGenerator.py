import os
import json
import random
import pathlib
from playwright.sync_api import sync_playwright

OUT_DIR = pathlib.Path("dataset_mixed")
IMG_DIR = OUT_DIR / "images"
CODE_DIR = OUT_DIR / "code"
META_DIR = OUT_DIR / "metadata"

for d in [IMG_DIR, CODE_DIR, META_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Famílias do Teste Cruel
FAMILIES = {
    "Header": [
        '<header style="{style}"><h1>{logo}</h1><nav><span>Home</span><span>About</span></nav></header>'
    ],
    "Card": [
        '<div style="{style}"><img src="https://via.placeholder.com/150" alt="img"><div style="padding:10px"><h3>{title}</h3><p>Description for this card.</p><button style="{btn_style}">Read More</button></div></div>'
    ],
    "Form": [
        '<form style="{style}"><h3>{title}</h3><input type="text" placeholder="Email" style="{input_style}"/><input type="password" placeholder="Password" style="{input_style}"/><button type="submit" style="{btn_style}">Submit</button></form>'
    ],
    "Navbar": [
        '<nav style="{style}"><span>{logo}</span><div><span>Link 1</span> | <span>Link 2</span></div></nav>'
    ],
    "Footer": [
        '<footer style="{style}"><p>&copy; 2026 {logo} Inc.</p><div><span>Privacy</span> | <span>Terms</span></div></footer>'
    ]
}

LOGOS = ["Logo", "Brand", "MySite", "App", "System"]
TITLES = ["Welcome", "Sign In", "Profile", "Product", "Details"]

def random_color():
    return f"#{random.randint(0, 0xFFFFFF):06x}"

def generate_style(family):
    bg_color = random_color()
    color = "#ffffff" if random.choice([True, False]) else "#000000"
    padding = f"{random.randint(10, 30)}px"
    border_radius = f"{random.randint(0, 16)}px"
    
    if family in ["Header", "Navbar", "Footer"]:
        return f"width: 100%; box-sizing: border-box; display: flex; justify-content: space-between; align-items: center; padding: {padding}; background-color: {bg_color}; color: {color}; font-family: sans-serif;"
    elif family == "Card":
        return f"width: 300px; border-radius: {border_radius}; overflow: hidden; background-color: {bg_color}; color: {color}; box-shadow: 0 4px 8px rgba(0,0,0,0.1); font-family: sans-serif;"
    elif family == "Form":
        return f"width: 300px; padding: {padding}; border-radius: {border_radius}; background-color: {bg_color}; color: {color}; display: flex; flex-direction: column; gap: 10px; font-family: sans-serif; box-shadow: 0 4px 8px rgba(0,0,0,0.1);"

def generate_sub_style(element):
    bg = random_color()
    if element == "btn":
        return f"padding: 10px; border-radius: 4px; border: none; background-color: {bg}; color: #fff; cursor: pointer; font-weight: bold;"
    elif element == "input":
        return f"padding: 8px; border-radius: 4px; border: 1px solid #ccc; width: 100%; box-sizing: border-box;"

def main():
    print("Gerando Dataset V8.5: O Teste Cruel (Mistura de Componentes)...")
    samples_per_family = 500 # 2500 total
    idx = 0
    
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1024, "height": 800}) 
        
        for family_name, templates in FAMILIES.items():
            print(f"Gerando {samples_per_family} amostras de {family_name}...")
            for _ in range(samples_per_family):
                idx += 1
                idx_str = f"{idx:06d}"
                template = random.choice(templates)
                
                logo = random.choice(LOGOS)
                title = random.choice(TITLES)
                
                style = generate_style(family_name)
                btn_style = generate_sub_style("btn")
                input_style = generate_sub_style("input")
                
                html_body = template.format(style=style, logo=logo, title=title, btn_style=btn_style, input_style=input_style)
                
                html_content = f"""
                <!DOCTYPE html>
                <html>
                <head><meta charset="utf-8"></head>
                <body style="margin: 0; padding: 20px; display: flex; justify-content: center; align-items: flex-start; background-color: #f0f0f0;">
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
                
                # Selecionar o primeiro elemento filho do body para capturar a bounding box exata
                element = page.locator("body > *").first
                if element.count() > 0:
                    element.screenshot(path=str(IMG_DIR / f"{idx_str}.png"))
                else:
                    page.screenshot(path=str(IMG_DIR / f"{idx_str}.png"))
                    
        browser.close()
    print(f"\nDataset V8_5 gerado! Total: {idx} componentes misturados salvos em ./dataset_mixed/")

if __name__ == "__main__":
    main()
