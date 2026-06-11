import os
import json
import random
import pathlib
from playwright.sync_api import sync_playwright

OUT_DIR = pathlib.Path("dataset_ultimate")
IMG_DIR = OUT_DIR / "images"
CODE_DIR = OUT_DIR / "code"
META_DIR = OUT_DIR / "metadata"

for d in [IMG_DIR, CODE_DIR, META_DIR]:
    d.mkdir(parents=True, exist_ok=True)

LOGOS = ["Logo", "Brand", "MySite", "App", "System"]
TITLES = ["Welcome", "Sign In", "Profile", "Product", "Details"]

def random_color():
    return f"#{random.randint(0, 0xFFFFFF):06x}"

def generate_flavors(family, idx_str):
    bg = random_color()
    color = "#ffffff" if random.choice([True, False]) else "#000000"
    
    logo = random.choice(LOGOS)
    title = random.choice(TITLES)
    
    if family == "Header":
        style = f"display: flex; justify-content: space-between; padding: 20px; background-color: {bg}; color: {color};"
        html_pure = f'<header style="{style}"><h1>{logo}</h1><nav><span>Link1</span></nav></header>'
        flavors = {
            "html": f"<LANG_HTML> <header style=\"{style}\">\n  <h1>{logo}</h1>\n  <nav><span>Link1</span></nav>\n</header>",
            "react": f"<LANG_REACT> export default function Header() {{\n  return (\n    <header style={{{{backgroundColor: '{bg}'}}}}>\n      <h1>{logo}</h1>\n    </header>\n  );\n}}",
            "nextjs": f"<LANG_NEXTJS> import Link from 'next/link';\nexport default function Header() {{\n  return (\n    <header style={{{{backgroundColor: '{bg}'}}}}>\n      <h1>{logo}</h1><Link href=\"/\">Link1</Link>\n    </header>\n  );\n}}",
            "tailwind": f"<LANG_TAILWIND> export default function Header() {{\n  return (\n    <header className=\"flex justify-between p-5 bg-gray-800\">\n      <h1>{logo}</h1>\n    </header>\n  );\n}}"
        }
        
    elif family == "Card":
        style = f"width: 300px; padding: 20px; background-color: {bg}; color: {color}; box-shadow: 0 4px 8px rgba(0,0,0,0.1); border-radius: 8px;"
        html_pure = f'<div style="{style}"><h3>{title}</h3><p>Desc</p><button>Action</button></div>'
        flavors = {
            "html": f"<LANG_HTML> <div style=\"{style}\">\n  <h3>{title}</h3>\n  <button>Action</button>\n</div>",
            "react": f"<LANG_REACT> export default function Card() {{\n  return (\n    <div style={{{{backgroundColor: '{bg}'}}}}>\n      <h3>{title}</h3>\n      <button>Action</button>\n    </div>\n  );\n}}",
            "nextjs": f"<LANG_NEXTJS> import Image from 'next/image';\nexport default function Card() {{\n  return (\n    <div style={{{{backgroundColor: '{bg}'}}}}>\n      <h3>{title}</h3>\n    </div>\n  );\n}}",
            "tailwind": f"<LANG_TAILWIND> export default function Card() {{\n  return (\n    <div className=\"w-64 p-5 shadow-lg rounded-lg bg-white\">\n      <h3>{title}</h3>\n    </div>\n  );\n}}"
        }
        
    elif family == "Form":
        style = f"display: flex; flex-direction: column; gap: 10px; width: 300px; padding: 20px; background-color: {bg}; color: {color};"
        html_pure = f'<form style="{style}"><h3>{title}</h3><input type="text" placeholder="Email" /><input type="password" placeholder="Pass" /><button type="submit">Send</button></form>'
        flavors = {
            "html": f"<LANG_HTML> <form style=\"{style}\">\n  <input type=\"text\" />\n  <button>Send</button>\n</form>",
            "react": f"<LANG_REACT> export default function Form() {{\n  const [val, setVal] = useState('');\n  return (\n    <form style={{{{flexDirection: 'column'}}}}> <input /> </form>\n  );\n}}",
            "nextjs": f"<LANG_NEXTJS> 'use client';\nexport default function Form() {{\n  return <form><input /></form>;\n}}",
            "tailwind": f"<LANG_TAILWIND> export default function Form() {{\n  return (\n    <form className=\"flex flex-col gap-2 p-4\">\n      <input className=\"border p-2\" />\n    </form>\n  );\n}}"
        }

    elif family == "Navbar":
        style = f"display: flex; padding: 15px; background-color: {bg}; color: {color};"
        html_pure = f'<nav style="{style}"><span>{logo}</span></nav>'
        flavors = {
            "html": f"<LANG_HTML> <nav style=\"{style}\">\n  <span>{logo}</span>\n</nav>",
            "react": f"<LANG_REACT> export default function Navbar() {{\n  return <nav><span>{logo}</span></nav>;\n}}",
            "nextjs": f"<LANG_NEXTJS> import Link from 'next/link';\nexport default function Navbar() {{\n  return <nav><Link href=\"/\">{logo}</Link></nav>;\n}}",
            "tailwind": f"<LANG_TAILWIND> export default function Navbar() {{\n  return <nav className=\"flex p-4\"><span>{logo}</span></nav>;\n}}"
        }

    else: # Footer
        style = f"padding: 20px; background-color: {bg}; color: {color}; text-align: center;"
        html_pure = f'<footer style="{style}"><p>&copy; 2026 {logo}</p></footer>'
        flavors = {
            "html": f"<LANG_HTML> <footer style=\"{style}\"><p>&copy; 2026 {logo}</p></footer>",
            "react": f"<LANG_REACT> export default function Footer() {{\n  return <footer><p>&copy; 2026 {logo}</p></footer>;\n}}",
            "nextjs": f"<LANG_NEXTJS> export default function Footer() {{\n  return <footer><p>&copy; 2026 {logo}</p></footer>;\n}}",
            "tailwind": f"<LANG_TAILWIND> export default function Footer() {{\n  return <footer className=\"p-4 text-center\"><p>&copy; 2026 {logo}</p></footer>;\n}}"
        }

    return html_pure, flavors

def main():
    print("Gerando Dataset V8.10: O Experimento Definitivo (5 Componentes x 4 Linguagens)...")
    samples_per_family = 200 # 1000 imagens -> 4000 amostras totais
    families = ["Header", "Card", "Form", "Navbar", "Footer"]
    idx = 0
    
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1024, "height": 600}) 
        
        for fam in families:
            print(f"Gerando {fam}...")
            for _ in range(samples_per_family):
                idx += 1
                idx_str = f"{idx:06d}"
                
                html_pure, flavors = generate_flavors(fam, idx_str)
                
                html_content = f"<!DOCTYPE html><html><body style='margin:0;padding:20px;'>{html_pure}</body></html>"
                html_path = CODE_DIR / f"{idx_str}.html"
                with open(html_path, "w", encoding="utf-8") as f:
                    f.write(html_content)
                    
                page.goto(f"file://{html_path.resolve()}")
                element = page.locator("body > *").first
                if element.count() > 0:
                    element.screenshot(path=str(IMG_DIR / f"{idx_str}.png"))
                else:
                    page.screenshot(path=str(IMG_DIR / f"{idx_str}.png"))
                    
                for lang, target_code in flavors.items():
                    meta = {
                        "id": idx_str,
                        "family": fam,
                        "language": lang,
                        "target_code": target_code,
                        "image_file": f"{idx_str}.png"
                    }
                    meta_filename = f"{idx_str}_{lang}.json"
                    with open(META_DIR / meta_filename, "w", encoding="utf-8") as f:
                        json.dump(meta, f, indent=2)
                        
        browser.close()
    print(f"\nDataset Ultimate gerado! 1000 imagens, 4000 amostras.")

if __name__ == "__main__":
    main()
