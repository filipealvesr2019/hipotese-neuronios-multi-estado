import json
from pathlib import Path

# Coloque aqui as pastas que quer ler
folders_to_scan = [
    "workspace_test/aiclinic",
    "workspace_test/aiforge",
    "workspace_test/auroradigital"
]

all_files = []

for folder in folders_to_scan:
    path = Path(folder)
    for file_path in path.rglob("*.*"):  # pega todos os arquivos
        try:
            content = file_path.read_text(encoding="utf-8")
            all_files.append({
                "path": str(file_path),
                "content": content
            })
        except:
            print(f"Erro lendo: {file_path}")

# Salva tudo em JSON
with open("workspace_docs.json", "w", encoding="utf-8") as f:
    json.dump(all_files, f, indent=2, ensure_ascii=False)

print(f"{len(all_files)} arquivos salvos em workspace_docs.json")