from pathlib import Path
import json
EXTENSOES = {
    ".py",
    ".js",
    ".ts",
    ".tsx",
    ".jsx",
    ".html",
    ".css",
    ".scss",
    ".php",
    ".cpp",
    ".c",
    ".h",
    ".hpp",
    ".cs",
    ".java",
    ".md",
    ".txt",
    ".json",
    ".xml",
    ".yml",
    ".yaml"
}

workspace = Path("workspace_test")

dados = []

for arquivo in workspace.rglob("*"):
    if arquivo.is_file() and arquivo.suffix.lower() in EXTENSOES:

        try:
            conteudo = arquivo.read_text(
                encoding="utf-8",
                errors="ignore"
            )

            dados.append({
                "path": str(arquivo),
                "ext": arquivo.suffix,
                "size": len(conteudo),
                "content": conteudo[:5000]
            })

        except:
            pass

with open("workspace.json", "w", encoding="utf-8") as f:
    json.dump(dados, f, indent=2)

print(f"Arquivos lidos: {len(dados)}")