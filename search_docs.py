import json

with open("workspace_docs.json", "r", encoding="utf-8") as f:
    documents = json.load(f)

query = input("Buscar: ").lower()

results = []

for doc in documents:

    content = doc["content"].lower()

    score = content.count(query)

    if score > 0:
        results.append((score, doc["path"]))

results.sort(reverse=True)

print("\nRESULTADOS:\n")

for score, path in results[:20]:
    print(f"[{score}] {path}")