import json

db = json.load(open("workspace.json", "r", encoding="utf8"))

while True:
    q = input("Buscar: ").lower()

    for item in db:
        if q in item["content"].lower():

            print("\n=== ENCONTRADO ===")
            print(item["path"])

            trecho = item["content"][:500]

            print("\n")
            print(trecho)
            print("\n" + "="*50)