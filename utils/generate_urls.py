import json

CLASSES_FILE = "data/CUB_200_2011/classes.txt"
OUTPUT_FILE = "services/data_collector/dags/config/urls.json"


def class_name_to_wiki_url(class_name: str) -> str:
    name = class_name.split(".", 1)[1]
    words = name.split("_")
    parts = []
    for word in words:
        if word and word[0].isupper():
            parts.append(("_", word))
        else:
            parts.append(("-", word))

    slug = parts[0][1]
    for sep, word in parts[1:]:
        slug += sep + word

    return f"https://en.wikipedia.org/wiki/{slug}"


def main():
    urls = []
    with open(CLASSES_FILE) as f:
        for line in f:
            parts = line.strip().split(" ", 1)
            if len(parts) < 2:
                continue
            class_name = parts[1].strip()
            url = class_name_to_wiki_url(class_name)
            urls.append(url)

    with open(OUTPUT_FILE, "w") as f:
        json.dump(
            {"urls": urls}, 
            f, 
            indent=2
        )

    print(f"Generated {len(urls)} URLs -> {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
