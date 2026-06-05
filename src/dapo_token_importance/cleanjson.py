import json
from collections import Counter
from pathlib import Path
input_path = Path("outputs/dapo_selected_tokens.json")
output_path = Path("outputs/dapo_tokens_anallysis.json")
tokens = json.loads(input_path.read_text(encoding="utf-8"))
def normalize_token(token: str) -> str:
    token = token.replace("Ġ", "")
    token = token.replace("▁", "")
    return token.strip()
counts = Counter()
for token in tokens:
    normalized = normalize_token(token)
    if normalized:
        counts[normalized] += 1
summary = [
    {"token": token, "frequency": frequency}
    for token, frequency in counts.most_common()
]
output_path.write_text(
    json.dumps(summary, ensure_ascii=False, indent=2),
    encoding="utf-8",
)
print(f"wrote {len(summary)} grouped tokens to {output_path}")
