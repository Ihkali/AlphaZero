"""Quick scan to see Event types and TimeControl values in the CSV."""
import csv
from collections import Counter

events = Counter()
tcs = Counter()

with open("chess_games.csv", "r", errors="replace") as f:
    r = csv.DictReader(f)
    for i, row in enumerate(r):
        events[row.get("Event", "").strip()] += 1
        tcs[row.get("TimeControl", "").strip()] += 1
        if i >= 500_000:
            break

print("=== Event types (top 20) ===")
for k, v in events.most_common(20):
    print(f"  {v:>10,}  {repr(k)}")

print()
print("=== TimeControl (top 30) ===")
for k, v in tcs.most_common(30):
    print(f"  {v:>10,}  {repr(k)}")
