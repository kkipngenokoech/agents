"""
ECE 462/662 - HW1 Part 2, Problem 2: Utility-Based Content Recommendation Agent
Relevance = 0.40*interest + 0.20*popularity + 0.25*duration + 0.15*freshness
"""
from typing import Dict, List, Tuple
import math

def recommend_content(user_profile: Dict, content_items: List[Dict], k: int) -> List[Tuple[str, float]]:
    """Recommend top k content items based on user profile."""
    interests = user_profile.get("interests", {})
    avg_time = user_profile.get("avg_watch_time", 15.0)
    scored = [(item["id"], calculate_relevance(interests, avg_time, item)) for item in content_items]
    return sorted(scored, key=lambda x: x[1], reverse=True)[:k]

def calculate_relevance(interests: Dict[str, int], avg_time: float, item: Dict) -> float:
    """Calculate relevance score (0-1) for a content item."""
    # Interest alignment (40%)
    max_int = max(interests.values()) if interests else 1
    interest = min(sum(interests.get(t, 0) for t in item.get("topics", [])) / max_int, 1.0)
    # Popularity (20%)
    views, likes = item.get("views", 1), item.get("likes", 0)
    popularity = math.sqrt(min(views / 50000, 1.0) * (likes / max(views, 1)))
    # Duration compatibility (25%)
    duration = math.exp(-((item.get("duration", avg_time) - avg_time) ** 2) / 72)
    # Freshness (15%)
    freshness = 2.0 ** (-item.get("days_since_upload", 30) / 30.0)
    return 0.40 * interest + 0.20 * popularity + 0.25 * duration + 0.15 * freshness

# Sample data
SAMPLE_USER = {"interests": {"AI": 8, "programming": 6, "machine_learning": 7, "cooking": 2}, "avg_watch_time": 15.0}
SAMPLE_CONTENT = [
    {"id": "video1", "topics": ["cooking"], "views": 50000, "likes": 3000, "duration": 20, "days_since_upload": 5},
    {"id": "video2", "topics": ["AI", "machine_learning"], "views": 120000, "likes": 8000, "duration": 18, "days_since_upload": 30},
    {"id": "video3", "topics": ["programming"], "views": 80000, "likes": 5500, "duration": 12, "days_since_upload": 10},
    {"id": "video4", "topics": ["AI", "data_science"], "views": 95000, "likes": 7200, "duration": 14, "days_since_upload": 3},
]

def run_test_cases():
    """Run with sample data."""
    print("\n[TEST] User interests: AI(8), ML(7), Programming(6), Cooking(2) | Avg time: 15 min\n")
    for i, (vid, score) in enumerate(recommend_content(SAMPLE_USER, SAMPLE_CONTENT, 3), 1):
        item = next(v for v in SAMPLE_CONTENT if v["id"] == vid)
        print(f"  {i}. {vid} (score: {score:.3f}) - {item['topics']}, {item['duration']} min")

def run_custom():
    """Run with custom user input."""
    print("\n[CUSTOM MODE] Enter your profile:\n")
    print("  Example format for interests: AI: 8, cooking: 3, sports: 5")
    raw = input("  Your interests (topic: level, ...): ")
    interests = {}
    for pair in raw.split(","):
        if ":" in pair:
            t, l = pair.split(":")
            interests[t.strip()] = int(l.strip())
    avg_time = float(input("  Avg watch time (minutes): ") or "15")

    print("\n  Enter content items (or 'done'):")
    print("  Format: id, topics(;sep), views, likes, duration, days_old")
    print("  Example: video1, AI;ML, 50000, 3000, 15, 7\n")
    items = []
    while True:
        line = input("  > ").strip()
        if line.lower() == "done" or not line:
            break
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 6:
            items.append({"id": parts[0], "topics": parts[1].split(";"), "views": int(parts[2]),
                         "likes": int(parts[3]), "duration": float(parts[4]), "days_since_upload": int(parts[5])})

    if not items:
        print("  No items entered.")
        return
    user = {"interests": interests, "avg_watch_time": avg_time}
    print(f"\n[RESULTS] Top {min(3, len(items))} recommendations:\n")
    for i, (vid, score) in enumerate(recommend_content(user, items, 3), 1):
        item = next(v for v in items if v["id"] == vid)
        print(f"  {i}. {vid} (score: {score:.3f}) - {item['topics']}")

if __name__ == "__main__":
    print("\n" + "="*50)
    print("  UTILITY-BASED CONTENT RECOMMENDATION AGENT")
    print("="*50)
    while True:
        print("\n  1. Run with test cases\n  2. Try with custom profile\n  3. Exit\n")
        choice = input("  Choice: ").strip()
        if choice == "1": run_test_cases()
        elif choice == "2": run_custom()
        elif choice == "3": print("\n  Goodbye!\n"); break
