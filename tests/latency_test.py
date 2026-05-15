import requests
import time
import statistics

API_URL = "PASTE_YOUR_API_URL_HERE"

image_urls = [
    {
        "label": "Cat",
        "url": "https://images.unsplash.com/photo-1514888286974-6c03e2ca1dba"
    },
    {
        "label": "Dog 1",
        "url": "https://images.unsplash.com/photo-1552053831-71594a27632d"
    },
    {
        "label": "Dog 2",
        "url": "https://images.unsplash.com/photo-1587300003388-59208cc962cb"
    },
    {
        "label": "Car",
        "url": "https://images.unsplash.com/photo-1494976388531-d1058494cdd8"
    },
    {
        "label": "Coffee",
        "url": "https://images.unsplash.com/photo-1509042239860-f550ce710b93"
    },
    {
        "label": "Apple",
        "url": "https://images.unsplash.com/photo-1560806887-1e4cd0b6cbd6"
    },
    {
        "label": "Bird",
        "url": "https://images.unsplash.com/photo-1444464666168-49d633b86797"
    },
    {
        "label": "Laptop",
        "url": "https://images.unsplash.com/photo-1496181133206-80ce9b88a853"
    },
    {
        "label": "Backpack",
        "url": "https://images.unsplash.com/photo-1553062407-98eeb64c6a62"
    },
    {
        "label": "Bicycle",
        "url": "https://images.unsplash.com/photo-1485965120184-e220f721d03e"
    }
]

latencies = []
rows = []

for i, item in enumerate(image_urls, start=1):
    start = time.time()
    response = requests.post(
        API_URL,
        json={"image_url": item["url"]},
        timeout=90
    )
    end = time.time()

    latency = end - start
    latencies.append(latency)

    try:
        result = response.json()
    except Exception:
        result = {"error": response.text}

    prediction = result.get("prediction", "ERROR")
    confidence = result.get("confidence", "N/A")

    rows.append({
        "request": i,
        "image": item["label"],
        "latency": latency,
        "prediction": prediction,
        "confidence": confidence,
        "status": response.status_code
    })

    print(f"Request {i}: {item['label']}")
    print(f"Status: {response.status_code}")
    print(f"Latency seconds: {latency:.3f}")
    print(f"Prediction: {prediction}")
    print(f"Confidence: {confidence}")
    print(f"Raw response: {response.text}")
    print("-" * 60)

print("SUMMARY")
print(f"Average latency: {statistics.mean(latencies):.3f} seconds")
print(f"Median latency: {statistics.median(latencies):.3f} seconds")
print(f"Minimum latency: {min(latencies):.3f} seconds")
print(f"Maximum latency: {max(latencies):.3f} seconds")

print("\nREPORT TABLE")
print("| Request | Image | Latency seconds | Prediction | Confidence | Status |")
print("|---:|---|---:|---|---:|---:|")

for row in rows:
    confidence_value = row["confidence"]

    if isinstance(confidence_value, float):
        confidence_value = f"{confidence_value:.4f}"

    print(
        f"| {row['request']} | {row['image']} | {row['latency']:.3f} | "
        f"{row['prediction']} | {confidence_value} | {row['status']} |"
    )
