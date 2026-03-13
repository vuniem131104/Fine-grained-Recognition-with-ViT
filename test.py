from __future__ import annotations

import concurrent.futures
import time

import requests

URL = 'http://localhost:8080/predict'
IMAGE_PATH = '/home/lehoangvu/Project_AIDE1/data/CUB_200_2011/images/003.Sooty_Albatross/Sooty_Albatross_0001_1071.jpg'
NUM_REQUESTS = 10


def send_request(request_id: int) -> dict:
    with open(IMAGE_PATH, 'rb') as f:
        files = {'file': (IMAGE_PATH, f, 'image/jpeg')}
        headers = {'accept': 'application/json'}
        start = time.time()
        try:
            response = requests.post(URL, headers=headers, files=files)
            elapsed = time.time() - start
            return {
                'id': request_id,
                'status_code': response.status_code,
                'response': response.json(),
                'elapsed': elapsed,
            }
        except Exception as e:
            elapsed = time.time() - start
            return {
                'id': request_id,
                'status_code': None,
                'error': str(e),
                'elapsed': elapsed,
            }


def main():
    print(f"Sending {NUM_REQUESTS} requests concurrently to {URL}...")
    start_all = time.time()

    with concurrent.futures.ThreadPoolExecutor(max_workers=NUM_REQUESTS) as executor:
        futures = {executor.submit(send_request, i): i for i in range(1, NUM_REQUESTS + 1)}
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            if 'error' in result:
                print(f"[Request {result['id']}] ERROR: {result['error']} ({result['elapsed']:.3f}s)")
            else:
                print(f"[Request {result['id']}] {result['status_code']} | {result['response']} ({result['elapsed']:.3f}s)")

    total = time.time() - start_all
    print(f"\nAll {NUM_REQUESTS} requests completed in {total:.3f}s")


if __name__ == '__main__':
    main()
