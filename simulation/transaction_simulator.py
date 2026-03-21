import requests
import random
import time

API_URL = "https://fraud-detection-ml-system.onrender.com/predict"

def generate_transaction():

    time = random.randint(0, 86400)

    if random.random() < 0.05:
        amount = random.uniform(5000,15000)
    else:
        amount = random.uniform(10,500)

    return {
        "Time": time,
        "Amount": round(amount,2)
    }

def send_transaction(txn):

    response = requests.post(API_URL, json=txn)

    if response.status_code == 200:

        result = response.json()

        print("Transaction:", txn)
        print("Fraud Probability:", result["fraud_probability"])
        print("Decision:", result["decision"])
        print("-"*40)

    else:
        
        print("API error:", response.text)
        
def simulate_transactions(n=20):

    for i in range(n):

        txn = generate_transaction()

        send_transaction(txn)

        time.sleep(1)

if __name__ == "__main__":

    simulate_transactions(20)