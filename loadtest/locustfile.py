"""Load test for the churn API.

Run (headless, 2 minutes, 50 users):
  locust -f loadtest/locustfile.py --host http://<alb-dns-name> \
         --headless -u 50 -r 5 -t 2m --csv loadtest/results
"""
from locust import HttpUser, between, task

CUSTOMER = {
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 5,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "Yes",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 95.5,
    "TotalCharges": 480.0,
}


class ChurnUser(HttpUser):
    wait_time = between(0.05, 0.2)

    @task(10)
    def predict(self):
        self.client.post("/predict", json=CUSTOMER)

    @task(1)
    def health(self):
        self.client.get("/health")
