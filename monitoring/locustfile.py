import random
from pathlib import Path
from locust import HttpUser, between, task

IMAGES = [(p.name, p.read_bytes()) for p in Path("data/test").glob("*/*.jpg")]


class ClinicUser(HttpUser):
    """One simulated person using the app."""
    wait_time = between(0.5, 1.5)

    # Weight predictions five times higher than health checks.
    @task(5)
    def predict(self):
        name, data = random.choice(IMAGES)
        self.client.post("/predict", files={"file": (name, data, "image/jpeg")})

    @task(1)
    def health(self):
        self.client.get("/health")