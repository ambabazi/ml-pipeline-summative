import random
from pathlib import Path
from locust import HttpUser, between, task

IMAGES = [(p.name, p.read_bytes()) for p in Path("data/test").glob("*/*.jpg")]  # load test images into memory once


class ClinicUser(HttpUser):
    """One simulated person using the app."""
    wait_time = between(0.5, 1.5)          # each user pauses 0.5–1.5 s between actions, like a real person

    @task(5)                               # weight 5: predictions are 5x more common than health checks
    def predict(self):
        name, data = random.choice(IMAGES)                                     # random test drawing
        self.client.post("/predict", files={"file": (name, data, "image/jpeg")})  # upload it for a prediction

    @task(1)                               # weight 1: occasional health check (like the dashboard polling)
    def health(self):
        self.client.get("/health")