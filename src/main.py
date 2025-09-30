from fastapi import FastAPI
from fastapi.responses import FileResponse
from starlette.staticfiles import StaticFiles
from helpers import get_all_kjv_syllables_str
import os

app = FastAPI()

backend_dir = os.path.dirname(os.path.abspath(__file__))  # /backend
project_root = os.path.dirname(backend_dir)  # /


@app.get("/")
def get_index():
    return FileResponse("src/frontend/dist/index.html")


app.mount(
    "/",
    StaticFiles(directory="src/frontend/dist"),
    name="/",
)
