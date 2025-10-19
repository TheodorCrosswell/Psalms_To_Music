from fastapi import FastAPI, Depends
from fastapi.responses import FileResponse
from starlette.staticfiles import StaticFiles
from data_readers import read_text
from functools import lru_cache
from models import WordsSyllables

app = FastAPI()


@lru_cache()
def get_kjv_words_syllables() -> WordsSyllables:
    kjv_text = read_text("./data/kjv.txt")
    kjv_words_syllables = WordsSyllables(kjv_text)
    return kjv_words_syllables


@app.get("/api/find_matches/kjv", response_model=list[dict])
async def read_items(
    searchText: str,
    scoreCutoff: int,
    kjv_words_syllables: WordsSyllables = Depends(get_kjv_words_syllables),
) -> list[dict]:
    search_text = searchText
    score_cutoff = scoreCutoff
    search_words_syllables = WordsSyllables(search_text)
    results = search_words_syllables.find_matches(kjv_words_syllables, score_cutoff)
    return results[0:50]


@app.get("/")
def get_index() -> FileResponse:
    return FileResponse("src/frontend/dist/index.html")


app.mount(
    "/",
    StaticFiles(directory="src/frontend/dist"),
    name="/",
)
