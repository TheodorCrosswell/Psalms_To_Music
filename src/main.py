from fastapi import FastAPI, Depends
from fastapi.responses import FileResponse
from starlette.staticfiles import StaticFiles
from helpers import (
    get_all_kjv_words_syllables_str,
    get_fuzzy_matches,
    match_results_to_json,
    text_to_words_syllables_str,
)
from functools import lru_cache
from models import WordDetail
from typing import List, Dict

app = FastAPI()


@lru_cache()
def get_kjv() -> tuple[list[str], str]:
    return get_all_kjv_words_syllables_str("data/kjv.txt")


@app.get("/fuzzy_search/kjv/{searchText}", response_model=list[list[WordDetail]])
async def read_items(
    searchText: str, kjv: tuple[list[str], str] = Depends(get_kjv)
) -> list[list[WordDetail]]:
    kjv_words, kjv_syllables = kjv
    searchText_words, searchText_syllables = text_to_words_syllables_str(searchText)
    matches = get_fuzzy_matches(searchText_syllables, kjv_syllables, 95)
    results = [
        match_results_to_json(
            kjv_words, searchText_words, kjv_syllables, searchText_syllables, match[2]
        )
        for match in matches
    ]
    return results


@app.get("/")
def get_index() -> FileResponse:
    return FileResponse("src/frontend/dist/index.html")


app.mount(
    "/",
    StaticFiles(directory="src/frontend/dist"),
    name="/",
)
