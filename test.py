from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import asyncio
import json
import random

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


async def event_generator():
    for i in range(10):
        data = {
            "event": random.choice(["message", "notification"]),
            "id": i,
            "value": random.randint(1, 100)
        }

        yield f"data: {json.dumps(data)}\n\n"

        await asyncio.sleep(1)


@app.get("/events")
async def events():
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream"
    )
