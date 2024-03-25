import uvicorn
import argparse
from fastapi import FastAPI
from contextlib import asynccontextmanager

from config import DEFAULT_PORT
from logic.SocketHandler import on_receive_video
from logic.Inference import init_onnx_sessions

@asynccontextmanager
async def lifespan(app: FastAPI):
    init_onnx_sessions()
    yield

parser = argparse.ArgumentParser()
parser.add_argument('--port', type=int, default=DEFAULT_PORT, help='enter your available port number!')

args = parser.parse_args()
app = FastAPI(lifespan=lifespan)
app.add_websocket_route("/ws", on_receive_video)
uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="info", ws="auto", lifespan="on")  