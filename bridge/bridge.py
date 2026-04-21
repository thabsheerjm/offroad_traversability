import asyncio
import zmq.asyncio
import json
import subprocess as sh
import os, signal
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from contextlib import asynccontextmanager
from datetime import datetime
import os, yaml

CONFIG_PATH = os.environ.get("OFFROAD_CONFIG", "/etc/offroad/offroad.yaml")
def load_config(cPath):
    with open(cPath) as f:
        return yaml.safe_load(f)

cfg = load_config(CONFIG_PATH)


BINARY    = cfg["backend"]["binary"]
MODEL     = cfg["backend"]["model"]
VIDEO_DIR = cfg["video"]["source_dir"]
OUT_VIDEO = os.path.join(cfg["video"]["output_dir"], "session_out.mp4")
LOG_DIR   = cfg["logging"]["log_dir"]
ZMQ_PORT  = cfg["server"]["zmq_port"]
BRIDGE_PORT = cfg["server"]["bridge_port"]

os.makedirs(LOG_DIR, exist_ok=True)

clients: set[WebSocket] = set()
queue: asyncio.Queue = asyncio.Queue(maxsize=10)
backend_proc = None
current_log_path = None
log_file = None

async def zmq_listener():
    ctx = zmq.asyncio.Context()
    sock = ctx.socket(zmq.SUB)
    sock.connect(f"tcp://localhost:{ZMQ_PORT}")
    sock.setsockopt(zmq.SUBSCRIBE, b"")
    print(f"zmq listener connected on port {ZMQ_PORT}")
    while True:
        try:
            msg = await sock.recv_string()
            if queue.full():
                queue.get_nowait()
            await queue.put(msg)

            if log_file:
                data = json.loads(msg)
                log_file.write(
                    f"{data['timestamp_us']},"
                    f"{data['model_id']},"
                    f"{data['confidence']},"
                    f"{data['inference_ms']},"
                    f"{data['preprocess_ms']},"
                    f"{data['display_fps']}\n"
                )
                log_file.flush()
        except Exception as e:
            print(f"ZMQ error: {e}")
            await asyncio.sleep(0.01)

async def broadcaster():
    global clients
    while True:
        msg = await queue.get()
        dead = set()
        for ws in clients:
            try:
                await ws.send_text(msg)
            except Exception:
                dead.add(ws)
        clients -= dead

def start_backend(video_path: str):
    global backend_proc
    if backend_proc and backend_proc.poll() is None:
        backend_proc.terminate()
        backend_proc.wait()
    env = os.environ.copy()
    env["OFFROAD_ZMQ_PORT"] = str(ZMQ_PORT)
    cmd = [BINARY, MODEL, video_path, OUT_VIDEO]
    backend_proc = sh.Popen(cmd, env=env)
    print(f"Backend started: PID = {backend_proc.pid}")

def stop_backend():
    global backend_proc
    if backend_proc and backend_proc.poll() is None:
        backend_proc.terminate()
        backend_proc.wait()
        backend_proc = None
        print("Backend stopped")

def open_log():
    global log_file, current_log_path
    fname = datetime.now().strftime("session_%Y%m%d_%H%M%S.csv")
    current_log_path = os.path.join(LOG_DIR, fname)
    log_file = open(current_log_path, "w")
    log_file.write("timestamp_us,model_id,confidence,inference_ms,preprocess_ms,fps\n")
    print(f"Logging to: {current_log_path}")

def close_log():
    global log_file
    if log_file:
        log_file.close()
        log_file = None
        print("Log closed")

async def handle_cmd(data: dict, websocket: WebSocket):
    cmd = data.get("cmd")

    if cmd == "start_session":
        video = data.get("video", "bicycle.mp4")
        video_path = os.path.join(VIDEO_DIR, video)
        open_log()
        start_backend(video_path)
        await websocket.send_text(json.dumps({"event": "session_started"}))

    elif cmd == "end_session":
        stop_backend()
        close_log()
        await websocket.send_text(json.dumps({
            "event": "session_ended",
            "log_path": current_log_path
        }))

    elif cmd == "get_videos":
        videos = [f for f in os.listdir(VIDEO_DIR)
                  if f.endswith(('.mp4', '.avi', '.mov'))]
        await websocket.send_text(json.dumps({
            "event": "video_list",
            "videos": sorted(videos)
        }))

@asynccontextmanager
async def lifespan(app: FastAPI):
    asyncio.create_task(zmq_listener())
    asyncio.create_task(broadcaster())
    yield
    stop_backend()
    close_log()

app = FastAPI(lifespan=lifespan)

app.add_middleware(CORSMiddleware,
                   allow_origins=["*"],
                   allow_methods=["*"],
                   allow_headers=["*"])

@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket):
    await websocket.accept()
    clients.add(websocket)
    print(f"Client connected. Total: {len(clients)}")
    try:
        while True:
            raw = await websocket.receive_text()
            try:
                data = json.loads(raw)
                await handle_cmd(data, websocket)
            except Exception as e:
                print(f"Command error: {e}")

    except WebSocketDisconnect:
        clients.discard(websocket)
        print(f"Client disconnected. Total: {len(clients)}")

@app.get("/download_log")
async def download_log():
    from fastapi.responses import FileResponse
    if current_log_path and os.path.exists(current_log_path):
        return FileResponse(
            current_log_path,
            media_type="text/csv",
            filename=os.path.basename(current_log_path)
        )
    return {"error": "No log available"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=BRIDGE_PORT)
