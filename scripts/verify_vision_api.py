"""Explicit real-model vision proof. Synthetic pixels and optional local image.

No input bytes or model descriptions are saved by this script. Run outside
the Metal sandbox against a separately started --vision server.
"""
import argparse
import base64
import io
import json
import random
import time
import urllib.error
import urllib.request
from pathlib import Path


def request(base, route, payload=None):
    raw = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(base + route, data=raw, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=240) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read())


def image_part(raw, mime="png"):
    return {"type": "image_url", "image_url": {"url": "data:image/" + mime + ";base64," + base64.b64encode(raw).decode()}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="http://127.0.0.1:8082")
    parser.add_argument("--image", type=Path, help="Optional private local photo; no bytes/content stored")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    from PIL import Image, ImageDraw
    status, cap = request(args.base, "/v1/capabilities")
    if status != 200 or not cap.get("vision", {}).get("supported"):
        raise SystemExit("Server does not advertise verified vision support")
    _, before = request(args.base, "/status")
    summary = {"model": before.get("model"), "runtime": before.get("runtime"), "probes": []}
    keys = ["top_left", "top_right", "bottom_left", "bottom_right"]
    schema = {"type": "object", "properties": {k: {"type": "string", "enum": ["red", "green", "blue", "yellow"]} for k in keys}, "required": keys, "additionalProperties": False}
    for _ in range(2):
        colors = random.SystemRandom().sample(["red", "green", "blue", "yellow"], 4)
        image = Image.new("RGB", (512, 512))
        draw = ImageDraw.Draw(image)
        for bounds, color in zip([(0, 0, 255, 255), (256, 0, 511, 255), (0, 256, 255, 511), (256, 256, 511, 511)], colors):
            draw.rectangle(bounds, fill=color)
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        payload = {"model": "local", "messages": [{"role": "user", "content": [{"type": "text", "text": "Identify each quadrant color in the image. Return JSON with top_left, top_right, bottom_left, bottom_right. Do not guess from this text."}, image_part(buffer.getvalue())]}], "temperature": 0, "max_tokens": 128, "response_format": {"type": "json_schema", "json_schema": {"name": "quadrants", "strict": True, "schema": schema}}}
        start = time.monotonic()
        code, output = request(args.base, "/v1/chat/completions", payload)
        content = json.loads(output["choices"][0]["message"]["content"]) if code == 200 else None
        _, state = request(args.base, "/status")
        summary["probes"].append({"kind": "random_color_quadrants", "status": code, "exact_match": content == dict(zip(keys, colors)), "latency_s": round(time.monotonic() - start, 2), "model_unloaded": state.get("model_loaded") is False, "model_available": state.get("model_available"), "metrics": state.get("stats", {}).get("last_generation"), "error": output.get("error")})
    if args.image:
        with Image.open(args.image) as source:
            source.thumbnail((1536, 1536))
            buffer = io.BytesIO()
            source.convert("RGB").save(buffer, format="JPEG", quality=90)
        schema = {"type": "object", "properties": {"scene": {"type": "string"}, "objects": {"type": "array", "items": {"type": "string"}}, "uncertainties": {"type": "array", "items": {"type": "string"}}}, "required": ["scene", "objects", "uncertainties"], "additionalProperties": False}
        code, output = request(args.base, "/v1/chat/completions", {"model": "local", "messages": [{"role": "user", "content": [{"type": "text", "text": "Describe only visible objects/materials/colors. State uncertainty explicitly. Return scene, objects, uncertainties as JSON."}, image_part(buffer.getvalue(), "jpeg")]}], "temperature": 0, "max_tokens": 384, "response_format": {"type": "json_schema", "json_schema": {"name": "visible_scene", "strict": True, "schema": schema}}})
        _, state = request(args.base, "/status")
        summary["probes"].append({"kind": "private_image_schema_only", "status": code, "content_not_persisted": True, "visual_accuracy_verified": False, "metrics": state.get("stats", {}).get("last_generation"), "error": output.get("error")})
    args.out.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    raise SystemExit(0 if all(p["status"] == 200 and p.get("exact_match", True) for p in summary["probes"]) else 1)


if __name__ == "__main__":
    main()
