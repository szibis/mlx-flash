"""Source-checkout proof: synthetic board plus independent Node arithmetic.

Requires the image-edit/dev dependencies and Node.js; runs no model or server.
"""

import argparse
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

repo = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo))
from mlx_flash_compress.studio_layers import prepare_studio_edge_correction, recolor_studio_scene  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output-dir", type=Path, required=True)
arguments = parser.parse_args()
arguments.output_dir.mkdir(parents=True, exist_ok=True)
spec = importlib.util.spec_from_file_location("edge_tests", repo / "tests/test_studio_edges.py")
fixture = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixture)
scene, matte, foreground, alpha = fixture.mixture_fixture()
basis, basis_receipt = prepare_studio_edge_correction(scene, matte)
board = Image.new("RGB", (4 * 320, 3 * 288), "white")
draw = ImageDraw.Draw(board)
rows = []
js = r"""
let input=''; process.stdin.on('data',c=>input+=c); process.stdin.on('end',()=>{
const d=JSON.parse(input), out=new Uint8ClampedArray(d.scene.length);
for(let p=0;p<d.alpha.length;p++) {
  const a=d.alpha[p]/255, base=[], candidate=[];
  for(let c=0;c<3;c++) {
    const gain=d.target[c]/d.reference[c], C=d.scene[p*3+c];
    base[c]=C*(a+(1-a)*gain);
    candidate[c]=C+(1-a)*(gain-1)*d.basis[p*4+c];
  }
  const supported=d.basis[p*4+3]===255 && candidate.every((v,c)=>Math.abs(v-base[c])<=64);
  for(let c=0;c<3;c++) out[p*3+c]=supported?candidate[c]:base[c];
} process.stdout.write(JSON.stringify(Array.from(out)));
});
"""
for row, target in enumerate([(24, 36, 30), (250, 240, 220), (150, 35, 220)]):
    old, old_receipt = recolor_studio_scene(scene, matte, target)
    new, receipt = recolor_studio_scene(scene, matte, target, edge_mode="local_unmix")
    expected = Image.fromarray(
        np.rint(alpha[..., None] / 255 * foreground + (1 - alpha[..., None] / 255) * target).astype(np.uint8)
    )
    payload = {
        "scene": np.asarray(scene).flatten().tolist(),
        "alpha": alpha.flatten().tolist(),
        "basis": np.asarray(basis).flatten().tolist(),
        "reference": receipt["reference_rgb"],
        "target": target,
    }
    proc = subprocess.run(["node", "-e", js], input=json.dumps(payload), text=True, capture_output=True, check=True)
    browser = np.asarray(json.loads(proc.stdout), dtype=np.uint8).reshape(np.asarray(scene).shape)
    assert np.array_equal(browser, np.asarray(new))
    edge = (alpha > 0) & (alpha < 255)
    edge[:25] = False
    edge[103:] = False
    errors = {}
    for label, img in [("legacy", old), ("local_unmix", new)]:
        error = np.abs(np.asarray(img).astype(int) - np.asarray(expected).astype(int))[edge]
        errors[label] = {"max_channel_error": int(error.max()), "mean_absolute_channel_error": float(error.mean())}
    rows.append({"target": target, "errors": errors, "browser_different_pixels": 0, "receipt": receipt})
    for col, (label, img) in enumerate(
        [("source", scene), ("legacy", old), ("local unmix", new), ("known mixture", expected)]
    ):
        board.paste(img.resize((320, 256), Image.Resampling.NEAREST), (col * 320, row * 288 + 32))
        draw.text((col * 320 + 6, row * 288 + 8), f"{label} / {target}", fill="black")
asset = arguments.output_dir / "studio-edge-synthetic-2026-10-16.png"
board.save(asset)
proof = {"board": asset.name, "basis": basis_receipt, "rows": rows}
(arguments.output_dir / "studio-edge-proof.json").write_text(json.dumps(proof, indent=2) + "\n")
print(json.dumps({"node_replay_different_pixels": 0, "targets": len(rows), "visual_quality_approved": False}))
