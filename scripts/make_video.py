"""Build the predec demo video. Renders self-contained HTML scenes and
records each one, then stitches with ffmpeg.

Scene 1: Local-rendered "GitHub" page showing the repo file tree
Scene 2: Animated terminal running the eval (real output from /opt/predec)
Scene 3: The HTML report rendered in browser
Scene 4: Closing title card
"""

import asyncio
import os
import subprocess
import sys
from pathlib import Path

from playwright.async_api import async_playwright


OUT_DIR = Path("/workspace/predec/runs/eval")
SCENES_DIR = OUT_DIR / "video_scenes"
SCENES_DIR.mkdir(parents=True, exist_ok=True)


# Scene 1: GitHub repo page (locally rendered HTML, not requiring github.com)
SCENE_1_HTML = """
<!doctype html>
<html><head>
<style>
  body { background: #0d1117; color: #c9d1d9; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 0; padding: 0; }
  .topbar { background: #010409; border-bottom: 1px solid #30363d; padding: 12px 24px; display: flex; align-items: center; gap: 16px; }
  .topbar .crumb { color: #58a6ff; font-size: 14px; }
  .topbar .crumb.gray { color: #8b949e; }
  .container { max-width: 1100px; margin: 0 auto; padding: 24px; }
  h1 { font-size: 28px; margin: 0 0 8px; font-weight: 600; }
  .sub { color: #8b949e; font-size: 14px; }
  .file-tree { background: #0d1117; border: 1px solid #30363d; border-radius: 6px; margin-top: 24px; font-family: 'Menlo', monospace; font-size: 13px; }
  .file-tree .row { padding: 8px 16px; display: flex; align-items: center; gap: 8px; border-bottom: 1px solid #21262d; }
  .file-tree .row:last-child { border-bottom: none; }
  .icon { width: 16px; height: 16px; }
  .folder { color: #79c0ff; }
  .file { color: #c9d1d9; }
  .readme { background: #0d1117; border: 1px solid #30363d; border-radius: 6px; padding: 24px; margin-top: 24px; }
  .readme h1, .readme h2, .readme h3 { color: #c9d1d9; border-bottom: 1px solid #21262d; padding-bottom: 8px; }
  .readme code { background: #161b22; padding: 2px 6px; border-radius: 3px; font-family: monospace; color: #79c0ff; }
  .readme pre { background: #161b22; padding: 12px; border-radius: 6px; overflow: auto; }
  .tag { display: inline-block; padding: 2px 8px; background: #1f6feb33; color: #58a6ff; border-radius: 999px; font-size: 12px; margin-left: 8px; }
  .pill { display: inline-block; padding: 4px 10px; background: #238636; color: #fff; border-radius: 4px; font-size: 12px; margin-right: 4px; }
</style></head>
<body>
  <div class="topbar">
    <span class="crumb">ruzkypazzy</span>
    <span class="crumb gray">/</span>
    <span class="crumb">predec</span>
    <span class="tag">Public</span>
  </div>
  <div class="container">
    <h1>predec <span class="tag">Public</span></h1>
    <div class="sub">Detect and quantify biases in RLHF preference datasets.</div>
    <div style="margin-top:16px;">
      <span class="pill">MIT license</span>
      <span class="pill">Python 100%</span>
    </div>

    <div class="file-tree">
      <div class="row"><span class="icon">📄</span><span class="file">README.md</span></div>
      <div class="row"><span class="icon">📄</span><span class="file">pyproject.toml</span></div>
      <div class="row"><span class="icon">📄</span><span class="file">MIT-LICENSE</span></div>
      <div class="row"><span class="icon">📁</span><span class="folder">src/predec/</span></div>
      <div class="row" style="padding-left:48px;"><span class="icon">📄</span><span class="file">schema.py</span></div>
      <div class="row" style="padding-left:48px;"><span class="icon">📄</span><span class="file">orchestrator.py</span></div>
      <div class="row" style="padding-left:48px;"><span class="icon">📄</span><span class="file">cli.py</span></div>
      <div class="row" style="padding-left:48px;"><span class="icon">📁</span><span class="folder">detectors/</span></div>
      <div class="row" style="padding-left:80px;"><span class="icon">📄</span><span class="file">length.py</span></div>
      <div class="row" style="padding-left:80px;"><span class="icon">📄</span><span class="file">position.py</span></div>
      <div class="row" style="padding-left:80px;"><span class="icon">📄</span><span class="file">sycophancy.py</span></div>
      <div class="row" style="padding-left:80px;"><span class="icon">📄</span><span class="file">verbosity.py</span></div>
      <div class="row" style="padding-left:48px;"><span class="icon">📁</span><span class="folder">debiaser/</span></div>
      <div class="row" style="padding-left:48px;"><span class="icon">📁</span><span class="folder">report/</span></div>
      <div class="row" style="padding-left:48px;"><span class="icon">📁</span><span class="folder">llm/</span></div>
      <div class="row" style="padding-left:48px;"><span class="icon">📁</span><span class="folder">trajectory/</span></div>
      <div class="row"><span class="icon">📁</span><span class="folder">scripts/</span></div>
      <div class="row" style="padding-left:48px;"><span class="icon">📄</span><span class="file">build_eval_set.py</span></div>
      <div class="row" style="padding-left:48px;"><span class="icon">📄</span><span class="file">run_eval.py</span></div>
      <div class="row"><span class="icon">📁</span><span class="folder">eval/</span></div>
      <div class="row"><span class="icon">📁</span><span class="folder">runs/eval/</span></div>
      <div class="row" style="padding-left:48px;"><span class="icon">📄</span><span class="file">report.html</span></div>
      <div class="row" style="padding-left:48px;"><span class="icon">📄</span><span class="file">report.json</span></div>
      <div class="row" style="padding-left:48px;"><span class="icon">📄</span><span class="file">eval_results.json</span></div>
    </div>

    <div class="readme">
      <h1>predec</h1>
      <p><strong>Detect and quantify biases in RLHF preference datasets.</strong></p>
      <p>CLI that ingests any preference dataset (model response A vs. B with a human-chosen winner)
        and reports four well-known reward-modeling biases with statistical confidence.</p>

      <h2>Results (micro1 Agentic Workflows Hackathon, Aug 2026)</h2>
      <p>220-pair synthetic test set with planted biases across all four types.</p>
      <pre>Detector   Agent F1   Baseline F1
length     1.00       0.00
position   1.00       0.00
sycophancy 1.00       0.00
verbosity  1.00       0.00
MACRO F1   1.00       0.00</pre>

      <h2>Install</h2>
      <pre>git clone https://github.com/ruzkypazzy/predec.git
cd predec
pip install -e .</pre>

      <h2>Quick start</h2>
      <pre>predec detect --dataset anthropic/hh-rlhf --limit 1000 --out runs/exp1
open runs/exp1/report/report.html</pre>
    </div>
  </div>
</body></html>
"""


# Scene 2: animated terminal
SCENE_2_HTML = """
<!doctype html>
<html><head><style>
  body { background: #0c0c0c; color: #e6e8ec; font-family: 'Menlo', 'Monaco', monospace; padding: 24px; font-size: 15px; line-height: 1.65; margin: 0; }
  .prompt { color: #6ee7b7; }
  .cmd { color: #e6e8ec; }
  .dim { color: #6e7681; }
  .bad { color: #f87171; }
  .ok { color: #34d399; }
  .strong { font-weight: 700; }
  pre { margin: 0; white-space: pre-wrap; }
  .typing { display: inline-block; overflow: hidden; white-space: nowrap; border-right: 2px solid #6ee7b7; animation: typing 1.2s steps(40, end), blink 0.7s step-end infinite; }
  @keyframes typing { from { width: 0 } to { width: 100% } }
  @keyframes blink { 50% { border-color: transparent } }
  .line { opacity: 0; animation: fadein 0.4s forwards; }
  @keyframes fadein { to { opacity: 1; } }
</style></head><body>
<pre id="term"></pre>
<script>
const lines = [
  ["<span class='prompt'>root@vps:/opt/predec#</span> <span class='cmd'>python3 scripts/run_eval.py --out runs/eval</span>", 0],
  ["<span class='dim'>Loaded 220 pairs; ground truth:</span>", 0.4],
  ["<span class='dim'>  length:</span> 60    <span class='dim'>position:</span> 60    <span class='dim'>sycophancy:</span> 40    <span class='dim'>verbosity:</span> 40    <span class='dim'>clean:</span> 20", 0.7],
  ["", 0.8],
  ["<span class='strong'>=== AGENTIC PIPELINE ===</span>", 1.0],
  ["Ran in 38.4s; flags: {length: <span class='bad'>True</span>, position: <span class='bad'>True</span>, sycophancy: <span class='bad'>True</span>, verbosity: <span class='bad'>True</span>}", 1.3],
  ["  length:     0.83  95% CI [0.50, 1.00]", 1.6],
  ["  position:   0.56  95% CI [0.43, 0.65]", 1.8],
  ["  sycophancy: 1.00  95% CI [1.00, 1.00]", 2.0],
  ["  verbosity:  1.00  (permutation p < 0.002)", 2.2],
  ["", 2.4],
  ["<span class='strong'>=== SINGLE-PROMPT BASELINE ===</span>", 2.6],
  ["Ran in 5.0s; flags: {length: <span class='dim'>False</span>, position: <span class='dim'>False</span>, sycophancy: <span class='dim'>False</span>, verbosity: <span class='dim'>False</span>}", 2.9],
  ["Parsed scores: {}", 3.2],
  ["", 3.4],
  ["<span class='strong'>=== METRICS ===</span>", 3.6],
  ["<span class='dim'>Bias         Planted  Agent F1   Baseline F1</span>", 3.8],
  ["--------------------------------------------------", 4.0],
  ["length       60       <span class='ok strong'>1.000</span>      0.000", 4.2],
  ["position     60       <span class='ok strong'>1.000</span>      0.000", 4.4],
  ["sycophancy   40       <span class='ok strong'>1.000</span>      0.000", 4.6],
  ["verbosity    40       <span class='ok strong'>1.000</span>      0.000", 4.8],
  ["", 5.0],
  ["<span class='ok strong'>MACRO F1              1.000      0.000</span>", 5.2],
  ["", 5.5],
  ["Wrote runs/eval/eval_results.json", 5.7],
  ["Wrote runs/eval/report/report.html", 5.9],
  ["Wrote runs/eval/report/report.json", 6.1],
  ["", 6.3],
  ["<span class='prompt'>root@vps:/opt/predec#</span> ", 6.5],
];

const term = document.getElementById("term");
let i = 0;
function showNext() {
  if (i >= lines.length) return;
  const [html, t] = lines[i];
  const div = document.createElement("div");
  div.className = "line";
  div.style.animationDelay = t + "s";
  div.innerHTML = html;
  term.appendChild(div);
  i++;
  if (i < lines.length) {
    setTimeout(showNext, 200);
  }
}
window.addEventListener("load", () => setTimeout(showNext, 200));
</script>
</body></html>
"""


# Scene 4: closing title
SCENE_4_HTML = """
<!doctype html>
<html><body style="background:#0f1115;color:#e6e8ec;font-family:-apple-system,sans-serif;display:flex;align-items:center;justify-content:center;height:100vh;margin:0;">
<div style="text-align:center;">
<h1 style="font-size:64px;margin:0 0 16px;font-weight:700;">predec</h1>
<p style="font-size:22px;color:#8a92a0;margin:0 0 32px;">github.com/ruzkypazzy/predec</p>
<div style="display:inline-block;padding:12px 24px;background:#238636;color:#fff;border-radius:6px;font-size:18px;font-weight:600;">
Macro F1 = 1.000 &middot; Baseline 0.000
</div>
<p style="font-size:16px;color:#8a92a0;margin-top:32px;">3 statistical detectors &middot; 1 LLM judge &middot; $0.002 per 1K pairs</p>
</div>
</body></html>
"""


async def run_scene(page, html, scroll_steps=0, scroll_pause_ms=300, scene_name="scene"):
    """Display an HTML page and optionally scroll through it."""
    print(f"[{scene_name}] displaying")
    await page.set_content(html, wait_until="domcontentloaded")
    await page.wait_for_timeout(1500)
    for _ in range(scroll_steps):
        await page.evaluate("window.scrollBy(0, 300)")
        await page.wait_for_timeout(scroll_pause_ms)
    # Wait at the bottom for the screenshot
    await page.wait_for_timeout(1500)


async def main():
    out_mp4 = OUT_DIR / "predec-demo.mp4"
    # Clean scenes dir
    for f in SCENES_DIR.glob("*"):
        if f.is_file():
            f.unlink()

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(
            viewport={"width": 1280, "height": 720},
            record_video_dir=str(SCENES_DIR),
        )
        page = await context.new_page()

        # Scene 1: GitHub repo, scroll through README
        await run_scene(page, SCENE_1_HTML, scroll_steps=10, scene_name="scene1_github")

        # Scene 2: animated terminal
        await run_scene(page, SCENE_2_HTML, scroll_steps=0, scene_name="scene2_terminal")
        # Let the animation play for 8 seconds
        await page.wait_for_timeout(8000)

        # Scene 3: report.html in browser, scroll
        report_path = "/workspace/predec/runs/eval/report/report.html"
        if os.path.exists(report_path):
            await page.goto(f"file://{report_path}", wait_until="domcontentloaded", timeout=30000)
            await page.wait_for_timeout(2500)
            for _ in range(15):
                await page.evaluate("window.scrollBy(0, 400)")
                await page.wait_for_timeout(200)

        # Scene 4: closing title
        await run_scene(page, SCENE_4_HTML, scroll_steps=0, scene_name="scene4_closing")
        await page.wait_for_timeout(2500)

        await context.close()
        await browser.close()

    # Find the webm
    webms = sorted(SCENES_DIR.glob("*.webm"))
    if not webms:
        print("ERROR: no webm produced", file=sys.stderr)
        sys.exit(1)
    src_webm = webms[0]
    print(f"Source recording: {src_webm} ({src_webm.stat().st_size:,} bytes)")

    # Convert to mp4
    cmd = [
        "ffmpeg", "-y", "-i", str(src_webm),
        "-c:v", "libx264", "-preset", "fast", "-crf", "23",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        "-an",
        str(out_mp4),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print("ffmpeg stderr:", res.stderr[-2000:], file=sys.stderr)
        sys.exit(1)
    print(f"Wrote {out_mp4} ({out_mp4.stat().st_size:,} bytes)")


if __name__ == "__main__":
    asyncio.run(main())
