"""Capture a screenshot of the predec HTML report using Playwright."""
import asyncio
import sys
from pathlib import Path

from playwright.async_api import async_playwright


async def main():
    html_path = Path("/workspace/predec/runs/eval/report/report.html").resolve()
    if not html_path.exists():
        print(f"ERROR: {html_path} does not exist", file=sys.stderr)
        sys.exit(1)

    out_path = Path("/workspace/predec/runs/eval/report/screenshot.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page(viewport={"width": 1280, "height": 1800})
        await page.goto(f"file://{html_path}")
        await page.wait_for_timeout(2000)
        await page.screenshot(path=str(out_path), full_page=True)
        await browser.close()
    print(f"Saved {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
