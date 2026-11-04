"""Page check of live.html in Chromium and WebKit, on the Mac, against a running server (manual 6).

NO_OPEN=1 ./live_spark2.sh --model etam-s &   # server on spark2, tunnel to localhost:8765, no other viewer
uv run --no-project --with playwright python tests/page_check.py [chromium] [webkit]
(once: uv run --no-project --with playwright playwright install chromium webkit)

The camera is a canvas stream (a red and a blue block), so both engines see the same scene. Not named test_*:
unittest on the hosts must not collect it (they have no browser).
"""

import sys
import time

from playwright.sync_api import sync_playwright

URL = "http://localhost:8765/"
FAKE = """navigator.mediaDevices.getUserMedia = async () => {
  const c = Object.assign(document.createElement('canvas'), { width: 640, height: 360 }), g = c.getContext('2d');
  let t = 0; setInterval(() => { g.fillStyle = '#ddd'; g.fillRect(0, 0, 640, 360); g.fillStyle = '#c33'; g.fillRect(120, 120, 90, 140);
    g.fillStyle = '#36c'; g.fillRect(380, 100 + 10 * Math.cos(t++ / 8), 110, 160); }, 33);
  return c.captureStream(30); };"""
CHIPS = "cs => cs.map(c => [c.firstChild.textContent, c.querySelector('.sw').style.background])"


def check(p, engine):
    b = getattr(p, engine).launch()
    pg = b.new_context(viewport={"width": 1200, "height": 820}).new_page()
    pg.add_init_script(FAKE)
    errors = []
    pg.on("pageerror", lambda e: errors.append(str(e)))
    chips = lambda: pg.eval_on_selector_all("#objs .chip", CHIPS)  # noqa: E731
    settle = lambda: time.sleep(1.5)  # noqa: E731  a few frames round trip
    pg.goto(URL)
    pg.wait_for_function("!document.getElementById('models').disabled", timeout=300000)
    bb = pg.locator("#overlay").bounding_box()
    assert bb
    at = lambda x, y: (bb["x"] + bb["width"] * x, bb["y"] + bb["height"] * y)  # noqa: E731

    pg.mouse.click(*at(0.26, 0.53))  # red block: new track
    settle()
    assert [c[0][:1] for c in chips()] == ["1"], chips()
    mod = "Meta" if engine == "webkit" else "Control"
    pg.keyboard.down(mod)
    pg.mouse.click(*at(0.68, 0.5))  # blue block: Cmd/Ctrl+click adds object 2
    pg.keyboard.up(mod)
    settle()
    two = chips()
    assert [c[0][:1] for c in two] == ["1", "2"] and two[0][1] != two[1][1], two
    pg.click("#objs .chip >> nth=0 >> .x")  # × on chip 1
    settle()
    one = chips()
    assert len(one) == 1 and one[0][0][:1] == "2" and one[0][1] == two[1][1], one  # id and colour stay
    pg.keyboard.press("Delete")  # removes the picked object
    settle()
    assert chips() == [], chips()

    pg.fill("#query", "   ")
    pg.press("#query", "Enter")
    assert "names" in pg.inner_text("#notice"), pg.inner_text("#notice")  # help text, nothing sent
    pg.fill("#query", "red square")
    pg.press("#query", "Enter")
    pg.wait_for_function("!/^Finding/.test(document.getElementById('notice').textContent)", timeout=300000)
    settle()
    notice = pg.inner_text("#notice")
    assert notice.startswith(("Found", "Nothing found")), notice
    if notice.startswith("Found"):
        assert all("red square" in c[0] for c in chips()), chips()  # chips carry the detector label

    pg.keyboard.press("Escape")  # leaves the query field
    pg.keyboard.press("Escape")  # clears the track
    settle()
    assert chips() == [], chips()
    pg.set_viewport_size({"width": 390, "height": 844})
    time.sleep(0.3)
    assert not pg.evaluate("document.documentElement.scrollWidth > innerWidth"), "horizontal scroll at phone width"
    assert not errors, errors
    b.close()
    print(f"{engine}: ok ({notice})")


if __name__ == "__main__":
    with sync_playwright() as p:
        for engine in sys.argv[1:] or ["chromium", "webkit"]:
            check(p, engine)
