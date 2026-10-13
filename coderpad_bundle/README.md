# How to run in Coderpad

## Classic single-file Python pad (editor + Run button, no terminal):

1. Open main.py, select all, copy.
2. Paste into the CoderPad editor, replacing whatever's there.
3. Hit Run — CoderPad runs it the same way python main.py does locally, which fires the embedded pytest suite (pytest.main([__file__, "-v"])) and prints results in the output pane.
4. You won't be able to actually serve it this way (no terminal, no port exposure) — this mode only gets you the tests.


## Sandbox-style pad (full container, file tree, terminal, port preview — common for full-stack/backend interviews):

1. Create main.py and fake_llm.py as separate files in the file tree, paste each in from your local copies.
2. Open the terminal tab, check what's preinstalled: python -c "import fastapi, httpx, pydantic_settings, uvicorn".
3. If anything's missing and the sandbox allows pip install, install it; if it doesn't allow installs, you're limited to whatever's baked into the image.
4. Run python fake_llm.py & then python -m uvicorn main:app --port 8000, and use CoderPad's port-forwarding/preview (if offered) or curl from the same terminal to hit it live.

### curl commands (run from the sandbox terminal, same host as uvicorn)

Health check:

    curl http://localhost:8000/healthz

Classify — clinical urgent:

    curl -X POST http://localhost:8000/classify \
      -H "x-api-key: dev-local-key" -H "Content-Type: application/json" \
      -d '{"message_id":"m1","patient_id":"p1","text":"I have chest pain"}'

Classify — billing:

    curl -X POST http://localhost:8000/classify \
      -H "x-api-key: dev-local-key" -H "Content-Type: application/json" \
      -d '{"message_id":"m2","patient_id":"p1","text":"question about my bill"}'

Classify — low confidence (no keyword match, falls to "general" at 0.5):

    curl -X POST http://localhost:8000/classify \
      -H "x-api-key: dev-local-key" -H "Content-Type: application/json" \
      -d '{"message_id":"m3","patient_id":"p1","text":"nothing in particular"}'

Metrics:

    curl http://localhost:8000/metrics

If CoderPad exposes a forwarded/preview URL instead of localhost, swap `http://localhost:8000` for that URL — the paths and headers stay the same.

# Make sure dependencies are installed

Check what's already in the sandbox image before assuming you need to install anything:

    python -c "import fastapi, httpx, pydantic, pydantic_settings, uvicorn, pytest; print('all present')"

If that fails and the sandbox allows installs:

    pip install fastapi httpx "pydantic>=2" pydantic-settings uvicorn pytest pytest-asyncio

If `pip install` is blocked in that pad, you're limited to whatever's baked into the image. 😅