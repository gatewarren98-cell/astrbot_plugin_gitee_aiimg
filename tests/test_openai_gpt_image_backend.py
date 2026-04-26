import base64
import sys
import types
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PNG_1X1 = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/p9sAAAAASUVORK5CYII="
)
JPEG_BYTES = b"\xff\xd8\xff\xe0" + (b"\x00" * 16)


class _Logger:
    def debug(self, *args, **kwargs):
        return None

    def info(self, *args, **kwargs):
        return None

    def warning(self, *args, **kwargs):
        return None

    def error(self, *args, **kwargs):
        return None


class _FakeResponse:
    status_code = 200
    headers = {"content-type": "application/json"}
    text = ""
    content = b""

    def json(self):
        return {"data": [{"b64_json": base64.b64encode(PNG_1X1).decode("ascii")}]}


class _FakeClient:
    def __init__(self):
        self.requests = []

    async def post(self, url, **kwargs):
        self.requests.append({"url": url, "kwargs": kwargs})
        return _FakeResponse()

    async def aclose(self):
        return None


class _FakeImageManager:
    def __init__(self):
        self.saved = []

    async def save_image(self, data: bytes):
        self.saved.append(data)
        return Path("/tmp/out.png")

    async def download_image(self, url: str):
        return Path("/tmp/out.png")


def _load_backend_class():
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    astrbot_mod = types.ModuleType("astrbot")
    api_mod = types.ModuleType("astrbot.api")
    api_mod.logger = _Logger()
    sys.modules["astrbot"] = astrbot_mod
    sys.modules["astrbot.api"] = api_mod

    from core.openai_gpt_image_backend import OpenAIGPTImageBackend

    return OpenAIGPTImageBackend


class OpenAIGPTImageBackendTests(unittest.IsolatedAsyncioTestCase):
    async def test_generate_posts_json_to_generations_endpoint(self):
        backend_cls = _load_backend_class()
        imgr = _FakeImageManager()
        client = _FakeClient()
        backend = backend_cls(
            imgr=imgr,
            base_url="https://api.openai.com",
            api_keys=["sk-test"],
            quality="high",
            output_format="webp",
            output_compression=70,
            moderation="low",
        )
        backend._client = client

        out = await backend.generate("a small test", size="1024x1024")

        self.assertEqual(out, Path("/tmp/out.png"))
        self.assertEqual(len(client.requests), 1)
        req = client.requests[0]
        self.assertEqual(req["url"], "https://api.openai.com/v1/images/generations")
        body = req["kwargs"]["json"]
        self.assertEqual(body["model"], "gpt-image-2")
        self.assertEqual(body["prompt"], "a small test")
        self.assertEqual(body["size"], "1024x1024")
        self.assertEqual(body["quality"], "high")
        self.assertEqual(body["output_format"], "webp")
        self.assertEqual(body["output_compression"], 70)
        self.assertEqual(body["moderation"], "low")
        self.assertEqual(imgr.saved, [PNG_1X1])

    async def test_edit_posts_each_reference_as_image_array_field(self):
        backend_cls = _load_backend_class()
        client = _FakeClient()
        backend = backend_cls(
            imgr=_FakeImageManager(),
            base_url="https://api.openai.com/v1",
            api_keys=["sk-test"],
        )
        backend._client = client

        await backend.edit("keep identity", [PNG_1X1, JPEG_BYTES], size="auto")

        self.assertEqual(len(client.requests), 1)
        req = client.requests[0]
        self.assertEqual(req["url"], "https://api.openai.com/v1/images/edits")
        self.assertEqual(req["kwargs"]["data"]["prompt"], "keep identity")
        self.assertEqual(req["kwargs"]["data"]["size"], "auto")
        files = req["kwargs"]["files"]
        self.assertEqual([name for name, _file in files], ["image[]", "image[]"])
        self.assertEqual(files[0][1][0], "input-1.png")
        self.assertEqual(files[0][1][2], "image/png")
        self.assertEqual(files[1][1][0], "input-2.jpg")
        self.assertEqual(files[1][1][2], "image/jpeg")


if __name__ == "__main__":
    unittest.main()
