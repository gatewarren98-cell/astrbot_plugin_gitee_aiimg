import importlib.util
import sys
import types
import unittest
from base64 import b64decode
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PACKAGE_NAME = "openai_responses_testpkg"
CORE_PACKAGE_NAME = f"{PACKAGE_NAME}.core"
MODULE_NAME = f"{CORE_PACKAGE_NAME}.openai_responses_image_backend"

TINY_PNG_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8"
    "/x8AAwMCAO+X2ioAAAAASUVORK5CYII="
)


class _Logger:
    def debug(self, *args, **kwargs):
        return None

    def info(self, *args, **kwargs):
        return None

    def warning(self, *args, **kwargs):
        return None

    def error(self, *args, **kwargs):
        return None


class _DummyImageManager:
    def __init__(self):
        self.saved_inputs: list[bytes] = []
        self.downloaded_urls: list[str] = []

    async def save_image(self, data: bytes):
        self.saved_inputs.append(data)
        return Path(f"/tmp/result_{len(self.saved_inputs)}.png")

    async def download_image(self, url: str):
        self.downloaded_urls.append(url)
        return Path("/tmp/downloaded.png")


def _clear_modules():
    for name in list(sys.modules):
        if name.startswith(PACKAGE_NAME) or name in {"astrbot", "astrbot.api"}:
            sys.modules.pop(name, None)


def _load_core_module(module_name: str):
    spec = importlib.util.spec_from_file_location(
        f"{CORE_PACKAGE_NAME}.{module_name}",
        ROOT / "core" / f"{module_name}.py",
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[f"{CORE_PACKAGE_NAME}.{module_name}"] = module
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def _load_module():
    _clear_modules()

    pkg = types.ModuleType(PACKAGE_NAME)
    pkg.__path__ = [str(ROOT)]
    sys.modules[PACKAGE_NAME] = pkg

    core_pkg = types.ModuleType(CORE_PACKAGE_NAME)
    core_pkg.__path__ = [str(ROOT / "core")]
    sys.modules[CORE_PACKAGE_NAME] = core_pkg

    astrbot_mod = types.ModuleType("astrbot")
    sys.modules["astrbot"] = astrbot_mod

    api_mod = types.ModuleType("astrbot.api")
    api_mod.logger = _Logger()
    sys.modules["astrbot.api"] = api_mod

    _load_core_module("gitee_sizes")
    _load_core_module("image_format")

    spec = importlib.util.spec_from_file_location(
        MODULE_NAME,
        ROOT / "core" / "openai_responses_image_backend.py",
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[MODULE_NAME] = module
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


class OpenAIResponsesImageBackendTests(unittest.IsolatedAsyncioTestCase):
    async def test_save_response_image_reads_image_generation_call_result(self):
        mod = _load_module()
        imgr = _DummyImageManager()
        backend = mod.OpenAIResponsesImageBackend(
            imgr=imgr,
            base_url="https://anyrouter.top/v1",
            api_keys=["test-key"],
        )

        out_path = await backend._save_response_image(
            {
                "output": [
                    {
                        "type": "image_generation_call",
                        "result": TINY_PNG_B64,
                    }
                ]
            }
        )

        self.assertEqual(out_path, Path("/tmp/result_1.png"))
        self.assertEqual(imgr.saved_inputs, [b64decode(TINY_PNG_B64)])

    async def test_save_response_image_downloads_url_candidate(self):
        mod = _load_module()
        imgr = _DummyImageManager()
        backend = mod.OpenAIResponsesImageBackend(
            imgr=imgr,
            base_url="https://anyrouter.top/v1/responses",
            api_keys=["test-key"],
        )

        out_path = await backend._save_response_image(
            {
                "output": [
                    {
                        "type": "image_generation_call",
                        "result": {"url": "https://cdn.example.com/out.png"},
                    }
                ]
            }
        )

        self.assertEqual(out_path, Path("/tmp/downloaded.png"))
        self.assertEqual(imgr.downloaded_urls, ["https://cdn.example.com/out.png"])

    async def test_save_response_image_extracts_markdown_url_from_text(self):
        mod = _load_module()
        imgr = _DummyImageManager()
        backend = mod.OpenAIResponsesImageBackend(
            imgr=imgr,
            base_url="https://anyrouter.top/v1",
            api_keys=["test-key"],
        )

        out_path = await backend._save_response_image(
            {
                "output": [
                    {
                        "type": "message",
                        "content": [
                            {
                                "type": "output_text",
                                "text": "done: ![image](https://cdn.example.com/final.png)",
                            }
                        ],
                    }
                ]
            }
        )

        self.assertEqual(out_path, Path("/tmp/downloaded.png"))
        self.assertEqual(imgr.downloaded_urls, ["https://cdn.example.com/final.png"])


class OpenAIResponsesPayloadTests(unittest.TestCase):
    def test_normalizes_responses_url(self):
        mod = _load_module()

        self.assertEqual(
            mod.normalize_responses_base_url("https://anyrouter.top/v1/responses"),
            "https://anyrouter.top/v1",
        )
        self.assertEqual(
            mod.normalize_responses_base_url("https://anyrouter.top"),
            "https://anyrouter.top/v1",
        )

    def test_build_generate_payload_uses_image_generation_tool(self):
        mod = _load_module()
        backend = mod.OpenAIResponsesImageBackend(
            imgr=_DummyImageManager(),
            base_url="https://anyrouter.top/v1",
            api_keys=["test-key"],
        )

        payload = backend._build_payload(
            prompt="画一只猫",
            images=None,
            model=None,
            size="2048x1152",
            resolution=None,
            extra_body=None,
        )

        self.assertEqual(payload["model"], "gpt-5.3-codex")
        self.assertEqual(payload["tools"], [{"type": "image_generation", "size": "1536x1024"}])
        self.assertEqual(payload["tool_choice"], {"type": "image_generation"})
        self.assertEqual(payload["input"][0]["content"][0]["type"], "input_text")

    def test_build_edit_payload_embeds_input_images(self):
        mod = _load_module()
        backend = mod.OpenAIResponsesImageBackend(
            imgr=_DummyImageManager(),
            base_url="https://anyrouter.top/v1",
            api_keys=["test-key"],
        )

        payload = backend._build_payload(
            prompt="改成水彩风",
            images=[b64decode(TINY_PNG_B64)],
            model="gpt-5.3-codex",
            size=None,
            resolution=None,
            extra_body=None,
        )

        content = payload["input"][0]["content"]
        self.assertEqual(content[0], {"type": "input_text", "text": "改成水彩风"})
        self.assertEqual(content[1]["type"], "input_image")
        self.assertTrue(content[1]["image_url"].startswith("data:image/png;base64,"))


if __name__ == "__main__":
    unittest.main()
