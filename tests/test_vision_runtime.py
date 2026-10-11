"""Universal bounded image input and adapter contract; no model inference."""
import base64
import io
import importlib.util
import unittest
import sys
import types
from unittest.mock import patch
from mlx_flash_compress.openai_compat import APIRequestError, api_capabilities, validate_chat_request


def image_url(color="red", size=(2, 2), fmt="PNG"):
    from PIL import Image
    output = io.BytesIO()
    Image.new("RGB", size, color).save(output, format=fmt)
    return "data:image/" + ("jpeg" if fmt == "JPEG" else "png") + ";base64," + base64.b64encode(output.getvalue()).decode()


def messages(url=None):
    return [{"role": "user", "content": [{"type": "text", "text": "Identify the color"}, {"type": "image_url", "image_url": {"url": url or image_url()}}]}]


@unittest.skipUnless(importlib.util.find_spec("PIL"), "optional vision dependency Pillow unavailable")
class VisionInputTests(unittest.TestCase):
    def test_enabled_vision_accepts_data_url_and_decodes_original_pixels(self):
        validate_chat_request({"messages": messages()}, vision=True)
        from mlx_flash_compress.vision_runtime import prepare_vision_messages
        text, images = prepare_vision_messages(messages())
        self.assertEqual(text, [{"role": "user", "content": [{"type": "text", "text": "Identify the color"}, {"type": "image"}]}])
        self.assertEqual(images[0].getpixel((0, 0)), (255, 0, 0))

    def test_text_runtime_remains_closed_to_image_content(self):
        with self.assertRaises(APIRequestError) as error:
            validate_chat_request({"messages": messages()})
        self.assertEqual(error.exception.code, "unsupported_modality")

    def test_remote_file_malformed_and_mismatched_images_are_rejected(self):
        from mlx_flash_compress.vision_runtime import prepare_vision_messages
        for url in ["https://example.test/x.jpg", "file:///tmp/x.jpg", "data:image/png;base64,!!!", "data:image/png;base64,YWJj", image_url().replace("image/png", "image/jpeg")]:
            with self.subTest(url=url[:40]), self.assertRaises(APIRequestError):
                prepare_vision_messages(messages(url))

    def test_extra_images_and_large_decoded_pixels_are_rejected(self):
        from mlx_flash_compress.vision_runtime import prepare_vision_messages
        multi = messages()
        multi[0]["content"].append(multi[0]["content"][1])
        with self.assertRaises(APIRequestError):
            prepare_vision_messages(multi)
        with patch("mlx_flash_compress.vision_runtime.MAX_IMAGE_PIXELS", 3), self.assertRaises(APIRequestError):
            prepare_vision_messages(messages())

    def test_untrusted_content_shapes_and_control_tokens_rejected(self):
        from mlx_flash_compress.vision_runtime import prepare_vision_messages, validate_vision_output
        for content in [[{"type": "audio"}], [{"type": "text", "text": 4}], [{"type": "image_url", "image_url": {"url": image_url(), "detail": "nonsense"}}]]:
            with self.subTest(content=content), self.assertRaises(APIRequestError):
                prepare_vision_messages([{"role": "user", "content": content}])
        for output in ["", "<|channel>analysis", "<think>hidden</think>blue", "blue<start_of_turn>", None]:
            with self.subTest(output=output), self.assertRaises(APIRequestError):
                validate_vision_output(output)
        validate_vision_output("The image is red.")

    def test_capabilities_require_loaded_vision_runtime(self):
        self.assertFalse(api_capabilities()["vision"]["supported"])
        cap = api_capabilities(vision=True)
        self.assertTrue(cap["vision"]["supported"])
        self.assertFalse(cap["image_operations"]["supported"])
        self.assertFalse(cap["endpoints"]["chat_completions"]["streaming"])

    def test_generation_releases_owned_model_on_success_and_native_failure(self):
        from mlx_flash_compress.vision_runtime import VisionRuntime
        result = types.SimpleNamespace(text="red", generation_tokens=1, prompt_tokens=80,
                                       finish_reason="stop", generation_tps=10, prompt_tps=100, peak_memory=5)
        model = types.SimpleNamespace(config={"model_type": "gemma4", "vision_config": {"layers": 1}})
        processor = types.SimpleNamespace(image_processor=object())
        def native_generate(model, processor, *, image, prompt, **kwargs):
            self.assertEqual(image[0].getpixel((0, 0)), (255, 0, 0))
            return result
        vlm = types.SimpleNamespace(load=lambda name: (model, processor), generate=native_generate)
        native = types.SimpleNamespace(synchronize=lambda: None, clear_cache=lambda: None)
        prompt = types.SimpleNamespace(apply_chat_template=lambda *args, **kwargs: "formatted")
        with patch.dict(sys.modules, {"mlx_vlm": vlm, "mlx_vlm.prompt_utils": prompt,
                                      "mlx": types.SimpleNamespace(core=native), "mlx.core": native}):
            runtime = VisionRuntime("test/vision")
            output = runtime.generate(messages(), 16, 0)
            self.assertEqual(output["output"], "red")
            self.assertEqual(output["prompt_tokens"], 80)
            self.assertTrue(runtime.ready)
            self.assertIsNone(runtime.model)
            self.assertIsNone(runtime.processor)
            def broken(*args, **kwargs):
                raise RuntimeError("native failed")
            vlm.generate = broken
            with self.assertRaisesRegex(RuntimeError, "native failed"):
                runtime.generate(messages(), 16, 0)
            self.assertIsNone(runtime.model)
            self.assertIsNone(runtime.processor)

    def test_text_checkpoint_cannot_advertise_vision_readiness(self):
        from mlx_flash_compress.vision_runtime import VisionRuntime
        native = types.SimpleNamespace(synchronize=lambda: None, clear_cache=lambda: None)
        vlm = types.SimpleNamespace(load=lambda name: (types.SimpleNamespace(config={"model_type": "qwen3"}), object()))
        with patch.dict(sys.modules, {"mlx_vlm": vlm, "mlx": types.SimpleNamespace(core=native), "mlx.core": native}):
            runtime = VisionRuntime("test/text")
            with self.assertRaisesRegex(RuntimeError, "no verified vision"):
                runtime.load()
            self.assertFalse(runtime.ready)
            self.assertIsNone(runtime.model)
