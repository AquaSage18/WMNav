import logging
from openai import OpenAI
import base64
import numpy as np
from PIL import Image
import io
import os
import re

logging.getLogger("openai").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)


def _clean_env_value(value):
    if value is None:
        return None
    value = value.strip().strip('"').strip("'")
    if not value or value.upper().startswith("YOUR_"):
        return None
    return value


def _first_env(*names, default=None):
    for name in names:
        value = _clean_env_value(os.environ.get(name))
        if value is not None:
            return value
    return default


def _openai_client(base_url, api_key):
    if not base_url:
        raise ValueError("Missing VLM base_url. Set a valid OpenAI-compatible endpoint.")
    if not api_key:
        raise ValueError("Missing VLM api_key. Set a valid key or use EMPTY for local services.")
    return OpenAI(api_key=api_key, base_url=base_url)


def _strip_thinking_content(text):
    if not isinstance(text, str):
        return "" if text is None else str(text)

    cleaned = text.strip()
    if not cleaned:
        return ""

    if "</think>" in cleaned:
        cleaned = cleaned.rsplit("</think>", 1)[-1].strip()

    cleaned = re.sub(r"<think>.*?</think>", "", cleaned, flags=re.IGNORECASE | re.DOTALL).strip()
    return cleaned


def _extract_message_text_value(value):
    if isinstance(value, str):
        return _strip_thinking_content(value)

    if isinstance(value, list):
        text_blocks = []
        for block in value:
            if isinstance(block, dict):
                if block.get("type") == "text":
                    text_blocks.append(block.get("text", ""))
            else:
                block_type = getattr(block, "type", None)
                text = getattr(block, "text", None)
                if text is not None and block_type in {None, "text"}:
                    text_blocks.append(text)
        if text_blocks:
            return _strip_thinking_content("".join(text_blocks))

    return ""


def _message_text(message):
    content = _extract_message_text_value(getattr(message, "content", None))
    if content:
        return content

    return ""


def _message_reasoning(message):
    for field in ("reasoning", "reasoning_content"):
        reasoning = _extract_message_text_value(getattr(message, field, None))
        if reasoning:
            return reasoning
    return ""


def encode_image(image):
    try:
        # 将numpy数组转换回PIL图像
        image = Image.fromarray(image[:, :, :3], mode='RGB')

        # 将图像保存到字节流中
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")

        # 将字节流编码为Base64
        return base64.b64encode(buffered.getvalue()).decode('utf-8')
    except Exception as e:
        raise RuntimeError(f"Failed to convert image to base64: {e}")


class GeminiVLM:
    """
    A specific implementation of a VLM using the Gemini API for image and text inference.
    """

    def __init__(self, model="gemini-2.0-flash", system_instruction=None, base_url=None, api_key=None):
        """
        Initialize the Gemini model with specified configuration.

        Parameters
        ----------
        model : str
            The model version to be used.
        system_instruction : str, optional
            System instructions for model behavior.
        """
        self.name = model
        self.client = _openai_client(
            _clean_env_value(base_url) or _first_env("GEMINI_BASE_URL", "OPENAI_BASE_URL"),
            _clean_env_value(api_key) or _first_env("GEMINI_API_KEY", "OPENAI_API_KEY"),
        )

        self.system_instruction = system_instruction

        self.spend = 0
        if '1.5-flash' in self.name:
            self.cost_per_input_token = 0.075 / 1_000_000
            self.cost_per_output_token = 0.3 / 1_000_000
        elif '1.5-pro' in self.name:
            self.cost_per_input_token = 1.25 / 1_000_000
            self.cost_per_output_token = 5 / 1_000_000
        else:
            self.cost_per_input_token = 0.1 / 1_000_000
            self.cost_per_output_token = 0.4 / 1_000_000

    def call_chat(self, image: list[np.array], text_prompt: str):
        base64_image = encode_image(image[0])
        try:
            response = self.client.chat.completions.create(
                model=self.name,
                messages=[
                    {
                        "role": "system",
                        "content": self.system_instruction
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": text_prompt},
                            {
                                "type": "image_url",
                                "image_url": f"data:image/png;base64,{base64_image}",
                            },
                        ],
                    }
                ],
                max_tokens=500,
                temperature=0,
                top_p=1,
                stream=False  # 是否开启流式输出
            )
            self.spend += (response.usage.prompt_tokens * self.cost_per_input_token +
                           response.usage.completion_tokens * self.cost_per_output_token)
        except Exception as e:
            print(f"{self.__class__.__name__} ERROR: {e}")
            return f"{self.__class__.__name__} ERROR"
        return _message_text(response.choices[0].message)

    def call(self, image: list[np.array], text_prompt: str):
        base64_image = encode_image(image[0])
        try:
            response = self.client.chat.completions.create(
                model=self.name,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": text_prompt},
                            {
                                "type": "image_url",
                                "image_url": f"data:image/png;base64,{base64_image}",
                            },
                        ],
                    }
                ],
                max_tokens=500,
                temperature=0,
                top_p=1,
                stream=False  # 是否开启流式输出
            )
            self.spend += (response.usage.prompt_tokens * self.cost_per_input_token +
                           response.usage.completion_tokens * self.cost_per_output_token)
        except Exception as e:
            print(f"{self.__class__.__name__} ERROR: {e}")
            return f"{self.__class__.__name__} ERROR"
        return _message_text(response.choices[0].message)

    def reset(self):
        """
        Reset the context state of the VLM agent.
        """
        pass

    def get_spend(self):
        """
        Retrieve the total spend on model usage.
        """
        return self.spend

class QwenVLM:
    """
    A specific implementation of a VLM using the Qwen API for image and text inference.
    Supports Qwen3 thinking mode through vLLM's OpenAI-compatible API.
    """

    supports_guided_json = True

    def __init__(self, model="Qwen/Qwen3.6-27B", system_instruction=None, base_url=None, api_key=None,
                 enable_thinking=None, max_tokens=4096, structured_max_tokens=1536,
                 disable_thinking_for_json=True):
        """
        Initialize the Qwen model with specified configuration.

        Parameters
        ----------
        model : str
            The model version to be used.
        system_instruction : str, optional
            System instructions for model behavior.
        enable_thinking : bool, optional
            Force Qwen3 thinking mode on or off. Defaults to auto-detect from the model name.
        """
        self.name = model
        if enable_thinking is None:
            self.enable_thinking = "qwen3" in self.name.lower()
        else:
            self.enable_thinking = enable_thinking
        self.client = _openai_client(
            _clean_env_value(base_url) or _first_env(
                "QWEN_BASE_URL",
                "GEMINI_BASE_URL",
                "OPENAI_BASE_URL",
                default="http://localhost:8000/v1",
            ),
            _clean_env_value(api_key) or _first_env(
                "QWEN_API_KEY",
                "GEMINI_API_KEY",
                "OPENAI_API_KEY",
                default="EMPTY",
            ),
        )

        self.system_instruction = system_instruction
        self.max_tokens = max_tokens
        self.structured_max_tokens = structured_max_tokens
        self.disable_thinking_for_json = disable_thinking_for_json
        self.last_reasoning = ""
        self.last_debug = {}

        self.spend = 0

    @staticmethod
    def _is_max_token_finish_reason(finish_reason):
        if finish_reason is None:
            return False
        return str(finish_reason).lower() in {"length", "max_tokens", "token_limit"}

    def _extra_body(self, response_schema=None):
        body = {}
        enable_thinking = self.enable_thinking
        if response_schema is not None and self.disable_thinking_for_json:
            enable_thinking = False

        body["chat_template_kwargs"] = {"enable_thinking": enable_thinking}

        if response_schema is not None:
            body["guided_json"] = response_schema

        return body or None

    def _create_completion(self, messages, response_schema=None):
        requested_max_tokens = self.structured_max_tokens if response_schema is not None else self.max_tokens
        kwargs = {
            "model": self.name,
            "messages": messages,
            "max_tokens": requested_max_tokens,
            "temperature": 0,
            "top_p": 1,
            "stream": False,
        }
        extra_body = self._extra_body(response_schema=response_schema)
        if extra_body is not None:
            kwargs["extra_body"] = extra_body

        self._last_request_debug = {
            "requested_max_tokens": requested_max_tokens,
            "response_schema": response_schema is not None,
            "enable_thinking": extra_body.get("chat_template_kwargs", {}).get("enable_thinking") if extra_body else None,
        }
        return self.client.chat.completions.create(
            **kwargs,
        )

    def _capture_response_debug(self, response):
        choice = response.choices[0]
        message = choice.message
        content = _message_text(message)
        reasoning = _message_reasoning(message)
        usage = getattr(response, "usage", None)
        finish_reason = getattr(choice, "finish_reason", None)
        self.last_reasoning = reasoning
        self.last_debug = {
            **getattr(self, "_last_request_debug", {}),
            "finish_reason": finish_reason,
            "truncated_by_max_tokens": self._is_max_token_finish_reason(finish_reason),
            "content_len": len(content),
            "reasoning_len": len(reasoning),
            "prompt_tokens": getattr(usage, "prompt_tokens", None),
            "completion_tokens": getattr(usage, "completion_tokens", None),
            "total_tokens": getattr(usage, "total_tokens", None),
            "raw_content_repr": repr(getattr(message, "content", None)),
            "has_reasoning": bool(reasoning),
        }
        logging.info("QwenVLM response debug: %s", self.last_debug)
        return content

    def call_chat(self, image: list[np.array], text_prompt: str, response_schema=None):
        base64_image = encode_image(image[0])
        try:
            messages = [
                {
                    "role": "system",
                    "content": self.system_instruction
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": text_prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_image}"
                            }
                        },
                    ],
                }
            ]
            response = self._create_completion(messages, response_schema=response_schema)
            self.spend += (response.usage.prompt_tokens + response.usage.completion_tokens)

        except Exception as e:
            self.last_reasoning = ""
            self.last_debug = {
                **getattr(self, "_last_request_debug", {}),
                "api_error": str(e),
                "finish_reason": None,
                "truncated_by_max_tokens": False,
            }
            print(f"{self.__class__.__name__} ERROR: {e}")
            return f"{self.__class__.__name__} ERROR"
        return self._capture_response_debug(response)

    def call(self, image: list[np.array], text_prompt: str, response_schema=None):
        base64_image = encode_image(image[0])
        try:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": text_prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_image}"
                            }
                        },
                    ],
                }
            ]
            response = self._create_completion(messages, response_schema=response_schema)
            self.spend += (response.usage.prompt_tokens + response.usage.completion_tokens)
        except Exception as e:
            self.last_reasoning = ""
            self.last_debug = {
                **getattr(self, "_last_request_debug", {}),
                "api_error": str(e),
                "finish_reason": None,
                "truncated_by_max_tokens": False,
            }
            print(f"{self.__class__.__name__} ERROR: {e}")
            return f"{self.__class__.__name__} ERROR"
        return self._capture_response_debug(response)

    def reset(self):
        """
        Reset the context state of the VLM agent.
        """
        self.last_reasoning = ""
        self.last_debug = {}

    def get_spend(self):
        """
        Retrieve the total spend on model usage.
        """
        return self.spend
