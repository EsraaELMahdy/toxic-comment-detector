from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import lru_cache

import torch
from PIL import Image

from config import BLIP_MAX_NEW_TOKENS, BLIP_MODEL

logger = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

FALLBACK_CAPTION = "image content"


@dataclass(frozen=True)
class CaptionResult:
    caption: str
    model_name: str
    device: str
    used_fallback: bool = False


def is_available() -> tuple[bool, str]:
    try:
        __import__("transformers")
    except ImportError:
        return False, "transformers is not installed."

    return True, f"Ready · {BLIP_MODEL} on {DEVICE.type}"


@lru_cache(maxsize=1)
def load_blip():
    from transformers import BlipForConditionalGeneration, BlipProcessor

    processor = BlipProcessor.from_pretrained(BLIP_MODEL)
    model = BlipForConditionalGeneration.from_pretrained(BLIP_MODEL)
    model.to(DEVICE)
    model.eval()

    logger.info("BLIP loaded on %s", DEVICE)
    return processor, model


def generate_caption(
    image: Image.Image,
    processor=None,
    model=None,
    max_new_tokens: int = BLIP_MAX_NEW_TOKENS,
) -> CaptionResult:
    if processor is None or model is None:
        try:
            processor, model = load_blip()
        except Exception as error:
            logger.warning("BLIP unavailable (%s).", error)
            return CaptionResult(
                caption=FALLBACK_CAPTION,
                model_name=BLIP_MODEL,
                device=DEVICE.type,
                used_fallback=True,
            )

    inputs = processor(images=image, return_tensors="pt")
    inputs = {key: value.to(DEVICE) for key, value in inputs.items()}

    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=max_new_tokens)

    caption = processor.decode(output[0], skip_special_tokens=True).strip()

    return CaptionResult(
        caption=caption or FALLBACK_CAPTION,
        model_name=BLIP_MODEL,
        device=DEVICE.type,
    )
