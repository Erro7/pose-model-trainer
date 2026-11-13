from __future__ import annotations

from dataclasses import dataclass

STYLE_PRESERVE = "沿用图库风格（推荐）"
STYLE_PHOTOREAL = "写实摄影"
STYLE_EDITORIAL = "时尚摄影"
STYLE_LIFESTYLE = "自然生活"
STYLE_STUDIO = "商业棚拍"
STYLE_OPTIONS = (
    STYLE_PRESERVE,
    STYLE_PHOTOREAL,
    STYLE_EDITORIAL,
    STYLE_LIFESTYLE,
    STYLE_STUDIO,
)

BODY_NATURAL = "自然比例（推荐）"
BODY_REFERENCE = "沿用人物参考比例"
BODY_SLIM = "纤细"
BODY_BALANCED = "匀称"
BODY_ATHLETIC = "运动型"
BODY_CURVY = "曲线型"
BODY_OPTIONS = (
    BODY_NATURAL,
    BODY_REFERENCE,
    BODY_SLIM,
    BODY_BALANCED,
    BODY_ATHLETIC,
    BODY_CURVY,
)

GALLERY_STRATEGY_FIXED_PERSON = "每批固定一人（推荐）"
GALLERY_STRATEGY_RANDOM_PER_ITEM = "每条独立随机"
GALLERY_STRATEGY_BALANCED = "多人均分随机"
GALLERY_STRATEGY_MULTI_REF = "同一人物多图参考"
GALLERY_STRATEGY_OPTIONS = (
    GALLERY_STRATEGY_FIXED_PERSON,
    GALLERY_STRATEGY_RANDOM_PER_ITEM,
    GALLERY_STRATEGY_BALANCED,
    GALLERY_STRATEGY_MULTI_REF,
)

GENERATION_BATCH_PRESETS = {
    "快速验证 · 4 张": 4,
    "标准训练 · 8 张（推荐）": 8,
    "大批量 · 16 张": 16,
    "自定义（使用高级设置）": None,
}
DEFAULT_GENERATION_BATCH_PRESET = "标准训练 · 8 张（推荐）"
CUSTOM_GENERATION_SIZE = "自定义尺寸（高级）"
DEFAULT_GENERATION_SEED = -1
DEFAULT_GENERATION_JOB_NAME = "image-auto-targets"


@dataclass(frozen=True)
class ImageModelSpec:
    key: str
    label: str
    backend: str
    available: bool
    summary: str


IMAGE_MODEL_SPECS = (
    ImageModelSpec(
        key="h3-image",
        label="MiniMax H3 图片 LoRA",
        backend="h3",
        available=True,
        summary="批量生成、审核建集和 H3 LoRA 训练已接入。",
    ),
    ImageModelSpec(
        key="sdxl-image",
        label="SDXL 图片 LoRA",
        backend="sdxl",
        available=False,
        summary="通用训练页已提供 SDXL LoRA；本页批量图片生成尚未接入。",
    ),
    ImageModelSpec(
        key="flux-image",
        label="FLUX.1 图片 LoRA",
        backend="flux",
        available=False,
        summary="通用训练页已提供 FLUX.1 LoRA；本页批量图片生成尚未接入。",
    ),
)
IMAGE_MODEL_LABELS = tuple(spec.label for spec in IMAGE_MODEL_SPECS)
IMAGE_MODEL_CHOICES = tuple(
    (f"{spec.label} · {'已接入' if spec.available else '待接入'}", spec.label)
    for spec in IMAGE_MODEL_SPECS
)
DEFAULT_IMAGE_MODEL = IMAGE_MODEL_SPECS[0].label


def image_model_spec(label: str) -> ImageModelSpec:
    text = str(label or "").strip()
    return next((spec for spec in IMAGE_MODEL_SPECS if spec.label == text), IMAGE_MODEL_SPECS[0])

STYLE_GUIDANCE = {
    STYLE_PRESERVE: "",
    STYLE_PHOTOREAL: (
        "Use a photorealistic photographic style with natural skin texture, realistic fabric, "
        "credible lighting and restrained color grading."
    ),
    STYLE_EDITORIAL: (
        "Use a polished fashion-editorial style with controlled studio lighting, refined styling "
        "and a clean contemporary composition."
    ),
    STYLE_LIFESTYLE: (
        "Use a natural lifestyle-photography style with soft available light, relaxed body language "
        "and an unforced everyday atmosphere."
    ),
    STYLE_STUDIO: (
        "Use a clean commercial studio style with balanced softbox lighting, accurate garment detail "
        "and a neutral seamless background."
    ),
}

BODY_GUIDANCE = {
    BODY_NATURAL: (
        "Depict a naturally proportioned adult build with realistic anatomy, independent of the "
        "person and pose references."
    ),
    BODY_REFERENCE: "",
    BODY_SLIM: "Depict a naturally slender adult build with realistic proportions and anatomy.",
    BODY_BALANCED: "Depict a naturally balanced adult build with realistic proportions and anatomy.",
    BODY_ATHLETIC: "Depict a toned athletic adult build with realistic proportions and anatomy.",
    BODY_CURVY: "Depict a naturally curvy adult build with realistic proportions and anatomy.",
}


@dataclass(frozen=True)
class CreativeProfile:
    style: str = STYLE_PRESERVE
    body: str = BODY_NATURAL
    gallery_strategy: str = GALLERY_STRATEGY_FIXED_PERSON

    def prompt_fragments(self) -> tuple[str, str]:
        return style_prompt(self.style), body_prompt(self.body)


def normalize_choice(value: str, options: tuple[str, ...], default: str) -> str:
    text = str(value or "").strip()
    return text if text in options else default


def normalize_creative_profile(
    style: str,
    body: str,
    gallery_strategy: str,
) -> CreativeProfile:
    return CreativeProfile(
        style=normalize_choice(style, STYLE_OPTIONS, STYLE_PRESERVE),
        body=normalize_choice(body, BODY_OPTIONS, BODY_NATURAL),
        gallery_strategy=normalize_choice(
            gallery_strategy,
            GALLERY_STRATEGY_OPTIONS,
            GALLERY_STRATEGY_FIXED_PERSON,
        ),
    )


def style_prompt(value: str) -> str:
    return STYLE_GUIDANCE.get(normalize_choice(value, STYLE_OPTIONS, STYLE_PRESERVE), "")


def body_prompt(value: str) -> str:
    return BODY_GUIDANCE.get(normalize_choice(value, BODY_OPTIONS, BODY_NATURAL), "")


def uses_reference_body(value: str) -> bool:
    return normalize_choice(value, BODY_OPTIONS, BODY_NATURAL) == BODY_REFERENCE

