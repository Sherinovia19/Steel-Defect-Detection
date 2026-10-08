"""
Qwen2.5-VL visual-language explanation module.

Qwen2.5-VL is used as an auxiliary visual-language component.
It provides a natural-language description of the visible steel
surface defect.

The YOLO26m + MSC detector remains the primary defect detector.
Qwen output is not used to calculate Precision, Recall, mAP50,
or mAP50-95.
"""

import argparse

import torch
from qwen_vl_utils import process_vision_info
from transformers import (
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
)


MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"

CLASS_NAMES = [
    "Pitted Surface",
    "Crazing",
    "Scratch",
    "Patch",
    "No confident defect",
]


def load_model():
    """Load Qwen2.5-VL-3B-Instruct."""

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype="auto",
        device_map="auto",
    )

    processor = AutoProcessor.from_pretrained(
        MODEL_ID
    )

    return model, processor


def explain_defect(
    model,
    processor,
    image_path,
    yolo_summary=None,
):
    """
    Ask Qwen to provide a short visual explanation.

    Args:
        model: Loaded Qwen2.5-VL model.
        processor: Qwen processor.
        image_path: Path to the steel image.
        yolo_summary: Optional YOLO detection summary.

    Returns:
        Qwen-generated text explanation.
    """

    prompt = """
Identify the visible steel surface defect.

Choose exactly one:

Pitted Surface
Crazing
Scratch
Patch
No confident defect

Give only the class name and one short visual reason.
"""

    if yolo_summary:
        prompt += (
            "\n\nThe YOLO detector produced the following "
            "auxiliary information:\n"
            f"{yolo_summary}\n"
            "\nUse this information as context, but make the "
            "visual explanation based on the image."
        )

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image_path,
                },
                {
                    "type": "text",
                    "text": prompt,
                },
            ],
        }
    ]

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    image_inputs, video_inputs = process_vision_info(
        messages
    )

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )

    inputs = inputs.to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=80,
        )

    generated_ids_trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(
            inputs.input_ids,
            generated_ids,
        )
    ]

    output = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )

    return output[0].strip()


def main():
    parser = argparse.ArgumentParser(
        description="Qwen2.5-VL steel defect explanation"
    )

    parser.add_argument(
        "--image",
        required=True,
        help="Path to the steel surface image",
    )

    args = parser.parse_args()

    model, processor = load_model()

    result = explain_defect(
        model=model,
        processor=processor,
        image_path=args.image,
    )

    print("\nQwen2.5-VL Explanation:")
    print(result)


if __name__ == "__main__":
    main()
