import torch
from typing import Dict, List, Any
from transformers import AutoProcessor, BitsAndBytesConfig, Idefics2ForConditionalGeneration, logging
from io import BytesIO
import base64
from PIL import Image

logging.set_verbosity_error()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class EndpointHandler():
    def __init__(self, path=""):
        peft_model_id = "rafly/idefics2-8b-ticket-ocr-OTEWE"
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16
        )
        
        self.processor = AutoProcessor.from_pretrained("HuggingFaceM4/idefics2-8b")
        
        model = Idefics2ForConditionalGeneration.from_pretrained(
            peft_model_id,
            torch_dtype=torch.float16,
            quantization_config=quantization_config,
        )
        
        self.model = model

    def __call__(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        data args:
            inputs (:obj: str) or file
        Return:
            A :obj:list | dict: will be serialized and returned
        """

        # get inputs
        image_base64 = data.pop("inputs", None)
        image_file = data.pop("file", None)
        max_token = data.pop("max_token", 200)

        if image_base64:
            # Decode the base64 image
            image_data = base64.b64decode(image_base64)
            image = Image.open(BytesIO(image_data))
        elif image_file:
            # Open the image file
            image = Image.open(image_file)
        else:
            raise ValueError("No valid image input provided")

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "give me the detail of this image"}
                ]
            }
        ]

        print(image)
        text = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.processor(text=[text.strip()], images=[image], return_tensors="pt", padding=True)
        generated_ids = self.model.generate(**inputs, max_new_tokens=max_token)
        
        generated_texts = self.processor.batch_decode(generated_ids[:, inputs["input_ids"].size(1):], skip_special_tokens=True)
        return generated_texts[0]
