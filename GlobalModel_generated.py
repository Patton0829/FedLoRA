import json
import os

os.environ["NO_PROXY"] = "127.0.0.1,localhost"
os.environ["no_proxy"] = "127.0.0.1,localhost"

import fire
import gradio as gr
import torch
import transformers

from peft import LoraConfig, PeftModel, set_peft_model_state_dict
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from utils.callbacks import Iteratorize, Stream
from utils.prompter import Prompter

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

try:
    if torch.backends.mps.is_available():
        device = "mps"
except Exception:
    pass


def _load_hf_token():
    token_path = "./HF_key.json"
    if not os.path.exists(token_path):
        return None

    with open(token_path, "r", encoding="utf-8") as file:
        keys = json.load(file)

    return keys.get("hf_token")


def _build_model_kwargs(load_8bit):
    hf_token = _load_hf_token()
    kwargs = {}
    if hf_token:
        kwargs["token"] = hf_token

    if device == "cuda":
        kwargs["device_map"] = "auto"
        kwargs["dtype"] = torch.float16
    elif device == "mps":
        kwargs["device_map"] = {"": device}
        kwargs["dtype"] = torch.float16
    else:
        kwargs["device_map"] = {"": device}
        kwargs["low_cpu_mem_usage"] = True
        kwargs["dtype"] = torch.float32

    return kwargs


def main(
    load_8bit: bool = False,
    base_model: str = "",
    lora_weights_path: str = "",
    lora_config_path: str = "",  # provide only the file path, excluding the file name 'adapter_config.json'
    prompt_template: str = "",  # The prompt template to use, will default to alpaca.
    server_name: str = "127.0.0.1",
    root_path: str = "",
    share_gradio: bool = False,
):
    base_model = base_model or os.environ.get("BASE_MODEL", "")
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='huggyllama/llama-7b'"

    prompter = Prompter(prompt_template)
    tokenizer_kwargs = {}
    hf_token = _load_hf_token()
    if hf_token:
        tokenizer_kwargs["token"] = hf_token

    tokenizer = AutoTokenizer.from_pretrained(base_model, **tokenizer_kwargs)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id or 0
    tokenizer.padding_side = "left"

    model_kwargs = _build_model_kwargs(load_8bit)
    if load_8bit:
        print("Warning: current local transformers/Qwen setup does not support `load_in_8bit` here. Falling back to non-8bit loading.")
    if not lora_weights_path.endswith(".bin"):
        model = AutoModelForCausalLM.from_pretrained(base_model, **model_kwargs)
        peft_kwargs = {}
        if hf_token:
            peft_kwargs["token"] = hf_token
        if device == "cuda":
            peft_kwargs["dtype"] = torch.float16
        elif device == "mps":
            peft_kwargs["device_map"] = {"": device}
            peft_kwargs["dtype"] = torch.float16
        else:
            peft_kwargs["device_map"] = {"": device}

        model = PeftModel.from_pretrained(model, lora_weights_path, **peft_kwargs)
    else:
        assert lora_config_path, "Please specify --lora_config_path when --lora_weights_path is a .bin file"
        model = AutoModelForCausalLM.from_pretrained(base_model, **model_kwargs)
        config = LoraConfig.from_pretrained(lora_config_path)
        lora_weights = torch.load(lora_weights_path, map_location="cpu")
        model = PeftModel(model, config)
        set_peft_model_state_dict(model, lora_weights, "default")
        del lora_weights

    model.config.pad_token_id = tokenizer.pad_token_id
    if tokenizer.bos_token_id is not None:
        model.config.bos_token_id = tokenizer.bos_token_id
    if tokenizer.eos_token_id is not None:
        model.config.eos_token_id = tokenizer.eos_token_id

    if not load_8bit and device != "cpu":
        model.half()

    model.eval()


    def evaluate(
        instruction,
        input=None,
        num_beams=4,
        max_new_tokens=128,
        stream_output=True,
        **kwargs,
    ):
        prompt = prompter.generate_prompt(instruction, input)
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        generation_config = GenerationConfig(
            do_sample=False,
            temperature=0.0,
            top_p=1.0,
            top_k=50,
            num_beams=num_beams,
            pad_token_id=tokenizer.pad_token_id,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            **kwargs,
        )

        generate_params = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "generation_config": generation_config,
            "return_dict_in_generate": True,
            "output_scores": True,
            "max_new_tokens": max_new_tokens,
        }

        if stream_output:
            # Stream the reply 1 token at a time.
            # This is based on the trick of using 'stopping_criteria' to create an iterator,
            # from https://github.com/oobabooga/text-generation-webui/blob/ad37f396fc8bcbab90e11ecf17c56c97bfbd4a9c/modules/text_generation.py#L216-L243.

            def generate_with_callback(callback=None, **kwargs):
                kwargs.setdefault(
                    "stopping_criteria", transformers.StoppingCriteriaList()
                )
                kwargs["stopping_criteria"].append(
                    Stream(callback_func=callback)
                )
                with torch.no_grad():
                    model.generate(**kwargs)

            def generate_with_streaming(**kwargs):
                return Iteratorize(
                    generate_with_callback, kwargs, callback=None
                )

            with generate_with_streaming(**generate_params) as generator:
                for output in generator:
                    decoded_output = tokenizer.decode(output, skip_special_tokens=True)

                    if tokenizer.eos_token_id is not None and output[-1] == tokenizer.eos_token_id:
                        break

                    yield prompter.get_response(decoded_output)
            return  # early return for stream_output

        # Without streaming
        with torch.no_grad():
            generation_output = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=max_new_tokens,
            )
        s = generation_output.sequences[0]
        output = tokenizer.decode(s, skip_special_tokens=True)
        yield prompter.get_response(output)

    sherpherd_UI = gr.Interface(
        fn=evaluate,
        inputs=[
            gr.components.Textbox(
                lines=2,
                label="Instruction",
                placeholder="Tell me about alpacas.",
            ),
            gr.components.Textbox(lines=2, label="Input", placeholder="none"),
            gr.components.Slider(
                minimum=1, maximum=4, step=1, value=4, label="Beams"
            ),
            gr.components.Slider(
                minimum=1, maximum=2000, step=1, value=128, label="Max tokens"
            ),
            gr.components.Checkbox(label="Stream output"),
        ],
        outputs=[
            gr.components.Textbox(
                lines=5,
                label="Output",
            )
        ],
        title="FederatedGPT-shepherd",
        description="Shepherd is a LLM that has been fine-tuned in a federated manner ",
    ).queue()

    launch_kwargs = {
        "server_name": server_name,
        "share": share_gradio,
    }
    if root_path:
        launch_kwargs["root_path"] = root_path

    sherpherd_UI.launch(**launch_kwargs)





if __name__ == "__main__":
    fire.Fire(main)
