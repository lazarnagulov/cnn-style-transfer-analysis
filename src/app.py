import io

import streamlit as st
from PIL import Image
import torch
import torchvision.transforms as transforms

from experiments.config import ExperimentConfig
from experiments.runner import run_experiment

CONTENT_LAYER_MAP = {
    "Shallow": ["conv2_2"],
    "Medium": ["conv4_2"],
    "Deep": ["conv5_2"],
}

STYLE_LAYER_MAP = {
    "Fine": ["conv1_1"],
    "Multi-scale": ["conv1_1", "conv2_1", "conv3_1"],
    "Full hierarchy": [
        "conv1_1", "conv2_1", "conv3_1",
        "conv4_1", "conv5_1"
    ],
}

def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    if tensor.ndim == 4:
        tensor = tensor.squeeze(0)

    tensor = torch.clamp(tensor.detach().cpu(), 0, 1)
    return transforms.ToPILImage()(tensor)


def main() -> None:
    st.set_page_config(page_title="Neural Style Transfer")
    st.title("Neural Style Transfer")
    content_file = st.file_uploader("Upload Content Image", type=["jpg", "png"])
    style_file   = st.file_uploader("Upload Style Image", type=["jpg", "png"])
    
    if content_file:
        content_img = Image.open(content_file).convert("RGB")
        st.subheader("Content Image")
        st.image(content_img, use_container_width=True)
    else:
        content_img = None

    if style_file:
        style_img = Image.open(style_file).convert("RGB")
        st.subheader("Style Image")
        st.image(style_img, use_container_width=True)
    else:
        style_img = None
    
    st.divider()

    st.subheader("Representation Settings")

    content_depth = st.radio(
        "Content Representation Depth:",
        ["Shallow", "Medium", "Deep"],
        index=1
    )

    style_scale = st.radio(
        "Style Representation Scale:",
        ["Fine", "Multi-scale", "Full hierarchy"],
        index=2
    )

    selected_content_layers = CONTENT_LAYER_MAP[content_depth]
    selected_style_layers   = STYLE_LAYER_MAP[style_scale]

    st.divider()

    alpha = st.slider("Content weight (α)", 0.1, 10.0, 1.0)
    beta  = st.slider("Style weight (β)", 1e4, 1e7, 1e6)
    steps = st.slider("Optimization steps", 20, 600, 300)

    if content_file and style_file:
        content_img = Image.open(content_file)
        style_img   = Image.open(style_file)
        
        if st.button("Run Style Transfer"):
            with st.spinner("Running optimization..."):
                result = run_experiment(ExperimentConfig(
                    content_image=content_img,
                    style_image=style_img,
                    steps=steps,
                    alpha=alpha,
                    beta=beta,
                    content_layers=selected_content_layers,
                    style_layers=selected_style_layers,
                ), return_history=False)

            output_image = tensor_to_pil(result)
            st.subheader("Stylized Output")
            st.image(output_image, width='stretch')
            buffer = io.BytesIO()
            output_image.save(buffer, format="PNG")
            buffer.seek(0)

            st.download_button(
                label="Download Stylized Image",
                data=buffer,
                file_name="stylized_output.png",
                mime="image/png",
            )

if __name__ == "__main__":
    main()