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
    st.set_page_config(page_title="Neural Style Transfer", layout="wide")
    st.title("Neural Style Transfer")
    st.markdown(
        "Explore how convolutional neural network depth and feature hierarchy "
        "affect style transfer representation."
    )

    st.sidebar.header("Configuration")

    content_depth = st.sidebar.radio(
        "Content Representation Depth",
        ["Shallow", "Medium", "Deep"],
        index=1,
        help="Shallow layers preserve edges and local structure. "
             "Deep layers preserve high-level semantics."
    )

    style_scale = st.sidebar.radio(
        "Style Representation Scale",
        ["Fine", "Multi-scale", "Full hierarchy"],
        index=2,
        help="Controls how many CNN layers contribute to style representation."
    )

    selected_content_layers = CONTENT_LAYER_MAP[content_depth]
    selected_style_layers   = STYLE_LAYER_MAP[style_scale]

    alpha = st.sidebar.slider("Content weight (α)", 0.1, 10.0, 1.0)
    beta  = st.sidebar.slider("Style weight (β)", 1e4, 1e7, 1e6)
    steps = st.sidebar.slider("Optimization steps", 20, 600, 300)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Content Image")
        content_file = st.file_uploader(
            "Upload Content Image",
            type=["jpg", "png"],
            key="content_upload"
        )

        content_img = (
            Image.open(content_file).convert("RGB")
            if content_file else None
        )

        if content_img:
            st.image(content_img, width='stretch')

    with col2:
        st.subheader("Style Image")
        style_file = st.file_uploader(
            "Upload Style Image",
            type=["jpg", "png"],
            key="style_upload"
        )

        style_img = (
            Image.open(style_file).convert("RGB")
            if style_file else None
        )

        if style_img:
            st.image(style_img, width='stretch')

    st.divider()

    run_disabled = not (content_img and style_img)

    if st.button("Run Style Transfer", disabled=run_disabled):
        with st.spinner("Running optimization... This may take a while."):
            result_tensor = run_experiment(
                ExperimentConfig(
                    content_image=content_img,
                    style_image=style_img,
                    steps=steps,
                    alpha=alpha,
                    beta=beta,
                    content_layers=selected_content_layers,
                    style_layers=selected_style_layers,
                ),
                return_history=False
            )

        output_image = tensor_to_pil(result_tensor)

        st.success("Style transfer completed successfully!")
        st.subheader("Result")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.image(content_img, caption="Content", width='stretch')

        with col2:
            st.image(style_img, caption="Style", width='stretch')

        with col3:
            st.image(output_image, caption="Stylized Output", width='stretch')

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