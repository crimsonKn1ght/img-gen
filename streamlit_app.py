import streamlit as st
from PIL import Image
import torch
from diffusers import AutoPipelineForText2Image
import random
from io import BytesIO

# ------------------------------
# Configuration
# ------------------------------
MODEL_OPTIONS = {
    "Stable Diffusion v1.5": "runwayml/stable-diffusion-v1-5",
    "Stable Diffusion XL": "stabilityai/stable-diffusion-xl-base-1.0",
    "Kandinsky 2.2": "kandinsky-community/kandinsky-2-2-decoder",
}

# ---------------------------------------------------
# Model Loading (with Streamlit's caching)
# ---------------------------------------------------
@st.cache_resource
def get_model(model_id):
    """Loads and caches the diffusion model to avoid reloading on every interaction."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if device == "cuda" else torch.float32

    # Display a user-friendly message while the model loads
    with st.spinner(f"Loading model: {model_id}... This can take a few minutes on the first run."):
        pipe = AutoPipelineForText2Image.from_pretrained(
            model_id, torch_dtype=torch_dtype
        ).to(device)
    return pipe, device

# ---------------------------------------------------
# Main App UI
# ---------------------------------------------------
st.set_page_config(layout="wide", page_title="IMG-GEN")

st.title("🖼️ IMG-GEN: AI Image Generator")
st.markdown("Bring your creative ideas to life! Describe the image you want in the sidebar, and the AI will generate it for you.")

# --- Sidebar for user inputs ---
with st.sidebar:
    st.header("⚙️ Settings")

    model_name = st.selectbox("Select Model:", options=list(MODEL_OPTIONS.keys()), help="Choose the AI model to generate the image.")
    model_id = MODEL_OPTIONS[model_name]

    prompt = st.text_area("1. Enter your Prompt:", "Photo of a majestic lion drinking from a river in the savanna, sunset, hyperrealistic, 8k", height=100, help="Describe the image you want to create.")
    neg_prompt = st.text_area("2. Enter a Negative Prompt:", "cartoon, blurry, low quality, watermark, text", height=100, help="Describe what you want to avoid in the image.")

    st.subheader("Advanced Settings")
    steps = st.slider("Inference Steps:", min_value=10, max_value=150, value=30, help="More steps can improve quality but take longer.")
    cfg = st.slider("Guidance (CFG):", min_value=1.0, max_value=20.0, value=7.5, step=0.1, help="How strongly the prompt influences the image.")

    # --- Seed input with a button for randomization ---
    if 'seed' not in st.session_state:
        st.session_state.seed = -1

    def set_random_seed():
        st.session_state.seed = random.randint(0, 2**32 - 1)

    seed = st.number_input("Seed:", value=st.session_state.seed, min_value=-1, max_value=2**32-1, help="Use -1 for a random seed, or set a specific number for reproducible results.")
    st.button("🎲 Randomize Seed", on_click=set_random_seed)

# --- Main content area ---
generate_button = st.button("Generate Image", type="primary", use_container_width=True)

# Use a placeholder for the image and download button for a cleaner layout
image_placeholder = st.empty()
download_placeholder = st.empty()

if generate_button:
    if not prompt:
        st.error("Please enter a prompt to generate an image.")
    else:
        try:
            # Load the model
            pipe, device = get_model(model_id)

            # --- Start generation ---
            with st.spinner("Generating image... Please wait."):
                final_seed = seed if seed != -1 else random.randint(0, 2**32 - 1)
                st.info(f"Using seed: {final_seed}")
                generator = torch.Generator(device=device).manual_seed(final_seed)

                with torch.inference_mode():
                    image = pipe(
                        prompt=prompt,
                        negative_prompt=neg_prompt,
                        num_inference_steps=steps,
                        guidance_scale=cfg,
                        generator=generator
                    ).images[0]

                # Store the generated image in the session state
                st.session_state.generated_image = image

        except Exception as e:
            st.error(f"An error occurred during image generation: {e}")

# --- Display the image and download button if it exists in the session state ---
if "generated_image" in st.session_state:
    image_placeholder.image(st.session_state.generated_image, caption="Generated Image", use_column_width=True)

    # --- Convert image to bytes for download button ---
    buf = BytesIO()
    st.session_state.generated_image.save(buf, format="PNG")
    byte_im = buf.getvalue()

    # --- Add a download button to its placeholder ---
    download_placeholder.download_button(
        label="Download Image",
        data=byte_im,
        file_name="generated_image.png",
        mime="image/png",
        use_container_width=True
    )
else:
    # Show a welcome message in the placeholder before the first generation
    image_placeholder.info("Your generated image will appear here. Adjust the settings in the sidebar and click 'Generate'.")

