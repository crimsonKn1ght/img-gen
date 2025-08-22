import tkinter as tk
from tkinter import ttk, filedialog
from PIL import Image, ImageTk
import torch
from diffusers import AutoPipelineForText2Image

# ------------------------------
# Supported Hugging Face Models
# ------------------------------
MODEL_OPTIONS = {
    "Stable Diffusion v1.5": "runwayml/stable-diffusion-v1-5",
    "Stable Diffusion XL": "stabilityai/stable-diffusion-xl-base-1.0",
    "Kandinsky 2.2": "kandinsky-community/kandinsky-2-2-decoder",
}

class Text2ImageApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Text-to-Image Generator")

        # Device selection
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # UI Layout
        self.prompt_label = tk.Label(root, text="Enter Prompt:")
        self.prompt_label.pack(pady=5)

        self.prompt_entry = tk.Entry(root, width=60)
        self.prompt_entry.pack(pady=5)

        # Model selection dropdown
        self.model_label = tk.Label(root, text="Select Model:")
        self.model_label.pack(pady=5)

        self.model_var = tk.StringVar(value=list(MODEL_OPTIONS.keys())[0])
        self.model_dropdown = ttk.Combobox(
            root, textvariable=self.model_var, values=list(MODEL_OPTIONS.keys()), state="readonly"
        )
        self.model_dropdown.pack(pady=5)

        # Generate button
        self.generate_button = tk.Button(root, text="Generate Image", command=self.generate_image)
        self.generate_button.pack(pady=10)

        # Image display
        self.image_label = tk.Label(root)
        self.image_label.pack(pady=10)

        # Save button
        self.save_button = tk.Button(root, text="Save Image", command=self.save_image, state=tk.DISABLED)
        self.save_button.pack(pady=5)

        # Pipeline placeholder
        self.pipe = None
        self.generated_image = None

    def load_model(self, model_choice):
        model_name = MODEL_OPTIONS[model_choice]
        print(f"Loading model: {model_choice} ({model_name})...")

        torch_dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.pipe = AutoPipelineForText2Image.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
        ).to(self.device)

    def generate_image(self):
        prompt = self.prompt_entry.get().strip()
        if not prompt:
            print("Please enter a prompt.")
            return

        model_choice = self.model_var.get()
        if self.pipe is None or self.loaded_model != model_choice:
            self.load_model(model_choice)
            self.loaded_model = model_choice

        print(f"Generating image with '{model_choice}' for prompt: {prompt}")

        with torch.inference_mode():
            image = self.pipe(prompt).images[0]

        self.generated_image = image

        # Resize for tkinter display
        display_img = image.resize((512, 512))
        tk_img = ImageTk.PhotoImage(display_img)
        self.image_label.configure(image=tk_img)
        self.image_label.image = tk_img

        self.save_button.config(state=tk.NORMAL)

    def save_image(self):
        if self.generated_image is None:
            return
        filepath = filedialog.asksaveasfilename(defaultextension=".png",
                                                filetypes=[("PNG files", "*.png")])
        if filepath:
            self.generated_image.save(filepath)
            print(f"Image saved at {filepath}")


# ------------------------------
# Run the App
# ------------------------------
if __name__ == "__main__":
    root = tk.Tk()
    app = Text2ImageApp(root)
    root.mainloop()
