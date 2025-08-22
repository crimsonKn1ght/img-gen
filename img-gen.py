import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import torch
from diffusers import AutoPipelineForText2Image
import threading
import queue
import json
import os
import random

# ------------------------------
# Configuration
# ------------------------------
CONFIG_FILE = "config.json"
MODEL_OPTIONS = {
    "Stable Diffusion v1.5": "runwayml/stable-diffusion-v1-5",
    "Stable Diffusion XL": "stabilityai/stable-diffusion-xl-base-1.0",
    "Kandinsky 2.2": "kandinsky-community/kandinsky-2-2-decoder",
}

# ---------------------------------------------------
# Model Class (Handles the AI/Diffusers Backend)
# ---------------------------------------------------
class GeneratorModel:
    """Manages loading diffusion models and generating images."""
    def __init__(self):
        self.pipe = None
        self.loaded_model_id = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch.float16 if self.device == "cuda" else torch.float32

    def generate(self, data, status_queue):
        """Loads model if needed and runs the generation pipeline."""
        model_id = data['model_id']

        # Load the pipeline if it's not the correct one
        if self.loaded_model_id != model_id:
            status_queue.put(f"Loading model: {data['model_name']}...")
            try:
                self.pipe = AutoPipelineForText2Image.from_pretrained(
                    model_id, torch_dtype=self.torch_dtype
                ).to(self.device)
                
                # --- Optional Performance Optimizations ---
                # For NVIDIA GPUs, uncomment the line below after `pip install xformers`
                # self.pipe.enable_xformers_memory_efficient_attention() 
                # For systems with low VRAM
                # self.pipe.enable_model_cpu_offload()

                self.loaded_model_id = model_id
            except Exception as e:
                return f"Error loading model: {e}"

        status_queue.put(f"Generating image for: '{data['prompt'][:40]}...'")
        
        # Set up the generator for reproducibility
        generator = torch.Generator(device=self.device).manual_seed(data['seed'])
        
        with torch.inference_mode():
            image = self.pipe(
                prompt=data['prompt'],
                negative_prompt=data['neg_prompt'],
                num_inference_steps=data['steps'],
                guidance_scale=data['cfg'],
                generator=generator
            ).images[0]
        
        return image

# ---------------------------------------------------
# View Class (Handles the Tkinter UI)
# ---------------------------------------------------
class UIView:
    """Manages all UI elements."""
    def __init__(self, root, controller):
        self.root = root
        self.controller = controller
        self.root.title("Next-Level Text-to-Image Generator")
        self.root.geometry("600x820")

        # --- Call the new method to set up the dark theme ---
        self.setup_style()

        # --- Create Frames for organization ---
        # The main_frame now uses the 'App.TFrame' style
        main_frame = ttk.Frame(root, padding="15", style='App.TFrame')
        main_frame.pack(fill=tk.BOTH, expand=True)

        top_frame = ttk.Frame(main_frame)
        top_frame.pack(fill=tk.X, pady=5)

        adv_frame = ttk.LabelFrame(main_frame, text="Advanced Settings", padding="10")
        adv_frame.pack(fill=tk.X, pady=10, ipady=5)

        image_frame = ttk.Frame(main_frame)
        image_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        # --- Top Frame Widgets (Prompts & Model) ---
        ttk.Label(top_frame, text="Prompt:").pack(anchor=tk.W)
        self.prompt_entry = ttk.Entry(top_frame, width=80, font=("Helvetica", 10))
        self.prompt_entry.pack(fill=tk.X, ipady=2)

        ttk.Label(top_frame, text="Negative Prompt:").pack(anchor=tk.W, pady=(10, 0))
        self.neg_prompt_entry = ttk.Entry(top_frame, width=80, font=("Helvetica", 10))
        self.neg_prompt_entry.pack(fill=tk.X, ipady=2)

        ttk.Label(top_frame, text="Select Model:").pack(anchor=tk.W, pady=(10, 0))
        self.model_var = tk.StringVar()
        self.model_dropdown = ttk.Combobox(
            top_frame, textvariable=self.model_var, values=list(MODEL_OPTIONS.keys()), state="readonly"
        )
        self.model_dropdown.pack(fill=tk.X)

        # --- Advanced Frame Widgets ---
        # Steps
        steps_frame = ttk.Frame(adv_frame)
        steps_frame.pack(fill=tk.X, expand=True, pady=2)
        ttk.Label(steps_frame, text="Steps:", width=14).pack(side=tk.LEFT)
        self.steps_var = tk.IntVar(value=25)
        self.steps_slider = ttk.Scale(steps_frame, from_=10, to=100, orient=tk.HORIZONTAL, variable=self.steps_var, command=lambda s: self.steps_var.set(int(float(s))))
        self.steps_slider.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.steps_label = ttk.Label(steps_frame, textvariable=self.steps_var, width=4)
        self.steps_label.pack(side=tk.LEFT)

        # CFG Scale
        cfg_frame = ttk.Frame(adv_frame)
        cfg_frame.pack(fill=tk.X, expand=True, pady=2)
        ttk.Label(cfg_frame, text="Guidance (CFG):", width=14).pack(side=tk.LEFT)
        self.cfg_var = tk.DoubleVar(value=7.5)
        self.cfg_slider = ttk.Scale(cfg_frame, from_=1.0, to=20.0, orient=tk.HORIZONTAL, variable=self.cfg_var, command=lambda s: self.cfg_var.set(round(float(s), 1)))
        self.cfg_slider.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.cfg_label = ttk.Label(cfg_frame, textvariable=self.cfg_var, width=4)
        self.cfg_label.pack(side=tk.LEFT)

        # Seed
        seed_frame = ttk.Frame(adv_frame)
        seed_frame.pack(fill=tk.X, expand=True, pady=2)
        ttk.Label(seed_frame, text="Seed:", width=14).pack(side=tk.LEFT)
        self.seed_var = tk.IntVar(value=-1)
        self.seed_entry = ttk.Entry(seed_frame, textvariable=self.seed_var)
        self.seed_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.random_seed_button = ttk.Button(seed_frame, text="🎲", width=3, command=self.controller.set_random_seed, style="Random.TButton")
        self.random_seed_button.pack(side=tk.LEFT, padx=(5,0))
        
        # --- Frame for side-by-side buttons ---
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=5)

        # --- Main Action Buttons ---
        self.generate_button = ttk.Button(button_frame, text="Generate Image", command=self.controller.start_generation, style="Accent.TButton")
        self.generate_button.pack(side=tk.LEFT, fill=tk.X, expand=True, ipady=5, padx=(0, 5))

        self.save_button = ttk.Button(button_frame, text="Save Image", command=self.controller.save_image, state=tk.DISABLED)
        self.save_button.pack(side=tk.LEFT, fill=tk.X, expand=True, ipady=5, padx=(5, 0))

        # --- Image Display ---
        self.image_label = ttk.Label(image_frame, text="\nYour generated image will appear here.", anchor=tk.CENTER, style="Image.TLabel", font=("Helvetica", 10))
        self.image_label.pack(fill=tk.BOTH, expand=True)

        # --- Status Bar ---
        self.status_var = tk.StringVar(value="Ready")
        self.status_bar = ttk.Label(root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W, padding=5, style="Status.TLabel")
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def setup_style(self):
        """Configures the dark theme for the entire application."""
        # --- Color Palette ---
        BG_COLOR = "#212121"         # Dark grey
        FRAME_COLOR = "#2c2c2c"     # Slightly lighter grey
        TEXT_COLOR = "#e0e0e0"       # Off-white
        ENTRY_BG = "#3c3c3c"         # Lighter grey for inputs
        ACCENT_COLOR = "#00bcd4"     # Cyan
        ACCENT_HOVER = "#00e5ff"     # Brighter Cyan
        BUTTON_COLOR = "#424242"     # Grey for standard buttons

        style = ttk.Style(self.root)
        self.root.configure(background=BG_COLOR)
        style.theme_use("clam")

        # --- Configure all widget styles ---
        style.configure(".", background=FRAME_COLOR, foreground=TEXT_COLOR, font=("Helvetica", 9))
        style.configure("App.TFrame", background=BG_COLOR)

        style.configure("TFrame", background=FRAME_COLOR)
        style.configure("TLabel", background=FRAME_COLOR, foreground=TEXT_COLOR)
        style.configure("Image.TLabel", background=ENTRY_BG, borderwidth=2, relief="sunken")
        style.configure("Status.TLabel", background=ACCENT_COLOR, foreground="#000000")
        
        style.configure("TLabelFrame", background=FRAME_COLOR, bordercolor=TEXT_COLOR)
        style.configure("TLabelFrame.Label", background=FRAME_COLOR, foreground=TEXT_COLOR, font=("Helvetica", 9, "bold"))

        # --- Entry and Dropdown Styles ---
        style.configure("TEntry", fieldbackground=ENTRY_BG, foreground=TEXT_COLOR, bordercolor=BUTTON_COLOR, insertcolor=TEXT_COLOR)
        style.map("TCombobox", fieldbackground=[("readonly", ENTRY_BG)], foreground=[("readonly", TEXT_COLOR)], selectbackground=[("readonly", ENTRY_BG)])
        self.root.option_add("*TCombobox*Listbox*Background", ENTRY_BG)
        self.root.option_add("*TCombobox*Listbox*Foreground", TEXT_COLOR)
        self.root.option_add("*TCombobox*Listbox*selectBackground", ACCENT_COLOR)

        # --- Button Styles ---
        style.configure("TButton", background=BUTTON_COLOR, foreground=TEXT_COLOR, padding=5, borderwidth=0)
        style.map("TButton",
            background=[("active", ENTRY_BG), ("disabled", "#505050")],
            foreground=[("disabled", "#909090")]
        )
        style.configure("Random.TButton", padding=0)

        # Accent button style
        style.configure("Accent.TButton", background=ACCENT_COLOR, foreground="#000000", font=("Helvetica", 10, "bold"))
        style.map("Accent.TButton",
            background=[("active", ACCENT_HOVER)],
        )

        # --- Slider Styles ---
        style.configure("TScale", background=FRAME_COLOR)


# ---------------------------------------------------
# Controller Class (Manages App Logic)
# ---------------------------------------------------
class AppController:
    def __init__(self, root):
        self.root = root
        self.model = GeneratorModel()
        self.view = UIView(root, self)
        
        self.comm_queue = queue.Queue()
        self.generated_image = None

        self.load_config()
        self.check_comm_queue()

        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def start_generation(self):
        """Gathers UI data and starts the generation thread."""
        self.view.generate_button.config(state=tk.DISABLED)
        self.view.save_button.config(state=tk.DISABLED)

        prompt_text = self.view.prompt_entry.get().strip()
        if not prompt_text:
            messagebox.showerror("Input Error", "Please enter a prompt.")
            self.view.generate_button.config(state=tk.NORMAL)
            return

        if self.view.seed_var.get() == -1:
            self.set_random_seed()

        # Data package for the backend thread
        generation_data = {
            'prompt': prompt_text,
            'neg_prompt': self.view.neg_prompt_entry.get().strip(),
            'model_name': self.view.model_var.get(),
            'model_id': MODEL_OPTIONS[self.view.model_var.get()],
            'steps': self.view.steps_var.get(),
            'cfg': self.view.cfg_var.get(),
            'seed': self.view.seed_var.get()
        }
        
        # Start generation in a separate thread
        threading.Thread(
            target=self.generation_worker, 
            args=(generation_data,),
            daemon=True
        ).start()

    def generation_worker(self, data):
        """Worker function that runs in a separate thread."""
        try:
            result = self.model.generate(data, self.comm_queue)
            self.comm_queue.put(result)
        except Exception as e:
            self.comm_queue.put(f"Unhandled Error: {e}")

    def check_comm_queue(self):
        """Checks the queue for messages from the worker thread."""
        try:
            message = self.comm_queue.get_nowait()
            
            if isinstance(message, Image.Image):
                self.display_generated_image(message)
                self.view.status_var.set("Image generated successfully. Ready.")
                self.view.generate_button.config(state=tk.NORMAL)
                self.view.save_button.config(state=tk.NORMAL)
            elif isinstance(message, str):
                self.view.status_var.set(message)
                # Re-enable button on error
                if "Error" in message:
                    self.view.generate_button.config(state=tk.NORMAL)

        except queue.Empty:
            pass
        finally:
            self.root.after(100, self.check_comm_queue)

    def display_generated_image(self, image):
        """Updates the UI with the newly generated image."""
        self.generated_image = image
        
        # Create a display-sized version
        w, h = self.view.image_label.winfo_width(), self.view.image_label.winfo_height()
        if w > 1 and h > 1: # Ensure the label has been rendered
             display_img = image.copy()
             display_img.thumbnail((w - 10, h - 10), Image.Resampling.LANCZOS)
             tk_img = ImageTk.PhotoImage(display_img)
        else: # Fallback for initial display
             tk_img = ImageTk.PhotoImage(image.resize((512, 512)))

        self.view.image_label.configure(image=tk_img, text="")
        self.view.image_label.image = tk_img

    def save_image(self):
        if self.generated_image is None:
            return
        filepath = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg"), ("All files", "*.*")],
            title="Save Image As"
        )
        if filepath:
            self.generated_image.save(filepath)
            self.view.status_var.set(f"Image saved to {os.path.basename(filepath)}")

    def set_random_seed(self):
        self.view.seed_var.set(random.randint(0, 2**32 - 1))

    def load_config(self):
        """Loads settings from the config file if it exists."""
        try:
            if os.path.exists(CONFIG_FILE):
                with open(CONFIG_FILE, 'r') as f:
                    config = json.load(f)
                    self.view.prompt_entry.insert(0, config.get("prompt", ""))
                    self.view.neg_prompt_entry.insert(0, config.get("neg_prompt", ""))
                    self.view.model_var.set(config.get("model_name", list(MODEL_OPTIONS.keys())[0]))
                    self.view.steps_var.set(config.get("steps", 25))
                    self.view.cfg_var.set(config.get("cfg", 7.5))
                    self.view.seed_var.set(config.get("seed", -1))
            else: # Set defaults if no config file
                self.view.model_dropdown.current(0)

        except (json.JSONDecodeError, KeyError) as e:
            print(f"Could not load config file: {e}. Using defaults.")
            self.view.model_dropdown.current(0)


    def save_config(self):
        """Saves current settings to the config file."""
        config = {
            "prompt": self.view.prompt_entry.get(),
            "neg_prompt": self.view.neg_prompt_entry.get(),
            "model_name": self.view.model_var.get(),
            "steps": self.view.steps_var.get(),
            "cfg": self.view.cfg_var.get(),
            "seed": self.view.seed_var.get()
        }
        with open(CONFIG_FILE, 'w') as f:
            json.dump(config, f, indent=4)

    def on_closing(self):
        """Handles the window close event."""
        self.save_config()
        self.root.destroy()

# ------------------------------
# Run the App
# ------------------------------
if __name__ == "__main__":
    root = tk.Tk()
    app = AppController(root)
    root.mainloop()
