<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=timeGradient&height=300&section=header&text=IMG-GEN&fontSize=60&fontAlign=50&fontAlignY=40&animation=twinkling" width="100%"/>
</p>

![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)
![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)
[![GitHub forks](https://img.shields.io/github/forks/crimsonKn1ght/img-gen.svg?style=social&label=Fork)](https://github.com/crimsonKn1ght/Code-OA-detection-model/network/members)
[![GitHub stars](https://img.shields.io/github/stars/crimsonKn1ght/img-gen.svg?style=social&label=★%20Star)](https://github.com/crimsonKn1ght/Code-OA-detection-model/stargazers)

## 🖼️ What is img-gen?

It creates images using diffusion models on your PC. It imports diffusion models (the first run is long as it downloads the model) and then creates images to your heart's content. It may take more or less time depending on your GPU.

Here are a few examples:

| Cyberpunk City | Couple dancing | Dog |
|---|---|---|
| <img src="https://github.com/user-attachments/assets/e11151ae-be50-4721-87e6-27d13c2c4137" width="250" height="250" /> | <img src="https://github.com/user-attachments/assets/d388344f-653a-4a4a-8ff0-277b15537ceb" width="250" height="250" /> | <img src="https://github.com/user-attachments/assets/a5eea5f7-ebb1-4c32-981c-456ce8ca4ea5" width="250" height="250" /> |

---

## 🚀 Installation & Setup

You can run this project using Docker (recommended for ease of use) or natively on your machine.

### Method 1: Using Docker (Recommended)

This is the easiest way to get started, as it handles all dependencies in an isolated environment.

**Prerequisites:**
* [Docker](https://docs.docker.com/get-docker/) installed.
* [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) for GPU support.

**1. Build the Docker Image:**
Clone this repository and navigate into the directory. Then, run the following command to build the image:
```bash
docker build -t img-gen-app .
```

**2. Run the Container:**
The command to run the container differs slightly based on your operating system.

**On Linux:**
First, allow Docker to connect to your host's display server:
```bash
xhost +local:docker
```
Then, run the container:
```bash
docker run --gpus all --rm -it \
       -v /tmp/.X11-unix:/tmp/.X11-unix \
       -e DISPLAY=$DISPLAY \
       img-gen-app
```

**On Windows:**
You'll need an X Server like **[VcXsrv](https://sourceforge.net/projects/vcxsrv/)** installed and running.
1.  Launch VcXsrv with the "Disable access control" option enabled.
2.  Find your PC's local IP address by running `ipconfig` in the command prompt.
3.  Run the container, replacing `YOUR_IP_ADDRESS` with the IP you found:
```powershell
docker run --gpus all --rm -it `
       -e DISPLAY=YOUR_IP_ADDRESS:0.0 `
       img-gen-app
```

### Method 2: Native Installation (Without Docker)

**1. Clone the Repository:**
```bash
git clone [https://github.com/crimsonKn1ght/img-gen.git](https://github.com/crimsonKn1ght/img-gen.git)
cd img-gen
```

**2. Install Dependencies:**

**On Linux (Debian/Ubuntu):**
First, install Python, pip, and Tkinter:
```bash
sudo apt-get update
sudo apt-get install -y python3.10 python3-pip python3-tk
```
Then, install the required Python packages:
```bash
pip install -r requirements.txt
```

**On Windows:**
1.  Install **[Python 3.10 or newer](https://www.python.org/downloads/)**. **Important:** Make sure to check the box that says "Add Python to PATH" during installation.
2.  Open a Command Prompt or PowerShell and install the required packages:
```powershell
pip install -r requirements.txt
```

**3. Run the Application:**
Once the dependencies are installed, you can start the application:
```bash
python img-gen.py
```

---

## 🤝 Contributions are greatly welcomed

Have any bright ideas? Open an issue! Make a pull request! Contributions are welcomed.


<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=timeGradient&height=200&section=footer&animation=twinkling" width="100%"/>
</p>