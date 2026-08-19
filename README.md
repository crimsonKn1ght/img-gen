<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=timeGradient&height=300&section=header&text=IMG-GEN&fontSize=60&fontAlign=50&fontAlignY=40&animation=twinkling" width="100%"/>
</p>

![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)
![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![GitHub forks](https://img.shields.io/github/forks/crimsonKn1ght/img-gen.svg?style=social&label=Fork)](https://github.com/crimsonKn1ght/img-gen/network/members)
[![GitHub stars](https://img.shields.io/github/stars/crimsonKn1ght/img-gen.svg?style=social&label=Stars)](https://github.com/crimsonKn1ght/img-gen/stargazers)

## What is img-gen?

It creates images using diffusion models on your PC. It imports diffusion models (the first run is long as it downloads the model) and then creates images to your heart's content. It may take more or less time depending on your GPU.

Here are a few examples:

| Cyberpunk City | Couple dancing | Dog |
|---|---|---|
| <img src="https://github.com/user-attachments/assets/e11151ae-be50-4721-87e6-27d13c2c4137" width="250" height="250" /> | <img src="https://github.com/user-attachments/assets/d388344f-653a-4a4a-8ff0-277b15537ceb" width="250" height="250" /> | <img src="https://github.com/user-attachments/assets/a5eea5f7-ebb1-4c32-981c-456ce8ca4ea5" width="250" height="250" /> |

---

Streamlit app link: https://img-gen-tool.streamlit.app/

## Models
- Stable Diffusion v1.5
- Stable Diffusion XL
- Kandinsky 2.2

## Run it locally

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

`app.py` is a thin launcher that starts the same app on port 7860 and binds all
interfaces, which is what container and Spaces-style hosting expects:

```bash
python app.py
```

The first run for a given model downloads its weights, which is the slow part. A CUDA
GPU is used in half precision when one is available, and it falls back to CPU in full
precision otherwise.

## How to use
1. Select a model in the sidebar
2. Enter your prompt + optional negative prompt
3. Adjust steps, CFG, and seed
4. Click "Generate"

Your image will appear on the main panel and can be downloaded.

---

## License

Apache License 2.0. See [LICENSE](LICENSE).

## ?? Contributing

Contributions from developers, researchers, students, designers, documentation
writers, and first-time open-source contributors are welcome.

You can help by improving the Streamlit interface, adding tests, documenting model
and hardware behavior, fixing bugs, or proposing carefully scoped features. If you
are new to the project, start with an issue labelled
[`good first issue`](https://github.com/crimsonKn1ght/img-gen/labels/good%20first%20issue)
or [`help wanted`](https://github.com/crimsonKn1ght/img-gen/labels/help%20wanted).

Before making a substantial change, comment on the relevant issue or open a feature
request so the approach can be discussed. See [CONTRIBUTING.md](CONTRIBUTING.md) for
the development workflow and pull-request checklist. By participating, you agree to
follow the [Code of Conduct](CODE_OF_CONDUCT.md).

Security concerns should be reported according to [SECURITY.md](SECURITY.md), not in
a public issue.

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=timeGradient&height=200&section=footer&animation=twinkling" width="100%"/>
</p>
