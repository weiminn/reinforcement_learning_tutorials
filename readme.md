<h1>Reinforcement Learning</h2>

<h2>Set up</h2>

Create Virtual Environment
> `python -m venv --copies venv`

Enter the virtual environment
```
source venv/bin/activate
```

Install tools
```
pip install brax==0.0.12 jax==0.3.14 jaxlib==0.3.14 pyglet==1.5.27 pyopengl matplotlib pandas numpy tqdm jupyter seaborn scikit-learn gym==0.21.0 gymnasium==0.26.3 pygame
```

Install torch
```
pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu
```

```
sudo apt-get install xvfb ffmpeg
```

```
pip install pytorch-lightning==1.6 pyvirtualdisplay
```