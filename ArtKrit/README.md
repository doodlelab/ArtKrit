# ArtKrit
ArtKrit is a plugin for Krita that helps artists enhance their drawing skills by scaffolding the process of replicating a reference image into three steps: composition, value, and color. At each stage, ArtKrit generates adaptive composition lines and provides feedback on value and color accuracy to help artists refine their work.

<img width="1062" height="618" alt="fig_teaser" src="https://github.com/user-attachments/assets/3bae40d9-9e91-4251-95b7-b7d251f5a3b5" />

**Bottom row**: Computational guidance and feedback provided by our system at each step. We offer object-based composition lines to assist with spatial positioning, and we visualize differences in value and color with verbal suggestions to guide the user.

*Reference image: "Interior Practice // Kitchen" by Loish (2021, digital).*

## Installation


### Krita
1. Install Krita from the official website: https://krita.org/en/download/ (tested on version 5.2.9)
2. To facilitate debugging, you can add the path to your Krita binary to your bash or zsh profile. On Mac, it should look like this (it would be bash for older MacOS versions):
    ```bash
    echo 'export PATH="/Applications/krita.app/Contents/MacOS/:$PATH"' >> ~/.zshrc
    source ~/.zshrc
    ```
    This allows you to run Krita from the terminal by typing `krita`. All the output from Krita and the plugin will be printed to the terminal.


### File Structure Setup
1. On Mac, the Python plugin folder is located at `~/Library/Application Support/Krita/pykrita/`. Navigate to this folder and `git clone` this repository. Note that for MacOS, ~/Library/Application Support and /Library/Application Support are different folders. If you don't find the Krita folder, make sure you are in the Application Support for your user.

2. Under the `pykrita` folder, create a `artkrit.desktop` file. The file structure now should look like this:
    ```
    pykrita/
        ArtKrit/
            ...
        artkrit.desktop
    ```

3. In the `artkrit.desktop` file, add the following content:
    ```ini
    # File: artkrit.desktop
    [Desktop Entry]
    Type=Service
    ServiceTypes=Krita/PythonPlugin
    X-KDE-Library=ArtKrit
    X-Python-2-Compatible=false
    X-Krita-Manual=Manual.html
    Name=ArtKrit
    Comment=Docker for ArtKrit
    ```


### Python Plugin Setup

1. Pick your favorite virtual environment tool (e.g. `uv`, `venv`, `conda`, etc.) and create a new environment with `python==3.10` at your home directory (`~`).
   - Make sure you use `python==3.10` for compatibility with Krita 5.2.9.
   - Name your environment `ddraw` for consistency. If you name it something else or place it elsewhere, update the system path at the top of `artkrit.py` and `value_color.py`.
   - I recommend using [`uv`](https://docs.astral.sh/uv/) to manage your environments for its simplicity and speed.
     - To install uv (on MacOS), `curl -LsSf https://astral.sh/uv/install.sh | sh`
     - To install the virtual environment, `uv venv ddraw --python 3.10`

2. Activate your environment (e.g., `source ddraw/bin/activate`). Now navigate (`cd`) to `~/Library/Application\ Support/Krita/pykrita/ArtKrit`

3. First, install pytorch with:
    ```bash
    pip install torch torchvision torchaudio

    # If you're using `uv`, you can use the following command:
    uv pip install torch torchvision torchaudio
    ```

4.  Then install the other required packages:
    ```bash
    pip install -r requirements.txt

    # If you're using `uv`, you can use the following command:
    uv pip install -r requirements.txt
    ```

## Running the Plugin
1. In one terminal, run Krita by typing `krita` in the terminal. This will allow you to see the output from the plugin. Directly opening Krita also works, but you won't see the output.

2. In another terminal, activate your environment and navigate to the `ArtKrit` folder. Then start the python server with:
    ```bash
    python script/composition/server.py
    ```
    Note, if you get an error, try running it with the specific python version: `python3.10 server.py`

3. On the first launch, enable the plugin by going to Preferences (cmd+`,`), scrolling down, selecting Python Plugin Manager, and checking the ArtKrit box. Then, relaunch Krita. The docker (window) for the plug-in can be found under Settings > Dockers > ArtKrit.
4. When setting up a Krita document, it is recommended to set it as the same size as the reference image. This will ensure that the plugin works as intended.
5. Make sure to click `Set Reference Image` button every time you reopen Krita.
6. If inferencing time for generating adaptive grids is too long, you can try to use smaller models listed in `run_models.py`. Note that smaller models will not be as performant.


## Helpful Resources
Krita API Documentation: [https://api.kde.org/krita/html/](https://api.kde.org/krita/html/)
Guide for plugins: [https://docs.krita.org/en/user_manual/python_scripting/krita_python_plugin_howto.html](https://docs.krita.org/en/user_manual/python_scripting/krita_python_plugin_howto.html)
