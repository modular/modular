# MAX Jupyter notebooks

This page explains how to get started with MAX example notebooks, and this
repo directory contains notebooks that demonstrate some of MAX features.

If you're not familiar with Jupyter notebooks, they're files that allow you to
create documents with live code, equations, visualizations, and explanatory
text. They're basically documents with executable code blocks, making them
great for sharing code experiments and programming tutorials.

## Get started in VS Code

Visual Studio Code is a great environment for programming with Jupyter notebooks.
Especially if you're developing with MAX on a remote system, using VS
Code is ideal because it allows you to edit and interact with notebooks on the
remote machine where you've installed MAX.

All you need is MAX (via Magic) and the Jupyter VS Code extension:

1. [Install Magic](https://docs.modular.com/magic).

2. Install [Visual Studio Code](https://code.visualstudio.com/) and the
   [Jupyter extension](https://marketplace.visualstudio.com/items?itemName=ms-toolsai.jupyter).

3. Open any `.ipynb` file in this repo and start running the code.

## Get started with JupyterLab

You can also run MAX notebooks in a local instance of JupyterLab. The following
is just a quick setup guide for Linux users, but it might not work with your
system (these instructions don't support remote access to the JupyterLab). For
more details about using JupyterLab, see the complete [JupyterLab installation
guide](https://jupyterlab.readthedocs.io/en/latest/getting_started/installation.html).

### 1. Launch JupyterLab

You can use either Magic or conda.

#### Using Magic

If you have [`magic`](https://docs.modular.com/magic) you can run the following
command to launch JupyterLab from this directory:

```sh
magic run jupyter lab
```

After a moment, it will open a browser window with JupterLab running.

#### Using conda

Create a Conda environment, activate that enviroment, and install JupyterLab.

``` sh
# Create a Conda environment if you don't have one
conda create -n max-repo
# Activate the environment
conda env update -n max-repo -f environment.yml --prune
# run JupyterLab
conda run -n max-repo jupyter lab
```

After a moment, it will open a browser window with JupterLab running.

### 2. Run the .ipynb notebooks

The left nav bar should show all the notebooks in this directory.
Open any `.ipynb` file and start running the code.
