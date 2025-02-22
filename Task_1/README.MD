# Task 1. Image classification + OOP

In this task, the goal was to train 3 separate models for Image classification using MNIST dataset such as:
* Random Forest;
* Feed-Forward Neural Network
* Convolutional Neural Network

To complete this task, next libraries were used:
- **pandas**, **scikit-learn**, **numpy**: for data manipulation and applying metrics as well as ready Random Forest architecture extraction.
- **PyTorch**, **torchvision**: for building and training neural networks and loading MNIST dataset.

As stated in the task description, the given models, specifically CNN with TinyVGG architecture, Feed-Forward Network and Random Forest, were built and trained. All three models showed good results: 
* CNN: 99.03%
* FFNN: 97.54%
* Random Forest: 96.73%

## Environment Setup

### 1. Install Dependencies

If you're using **Google Colab**, most dependencies are already pre-installed. However, you may need to install additional libraries, such as **torchmetrics** to use accuracy metric, because specifically this lib isn't integrated. You can do this by running the following command in a code cell:

```bash
pip install -r requirements.txt
```
### 2. Using Google Colab

If you are using **Google Colab**, you don't need to worry about the environment setup or installing the kernel. Just open the notebook and connect to the Colab runtime. But don't forget to have image_matching_algorithm.py script in your directory.

### 3. Install Jupyter Kernel (for Local Development)
```bash
pip install ipykernel
python -m ipykernel install --user --name <your_env_name> --display-name "Python (<your_env_name>)"
```

### 4. Launch Jupyter Notebook (for Local Development)

```bash
jupyter notebook
```
