# Task 2. Named entity recognition + image classification

In this task, the goal was to train two models: one for performing the NER task, and the other for image classification. After that, a pipeline was built that would accept two user inputs, text and the image itself, and provide an answer on whether the user is correct or not.

To accomplish this, the following libraries were used:

Additionally, to train these models, the Animals-10 image dataset and the generated animals_ner_dataset via ChatGPT LLM were used. Just in case, I’ll attach the prompt here:

Furthermore, the pre-trained architecture CNN ResNet18 and BERT were used for image and text classification, respectively. Both networks showed good results after training. To view examples of their work, go to the Task_2_Demo notebook. For dataset analysis, you can view the EDA_Notebook.

Additionally, the same pipeline was built that accepts text and an image and outputs a list of boolean values corresponding to each animal name mentioned in the text.

## Environment Setup

### 1. Install Dependencies

If you're using **Google Colab**, most dependencies are already pre-installed. However, you may need to install additional libraries, such as torchvision. You can do this by running the following command in a code cell:

```bash
pip install -r requirements.txt
```
### 2. Using Google Colab

You can run the scripts manually, go to Task_2_Training_Notebook.ipynb to see CLI commands for each script.

If you are using **Google Colab**, you don't need to worry about the environment setup or installing the kernel. Just open the notebook and connect to the Colab runtime. But upload the dataset to the Colab directory if you want to run training script for BERT.

### 3. Install Jupyter Kernel (for Local Development)
```bash
pip install ipykernel
python -m ipykernel install --user --name <your_env_name> --display-name "Python (<your_env_name>)"
```

### 4. Launch Jupyter Notebook (for Local Development)

```bash
jupyter notebook
```

