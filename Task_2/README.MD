# Task 2. Named entity recognition + image classification

In this task, the goal was to train two models: one for performing the NER task, and the other for image classification. After that, a pipeline was built that would accept two user inputs, text and the image itself, and provide an answer on whether the user is correct or not.

To accomplish this, the following libraries were used:
- **pandas**, **scikit-learn**, *Pillow**, **torchmetrics**, **numpy**, **Seaborn**: for data manipulation, visualization, and metrics application.
- **PyTorch**, **Transformers**: for training and downloading pre-trained models, like BERT and CNN ResNet-18.


Additionally, to train these models, the Animals-10 image dataset and the generated animals_ner_dataset via ChatGPT LLM were used. Just in case, I’ll attach the prompt here, but note that it is just a united prompt, which I used as separate parts for LLM:

**"Create a dataset for a Named Entity Recognition task that requires each sentence to be distinct and use a diverse vocabulary, ensuring that no sentence is overly similar to another. The sentences must be from 5 to 50 words in length. Some sentences should mention animals, and these mentions can vary in number: sometimes a sentence will include one animal, sometimes two or even three, but not every sentence must contain an animal. In addition, include some empty sentences. Make sure that the sentences introduce the diversity of context, like news, facts and so on. Do not always start any sentence with “The” or “A” immediately followed by an animal name. When tagging animal names with the B-ANIMAL label, only use names from the following set: dog, horse, elephant, butterfly, chicken, cat, cow, sheep, spider, and squirrel. Each sentence should be formatted with token-level labels, pairing each token with either "O" or "B-ANIMAL", and the dataset as a whole should reflect a rich mixture of styles and contexts while strictly adhering to these guidelines."**

Furthermore, the pre-trained architecture CNN ResNet-18 and BERT were used for image and text classification, respectively. Both networks showed good results after training. To view examples of their work, go to the Task_2_Demo notebook. For dataset analysis, you can view the EDA_Notebook.

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

