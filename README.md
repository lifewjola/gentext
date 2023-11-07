# GenText - Text Generation with TensorFlow

`GenText` is an open-source library that utilizes TensorFlow to generate text using Recurrent Neural Networks (RNNs). With `GenText`, you can create text-based models and generate text sequences based on a given input. 

## Sample Output
```
Talk to me. It gives my hands sliding my forehead again my walletting them. “Don’t forget the ex-wife is sitting
on track, Lucy. How long and Old-you know where a few weeks back.” “Oh, honey. I’ll never be a depart for the fact
that you like, being Finish dates and bring something.” Trombing up my legs and out of sight, I drinked. “You
because he’s a good thing.” Jason’s styma, revealing them walker. He does a shotted sigh and quickly very curve.
“Sophia, that’s what you think.” He leans his eyes to mine, looking up into his eyes. “I knew you’re not serious.”
Annie was trying to stay awake but he kept it had no longer hated that being out on the pork, just as I could. Wildly,
there his softness, over his. A flood of list on the center of the kiss, I knew he was looking dealing with a ceiling
and adventounce. Heat she shapes his hand and stagged the TVsolu box to pinch and grabbed his hand. “I’ve brought you
a few cream.” Josh digres and stuck them already black carted into
```

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)

## Installation

You can install `GenText` via pip:

```bash
pip install gentext
```

## Usage

### Read Your Data

```python
path_to_text_data = '/content/books.txt'  # replace with your own file path
text = open(path_to_text_data, 'rb').read().decode(encoding='utf-8')
```

In this step, you specify the path to your text data file, which should be a plain text file (.txt) containing the text you want to use for training and text generation.
You read the content of the file and decode it using UTF-8 encoding. The text variable will be used as the dataset for your text generation model.

### Create the text generation model

```python
model = BuildModel(text)
```
You create an instance of the BuildModel class, passing the text dataset as input. This class sets up the core components of your text generation model, including embedding layers, GRU units, and a dense layer.

### Instantiate the GenText class 
 
```python
gen = GenText()
```
Here, you create an instance of the GenText class, which provides high-level functionality for training and generating text. 
This class helps manage your text generation model and training process.

### Train the Model

```python
gen.train_model(model, EPOCHS=30)
```
You call the train_model method of the GenText class to train your text generation model. 
The model you created in the previous step is passed as an argument. 
You specify the number of training epochs (20 and above recommended). This step is essential to teach your model to generate text based on the provided dataset.

### Generate some text

```python
gen_model = OneStep(text, model)
input_text = 'Tal to me'
length = 1000
gen.generate_text(gen_model, input_text, length)
```

To generate text, you first create an instance of the OneStep class, which is designed for generating text based on the trained model. 
You provide an initial input text, in this case, 'I love,' and specify the desired length of the generated text (in this example, 1000 characters). 
The generate_text method generates text based on the provided input and length, and it returns the generated text.

## Did you notice?
- There was no text preprocessing step. Yep, that's right, the GenText class does all that heavy lifting for you so you can focus on 'building'.
- The output doesn't make a lot of sense but the model was able to identify real words and sentences.

## Pro Tips
- Use a T4 GPU or TPU for faster processing
- Use high quality data and decent EPOCHS for better output
- Tune the other paramters for better perfomance

## Extra Functions
These steps aren't compulsory when using the gentext library but they WILL be useful.

#### Recall that we instantiated the `GenText` class as `gen`

### Save the model

```python
gen.save_model(model)
```
After training your model, you can use the save_model method to save the model's weights. 
This allows you to persist the trained model for future use without the need to retrain it every time you want to generate text. 


### Load the model

```python
gen.load_model()
```
You can load the saved model using the load_model method. 
This step is useful when you want to use the pre-trained model for text generation without going through the training process again.

Happy generating :)

## Contributing

Contributions are welcome! If you would like to contribute to `GenText`, please follow the [contribution guidelines](CONTRIBUTING.md).

