# DermaDetectAI


## Pre-trained Models

The repository includes pre-trained models for skin disease detection:

1. **Model 1**: Detects 5 diseases. Trained on a ~69MB dataset with 98% validation accuracy.
2. **Model 2**: Detects 10 diseases. Trained on a ~2GB dataset with 85% validation accuracy.



### Install Dependencies

Each model has its own `requirements.txt` file. To install the dependencies for a specific model, navigate to the respective model directory and run:

```bash
pip install -r requirements.txt
```

### Running the Application

To start the Flask application for a specific model, navigate to its directory and execute:

```bash
python app.py
```

The Flask server will start, and you can access the application at `http://127.0.0.1:5000`. Use the web interface to upload an image and receive disease predictions.

## Using the Pre-trained Models

The pre-trained models are included in the repository, allowing you to use them directly without additional training.

## Training the Models

To train the models from scratch, navigate to the `src` directory of the respective model and run `main.py`. Ensure that you have the dataset in the appropriate directory and adjust the `num_classes` parameter according to your dataset's number of classes.

```bash
python src/main.py
```

## Project Structure

Here’s an overview of the project structure:

```
DermaDetectAI/
├── LICENSE
├── README.md
├── model-X/
│   ├── app.py
│   ├── models/
│   │   └── skin_disease_model.pth
│   ├── requirements.txt
│   ├── src/
│   │   └── main.py
│   ├── templates/
│   │   ├── result.html
│   │   └── upload.html
│   └── uploads/
│       └── [user_uploaded_files]
└── [other_files_and_directories]
```





