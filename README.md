# AI-Face-Mask-Detector
COMP6721 project : AI Face Mask Detector
To install dependencies run:

`pip install -r requirements.txt`


# Instructions to train models:

1 - Clone the repository 
2 - Choose the CNN version to be trained i.e CNNV1, CNNV2, CNNV3 by instantiating the variable `model`
    `model = CNNV1(num_classes)` in train.py and run the file via the command `python app/train.py`

# Instructions to evaluate the models:

1 - Make sure to have the model directory present on the same level of app and data with the trained models present:
  - model/CNNV1.pb
  - model/CNNV2.pb
  - model/CNNV3.pb
2 - To test a single image place the image under data/single/single-image (make sure to have only one image there) then run eval.py and it will be the last statement of the output
