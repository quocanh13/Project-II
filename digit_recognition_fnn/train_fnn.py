from PIL import Image
from digit_recognition_fnn.FNN import *
from digit_recognition_fnn.images.ImageDataset import ImageDataset
from digit_recognition_fnn.images.data import get_img_data, get_image_path
# python -m digit_recognition_fnn.train_fnn
nn = NeuralNetwork(
    layers=[
        (400, 10000, ActivationFunction.relu),
        (10, 400, ActivationFunction.copy)
    ], 
    loss_func=LossFunction.softmax_cross_entropy,
    output_func=ActivationFunction.softmax
)
LossFunction.class_weight = tc.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 8.0, 1.0, 1.0, 8.0, 1.0])

img_dataset = ImageDataset(
    img_data=get_img_data(), 
    img_encoder=ImageDataset.white_black_numpy_encoder, 
    flatten=True,
    rotate_info=(-20, 20, 5),
    translate_info=(0, 30, 5)
)

def main():
    nn.train(
        train_data=img_dataset, 
        epoch=3, 
        alpha=0.03, 
        batch_size=32, 
    )
    for i in range(0, 10):
        with Image.open(get_image_path(f"{i}", 10)) as img:
            res = nn.predict(ImageDataset.white_black_numpy_encoder(img).reshape(1, 10000))[0].tolist()
            max_prob = max(res)
            print(f"Number : {res.index(max_prob)} - {max_prob}")
            print(f"Prob : {res}")

if __name__ == "__main__":
    main()