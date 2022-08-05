import numpy as np
import torch
import torch.nn as nn

from PIL import Image
import torchvision.transforms as transforms
import streamlit as st
from PIL import Image
from io import BytesIO
import numpy as np
import matplotlib.pyplot as plt
import base64


st.set_option('deprecation.showfileUploaderEncoding', False)

st.set_page_config(
    page_title="Informative Drawings", layout="wide", page_icon="./images/icon.png"
)

norm_layer = nn.InstanceNorm2d


# def get_image_download_link(img):
#     buffered = BytesIO()
#     img.save(buffered, format="JPEG")
#     img_str = base64.b64encode(buffered.getvalue()).decode()
#     href = f'<a href="data:image/jpg;base64,{img_str}" target="_blank">Download result</a>'
#     return href


def pil_to_bytes(model_output):
    pil_image = Image.fromarray(np.squeeze(model_output).astype(np.uint8))
    buffer = BytesIO()
    pil_image.save(buffer, format="PNG")
    byte_image = buffer.getvalue()
    return byte_image


class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        conv_block = [nn.ReflectionPad2d(1),
                      nn.Conv2d(in_features, in_features, 3),
                      norm_layer(in_features),
                      nn.ReLU(inplace=True),
                      nn.ReflectionPad2d(1),
                      nn.Conv2d(in_features, in_features, 3),
                      norm_layer(in_features)
                      ]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)


class Generator(nn.Module):
    def __init__(self, input_nc, output_nc, n_residual_blocks=9, sigmoid=True):
        super(Generator, self).__init__()

        # Initial convolution block
        model0 = [nn.ReflectionPad2d(3),
                  nn.Conv2d(input_nc, 64, 7),
                  norm_layer(64),
                  nn.ReLU(inplace=True)]
        self.model0 = nn.Sequential(*model0)

        # Downsampling
        model1 = []
        in_features = 64
        out_features = in_features * 2
        for _ in range(2):
            model1 += [nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                       norm_layer(out_features),
                       nn.ReLU(inplace=True)]
            in_features = out_features
            out_features = in_features * 2
        self.model1 = nn.Sequential(*model1)

        model2 = []
        # Residual blocks
        for _ in range(n_residual_blocks):
            model2 += [ResidualBlock(in_features)]
        self.model2 = nn.Sequential(*model2)

        # Upsampling
        model3 = []
        out_features = in_features // 2
        for _ in range(2):
            model3 += [nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
                       norm_layer(out_features),
                       nn.ReLU(inplace=True)]
            in_features = out_features
            out_features = in_features // 2
        self.model3 = nn.Sequential(*model3)

        # Output layer
        model4 = [nn.ReflectionPad2d(3),
                  nn.Conv2d(64, output_nc, 7)]
        if sigmoid:
            model4 += [nn.Sigmoid()]

        self.model4 = nn.Sequential(*model4)

    def forward(self, x, cond=None):
        out = self.model0(x)
        out = self.model1(out)
        out = self.model2(out)
        out = self.model3(out)
        out = self.model4(out)

        return out


model1 = Generator(3, 1, 3)
model1.load_state_dict(torch.load('model.pth', map_location=torch.device('cpu')))
model1.eval()

model2 = Generator(3, 1, 3)
model2.load_state_dict(torch.load('model2.pth', map_location=torch.device('cpu')))
model2.eval()


def predict(input_img, ver):
    input_img = Image.open(input_img)
    transform = transforms.Compose([transforms.Resize(512, Image.BICUBIC), transforms.ToTensor()])
    input_img = transform(input_img)
    input_img = torch.unsqueeze(input_img, 0)
    drawing = 0
    with torch.no_grad():
        if ver == 'style 2':
            drawing = model2(input_img)[0].detach()
        else:
            drawing = model1(input_img)[0].detach()
    return drawing


st.sidebar.title("Informative Drawings")
# st.sidebar.markdown("Neural style transfer is an optimization technique used to take two images:</br>- **Content image** </br>- **Style reference image** (such as an artwork by a famous painter)</br>Blend them together so the output image looks like the content image, but “painted” in the style of the style reference image.", unsafe_allow_html=True)

content_image_buffer = st.sidebar.file_uploader("upload content image", type=["png", "jpeg", "jpg"],
                                                accept_multiple_files=False, key=None, help="content image")

style = st.sidebar.radio(
    "select style type",
    ('anime_style','style 2'))

col1, col2 = st.columns(2)



with st.spinner("Loading Input image.."):
    if content_image_buffer:
        col1.header("Input  Image")
        col1.image(content_image_buffer, use_column_width=True)
        content_img_size = (500, 500)
        content_image = content_image_buffer

if st.sidebar.button(label="Generate"):

    if content_image_buffer and style:
        with st.spinner('Generating Stylized image ...'):
            stylized_image = predict(content_image, style)
            stylized_image = transforms.ToPILImage()(stylized_image)
            col2.header("output Image")
            col2.image(np.array(stylized_image))
            st.download_button(label="Download result", data=pil_to_bytes(stylized_image),
                               file_name="stylized_image.png", mime="image/png")

    else:
        st.sidebar.markdown("Please chose content and style pictures.")
