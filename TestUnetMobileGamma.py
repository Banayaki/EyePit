from makiflow.layers import *
from makiflow.models.segmentation.segmentator import Segmentator
import makiflow

import tensorflow as tf
import numpy as np
import glob
import cv2
import seaborn as sns
from tqdm import tqdm
from matplotlib import pyplot as plt
import random
from makiflow.models.classificator import Classificator

from sklearn.utils import shuffle
from scipy.ndimage import gaussian_filter
from makiflow.metrics import dice_coeff


layer_name = ['input',
            'Conv/weights',
            'Conv/BatchNorm',
            'Conv_relu',
            'expanded_conv/depthwise/depthwise_weights',
            'expanded_conv/depthwise/BatchNorm',
            'expanded_conv/depthsiwe_relu',
            'expanded_conv/project/weights',
            'expanded_conv/project/BatchNorm',
            'expanded_conv_1/expand/weights',
            'expanded_conv_1/expand/BatchNorm',
            'expanded_conv_1/expand_relu',
            'expanded_conv_1/depthwise/depthwise_weights',
            'expanded_conv_1/depthwise/BatchNorm',
            'expanded_conv_1/depthsiwe_relu',
            'expanded_conv_1/project/weights',
            'expanded_conv_1/project/BatchNorm',
            'expanded_conv_2/expand/weights',
            'expanded_conv_2/expand/BatchNorm',
            'expanded_conv_2/expand_relu',
            'expanded_conv_2/depthwise/depthwise_weights',
            'expanded_conv_2/depthwise/BatchNorm',
            'expanded_conv_2/depthsiwe_relu',
            'expanded_conv_2/project/weights',
            'expanded_conv_2/project/BatchNorm',
            'expanded_conv_2/add',
            'expanded_conv_3/expand/weights',
            'expanded_conv_3/expand/BatchNorm',
            'expanded_conv_3/expand_relu',
            'expanded_conv_3/depthwise/depthwise_weights',
            'expanded_conv_3/depthwise/BatchNorm',
            'expanded_conv_3/depthsiwe_relu',
            'expanded_conv_3/project/weights',
            'expanded_conv_3/project/BatchNorm',
            'expanded_conv_4/expand/weights',
            'expanded_conv_4/expand/BatchNorm',
            'expanded_conv_4/expand_relu',
            'expanded_conv_4/depthwise/depthwise_weights',
            'expanded_conv_4/depthwise/BatchNorm',
            'expanded_conv_4/depthsiwe_relu',
            'expanded_conv_4/project/weights',
            'expanded_conv_4/project/BatchNorm',
            'expanded_conv_4/add',
            'expanded_conv_5/expand/weights',
            'expanded_conv_5/expand/BatchNorm',
            'expanded_conv_5/expand_relu',
            'expanded_conv_5/depthwise/depthwise_weights',
            'expanded_conv_5/depthwise/BatchNorm',
            'expanded_conv_5/depthsiwe_relu',
            'expanded_conv_5/project/weights',
            'expanded_conv_5/project/BatchNorm',
            'expanded_conv_5/add',
            'expanded_conv_6/expand/weights',
            'expanded_conv_6/expand/BatchNorm',
            'expanded_conv_6/expand_relu',
            'expanded_conv_6/depthwise/depthwise_weights',
            'expanded_conv_6/depthwise/BatchNorm',
            'expanded_conv_6/depthsiwe_relu',
            'expanded_conv_6/project/weights',
            'expanded_conv_6/project/BatchNorm',
            'expanded_conv_7/expand/weights',
            'expanded_conv_7/expand/BatchNorm',
            'expanded_conv_7/expand_relu',
            'expanded_conv_7/depthwise/depthwise_weights',
            'expanded_conv_7/depthwise/BatchNorm',
            'expanded_conv_7/depthsiwe_relu',
            'expanded_conv_7/project/weights',
            'expanded_conv_7/project/BatchNorm',
            'expanded_conv_7/add',
            'expanded_conv_8/expand/weights',
            'expanded_conv_8/expand/BatchNorm',
            'expanded_conv_8/expand_relu',
            'expanded_conv_8/depthwise/depthwise_weights',
            'expanded_conv_8/depthwise/BatchNorm',
            'expanded_conv_8/depthsiwe_relu',
            'expanded_conv_8/project/weights',
            'expanded_conv_8/project/BatchNorm',
            'expanded_conv_8/add',
            'expanded_conv_9/expand/weights',
            'expanded_conv_9/expand/BatchNorm',
            'expanded_conv_9/expand_relu',
            'expanded_conv_9/depthwise/depthwise_weights',
            'expanded_conv_9/depthwise/BatchNorm',
            'expanded_conv_9/depthsiwe_relu',
            'expanded_conv_9/project/weights',
            'expanded_conv_9/project/BatchNorm',
            'expanded_conv_9/add',
            'expanded_conv_10/expand/weights',
            'expanded_conv_10/expand/BatchNorm',
            'expanded_conv_10/expand_relu',
            'expanded_conv_10/depthwise/depthwise_weights',
            'expanded_conv_10/depthwise/BatchNorm',
            'expanded_conv_10/depthsiwe_relu',
            'expanded_conv_10/project/weights',
            'expanded_conv_10/project/BatchNorm',
            'expanded_conv_11/expand/weights',
            'expanded_conv_11/expand/BatchNorm',
            'expanded_conv_11/expand_relu',
            'expanded_conv_11/depthwise/depthwise_weights',
            'expanded_conv_11/depthwise/BatchNorm',
            'expanded_conv_11/depthsiwe_relu',
            'expanded_conv_11/project/weights',
            'expanded_conv_11/project/BatchNorm',
            'expanded_conv_11/add',
            'expanded_conv_12/expand/weights',
            'expanded_conv_12/expand/BatchNorm',
            'expanded_conv_12/expand_relu',
            'expanded_conv_12/depthwise/depthwise_weights',
            'expanded_conv_12/depthwise/BatchNorm',
            'expanded_conv_12/depthsiwe_relu',
            'expanded_conv_12/project/weights',
            'expanded_conv_12/project/BatchNorm',
            'expanded_conv_12/add',
            'expanded_conv_13/expand/weights',
            'expanded_conv_13/expand/BatchNorm',
            'expanded_conv_13/expand_relu',
            'expanded_conv_13/depthwise/depthwise_weights',
            'expanded_conv_13/depthwise/BatchNorm',
            'expanded_conv_13/depthsiwe_relu',
            'expanded_conv_13/project/weights',
            'expanded_conv_13/project/BatchNorm',
            'expanded_conv_14/expand/weights',
            'expanded_conv_14/expand/BatchNorm',
            'expanded_conv_14/expand_relu',
            'expanded_conv_14/depthwise/depthwise_weights',
            'expanded_conv_14/depthwise/BatchNorm',
            'expanded_conv_14/depthsiwe_relu',
            'expanded_conv_14/project/weights',
            'expanded_conv_14/project/BatchNorm',
            'expanded_conv_14/add',
            'expanded_conv_15/expand/weights',
            'expanded_conv_15/expand/BatchNorm',
            'expanded_conv_15/expand_relu',
            'expanded_conv_15/depthwise/depthwise_weights',
            'expanded_conv_15/depthwise/BatchNorm',
            'expanded_conv_15/depthsiwe_relu',
            'expanded_conv_15/project/weights',
            'expanded_conv_15/project/BatchNorm',
            'expanded_conv_15/add',
            'expanded_conv_16/expand/weights',
            'expanded_conv_16/expand/BatchNorm',
            'expanded_conv_16/expand_relu',
            'expanded_conv_16/depthwise/depthwise_weights',
            'expanded_conv_16/depthwise/BatchNorm',
            'expanded_conv_16/depthsiwe_relu',
            'expanded_conv_16/project/weights',
            'expanded_conv_16/project/BatchNorm',
            'Conv_1/weights',
            'Conv_1/BatchNorm',
            'out_relu',
]


def elastic_transform(image, alpha, sigma, alpha_affine, random_state=None):
    """Elastic deformation of images as described in [Simard2003]_ (with modifications).
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
         Convolutional Neural Networks applied to Visual Document Analysis", in
         Proc. of the International Conference on Document Analysis and
         Recognition, 2003.

     Based on https://gist.github.com/erniejunior/601cdf56d2b424757de5
    """
    if random_state is None:
        random_state = np.random.RandomState(None)

    shape = image.shape
    shape_size = shape[:2]

    # Random affine
    center_square = np.float32(shape_size) // 2
    square_size = min(shape_size) // 3
    pts1 = np.float32([center_square + square_size, [center_square[0] + square_size, center_square[1] - square_size],
                       center_square - square_size])
    pts2 = pts1 + random_state.uniform(-alpha_affine, alpha_affine, size=pts1.shape).astype(np.float32)
    M = cv2.getAffineTransform(pts1, pts2)
    image = cv2.warpAffine(image, M, shape_size[::-1], borderMode=cv2.BORDER_REFLECT_101,
                           flags=cv2.INTER_NEAREST)

    # *shape = height, width
    dx = gaussian_filter((random_state.rand(shape_size[0], shape_size[1]) * 2 - 1), sigma, mode='nearest') * alpha
    dy = gaussian_filter((random_state.rand(shape_size[0], shape_size[1]) * 2 - 1), sigma, mode='nearest') * alpha

    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    mapx = np.float32(x + dx)
    mapy = np.float32(y + dy)
    return cv2.remap(image, mapx, mapy, cv2.INTER_NEAREST, borderMode=cv2.BORDER_REFLECT101)


def augment_data(images, masks, num_pos):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cols, rows = np.array(images.shape[1:3])

    augmented_images = []
    augmented_masks = []
    aug_num_pos = []

    for index, (image, mask) in tqdm(enumerate(zip(images, masks))):
        for i in range(3):
            image[:, :, i] = clahe.apply(image[:, :, i])
        
        for i in range(150):
            flip_param = random.randint(-1, 2)
            rotate_param = random.randint(0, 360)
            scale_param = np.random.random() + 1 # [1, 2)
            M = cv2.getRotationMatrix2D((cols / 2, rows / 2), rotate_param, scale_param)

            if flip_param != 2:
                img_mask = cv2.flip(mask, flip_param)
                img_target = cv2.flip(image, flip_param)
            else:
                img_mask = mask
                img_target = image

            img_mask = cv2.warpAffine(img_mask, M, (img_mask.shape[:2]), borderMode=cv2.BORDER_REFLECT101, flags=cv2.INTER_NEAREST)
            img_target = cv2.warpAffine(img_target, M, (img_target.shape[:2]), borderMode=cv2.BORDER_REFLECT101, flags=cv2.INTER_NEAREST)

            img_merge = np.concatenate((img_target[...], img_mask[...]), axis=2)

            img_merge_p = elastic_transform(img_merge, img_merge.shape[1] * 2, img_merge.shape[1] * 0.1, img_merge.shape[1] * 0.1)
            img_target = img_merge_p[..., :3]
            img_mask = img_merge_p[..., 3:]

            augmented_images.append(img_target)
            augmented_masks.append(cv2.cvtColor(img_mask, cv2.COLOR_BGR2GRAY))
            aug_num_pos.append(num_pos[index])
    
    return augmented_images, augmented_masks, aug_num_pos


def load_data(resize_into=256):
    Xtrain = []
    Ytrain = []
    Xtest = []
    Ytest = []
    num_positives = []

    path_to_salt = './dataset'
    path_to_images = f'{path_to_salt}/imgs'
    path_to_masks = f'{path_to_salt}/mask'
    imgs = glob.glob(f'{path_to_images}/*')
    masks = glob.glob(f'{path_to_masks}/*')
    last_index = 0
    for i in tqdm(range(len(imgs))):
        mask_name = masks[i]
        img_name = mask_name.replace('mask', 'imgs').replace('_m', '')

        img = cv2.imread(img_name)
        img = cv2.resize(img, (resize_into, resize_into), interpolation=cv2.INTER_NEAREST)
        mask = cv2.imread(mask_name)
        mask = cv2.resize(mask, (resize_into, resize_into), interpolation=cv2.INTER_NEAREST)

        Xtrain.append(img)
        Ytrain.append(mask)

        true_map = np.array(cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)) == 0
        num_positives.append(resize_into*resize_into - true_map.sum())

    return np.array(Xtrain, dtype=np.uint8), np.array(Ytrain, dtype=np.uint8) // 10, num_positives


def make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def inverted_res_block(inputs, in_f,out_f, alpha, stride,expansion, block_id):
    global layer_name
    x = inputs
    pointwise_conv_filters = int(out_f*alpha)
    pointwise_f = make_divisible(pointwise_conv_filters,8)
    prefix = 'expanded_conv_{}/'.format(block_id)
    exp_f = expansion * in_f
    if block_id:#check it's zero id or not
        # Expand
        x = ConvLayer(kw=1, 
                    kh=1,
                    in_f=in_f,
                    out_f=exp_f, 
                    name=prefix + 'expand/weights',
                    stride=1,
                    use_bias=False,
                    padding='SAME',
                    activation=None)(x)

        x = BatchNormLayer(D=exp_f,name=prefix+'expand/BatchNorm')(x)

        x = ActivationLayer(activation=tf.nn.relu6,name=prefix+'expand_relu')(x)
    else:
        prefix = 'expanded_conv/'

    # Depthwise

    x = DepthWiseConvLayer(kw=3,
                        kh=3,
                        in_f=exp_f,
                        multiplier = 1,
                        activation=None,
                        stride=stride,
                        padding='SAME',
                        use_bias=False,
                        name=prefix + 'depthwise/depthwise_weights')(x)

    x = BatchNormLayer(D=exp_f,name=prefix+'depthwise/BatchNorm')(x)

    x = ActivationLayer(activation=tf.nn.relu6,name=prefix+'depthsiwe_relu')(x)

    # Project
    x = ConvLayer(kw=1,
                kh=1,
                in_f=exp_f,
                out_f=pointwise_f,
                stride=1,
                padding='SAME',
                use_bias=False,
                activation=None,
                name=prefix+'project/weights')(x)
    x = BatchNormLayer(D=pointwise_f,name=prefix+'project/BatchNorm')(x)


    if stride == 1 and in_f == pointwise_f:
        return SumLayer(name=prefix+'add')([inputs,x]),pointwise_f
    else:
        return x,pointwise_f
    

def upconv(input_tensor, in_f, out_f, stride, block_id, concat_tensor=None, dilation=1):
    tensor = input_tensor
    if concat_tensor is not None:
        # tensor = UpConvLayer(kh=2, kw=2, in_f=new_in_f, out_f=filters, name=f'UpconvBlock_{block_id}')(tensor)
        tensor = UpSamplingLayer(name=f'UpSampling_{block_id}')(tensor)

        # Ugly solution for input shape=(401,401,3)
        # if block_id == 25:
            # tensor = Lambda(lambda x: x[:, :-1, :-1, :])(tensor)

        tensor = ConcatLayer(name=f'concat_{block_id}')([tensor, concat_tensor])
    dec_conv, in_new_f = inverted_res_block(tensor, in_f=tensor.get_shape()[-1], out_f=out_f, alpha=1, stride=stride, 
                                            expansion=6, block_id=block_id)
    return dec_conv, in_new_f


def get_MobileNetV2_1_224(picture_size=256 ,batch_size=64, include_top=True, num_classes=10, build_Classificator=False):
    # If build_Classificator is False, method return input as MakiTensor and output as MakiTensor
    # NOTICE checkpoint have "prediction" layer at the end for 1001 classes
    alpha = 1
    first_filt = make_divisible(32 * alpha, 8)

    in_x = InputLayer(input_shape=[batch_size ,picture_size, picture_size,3],name='input')
    x = ConvLayer(kw=3, kh=3, in_f=3, out_f=first_filt, stride=2, padding='SAME', activation=None, use_bias=False, name='Conv/weights')(in_x)

    #128
    x = BatchNormLayer(D=first_filt, name='Conv/BatchNorm')(x)
    x = ActivationLayer(activation=tf.nn.relu6, name='Conv_relu')(x)

    enc_1, in_new_f = inverted_res_block(inputs=x, in_f=first_filt, out_f=16, alpha=alpha, stride=1, expansion=1, block_id=0)
    enc_2, in_new_f = inverted_res_block(inputs=enc_1, in_f=in_new_f,out_f=24, alpha=alpha, stride=2, expansion=6, block_id=1)
    #64
    enc_3, in_new_f = inverted_res_block(inputs=enc_2, in_f=in_new_f, out_f=24, alpha=alpha, stride=1, expansion=6,block_id=2)
    enc_4, in_new_f = inverted_res_block(inputs=enc_3, in_f=in_new_f, out_f=32, alpha=alpha, stride=2, expansion=6,block_id=3)
    #32
    enc_5, in_new_f = inverted_res_block(inputs=enc_4, in_f=in_new_f, out_f=32, alpha=alpha, stride=1, expansion=6,block_id=4)
    enc_6, in_new_f = inverted_res_block(inputs=enc_5, in_f=in_new_f, out_f=32, alpha=alpha, stride=1, expansion=6,block_id=5)
    enc_7, in_new_f = inverted_res_block(inputs=enc_6, in_f=in_new_f, out_f=64, alpha=alpha, stride=2, expansion=6,block_id=6)
    #, 16
    enc_8, in_new_f = inverted_res_block(inputs=enc_7, in_f=in_new_f, out_f=64, alpha=alpha, stride=1, expansion=6,block_id=7)
    enc_9, in_new_f = inverted_res_block(inputs=enc_8, in_f=in_new_f, out_f=64, alpha=alpha, stride=1, expansion=6,block_id=8)
    enc_10, in_new_f = inverted_res_block(inputs=enc_9, in_f=in_new_f, out_f=64, alpha=alpha, stride=1, expansion=6,block_id=9)
    enc_11, in_new_f = inverted_res_block(inputs=enc_10, in_f=in_new_f, out_f=96, alpha=alpha, stride=1, expansion=6,block_id=10)
    #, 
    enc_12, in_new_f = inverted_res_block(inputs=enc_11, in_f=in_new_f, out_f=96, alpha=alpha, stride=1, expansion=6,block_id=11)
    enc_13, in_new_f = inverted_res_block(inputs=enc_12, in_f=in_new_f, out_f=96, alpha=alpha, stride=1, expansion=6,block_id=12)
    enc_14, in_new_f = inverted_res_block(inputs=enc_13, in_f=in_new_f, out_f=160, alpha=alpha, stride=2, expansion=6,block_id=13)
    #, 8
    enc_15, in_new_f = inverted_res_block(inputs=enc_14, in_f=in_new_f, out_f=160, alpha=alpha, stride=1, expansion=6,block_id=14)
    enc_16, in_new_f = inverted_res_block(inputs=enc_15, in_f=in_new_f, out_f=160, alpha=alpha, stride=1, expansion=6,block_id=15)
    enc_17, in_new_f = inverted_res_block(inputs=enc_16, in_f=in_new_f, out_f=320, alpha=alpha, stride=1, expansion=6,block_id=16)
    # UP

    x = ConvLayer(kw=1, kh=1, in_f=in_new_f, out_f=1280, use_bias=False, activation=None, name='Conv_1/weights')(enc_17)
    x = BatchNormLayer(D=1280, name='Conv_1/BatchNorm')(x)
    x = ActivationLayer(activation=tf.nn.relu6, name='out_relu')(x)

    bottleneck, in_new_f = inverted_res_block(inputs=x, in_f=1280, out_f=160, alpha=alpha, stride=1, expansion=6, block_id=17)
    # 

    upconv1, in_new_f = upconv(input_tensor=bottleneck, in_f=in_new_f, out_f=160, stride=1, block_id=18)
    upconv2, in_new_f = upconv(input_tensor=upconv1, in_f=in_new_f, out_f=160, stride=1, block_id=19)
    upconv3, in_new_f = upconv(input_tensor=upconv2, in_f=in_new_f, out_f=160, stride=1, block_id=20)

    upconv4, in_new_f = upconv(input_tensor=upconv3, in_f=in_new_f, out_f=96, stride=1, block_id=21)
    upconv5, in_new_f = upconv(input_tensor=upconv4, in_f=in_new_f, out_f=96, stride=1, block_id=22)
    upconv6, in_new_f = upconv(input_tensor=upconv5, in_f=in_new_f, out_f=96, stride=1, block_id=23)
    upconv7, in_new_f = upconv(input_tensor=upconv6, in_f=in_new_f, out_f=64, stride=1, block_id=24)
    upconv8, in_new_f = upconv(input_tensor=upconv7, in_f=in_new_f, out_f=64, stride=1, block_id=25, concat_tensor=enc_9)

    upconv9, in_new_f = upconv(input_tensor=upconv8, in_f=in_new_f, out_f=64, stride=1, block_id=26)
    upconv10, in_new_f = upconv(input_tensor=upconv9, in_f=in_new_f, out_f=64, stride=1, block_id=27)
    upconv11, in_new_f = upconv(input_tensor=upconv10, in_f=in_new_f, out_f=64, stride=1, block_id=28, concat_tensor=enc_6)

    upconv12, in_new_f = upconv(input_tensor=upconv11, in_f=in_new_f, out_f=32, stride=1, block_id=29)
    upconv13, in_new_f = upconv(input_tensor=upconv12, in_f=in_new_f, out_f=32, stride=1, block_id=30)
    upconv14, in_new_f = upconv(input_tensor=upconv13, in_f=in_new_f, out_f=32, stride=1, block_id=31, concat_tensor=enc_3)

    upconv15, in_new_f = upconv(input_tensor=upconv14, in_f=in_new_f, out_f=24, stride=1, block_id=32)
    upconv16, in_new_f = upconv(input_tensor=upconv15, in_f=in_new_f, out_f=24, stride=1, block_id=33, concat_tensor=enc_1)

    upconv17, in_new_f = upconv(input_tensor=upconv16, in_f=in_new_f, out_f=16, stride=1, block_id=34)

    upconv18, in_new_f = upconv(input_tensor=upconv17, in_f=in_new_f, out_f=16, stride=1, block_id=35, concat_tensor=in_x)

    output_x = ConvLayer(kw=1, kh=1, in_f=16, out_f=10, name='output')(upconv18)

    return in_x, output_x


if __name__ == "__main__":
    gamma_list = [0.5, 0.75, 1.0, 1.5, 2.0]
    
    makiflow.set_main_gpu(1)
    Xtrain, Ytrain, num_pos = load_data()
    
    Xtest = Xtrain[-5:]
    Ytest = Ytrain[-5:]
    num_pos_test = num_pos[-5:]
    
#     15
    Xtrain, Ytrain, num_pos = augment_data(Xtrain[:-5], Ytrain[:-5], num_pos[:-5])
    Xtrain, Ytrain, num_pos = shuffle(Xtrain, Ytrain, num_pos)
    
    Xtest, Ytest, num_pos_test = augment_data(Xtest, Ytest, num_pos_test)
    Xtest, Ytest, num_pos_test = shuffle(Xtest, Ytest, num_pos_test)
    
    Xtrain = np.array(Xtrain, dtype=np.float32) / 255
    Ytrain = np.array(Ytrain, dtype=np.uint8)
    Xtest = np.array(Xtest, dtype=np.float32) / 255
    Ytest = np.array(Ytest, dtype=np.uint8)
    
    for img_index in range(5):
        sns.heatmap(Ytest[img_index]).get_figure().savefig(f'result/test_{img_index}.png')
        plt.clf()
    
    in_x, output_x = get_MobileNetV2_1_224(batch_size=32)
    model = Segmentator(input_s=in_x, output=output_x)
    untrainable = [(name, False) for name in layer_name]
    model.set_layers_trainable(untrainable)
    model.set_session(tf.Session())
    
    model.load_weights('weights/weights_30.ckpt', layer_name)
    
    optimizer = tf.train.AdamOptimizer(learning_rate=10e-4)
    
    with open('test.txt', 'w') as test_out:
        for gamma in gamma_list:
            model.set_session(tf.Session())
            model.load_weights('weights/weights_30.ckpt', layer_name)
            optimizer = tf.train.AdamOptimizer(learning_rate=10e-4)

            loss_list = []
            iou_list = []
            for i in range(10):
                loss_list += model.fit_focal(images=Xtrain, labels=Ytrain, gamma=gamma, num_positives=num_pos, optimizer=optimizer, epochs=5)
                pred = model.predict(Xtest[:32])
                iou_list += dice_coeff(Ytest[:32], pred, 10, True)
                for img_index in range(5):
                    sns.heatmap(pred[img_index].argmax(axis=2)).get_figure().savefig(f'result/gamma={gamma}_epochs={5 * (i+1)}_{img_index}.png')
                    plt.clf()
            test_out.write(str(loss_list))
            test_out.write(str(iou_list))
