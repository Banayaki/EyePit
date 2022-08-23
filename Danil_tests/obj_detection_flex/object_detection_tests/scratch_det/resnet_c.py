activation = tf.nn.relu
decay_ssd = 0.9
decay_back = 0.997
eps = 1.001e-05
eps_resnet = 1.001e-05
# detector_bias=False
boxes_old = [
                (1, 1), (1, 2), (2, 1),
                (1.26, 1.26), (1.26, 2.52), (2.52, 1.26),
                (1.59, 1.59), (1.59, 3.17), (3.17, 1.59)
            ]

boxes = [
                (1.0, 1.0), (1.0, 2.0), (2.0, 1.0),
                (1.41, 1.41), (2.83, 1.41), (1.41, 2.83)
            ]

def mini_block(x, in_f, out_f, stage_id,unit_id,num_block):
    prefix = 'mini_stage' + str(stage_id) + '_unit' + str(unit_id) + '_'

    mx = ConvLayer(kw=3,kh=3,in_f=in_f,out_f=out_f,activation=None,
                                    padding='VALID',use_bias=False,name=prefix + 'conv1')(x)
                                                                                    
    mx = BatchNormLayer(D=out_f, eps=eps_resnet,name=prefix + 'bn2')(mx)
    
    mx = ActivationLayer(name=prefix + 'activation_1')(mx)
    return mx


def identity_block(x,in_f,stage_id,unit_id,num_block):                  
    prefix = 'stage' + str(stage_id) + '_unit' + str(unit_id) + '_'

    mx = BatchNormLayer(D=in_f, eps=eps_resnet, name=prefix + 'bn1')(x)
                                        
    mx = ActivationLayer(name=prefix + 'activation_1')(mx)

    mx = ZeroPaddingLayer(padding=[[1,1],[1,1]], name=prefix + 'zero_pad_1')(mx)

    mx = ConvLayer(kw=3,kh=3,in_f=in_f,out_f=in_f,activation=None,
                                    padding='VALID',use_bias=False,name=prefix + 'conv1')(mx)
                                                                                    
    mx = BatchNormLayer(D=in_f, eps=eps_resnet,name=prefix + 'bn2')(mx)
    
    mx = ActivationLayer(name=prefix + 'activation_2')(mx)

    mx = ZeroPaddingLayer(padding=[[1,1],[1,1]], name=prefix + 'zero_pad_2')(mx)

    mx = ConvLayer(kw=3,kh=3,in_f=in_f,out_f=in_f,activation=None,
                                    padding='VALID',use_bias=False,name=prefix + 'conv2')(mx)                        

    x = SumLayer(name='add' + str(num_block))([mx,x])

    return x

def conv_block(x,in_f,stage_id,unit_id,num_block,out_f=None,stride=2):
              
    prefix = 'stage' + str(stage_id) + '_unit' + str(unit_id) + '_'
    double_in_f = None
    if out_f is None:
        double_in_f = int(2*in_f)
    else:
        double_in_f = out_f

    x = BatchNormLayer(D=in_f, eps=eps_resnet, name=prefix + 'bn1')(x)
    x = ActivationLayer(name=prefix + 'activation_1')(x)

    mx = ZeroPaddingLayer(padding=[[1,1],[1,1]], name=prefix + 'zero_pad_1')(x)

    mx = ConvLayer(kw=3, kh=3, in_f=in_f, out_f=double_in_f, activation=None, stride=stride,
                                    padding='VALID', use_bias=False, name=prefix + 'conv1')(mx)
                                                                                
    mx = BatchNormLayer(D=double_in_f, eps=eps_resnet, name=prefix + 'bn2')(mx)
    mx = ActivationLayer(name=prefix + 'activation_2')(mx)

    mx = ZeroPaddingLayer(padding=[[1,1],[1,1]], name=prefix + 'zero_pad_2')(mx)
    mx = ConvLayer(kw=3, kh=3, in_f=double_in_f, out_f=double_in_f, activation=None,
                                    padding='VALID', use_bias=False, name=prefix + 'conv2')(mx)
                                                                                

    sx = ConvLayer(kw=1, kh=1, in_f=in_f, out_f=double_in_f, stride=stride,
                                    padding='VALID', activation=None, use_bias=False, name=prefix + 'sc/conv')(x)
                                                                               
    x = SumLayer(name='add' + str(num_block))([mx,sx])

    return x    


def create_resnet34(batch_size, picture_size=300):

    in_x = InputLayer(input_shape=[batch_size,picture_size,picture_size,3], name='Input')

    # STRIDE 1
    x = BatchNormLayer(D=3, eps=eps_resnet, use_gamma=False, name='bn_data')(in_x)
    x = mini_block(x=x, in_f=3, out_f=64,  stage_id=0, unit_id=1, num_block=1)
    x = mini_block(x=x, in_f=64, out_f=64,  stage_id=0, unit_id=2, num_block=2)
    # STRIDE 2
    x = MaxPoolLayer(name='max_pooling2d_first')(x)
    # size_picture = 150
    x = identity_block(x=x, in_f=64, stage_id=1, unit_id=1, num_block=3)
    x = identity_block(x=x, in_f=64, stage_id=1, unit_id=2, num_block=4)

    # STRIDE 4
    x = conv_block(x=x, in_f=64, stage_id=2, unit_id=1, num_block=5)
    # size_picture = 75
    x = identity_block(x=x, in_f=128, stage_id=2, unit_id=2, num_block=6)

    # STRIDE 8
    x = conv_block(x=x, in_f=128, stage_id=3, unit_id=1, num_block=7)
    # 38 out_f = 128
    x = identity_block(x=x, in_f=256, stage_id=3, unit_id=2, num_block=8)
    x_38 = x
    # STRIDE 16
    x = conv_block(x=x, in_f=256, stage_id=4, unit_id=1, num_block=9)
    # 19 out_f = 256
    x = identity_block(x=x, in_f=512, stage_id=4, unit_id=2, num_block=10)
    x_19 = x

    return in_x, x_38, x_19


in_x, x_38, x_19 = create_resnet34(32, 300)

# 38x38
x_38 = BatchNormLayer(D=256, eps=eps_resnet,name='end_bn1')(x_38)
dc1 = DetectorClassifier(
    reg_fms=x_38, rkw=3, rkh=3, rin_f=256,
    class_fms=x_38, ckw=3, ckh=3, cin_f=256, num_classes=len(num2name) + 1, 
    dboxes = boxes,name='dc1', use_reg_bias=detector_bias, use_class_bias=detector_bias
)

# 19x19
x_19 = BatchNormLayer(D=512, eps=eps_resnet,name='end_bn2')(x_19)
dc2 = DetectorClassifier(
    reg_fms=x_19, rkw=3, rkh=3, rin_f=512,
    class_fms=x_19, ckw=3, ckh=3, cin_f=512, num_classes=len(num2name) + 1, 
    dboxes = boxes, name='dc2', use_reg_bias=detector_bias, use_class_bias=detector_bias
)

x = conv_block(x=x_19, in_f=512, out_f=128, stage_id=6, unit_id=1, num_block=13)
x_b = BatchNormLayer(D=128, eps=eps_resnet,name='end_bn3')(x)

# 10x10
dc3 = DetectorClassifier(
    reg_fms=x_b, rkw=3, rkh=3, rin_f=128,
    class_fms=x_b, ckw=3, ckh=3, cin_f=128, num_classes=len(num2name) + 1, 
    dboxes = boxes,name='dc3', use_reg_bias=detector_bias, use_class_bias=detector_bias
)

x = conv_block(x=x, in_f=128, out_f=128, stage_id=7, unit_id=1, num_block=14)
x_b = BatchNormLayer(D=128, eps=eps_resnet,name='end_bn4')(x)

# 5x5
dc4 = DetectorClassifier(
    reg_fms=x_b, rkw=3, rkh=3, rin_f=128,
    class_fms=x_b, ckw=3, ckh=3, cin_f=128, num_classes=len(num2name) + 1, 
    dboxes = boxes,name='dc4', use_reg_bias=detector_bias, use_class_bias=detector_bias
)

x = conv_block(x=x, in_f=128, out_f=128, stage_id=8, unit_id=1, num_block=15)
x_b = BatchNormLayer(D=128, eps=eps_resnet,name='end_bn5')(x)
# 3x3
dc5 = DetectorClassifier(
    reg_fms=x_b, rkw=1, rkh=1, rin_f=128,
    class_fms=x_b, ckw=1, ckh=1, cin_f=128, num_classes=len(num2name) + 1, 
    dboxes=boxes,name='dc5', use_reg_bias=detector_bias, use_class_bias=detector_bias

)

x = conv_block(x=x, in_f=128, out_f=128, stage_id=9, unit_id=1, num_block=16)
x_b = BatchNormLayer(D=128, eps=eps_resnet,name='end_bn6')(x)

# 2x2
dc6 = DetectorClassifier(
    reg_fms=x_b, rkw=1, rkh=1, rin_f=128,
    class_fms=x_b, ckw=1, ckh=1, cin_f=128, num_classes=len(num2name) + 1, 
    dboxes=boxes,name='dc6', use_reg_bias=detector_bias, use_class_bias=detector_bias

)

x = conv_block(x=x, in_f=128, out_f=128, stage_id=10, unit_id=1, num_block=17)
x_b = BatchNormLayer(D=128, eps=eps_resnet,name='end_bn7')(x)

# 1x1
dc7 = DetectorClassifier(
    reg_fms=x_b, rkw=1, rkh=1, rin_f=128,
    class_fms=x_b, ckw=1, ckh=1, cin_f=128, num_classes=len(num2name) + 1, 
    dboxes = [(1.0,1.0),(0.7,0.7),(0.5, 0.5),(1.0,0.5),(0.5,1.0)] ,name='dc7', use_reg_bias=detector_bias, use_class_bias=detector_bias

)

model = SSDModel(dcs=[dc1, dc2, dc3, dc4, dc5, dc6, dc7],input_s=in_x,name='ScratchDet')