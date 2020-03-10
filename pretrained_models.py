
def get_model(model_name):

    if model_name == 'VGG16':
        from tensorflow.keras.applications.vgg16 import VGG16
        from tensorflow.keras.applications.vgg16 import preprocess_input, decode_predictions
        model = VGG16(weights='imagenet')
    if model_name == 'VGG19':
        from tensorflow.keras.applications.vgg19 import VGG19
        from tensorflow.keras.applications.vgg19 import preprocess_input, decode_predictions
        model = VGG19(weights='imagenet')
    elif model_name == 'ResNet50':
        from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
        from tensorflow.keras.applications.resnet50 import ResNet50
        model = ResNet50(weights='imagenet')
    elif model_name == 'InceptionV3':
        from tensorflow.keras.applications.inception_v3 import preprocess_input, decode_predictions
        from tensorflow.keras.applications.inception_v3 import InceptionV3
        model = InceptionV3(weights='imagenet')
    elif model_name == 'InceptionResNetV2':
        from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input, decode_predictions
        from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2
        model = InceptionResNetV2(weights='imagenet')
    elif model_name == 'Xception':
        from tensorflow.keras.applications.xception import preprocess_input, decode_predictions
        from tensorflow.keras.applications.xception import Xception
        model = Xception(weights='imagenet')
    elif model_name == 'MobileNet':
        from tensorflow.keras.applications.mobilenet import preprocess_input, decode_predictions
        from tensorflow.keras.applications.mobilenet import MobileNet
        model = MobileNet(weights='imagenet')
    elif model_name == 'MobileNetV2':
        from tensorflow.keras.applications.mobilenetv2 import preprocess_input, decode_predictions
        from tensorflow.keras.applications.mobilenetv2 import MobileNetV2
        model = MobileNetV2(weights='imagenet')
    elif model_name == 'DenseNet':
        from tensorflow.keras.applications.densenet import preprocess_input, decode_predictions
        from tensorflow.keras.applications.densenet import DenseNet121
        model = DenseNet121(weights='imagenet')
    elif model_name == 'NASNet':
        from tensorflow.keras.applications.nasnet import preprocess_input, decode_predictions
        from tensorflow.keras.applications.nasnet import NASNetMobile
        model = NASNetMobile(weights='imagenet')
    elif model_name == 'EfficientNet':
        from efficientnet.tfkeras import EfficientNetB0
        from keras.applications.imagenet_utils import decode_predictions
        from efficientnet.tfkeras import preprocess_input
        model = EfficientNetB7(weights='imagenet')
    else:
        print("[INFO] No model selected")
        

    return model, preprocess_input, decode_predictions