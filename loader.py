#nltk.download('stopwords')
from nltk.corpus import stopwords
import string
import os
from torch.distributions import uniform
import torch
import numpy as np
import re
from pyvi import ViTokenizer
import string
from vncorenlp import VnCoreNLP
annotator = VnCoreNLP("/content/drive/My Drive/Sentiment-analysis/VnCoreNLP-master/VnCoreNLP-1.1.1.jar", annotators="wseg", max_heap_size='-Xmx500m') 

wrong_terms = {
    'ô kêi': ' ok ', 'okie': ' ok ', 'o kê': ' ok ', 'okey': ' ok ', 'ôkê': ' ok ', 'oki': ' ok ', 'oke': ' ok ',
    'okay': ' ok ', 'okê': ' ok ', 'ote': ' ok ',
    'kg ': u' không ', 'not': u' không ', u' kg ': u' không ', '"k ': u' không ', ' kh ': u' không ',
    'kô': u' không ', 'hok': u' không ', ' kp ': u' không phải ', u' kô ': u' không ', '"ko ': u' không ',
    ' cam ': ' camera ', ' cameera ': ' camera ', 'thuết kế': u'thiết_kế', 'ết_kế ': u' thiết_kế ',
    'gud': u' tốt ', 'god': u' tốt ', 'wel done': ' tốt ', 'good': u' tốt ', 'gút': u' tốt ',
    'sấu': u' xấu ', 'gut': u' tốt ', u' tot ': u' tốt ', u' nice ': u' tốt ', 'perfect': 'rất tốt',
    'bt': u' bình thường ','sp':u'sản phẩm ',
    ' m ': u' mình ', u' mik ': u' mình ', 'mìn': u'mình', u' mìnhh ': u' mình ', u' mềnh ': ' mình ',
    ' mk ': u' mình ', ' mik ': ' mình ',
    ' ko ': u' không ', u' k ': u' không ', 'khong': u' không ', u' hok ': u' không ',
    u' wá ': u' quá ', ' wa ': u' quá ', u'qá': u' quá ',
    ' cute ': 'dễ_thương',
    u' tẹc vời ': ' tuyệt_vời ', u'tiệc dời': ' tuyệt_vời ', u'tẹc zời': ' tuyệt_vời ',
    ' dc ': u' được ', u' đc ': u' được ', ' j ': ' gì ',
    ' màn hìn ': ' màn_hình ', u' màng hình ': u' màn_hình ', ' dt ': u' điện_thoại ',
    ' đt ': u' điện_thoại ',
    ' tet ': u' kiểm_tra ', ' test ': u' kiểm_tra ', ' tét ': u' kiểm_tra ',
    u' cẩm ': u' cầm ', u' cấm ': u' cầm ', u' sước ': u' xước ', u' xướt ': u' xước ',
    u'sài ': ' xài ',
    u' mựơt ': u' mượt ',
    u' sức sắc ': u' xuất_sắc ', u' xức sắc ': u' xuất_sắc ',
    ' fai ': u' phải ', u' fải ': u' phải ',
    u' bây_h ': u' bây_giờ ',
    u' mội ': u' mọi ', ' moi ': u' mọi ',
    u'ợc điểm ': u' nhược điểm ',
    u' sámsumg ': ' samsung ', ' sam ': ' samsung ', 'sam_sung ': ' samsung ',
    u' kbiết ': u' không_biết ', u' rất tiết ': u' rất_tiếc ', u' rất_tiết ': u' rất_tiếc ',
    u' rất tiêc ': u' rất_tiếc ',
    u' lát ': ' lag ', u' lác ': ' lag ', ' lat ': ' lag ', ' lac ': ' lag ', u' khựng ': ' lag ',
    u' giật ': ' lag ',
    u' văng ra ': ' lag ', u' đơ ': ' lag ', u' lắc ': ' lag ',
    u' film ': ' phim ', ' phin ': ' phim ', ' fim ': ' phim ',
    ' nhung ': u' nhưng ', u' ấu hình ': u' cấu_hình ',
    ' sd ': u' sử_dụng ', u' mài ': u' màu ', u' lấm ': u' lắm ',
    u' tôt ': ' tốt ', u' tôn ': u' tốt ', 'aple ': ' apple ', "gja": u" giá ", u"sục": u"sụt",
    u' âm_lượ ': u' âm_lượng ', u' thất_vọ ': u' thất_vọng ', u' dùg ': u' dùng ',
    u' bỗ ': u' bổ ',
    u' sụt ': u' tụt ', u' tuột ': u' tụt ', u' xuống ': u' tụt ',
    u'chíp ': ' chip ',
    ' bin ': ' pin '
}
emotion_icons = {
    "👹": "negative", "👻": "positive", "💃": "positive",'🤙': ' positive ', '👍': ' positive ',
    "💄": "positive", "💎": "positive", "💩": "positive","😕": "negative", "😱": "negative", "😸": "positive",
    "😾": "negative", "🚫": "negative",  "🤬": "negative","🧚": "positive", "🧡": "positive",'🐶':' positive ',
    '👎': ' negative ', '😣': ' negative ','✨': ' positive ', '❣': ' positive ','☀': ' positive ',
    '♥': ' positive ', '🤩': ' positive ', 'like': ' positive ', '💌': ' positive ',
    '🤣': ' positive ', '🖤': ' positive ', '🤤': ' positive ', ':(': ' negative ', '😢': ' negative ',
    '❤': ' positive ', '😍': ' positive ', '😘': ' positive ', '😪': ' negative ', '😊': ' positive ',
    '?': ' ? ', '😁': ' positive ', '💖': ' positive ', '😟': ' negative ', '😭': ' negative ',
    '💯': ' positive ', '💗': ' positive ', '♡': ' positive ', '💜': ' positive ', '🤗': ' positive ',
    '^^': ' positive ', '😨': ' negative ', '☺': ' positive ', '💋': ' positive ', '👌': ' positive ',
    '😖': ' negative ', '😀': ' positive ', ':((': ' negative ', '😡': ' negative ', '😠': ' negative ',
    '😒': ' negative ', '🙂': ' positive ', '😏': ' negative ', '😝': ' positive ', '😄': ' positive ',
    '😙': ' positive ', '😤': ' negative ', '😎': ' positive ', '😆': ' positive ', '💚': ' positive ',
    '✌': ' positive ', '💕': ' positive ', '😞': ' negative ', '😓': ' negative ', '️🆗️': ' positive ',
    '😉': ' positive ', '😂': ' positive ', ':v': '  positive ', '=))': '  positive ', '😋': ' positive ',
    '💓': ' positive ', '😐': ' negative ', ':3': ' positive ', '😫': ' negative ', '😥': ' negative ',
    '😃': ' positive ', '😬': ' 😬 ', '😌': ' 😌 ', '💛': ' positive ', '🤝': ' positive ', '🎈': ' positive ',
    '😗': ' positive ', '🤔': ' negative ', '😑': ' negative ', '🔥': ' negative ', '🙏': ' negative ',
    '🆗': ' positive ', '😻': ' positive ', '💙': ' positive ', '💟': ' positive ',
    '😚': ' positive ', '❌': ' negative ', '👏': ' positive ', ';)': ' positive ', '<3': ' positive ',
    '🌝': ' positive ',  '🌷': ' positive ', '🌸': ' positive ', '🌺': ' positive ',
    '🌼': ' positive ', '🍓': ' positive ', '🐅': ' positive ', '🐾': ' positive ', '👉': ' positive ',
    '💐': ' positive ', '💞': ' positive ', '💥': ' positive ', '💪': ' positive ','😌':'negative',
    '💰': ' positive ',  '😇': ' positive ', '😛': ' positive ', '😜': ' positive ',
    '🙃': ' positive ', '🤑': ' positive ', '🤪': ' positive ', '☹': ' negative ',  '💀': ' negative ',
    '😔': ' negative ', '😧': ' negative ', '😩': ' negative ', '😰': ' negative ', '😳': ' negative ',
    '😵': ' negative ', '😶': ' negative ', '🙁': ' negative ', ':))': ' positive ', ':)': ' positive ',
    'he he': ' positive ', 'hehe': ' positive ', 'hihi': ' positive ', 'haha': ' positive ',
    'hjhj': ' positive ', ' lol ': ' negative ', 'huhu': ' negative ', ' 4sao ': ' positive ', ' 5sao ': ' positive ',
    ' 1sao ': ' negative ', ' 2sao ': ' negative ',
    ': ) )': ' positive ', ' : ) ': ' positive ','🌟':'sao','🏻':"",
}
def preprecess_feature(sentence):
    # replace_wrong_terms
    for key, value in wrong_terms.items():
        sentence = sentence.replace(key, value)
    # replace_emotion_icons
    for key, value in emotion_icons.items():
        if sentence.find(key) >= 0:
            sentence = sentence.replace(key, value)
    # remove_repeated_characters
    sentence = re.sub(r'([A-Z])\1+', lambda m: m.group(1), sentence, flags=re.IGNORECASE)

    punctuation = """!"#$%&\'()*+,-./:;<=>?@[\\]^`{|}~"""
    translator = str.maketrans(punctuation, ' ' * len(punctuation))
    sentence = sentence.translate(translator)
    #sentence = annotator.tokenize(sentence)
    return sentence
def preprocess_data(lines):

    #lines = remove_special_text(lines)

    #print(lines)
    formatted_lines = []
    for tokens in lines:   
        tokens = preprecess_feature(tokens)
        #tokens = annotator.tokenize(tokens)
        #tokens_list = []
        #for sublist in tokens:
        #    for item in sublist:
        #        tokens_list.append(item)
        #print(tokens)
        tokens = tokens.split(" ")
        tokens_list = [word for word in tokens if len(word) > 1]
        #tokens = [word for word in tokens if word.isalpha()]
        stopwords = open("Vietnamese-stopword.txt", "r", encoding='utf-8')
        stop_words = set(stopwords.read().split())
        tokens = [w for w in tokens_list if w not in stop_words]
        tokens = u" ".join(tokens)
        formatted_lines.append(tokens)
    #print(len(formatted_lines))
    return formatted_lines


def load_data(dataset_path):
    """Load test, validation and train data with their labels"""
    x_train = []
    y_train = []
    x_val = []
    y_val = []
    x_test = []
    y_test = []

    with open(os.path.join(dataset_path, "train.txt"), 'r', encoding='utf-8') as f:
        for line in f.readlines():
            label = int(line[0])
            sentence = line[1:].strip()
            x_train.append(sentence)
            y_train.append(label)
    with open(os.path.join(dataset_path, "dev.txt"), 'r', encoding='utf-8') as f:
        for line in f.readlines():
            label = int(line[0])
            sentence = line[1:].strip()
            x_val.append(sentence)
            y_val.append(label)
    with open(os.path.join(dataset_path, "test.txt"), 'r', encoding='utf-8') as f:
        for line in f.readlines():
            label = int(line[0])
            sentence = line[1:].strip()
            x_test.append(sentence)
            y_test.append(label)
    #print(x_train.size(0),y_train.size(0))
    return x_train, x_val, x_test, y_train, y_val, y_test


def load_embedding(embed_path):
    """Load word embedding and return word-embedding vocabulary"""
    embedding2index = {}
    with open(embed_path, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            lexicons = line.split()
            #word = lexicons[0]
            word = ''.join(lexicons[:-300])
            #print(word)
            embedding2index[word] = torch.from_numpy(np.asarray(lexicons[-300:], dtype='float64'))
        embedding_size = len(lexicons) - 1
        #print(embedding_size, embedding2index)
    return embedding2index, embedding_size


def load_embedding_matrix(embedding, words, embedding_size):
    """Add new words in the embedding matrix and return it"""
    embedding_matrix = torch.zeros(len(words), embedding_size)
    for i, word in enumerate(words):
        # Note: PAD embedded as sequence of zeros
        if word not in embedding:
            if word != 'PAD':
                embedding_matrix[i] = uniform.Uniform(-0.25, 0.25).sample(torch.Size([embedding_size]))
        else:
            embedding_matrix[i] = embedding[word]
    return embedding_matrix


def get_loaders(x_train, x_val, x_test, y_train, y_val, y_test, batch_size, device):
    """Return iterables over train, validation and test dataset"""

    # convert labels to vectors and put on device
    y_train = torch.from_numpy(np.asarray(y_train,dtype='int32')).to(device)
    y_val = torch.from_numpy(np.asarray(y_val,dtype='int32')).to(device)
    y_test = torch.from_numpy(np.asarray(y_test,dtype='int32')).to(device)

    # convert sequences of indexes to tensors and put on device
    x_train = torch.from_numpy(np.asarray(x_train, dtype='int32')).to(device)
    x_val = torch.from_numpy(np.asarray(x_val, dtype='int32')).to(device)
    x_test = torch.from_numpy(np.asarray(x_test, dtype='int32')).to(device)

    train_array = torch.utils.data.TensorDataset(x_train, y_train)
    #print(x_train.size(0), y_train.size(0))
    train_loader = torch.utils.data.DataLoader(train_array, batch_size)

    val_array = torch.utils.data.TensorDataset(x_val, y_val)
    val_loader = torch.utils.data.DataLoader(val_array, batch_size)

    test_array = torch.utils.data.TensorDataset(x_test, y_test)
    test_loader = torch.utils.data.DataLoader(test_array, batch_size)
    #print(x_train.size(0),y_train.size(0))
    return train_loader, val_loader, test_loader