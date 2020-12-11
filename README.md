# Fine-Grained Sentiment Analysis for Vietnamese social media.
Code model CNN được clone từ https://github.com/davide97l/Sentiment-analysis có chỉnh sửa và thêm một số file để support project.
## Dataset preparation
Dataset được chia thành 3 tập : `train.txt`, `dev.txt`, `test.txt`, mỗi tập chứa các sample như sau.
```
  nhãn_1 câu1
  nhãn_2 câu2
  ...
  nhãn_N câu_N
```
## Data preprocessing
- Chuẩn hóa tiếng Việt, xử lý emoj, chuẩn hóa tiếng Anh, thuật ngữ
- Chuyển thành chữ thường
- Loại bỏ các ký tự kéo dài: vd: đẹppppppp
- Loại bỏ stopwords
- Chuẩn hóa 1 số sentiment words/English words

Xem chi tiết file `loader.py`.

## Pretrained embedding
Sử dụng [**Phoword2Vec**](https://github.com/datquocnguyen/PhoW2V) pretrained model từ VinAI Research. Tải về giải nén và để vào thư mục /embed.


## CNN (Convolutional Neural Network)
Huấn luyện mạng Convolutional Neural Network.
```
  python main.py -m "cnn" -dp "dataset/sst5"
```
Huấn luyện mạng Convolutional Neural Network với dropout và embedding dropout.
```
  python main.py -m "cnn" -dp "dataset/sst5" -dr 0.2 -ed 0.2
```
Sử dụng pretrained embedding.
```
  python main.py -m "cnn" -dp "dataset/sst5" -ep "embed/word2vec_vi_syllables_300dims.txt"
```

