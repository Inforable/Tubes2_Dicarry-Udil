# IF3270 Tubes 2 - Dicarry-Udil

## Deskripsi Singkat
Repositori ini berisi implementasi **Convolutional Neural Network (CNN)**, **Simple Recurrent Neural Network (RNN)**, dan **Long Short-Term Memory (LSTM)** yang dibangun dari awal (*from scratch*) menggunakan library **NumPy**. 

Proyek ini terbagi menjadi dua bagian utama:
1. **Image Classification (CNN)**: Melakukan klasifikasi gambar menggunakan dataset Intel Image Classification.
2. **Image Captioning (RNN & LSTM)**: Menggunakan arsitektur Encoder-Decoder (Pretrained CNN sebagai encoder, dan RNN/LSTM *from scratch* sebagai decoder dengan *pre-inject method*) untuk menghasilkan deskripsi teks dari sebuah gambar menggunakan dataset Flickr8k.

---

## Cara Setup dan Run Program

### 1. Persiapan Environment
Pastikan Anda menggunakan Python 3.x. Disarankan untuk menggunakan virtual environment.
```bash
# Clone the repository
git clone <repository-url>
cd Tubes2_Dicarry-Udil

# Create and activate virtual environment (macOS/Linux)
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Dataset Preparation

#### Bagian CNN (Intel Image Classification)
1. Download dataset dari [Kaggle](https://www.kaggle.com/datasets/puneet6060/intel-image-classification).
2. Pindahkan file `archive.zip` hasil unduhan ke dalam folder `data/intel/`.
3. Jalankan script setup untuk mengekstrak dan membagi data:
   ```bash
   python3 setup_intel.py
   ```

#### Bagian RNN & LSTM (Flickr8k Image Captioning)
1. Download dataset dari [Kaggle](https://www.kaggle.com/datasets/adityajn105/flickr8k).
2. Pindahkan file `archive.zip` hasil unduhan ke dalam folder `data/flickr8k/`.
3. Jalankan script setup untuk mengekstrak dataset dan membersihkan struktur folder:
   ```bash
   python3 setup_flickr8k.py
   ```

### 3. Cara Menjalankan Program

#### Untuk Task CNN (Image Classification)
1. Buka Jupyter Notebook: `jupyter notebook`
2. Jalankan `src/cnn/train.ipynb` untuk melatih berbagai konfigurasi CNN menggunakan Keras dan menyimpan bobot model.
3. Jalankan `src/cnn/evaluate.ipynb` untuk mengevaluasi performa model *from scratch* dengan memuat bobot dari model Keras.

#### Untuk Task RNN & LSTM (Image Captioning)
**Penting:** Anda harus melakukan ekstraksi fitur gambar dan preprocessing teks terlebih dahulu.
1. Ekstraksi fitur CNN dari gambar (hanya dijalankan sekali):
   ```bash
   python3 src/captioning/common/feature_extraction.py
   ```
2. Tokenisasi dan preprocessing caption:
   ```bash
   python3 src/captioning/common/preprocessing.py
   ```
3. Buka Jupyter Notebook: `jupyter notebook`
4. Untuk **RNN**: 
   - Jalankan `src/captioning/rnn/train.ipynb` untuk melatih 6 variasi arsitektur RNN menggunakan Keras.
   - Jalankan `src/captioning/rnn/evaluate.ipynb` untuk menghitung skor BLEU-4 dan METEOR menggunakan arsitektur *from scratch*.
5. Untuk **LSTM**:
   - Jalankan `src/captioning/lstm/train.ipynb` untuk melatih variasi arsitektur LSTM.
   - Jalankan `src/captioning/lstm/evaluate.ipynb` untuk evaluasi dan analisis perbandingan Keras vs Scratch dan RNN vs LSTM.

---

## Pembagian Tugas

- **NIM: 13523156**
  - Mengimplementasikan arsitektur Dense, LSTM Cell, dan *Dense Projection* *from scratch*.
  - Melakukan pelatihan LSTM, analisis perbandingan Keras vs Scratch (LSTM), dan analisis komparatif performa/kualitas *caption* antara arsitektur RNN dan LSTM.
  - Mengimplementasikan bonus Beam Search Decoder.

- **NIM: 13523160**
  - Membangun pipeline ekstraksi fitur gambar menggunakan pre-trained CNN (InceptionV3).
  - Mengimplementasikan fungsionalitas preprocessing caption (Tokenisasi, Padding).
  - Membangun arsitektur *Embedding Layer* dan *Simple RNN* *from scratch* secara modular dengan dukungan variasi *num_layers* dan *hidden_dim*.
  - Melakukan eksperimen, pelatihan, perhitungan metrik (BLEU-4, METEOR), dan visualisasi *loss* untuk pipeline *Image Captioning* berbasis RNN.

- **NIM: 18223121**
  - Mengimplementasikan seluruh modul CNN *from scratch* (Conv2D, LocallyConnected2D, Pooling, Flatten, Activations).
  - Mengimplementasikan *utility functions* (Image loader, Batch loader).
  - Melakukan pelatihan, eksperimen (16 variasi), dan evaluasi Keras vs Scratch untuk task *Image Classification*.
