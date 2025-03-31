import os
import numpy as np
import librosa
import matplotlib.pyplot as plt
import itertools
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix

# Adicionando o download do dataset via Kaggle Hub
import kagglehub

# Parâmetros de processamento de áudio
sr = 22050                   # Taxa de amostragem padrão
win_length = int(0.02 * sr)  # Janela de ~20ms (aproximadamente 441 samples)
hop_length = win_length // 2 # 50% de sobreposição

###########################################
# Funções de extração de características  #
###########################################

def extract_features(file_path):
    """
    Carrega um arquivo de áudio e extrai:
    - Zero Crossing Rate (média e desvio)
    - Spectral Centroid (média e desvio)
    - MFCCs (13 coeficientes com média e desvio)
    - ΔMFCC e ΔΔMFCC (média e desvio para cada coeficiente)
    Retorna um vetor de features.
    """
    y, _ = librosa.load(file_path, sr=sr)
    
    # Zero Crossing Rate
    zcr = librosa.feature.zero_crossing_rate(y, frame_length=win_length, hop_length=hop_length)
    zcr_mean = np.mean(zcr)
    zcr_std = np.std(zcr)
    
    # Spectral Centroid
    spec_centroid = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=win_length, hop_length=hop_length)
    centroid_mean = np.mean(spec_centroid)
    centroid_std = np.std(spec_centroid)
    
    # MFCCs
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, n_fft=win_length, hop_length=hop_length)
    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)
    
    # Delta MFCC
    delta_mfcc = librosa.feature.delta(mfcc)
    delta_mfcc_mean = np.mean(delta_mfcc, axis=1)
    delta_mfcc_std = np.std(delta_mfcc, axis=1)
    
    # Delta-Delta MFCC
    delta2_mfcc = librosa.feature.delta(mfcc, order=2)
    delta2_mfcc_mean = np.mean(delta2_mfcc, axis=1)
    delta2_mfcc_std = np.std(delta2_mfcc, axis=1)
    
    # Concatena todos os recursos em um único vetor
    features = np.hstack([
        [zcr_mean, zcr_std],
        [centroid_mean, centroid_std],
        mfcc_mean, mfcc_std,
        delta_mfcc_mean, delta_mfcc_std,
        delta2_mfcc_mean, delta2_mfcc_std
    ])
    return features

def extract_features_from_signal(y, sr):
    """
    Extrai as mesmas features utilizadas na função extract_features,
    mas a partir de um sinal de áudio já carregado.
    """
    # Zero Crossing Rate
    zcr = librosa.feature.zero_crossing_rate(y, frame_length=win_length, hop_length=hop_length)
    zcr_mean = np.mean(zcr)
    zcr_std = np.std(zcr)
    
    # Spectral Centroid
    spec_centroid = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=win_length, hop_length=hop_length)
    centroid_mean = np.mean(spec_centroid)
    centroid_std = np.std(spec_centroid)
    
    # MFCCs
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, n_fft=win_length, hop_length=hop_length)
    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)
    
    # Delta MFCC
    delta_mfcc = librosa.feature.delta(mfcc)
    delta_mfcc_mean = np.mean(delta_mfcc, axis=1)
    delta_mfcc_std = np.std(delta_mfcc, axis=1)
    
    # Delta-Delta MFCC
    delta2_mfcc = librosa.feature.delta(mfcc, order=2)
    delta2_mfcc_mean = np.mean(delta2_mfcc, axis=1)
    delta2_mfcc_std = np.std(delta2_mfcc, axis=1)
    
    features = np.hstack([
        [zcr_mean, zcr_std],
        [centroid_mean, centroid_std],
        mfcc_mean, mfcc_std,
        delta_mfcc_mean, delta_mfcc_std,
        delta2_mfcc_mean, delta2_mfcc_std
    ])
    return features

#############################################
# Funções para efeitos de ruído e eco       #
#############################################

def add_noise(y, noise_factor=0.005):
    """
    Adiciona ruído branco ao sinal.
    """
    noise = np.random.randn(len(y))
    y_noise = y + noise_factor * noise
    return y_noise

def add_echo(y, sr, delay=0.2, decay=0.6):
    """
    Adiciona efeito de eco ao sinal.
    delay: atraso em segundos
    decay: fator de decaimento do eco
    """
    delay_samples = int(delay * sr)
    echo_signal = np.zeros(len(y) + delay_samples)
    echo_signal[:len(y)] = y
    echo_signal[delay_samples:] += decay * y
    # Retorna o sinal com mesmo comprimento do original
    return echo_signal[:len(y)]

#############################################
# Carregamento do dataset (GTZAN)           #
#############################################

def load_dataset(base_dir):
    """
    Percorre a estrutura de diretórios do dataset GTZAN.
    Supõe que cada subpasta representa um gênero e contém arquivos .wav.
    Retorna X (features) e y (labels).
    """
    genres = [g for g in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, g))]
    features = []
    labels = []
    
    for genre in genres:
        genre_dir = os.path.join(base_dir, genre)
        for filename in os.listdir(genre_dir):
            if filename.lower().endswith('.wav'):
                file_path = os.path.join(genre_dir, filename)
                try:
                    feat = extract_features(file_path)
                    features.append(feat)
                    labels.append(genre)
                except Exception as e:
                    print(f"Erro ao processar {file_path}: {e}")
    return np.array(features), np.array(labels)

#############################################
# Função para plotar a matriz de confusão   #
#############################################

def plot_confusion_matrix(cm, classes, title='Matriz de Confusão', cmap=plt.cm.Blues):
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    
    plt.ylabel('Rótulo verdadeiro')
    plt.xlabel('Rótulo previsto')
    plt.tight_layout()
    plt.show()

#############################################
# Execução Principal do Script            #
#############################################

if __name__ == '__main__':
    # Defina o caminho para o dataset GTZAN (ajuste conforme necessário)
    # Download da versão mais recente do dataset GTZAN
    path = kagglehub.dataset_download("andradaolteanu/gtzan-dataset-music-genre-classification")
    print("Path to dataset files:", path)

    # Define o caminho para o dataset baixado
    # dataset_path = path  # ou ajuste para o subdiretório correto, se necessário
    # dataset_path = os.path.join(path, "genres_original")
    dataset_path = "C://Users//Pichau//.cache//kagglehub//datasets//andradaolteanu//gtzan-dataset-music-genre-classification//versions//1//Data//genres_original"

    
    print("Carregando o dataset GTZAN...")
    X, y = load_dataset(dataset_path)
    # print("X, y", X, y)
    print(f"Dataset carregado: {X.shape[0]} amostras, cada uma com {X.shape[1]} features.")
    
    # Divisão em conjunto de treino e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Treinamento do classificador SVM
    print("Treinando o classificador SVM...")
    clf = SVC(kernel='linear', probability=True)
    clf.fit(X_train, y_train)
    
    # Avaliação no conjunto de teste
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Acurácia no conjunto de teste: {acc*100:.2f}%")
    
    # Cálculo e exibição da matriz de confusão
    classes = np.unique(y)
    cm = confusion_matrix(y_test, y_pred, labels=classes)
    print("Matriz de Confusão:")
    print(cm)
    plot_confusion_matrix(cm, classes)
    
    #########################################
    # Validação com Áudios com Ruído e Eco  #
    #########################################
    
    # Para demonstrar, selecionamos um áudio de exemplo a partir do dataset.
    # OBS.: Como não armazenamos o caminho original durante o carregamento,
    # selecionamos manualmente o primeiro arquivo encontrado.
    genre_folders = [g for g in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, g))]
    if genre_folders:
        exemplo_folder = os.path.join(dataset_path, genre_folders[0])
        arquivos_exemplo = [f for f in os.listdir(exemplo_folder) if f.lower().endswith('.wav')]
        if arquivos_exemplo:
            exemplo_file = os.path.join(exemplo_folder, arquivos_exemplo[0])
            print(f"\nÁudio de exemplo para validação: {exemplo_file}")
            y_original, _ = librosa.load(exemplo_file, sr=sr)
            
            # Aplicar ruído
            y_noise = add_noise(y_original, noise_factor=0.005)
            features_noise = extract_features_from_signal(y_noise, sr)
            pred_noise = clf.predict([features_noise])
            print(f"Predição para áudio com RUÍDO: {pred_noise[0]}")
            
            # Aplicar eco
            y_echo = add_echo(y_original, sr, delay=0.2, decay=0.6)
            features_echo = extract_features_from_signal(y_echo, sr)
            pred_echo = clf.predict([features_echo])
            print(f"Predição para áudio com ECO: {pred_echo[0]}")
        else:
            print("Nenhum arquivo .wav encontrado na pasta de exemplo.")
    else:
        print("Nenhuma pasta de gênero encontrada no dataset.")
