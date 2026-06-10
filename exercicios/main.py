# =============================================================
# Classificação de Cantos de Pássaros com SVM 
# =============================================================

import os
import numpy as np
import librosa
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# =============================================================
# 1. CONFIGURAÇÕES
# =============================================================

ESPECIES = [
    "sabia-laranjeira",
    "bem-te-vi",
    "tico-tico"
]

PASTA_AUDIOS = "audios_passaros"
DURACAO_SEG  = 30
SR           = 22050
N_MFCC       = 13

# =============================================================
# 2. CARREGAR ÁUDIOS LOCAIS
# =============================================================

def carregar_audios_locais():
    arquivos = []
    rotulos = []

    print("Carregando áudios locais")

    for especie in ESPECIES:
        pasta = os.path.join(PASTA_AUDIOS, especie)

        if not os.path.exists(pasta):
            print(f"Pasta não encontrada: {pasta}")
            continue

        for nome_arquivo in os.listdir(pasta):
            if nome_arquivo.endswith((".mp3", ".wav")):
                caminho = os.path.join(pasta, nome_arquivo)
                arquivos.append(caminho)
                rotulos.append(especie)

    print(f"Total de arquivos carregados: {len(arquivos)}")
    return arquivos, rotulos

# =============================================================
# 3. EXTRAÇÃO DE FEATURES 
# =============================================================

def extrair_features(caminho_arquivo):
    try:
        y, sr = librosa.load(caminho_arquivo, sr=SR, mono=True)

        if len(y) < sr * 1:
            print(f"Áudio muito curto: {caminho_arquivo}")
            return None

        duracao_total = len(y) / sr
        if duracao_total > DURACAO_SEG:
            inicio = int((duracao_total / 2 - DURACAO_SEG / 2) * sr)
            fim = inicio + int(DURACAO_SEG * sr)
            y = y[inicio:fim]


        y = librosa.util.normalize(y)

        # =========================
        # FEATURES
        # =========================

        # MFCC
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
        mfcc_media = np.mean(mfccs, axis=1)
        mfcc_std   = np.std(mfccs, axis=1)

        # Energia
        rms = librosa.feature.rms(y=y)
        rms_media = np.mean(rms)

        # Centróide
        centroide = librosa.feature.spectral_centroid(y=y, sr=sr)
        centroide_media = np.mean(centroide)

        # ZCR
        zcr = librosa.feature.zero_crossing_rate(y)
        zcr_media = np.mean(zcr)

        # Bandwidth
        bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        bandwidth_media = np.mean(bandwidth)

        # Rolloff
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        rolloff_media = np.mean(rolloff)

        return np.concatenate([
            mfcc_media,
            mfcc_std,
            [rms_media, centroide_media, zcr_media, bandwidth_media, rolloff_media]
        ])

    except Exception as e:
        print(f"Erro ao processar {caminho_arquivo}: {e}")
        return None

# =============================================================
# 4. PIPELINE PRINCIPAL
# =============================================================

def main():

    arquivos, rotulos = carregar_audios_locais()

    if not arquivos:
        print("Nenhum áudio encontrado.")
        return

    print("\nExtraindo features...")
    X, y = [], []

    for arquivo, rotulo in zip(arquivos, rotulos):
        feat = extrair_features(arquivo)
        if feat is not None:
            X.append(feat)
            y.append(rotulo)

    X = np.array(X)

    if len(X) < 6:
        print("Poucos dados.")
        return

    print(f"\n📊 Dataset: {X.shape[0]} amostras x {X.shape[1]} features")

    # encoding
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    # split
    X_treino, X_teste, y_treino, y_teste = train_test_split(
        X, y_enc, test_size=0.25, random_state=42, stratify=y_enc
    )

    # normalização
    scaler = StandardScaler()
    X_treino = scaler.fit_transform(X_treino)
    X_teste  = scaler.transform(X_teste)

    # modelo
    print("\n🤖 Treinando SVM...")
    modelo = SVC(kernel="rbf", C=10, gamma="scale", class_weight="balanced")
    modelo.fit(X_treino, y_treino)

    # avaliação
    y_pred = modelo.predict(X_teste)
    acc = accuracy_score(y_teste, y_pred)

    print(f"\nAcurácia: {acc:.2%}")
    print("\nRelatório:")
    print(classification_report(y_teste, y_pred, target_names=le.classes_))

    # =============================================================
    # 5. GRÁFICOS
    # =============================================================

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Classificação de Cantos de Pássaros (SVM)", fontsize=14)

    # matriz
    cm = confusion_matrix(y_teste, y_pred)
    sns.heatmap(cm, annot=True, fmt="d",
                xticklabels=le.classes_,
                yticklabels=le.classes_,
                ax=axes[0])

    axes[0].set_title("Matriz de Confusão")
    axes[0].set_xlabel("Previsto")
    axes[0].set_ylabel("Real")

    nomes_feat = (
        [f"MFCC_mean {i+1}" for i in range(N_MFCC)] +
        [f"MFCC_std {i+1}" for i in range(N_MFCC)] +
        ["RMS", "Centroide", "ZCR", "Bandwidth", "Rolloff"]
    )

    variancia = np.var(X_treino, axis=0)

    axes[1].barh(nomes_feat, variancia)
    axes[1].set_title("Importância das Features")
    axes[1].invert_yaxis()

    plt.tight_layout()
    plt.savefig("resultado.png")
    plt.show()

    print("\nGráfico salvo como resultado.png")

# =============================================================
if __name__ == "__main__":
    main()