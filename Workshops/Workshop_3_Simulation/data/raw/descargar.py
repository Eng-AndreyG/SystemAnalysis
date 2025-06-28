import kaggle
from tqdm import tqdm
import os
import shutil

# Configura rutas
os.makedirs('data/raw', exist_ok=True)
files = [
    'train.csv',
    'test.csv',
    'sample_submission.csv',
    'identity_individual_annotations.csv'
]

def download_and_verify(file):
    try:
        print(f"\n⏳ Descargando {file}...")
        
        # Descarga el archivo
        kaggle.api.competition_download_file(
            'jigsaw-unintended-bias-in-toxicity-classification',
            file,
            path='data/raw',
            quiet=False
        )
        
        # Kaggle a veces descarga sin extensión .zip, así que verificamos ambos casos
        zip_file = f"data/raw/{file}.zip"
        no_zip_file = f"data/raw/{file}"
        
        # Renombrar si es necesario
        if os.path.exists(no_zip_file):
            os.rename(no_zip_file, zip_file)
            print(f"✔ Archivo renombrado a {file}.zip")
        
        # Verificar que existe
        if os.path.exists(zip_file):
            size = os.path.getsize(zip_file) / (1024*1024)  # Tamaño en MB
            print(f"✅ {file}.zip descargado correctamente | Tamaño: {size:.2f} MB")
            
            # Descomprimir
            shutil.unpack_archive(zip_file, 'data/raw')
            print(f"📦 Archivo descomprimido: data/raw/{file}")
            
            return True
        else:
            print(f"❌ No se encontró {file}.zip después de la descarga")
            return False
            
    except Exception as e:
        print(f"⚠️ Error con {file}: {str(e)}")
        return False

# Ejecutar descargas
for file in files:
    download_and_verify(file)