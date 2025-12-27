"""
Script pour télécharger le modèle hand_landmarker.task
"""

import urllib.request
import os
import sys

def download_model():
    
    model_url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
    model_filename = "hand_landmarker.task"
    
    print("=" * 60)
    print("Téléchargement du modèle MediaPipe Hand Landmarker")
    print("=" * 60)
    print()
    
    if os.path.exists(model_filename):
        print(f"✓ Le fichier '{model_filename}' existe déjà.")
        response = input("Voulez-vous le re-télécharger ? (o/n): ")
        if response.lower() != 'o':
            print("Téléchargement annulé.")
            return
        print("Re-téléchargement...")
    
    print(f"URL: {model_url}")
    print(f"Destination: {model_filename}")
    print()
    print("Téléchargement en cours...")
    
    try:
        def report_progress(block_num, block_size, total_size):
            downloaded = block_num * block_size
            percent = min(100, (downloaded / total_size) * 100)
            bar_length = 50
            filled_length = int(bar_length * downloaded // total_size)
            bar = '█' * filled_length + '-' * (bar_length - filled_length)
            
            downloaded_mb = downloaded / (1024 * 1024)
            total_mb = total_size / (1024 * 1024)
            
            sys.stdout.write(f'\r[{bar}] {percent:.1f}% ({downloaded_mb:.1f}/{total_mb:.1f} MB)')
            sys.stdout.flush()
        
        urllib.request.urlretrieve(model_url, model_filename, report_progress)
        
        print()
        print()
        print("=" * 60)
        print("✓ Téléchargement terminé avec succès!")
        print("=" * 60)
        
        file_size = os.path.getsize(model_filename) / (1024 * 1024)
        print(f"Taille du fichier: {file_size:.2f} MB")
        print(f"Emplacement: {os.path.abspath(model_filename)}")
        
    except Exception as e:
        print(f"✗ Erreur lors du téléchargement: {e}")


if __name__ == "__main__":
    try:
        download_model()
    except KeyboardInterrupt:
        print("\n\nTéléchargement annulé par l'utilisateur")