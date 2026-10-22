"""
Gera e salva figuras comparativas de Visão Computacional:
- Inspeção Geométrica com Canny Edge Detection
- Inspeção Térmica com Segmentação de Hotspots em HSV
"""
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
from vision_inspection import IndustrialVisionInspector

# Cria diretório para salvar as imagens
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "docs", "images")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# 1. Demonstração Visual: Inspeção Geométrica (Canny)
# ---------------------------------------------------------------------------
# Simula imagem de uma pá/rotor perfeito (círculo) e um defeituoso (entalhe/achatamento)
img_peca_ok = np.zeros((300, 300), dtype=np.uint8)
cv2.circle(img_peca_ok, (150, 150), 70, 255, -1)

img_peca_defeito = np.zeros((300, 300), dtype=np.uint8)
cv2.ellipse(img_peca_defeito, (150, 150), (100, 45), 25, 0, 360, 255, -1)

# Aplicação dos passos de Canny
blur_ok = cv2.GaussianBlur(img_peca_ok, (5, 5), 0)
edges_ok = cv2.Canny(blur_ok, 50, 150)

blur_def = cv2.GaussianBlur(img_peca_defeito, (5, 5), 0)
edges_def = cv2.Canny(blur_def, 50, 150)

_, met_ok = IndustrialVisionInspector.inspecao_geometrica(img_peca_ok)
_, met_def = IndustrialVisionInspector.inspecao_geometrica(img_peca_defeito)

fig, axes = plt.subplots(2, 3, figsize=(14, 9))

# Linha 1: Peça Aprovada
axes[0, 0].imshow(img_peca_ok, cmap="gray")
axes[0, 0].set_title("1. Componente Nominal (Original)")
axes[0, 0].axis("off")

axes[0, 1].imshow(edges_ok, cmap="inferno")
axes[0, 1].set_title("2. Detecção de Bordas (Canny)")
axes[0, 1].axis("off")

# Contorno sobreposto
overlay_ok = cv2.cvtColor(img_peca_ok, cv2.COLOR_GRAY2RGB)
cnts_ok, _ = cv2.findContours(edges_ok, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(overlay_ok, cnts_ok, -1, (0, 255, 0), 2)
axes[0, 2].imshow(overlay_ok)
axes[0, 2].set_title(f"3. Diagnóstico: {met_ok['status']}\nCircularidade: {met_ok['circularidade']}")
axes[0, 2].axis("off")

# Linha 2: Peça Defeituosa
axes[1, 0].imshow(img_peca_defeito, cmap="gray")
axes[1, 0].set_title("4. Componente com Deformação (Original)")
axes[1, 0].axis("off")

axes[1, 1].imshow(edges_def, cmap="inferno")
axes[1, 1].set_title("5. Detecção de Bordas (Canny)")
axes[1, 1].axis("off")

overlay_def = cv2.cvtColor(img_peca_defeito, cv2.COLOR_GRAY2RGB)
cnts_def, _ = cv2.findContours(edges_def, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(overlay_def, cnts_def, -1, (255, 0, 0), 2)
axes[1, 2].imshow(overlay_def)
axes[1, 2].set_title(f"6. Diagnóstico: {met_def['status']}\nCircularidade: {met_def['circularidade']}")
axes[1, 2].axis("off")

plt.suptitle("Inspeção Óptica de Precisão - Detecção Canny e Análise de Circularidade", fontsize=14)
plt.tight_layout()
caminho_geom = os.path.join(OUTPUT_DIR, "inspecao_geometrica_canny.png")
plt.savefig(caminho_geom, dpi=200)
plt.close()
print(f"-> Painel de Canny salvo em: {caminho_geom}")

# ---------------------------------------------------------------------------
# 2. Demonstração Visual: Inspeção Térmica no Espaço HSV
# ---------------------------------------------------------------------------
# Simula imagem termográfica (Ironbow): fundo azul/roxo e um ponto vermelho/quente no centro
img_thermal = np.full((300, 300, 3), (130, 60, 20), dtype=np.uint8)  # Tom frio
cv2.circle(img_thermal, (150, 150), 55, (0, 0, 255), -1)              # Hotspot avermelhado

hsv = cv2.cvtColor(img_thermal, cv2.COLOR_BGR2HSV)
lower_red1 = np.array([0, 120, 150])
upper_red1 = np.array([10, 255, 255])
lower_red2 = np.array([170, 120, 150])
upper_red2 = np.array([180, 255, 255])
mask = cv2.bitwise_or(cv2.inRange(hsv, lower_red1, upper_red1), cv2.inRange(hsv, lower_red2, upper_red2))

_, met_term = IndustrialVisionInspector.inspecao_termica(img_thermal)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(cv2.cvtColor(img_thermal, cv2.COLOR_BGR2RGB))
axes[0].set_title("1. Termografia Infravermelha (Entrada)")
axes[0].axis("off")

axes[1].imshow(hsv[:, :, 0], cmap="jet")
axes[1].set_title("2. Canal Matiz (Hue) no Espaço HSV")
axes[1].axis("off")

axes[2].imshow(mask, cmap="Reds")
axes[2].set_title(f"3. Segmentação de Hotspot ({met_term['status']})\nÁrea Afetada: {met_term['taxa_area_critica']*100:.2f}%")
axes[2].axis("off")

plt.suptitle("Inspeção Termográfica - Delimitação de Superaquecimento em HSV", fontsize=14)
plt.tight_layout()
caminho_term = os.path.join(OUTPUT_DIR, "inspecao_termica_hsv.png")
plt.savefig(caminho_term, dpi=200)
plt.close()
print(f"-> Painel Térmico salvo em: {caminho_term}")