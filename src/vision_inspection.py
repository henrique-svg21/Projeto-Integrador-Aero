"""
Módulo de Visão Computacional para Inspeção Multimodal de Turbinas (OpenCV).
Inclui rotinas de validação geométrica e análise termográfica HSV.
"""
from typing import Dict, Tuple
import cv2
import numpy as np


class IndustrialVisionInspector:
    """Pipeline de inspeção óptica e termográfica para controle de qualidade."""

    @staticmethod
    def inspecao_geometrica(image: np.ndarray) -> Tuple[bool, Dict[str, float]]:
        """
        Inspeção Geométrica de Pás/Componentes de Turbina:
        Aplica suavização gaussiana, Canny Edge Detection e extrai contornos
        validando circularidade e dimensionalidade.
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # Suavização para remoção de ruído de alta frequência
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Canny Edge Detection
        edges = cv2.Canny(blurred, threshold1=50, threshold2=150)

        # Extração de contornos
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return False, {"area_px": 0.0, "circularidade": 0.0, "status": "Sem contorno detectado"}

        # Identifica o maior componente
        c = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(c)
        perimeter = cv2.arcLength(c, True)

        # Circularidade: 4 * pi * (Área / Perímetro^2)
        circularidade = (4 * np.pi * area / (perimeter ** 2)) if perimeter > 0 else 0.0

        # Tolerância: área mínima e circularidade mínima de 0.70
        aprovado = (area > 500.0) and (circularidade >= 0.70)

        metricas = {
            "area_px": float(round(area, 2)),
            "perimetro_px": float(round(perimeter, 2)),
            "circularidade": float(round(circularidade, 4)),
            "status": "APROVADO" if aprovado else "REJEITADO"
        }
        return aprovado, metricas

    @staticmethod
    def inspecao_termica(image_thermal_bgr: np.ndarray, temp_threshold_ratio: float = 0.85) -> Tuple[bool, Dict[str, float]]:
        """
        Inspeção Térmica de Imagens Infravermelhas:
        Converte para HSV e segmenta zonas térmicas críticas (Hotspots).
        """
        hsv = cv2.cvtColor(image_thermal_bgr, cv2.COLOR_BGR2HSV)

        # Segmentação de tons avermelhados no espaço HSV
        lower_red1 = np.array([0, 120, 150])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 120, 150])
        upper_red2 = np.array([180, 255, 255])

        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        hotspot_mask = cv2.bitwise_or(mask1, mask2)

        total_pixels = image_thermal_bgr.shape[0] * image_thermal_bgr.shape[1]
        hotspot_pixels = int(np.count_nonzero(hotspot_mask))
        hotspot_ratio = hotspot_pixels / total_pixels

        superaquecimento = hotspot_ratio > (1.0 - temp_threshold_ratio) * 0.1

        metricas = {
            "hotspot_pixels": hotspot_pixels,
            "taxa_area_critica": float(round(hotspot_ratio, 4)),
            "alarme_termico": superaquecimento,
            "status": "CRITICO_SUPERAQUECIDO" if superaquecimento else "NORMAL"
        }
        return superaquecimento, metricas


if __name__ == "__main__":
    print("=======================================================")
    print("      TESTE DO PIPELINE DE VISÃO COMPUTACIONAL        ")
    print("=======================================================")
    inspector = IndustrialVisionInspector()

    # 1. Teste Geométrico: Componente Circular Aprovado
    img_ok = np.zeros((300, 300, 3), dtype=np.uint8)
    cv2.circle(img_ok, (150, 150), 60, (255, 255, 255), -1)
    status_ok, met_ok = inspector.inspecao_geometrica(img_ok)
    print(f"\n[Peça Perfeita] Resultado: {met_ok['status']} | Circularidade: {met_ok['circularidade']}")

    # 2. Teste Geométrico: Componente Rejeitado (Deformação Elíptica/Irregular)
    img_bad = np.zeros((300, 300, 3), dtype=np.uint8)
    cv2.ellipse(img_bad, (150, 150), (90, 25), 30, 0, 360, (255, 255, 255), -1)
    status_bad, met_bad = inspector.inspecao_geometrica(img_bad)
    print(f"[Peça Defeituosa] Resultado: {met_bad['status']} | Circularidade: {met_bad['circularidade']}")

    # 3. Teste Térmico: Detecção de Hotspot
    img_thermal = np.full((300, 300, 3), (120, 50, 20), dtype=np.uint8)
    # Simulação de ponto quente avermelhado (BGR)
    cv2.circle(img_thermal, (150, 150), 45, (0, 0, 255), -1)
    alerta, met_term = inspector.inspecao_termica(img_thermal)
    print(f"[Inspeção Térmica IR] Alarme: {met_term['status']} | Taxa Crítica: {met_term['taxa_area_critica']*100}%")
    print("=======================================================")