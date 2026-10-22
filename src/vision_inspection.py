"""
Módulo de Visão Computacional para Inspeção Multimodal de Turbinas (OpenCV).
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

        # Canny Edge Detection com limiares industriais
        edges = cv2.Canny(blurred, threshold1=50, threshold2=150)

        # Extração de contornos externos
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return False, {"area": 0.0, "circularidade": 0.0, "status": "Sem contorno detectado"}

        # Identifica o maior componente geométrico
        c = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(c)
        perimeter = cv2.arcLength(c, True)

        # Cálculo da Circularidade: 4 * pi * (Área / Perímetro^2)
        circularidade = (4 * np.pi * area / (perimeter ** 2)) if perimeter > 0 else 0.0

        # Tolerância industrial de controle de qualidade
        criterio_aprovacao = (area > 500.0) and (circularidade >= 0.70)

        metricas = {
            "area_px": float(area),
            "perimetro_px": float(perimeter),
            "circularidade": float(round(circularidade, 4)),
            "status": "APROVADO" if criterio_aprovacao else "REJEITADO"
        }
        return criterio_aprovacao, metricas

    @staticmethod
    def inspecao_termica(image_thermal_bgr: np.ndarray, temp_threshold_ratio: float = 0.85) -> Tuple[bool, Dict[str, float]]:
        """
        Inspeção Térmica de Imagens Infravermelhas:
        Converte para espaço de cor HSV e segmenta zonas térmicas críticas (Hotspots).
        """
        hsv = cv2.cvtColor(image_thermal_bgr, cv2.COLOR_BGR2HSV)

        # No espaço térmico padrão (Ironbow/Jet), temperaturas críticas concentram-se
        # nos canais de alta saturação e tons avermelhados/brancos (faixa 0-10 e 170-180 de Matiz)
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

        # Superaquecimento crítico detectado se ultrapassar o limiar de tolerância
        superaquecimento = hotspot_ratio > (1.0 - temp_threshold_ratio) * 0.1

        metricas = {
            "hotspot_pixels": hotspot_pixels,
            "taxa_area_critica": float(round(hotspot_ratio, 4)),
            "alarme_termico": superaquecimento
        }
        return superaquecimento, metricas