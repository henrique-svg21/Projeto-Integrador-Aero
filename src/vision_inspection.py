"""
Módulo de Visão Computacional para Inspeção Multimodal de Turbinas (OpenCV).
Suporte a tolerâncias geométricas e de desgaste parametrizáveis.
"""
import base64
from typing import Any, Dict, Tuple
import cv2
import numpy as np


class IndustrialVisionInspector:
    """Pipeline de inspeção óptica e termográfica para controle de qualidade."""

    @staticmethod
    def _to_base64_png(image: np.ndarray) -> str:
        """Converte matriz OpenCV em string Data URI Base64 para renderização HTML."""
        success, buffer = cv2.imencode(".png", image)
        if not success:
            return ""
        encoded = base64.b64encode(buffer).decode("utf-8")
        return f"data:image/png;base64,{encoded}"

    @staticmethod
    def inspecao_geometrica(image: np.ndarray) -> Tuple[bool, Dict[str, Any]]:
        """Inspeção de uma única peça via Canny e Circularidade."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image.copy()
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return False, {"area_px": 0.0, "circularidade": 0.0, "status": "Sem contorno"}

        c = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(c)
        perimeter = cv2.arcLength(c, True)
        circularidade = (4 * np.pi * area / (perimeter ** 2)) if perimeter > 0 else 0.0
        aprovado = bool((area > 500.0) and (circularidade >= 0.70))

        overlay = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(overlay, [c], -1, (0, 255, 0) if aprovado else (0, 0, 255), 2)

        metricas = {
            "area_px": float(round(area, 2)),
            "perimetro_px": float(round(perimeter, 2)),
            "circularidade": float(round(circularidade, 4)),
            "status": "APROVADO" if aprovado else "REJEITADO",
            "canny_b64": IndustrialVisionInspector._to_base64_png(edges),
            "overlay_b64": IndustrialVisionInspector._to_base64_png(overlay)
        }
        return aprovado, metricas

    @staticmethod
    def comparar_componentes(
        img_ref: np.ndarray,
        img_test: np.ndarray,
        min_circularidade: float = 0.70,
        max_desgaste_pct: float = 10.0,
        diff_threshold: int = 35,
        canny_low: int = 50,
        canny_high: int = 150
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Compara a Peça de Referência com a Peça em Teste utilizando tolerâncias dinâmicas.
        """
        h, w = img_ref.shape[:2]
        if img_test.shape[:2] != (h, w):
            img_test = cv2.resize(img_test, (w, h))

        gray_ref = cv2.cvtColor(img_ref, cv2.COLOR_BGR2GRAY) if len(img_ref.shape) == 3 else img_ref
        gray_test = cv2.cvtColor(img_test, cv2.COLOR_BGR2GRAY) if len(img_test.shape) == 3 else img_test

        # 1. Canny com limiares parametrizáveis
        blur_ref = cv2.GaussianBlur(gray_ref, (5, 5), 0)
        blur_test = cv2.GaussianBlur(gray_test, (5, 5), 0)
        edges_ref = cv2.Canny(blur_ref, canny_low, canny_high)
        edges_test = cv2.Canny(blur_test, canny_low, canny_high)

        # 2. Métricas da Referência
        cnts_ref, _ = cv2.findContours(edges_ref, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        c_ref = max(cnts_ref, key=cv2.contourArea) if cnts_ref else None
        area_ref = cv2.contourArea(c_ref) if c_ref is not None else 1.0
        peri_ref = cv2.arcLength(c_ref, True) if c_ref is not None else 1.0
        circ_ref = (4 * np.pi * area_ref / (peri_ref ** 2)) if peri_ref > 0 else 0.0

        # 3. Métricas do Teste
        cnts_test, _ = cv2.findContours(edges_test, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        c_test = max(cnts_test, key=cv2.contourArea) if cnts_test else None
        area_test = cv2.contourArea(c_test) if c_test is not None else 0.0
        peri_test = cv2.arcLength(c_test, True) if c_test is not None else 0.0
        circ_test = (4 * np.pi * area_test / (peri_test ** 2)) if peri_test > 0 else 0.0

        # 4. Diferença Absoluta com Limiar de Sensibilidade
        diff = cv2.absdiff(gray_ref, gray_test)
        _, diff_mask = cv2.threshold(diff, diff_threshold, 255, cv2.THRESH_BINARY)

        # Destaque em vermelho néon sobre as falhas
        overlay_desgaste = cv2.cvtColor(gray_test, cv2.COLOR_GRAY2BGR) if len(img_test.shape) == 2 else img_test.copy()
        overlay_desgaste[diff_mask > 0] = [0, 0, 255]

        # Sobreposição de bordas: Ciano (Nominal) vs Amarelo (Teste)
        overlay_bordas = np.zeros((h, w, 3), dtype=np.uint8)
        overlay_bordas[edges_ref > 0] = [255, 255, 0]
        overlay_bordas[edges_test > 0] = [0, 255, 255]

        desgaste_px = int(np.count_nonzero(diff_mask))
        desgaste_pct = (desgaste_px / area_ref) * 100 if area_ref > 0 else 0.0
        delta_circularidade = circ_test - circ_ref

        # Avaliação contra as tolerâncias ajustadas pelo operador
        falha_circularidade = circ_test < min_circularidade
        falha_desgaste = desgaste_pct > max_desgaste_pct
        aprovado = bool(not falha_circularidade and not falha_desgaste)

        motivos = []
        if falha_desgaste:
            motivos.append(f"Desgaste ({desgaste_pct:.1f}%) > Tolerância ({max_desgaste_pct:.1f}%)")
        if falha_circularidade:
            motivos.append(f"Circularidade ({circ_test:.3f}) < Mínimo ({min_circularidade:.3f})")

        status_texto = "APROVADO" if aprovado else "REJEITADO: " + " | ".join(motivos)

        metricas = {
            "aprovado": aprovado,
            "status": status_texto,
            "circularidade_referencia": float(round(circ_ref, 4)),
            "circularidade_teste": float(round(circ_test, 4)),
            "delta_circularidade": float(round(delta_circularidade, 4)),
            "area_referencia_px": float(round(area_ref, 2)),
            "area_teste_px": float(round(area_test, 2)),
            "desgaste_superficie_pct": float(round(desgaste_pct, 2)),
            "tolerancias_aplicadas": {
                "min_circularidade": min_circularidade,
                "max_desgaste_pct": max_desgaste_pct,
                "diff_threshold": diff_threshold,
                "canny_range": [canny_low, canny_high]
            },
            "canny_ref_b64": IndustrialVisionInspector._to_base64_png(edges_ref),
            "canny_test_b64": IndustrialVisionInspector._to_base64_png(edges_test),
            "overlay_desgaste_b64": IndustrialVisionInspector._to_base64_png(overlay_desgaste),
            "overlay_bordas_b64": IndustrialVisionInspector._to_base64_png(overlay_bordas)
        }
        return aprovado, metricas

    @staticmethod
    def inspecao_termica(image_thermal_bgr: np.ndarray, temp_threshold_ratio: float = 0.85) -> Tuple[bool, Dict[str, Any]]:
        """Inspeção Térmica de Imagens Infravermelhas no Espaço HSV."""
        hsv = cv2.cvtColor(image_thermal_bgr, cv2.COLOR_BGR2HSV)
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
        superaquecimento = bool(hotspot_ratio > (1.0 - temp_threshold_ratio) * 0.1)

        thermal_overlay = image_thermal_bgr.copy()
        thermal_overlay[hotspot_mask > 0] = [255, 255, 255]

        metricas = {
            "alarme_termico": superaquecimento,
            "status": "CRITICO_SUPERAQUECIDO" if superaquecimento else "NORMAL_NOMINAL",
            "hotspot_pixels": hotspot_pixels,
            "taxa_area_critica_pct": float(round(hotspot_ratio * 100, 2)),
            "mask_b64": IndustrialVisionInspector._to_base64_png(hotspot_mask),
            "overlay_b64": IndustrialVisionInspector._to_base64_png(thermal_overlay)
        }
        return superaquecimento, metricas