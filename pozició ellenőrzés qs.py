import cv2
import numpy as np

# --- KONFIGURÁCIÓ ---
SHAPE_INVENTORY = [
    {"name": "Ful/Kicsi", "count": 2, "min": 8000, "max": 11000},
    {"name": "kozepes 3 szög", "count": 1, "min": 16600, "max": 18200}, 
    {"name": "fej", "count": 1, "min": 15600, "max": 17200}, 
    {"name": "nagy3szog", "count": 2, "min": 35000, "max": 42000},
    {"name": "nyak/paralelogramma", "count": 1, "min": 19000, "max": 22000}
]

def get_stable_angle(cnt):
    """Visszaadja a kontúr fő irányát (RotatedRect-tel)."""
    rect = cv2.minAreaRect(cnt)
    angle = rect[2]
    # OpenCV szög-korrekció (verziótól függhet, így stabilizáljuk)
    if rect[1][0] < rect[1][1]:
        angle += 90
    return angle

def get_parallel_score(angle1, angle2):
    """Kiszámolja a párhuzamosságot 0.0 és 1.0 között, figyelembe véve a 90 fokos szimmetriát."""
    diff = abs(angle1 - angle2) % 90
    if diff > 45: 
        diff = abs(90 - diff)
    
    # 0 fok diff = 1.0 pont, 10 fok diff = 0.0 pont (szigorúbb skála)
    score = 1.0 - (diff / 10.0)
    return max(0.0, round(score, 2))

def analyze_frame_no_roi(frame):
    # Képjavítás
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    enhanced = cv2.merge((h, cv2.add(s, 20), cv2.add(v, 40)))
    work_frame = cv2.cvtColor(enhanced, cv2.COLOR_HSV2BGR)
    
    mask = cv2.inRange(enhanced, np.array([145, 40, 85]), np.array([171, 250, 255]))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8))
    
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if not contours or hierarchy is None: return 0, work_frame

    # 1. ALAP (KÉK)
    base_idx = -1
    max_a = 0
    for i, c in enumerate(contours):
        if cv2.contourArea(c) > max_a:
            max_a = cv2.contourArea(c)
            base_idx = i
    
    if base_idx == -1: return 0, work_frame
    
    base_angle = get_stable_angle(contours[base_idx])
    cv2.drawContours(work_frame, [contours[base_idx]], -1, (255, 0, 0), 2)

    # 2. ALKATRÉSZEK
    detected_parts = []
    total_score = 0
    found_count = 0
    
    for i, cnt in enumerate(contours):
        if hierarchy[0][i][3] == base_idx:
            area = cv2.contourArea(cnt)
            if area > 1000:
                p_angle = get_stable_angle(cnt)
                score = get_parallel_score(base_angle, p_angle)
                detected_parts.append({"cnt": cnt, "area": area, "score": score, "used": False})

    # 3. ÖSSZESÍTETT PONTZÁM SZÁMÍTÁSA
    for inv in SHAPE_INVENTORY:
        m = 0
        for p in detected_parts:
            if not p["used"] and inv["min"] <= p["area"] <= inv["max"]:
                p["used"] = True
                m += 1
                total_score += p["score"]
                found_count += 1
                
                # Színezés és egyedi pontszám
                color = (0, int(255 * p["score"]), int(255 * (1 - p["score"])))
                cv2.drawContours(work_frame, [p["cnt"]], -1, color, 2)
                
                M = cv2.moments(p["cnt"])
                if M["m00"] != 0:
                    cX, cY = int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"])
                    cv2.putText(work_frame, f"{p['score']}", (cX-20, cY), 0, 0.5, (255,255,255), 1)
                if m == inv["count"]: break

    # VÉGSŐ ÖSSZESÍTETT INDEX (0-100% vagy 0-1.0)
    final_index = round(total_score / 7, 2) if found_count > 0 else 0.0
    
    # Kijelzés
    cv2.putText(work_frame, f"OSSZESITETT INDEX: {final_index}", (30, 40), 1, 1.5, (255, 255, 255), 2)
    return final_index, work_frame
