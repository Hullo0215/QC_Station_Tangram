import cv2
import numpy as np
import time

# --- KONFIGURÁCIÓ ---
# Az inventory-t a beküldött képeid és kódod alapján hagytam meg
SHAPE_INVENTORY = [
    {"name": "Ful/Kicsi", "count": 2, "min": 8000, "max": 11000},
    {"name": "kozepes 3 szög", "count": 1, "min": 16600, "max": 18200}, 
    {"name": "fej", "count": 1, "min": 15600, "max": 17200}, 
    {"name": "nagy3szog", "count": 2, "min": 35000, "max": 42000},
    {"name": "nyak/paralelogramma", "count": 1, "min": 19000, "max": 22000}
]

# Pontozási határok (fokban)
MAX_ANGLE_ERROR = 12.0  # 12 fokos hiba felett a pontszám 0.0
MIN_ANGLE_ERROR = 1.0   # 1 fokon belül a pontszám 1.0

def get_stable_angle(cnt):
    """Visszaadja a kontúr fő irányát, korrigálva az OpenCV szögugrásait."""
    rect = cv2.minAreaRect(cnt)
    (x, y), (w, h), angle = rect
    if w < h:
        angle = angle + 90
    return angle

def get_parallel_score(base_angle, part_angle):
    """Kiszámolja a párhuzamosságot 0.0 és 1.0 között az alaphoz képest."""
    diff = abs(base_angle - part_angle) % 90
    if diff > 45: 
        diff = abs(90 - diff)
    
    # Skálázás: 0 fok diff = 1.0, MAX_ANGLE_ERROR diff = 0.0
    if diff <= MIN_ANGLE_ERROR: return 1.0
    score = 1.0 - (diff / MAX_ANGLE_ERROR)
    return max(0.0, round(score, 2))

def enhance_image(image):
    """Szoftveres kontraszt és telítettség fokozás a detektáláshoz."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    s = cv2.add(s, 20) # Telítettség +20
    v = cv2.add(v, 40) # Fényerő +40
    return cv2.cvtColor(cv2.merge((h, s, v)), cv2.COLOR_HSV2BGR)

def analyze_frame_no_roi(frame):
    # 1. ELŐFELDOLGOZÁS
    enhanced = enhance_image(frame)
    work_frame = enhanced.copy()
    hsv = cv2.cvtColor(work_frame, cv2.COLOR_BGR2HSV)
    
    # Maszkolás (rózsaszín alap kontúrjaihoz)
    mask = cv2.inRange(hsv, np.array([145, 40, 85]), np.array([171, 250, 255]))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8))
    
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if not contours or hierarchy is None: 
        return work_frame

    # 2. ALAP (KÉK KERET) AZONOSÍTÁSA
    base_idx = -1
    max_area = 0
    for i, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        if area > max_area:
            max_area = area
            base_idx = i

    if base_idx == -1: 
        return work_frame

    # Az alap szöge lesz a referencia (1.00)
    base_angle = get_stable_angle(contours[base_idx])
    cv2.drawContours(work_frame, [contours[base_idx]], -1, (255, 0, 0), 2)

    # 3. BELSŐ ALKATRÉSZEK GYŰJTÉSE
    detected_parts = []
    for i, cnt in enumerate(contours):
        if hierarchy[0][i][3] == base_idx: # Csak ami az alapon belül van
            area = cv2.contourArea(cnt)
            if area > 1000:
                p_angle = get_stable_angle(cnt)
                score = get_parallel_score(base_angle, p_angle)
                detected_parts.append({
                    "cnt": cnt, 
                    "area": area, 
                    "score": score, 
                    "identified": False
                })

    # 4. KIÉRTÉKELÉS (INVENTORY + PONTOZÁS)
    total_score = 0
    found_count = 0
    
    temp_inventory = [dict(item) for item in SHAPE_INVENTORY]
    for inv in temp_inventory:
        matches = 0
        for part in detected_parts:
            if not part["identified"] and inv["min"] <= part["area"] <= inv["max"]:
                part["identified"] = True
                matches += 1
                total_score += part["score"]
                found_count += 1
                
                # Vizuális visszajelzés: Színkód a pontszám alapján
                # Zöld (1.0) -> Piros (0.0) átmenet
                color = (0, int(255 * part["score"]), int(255 * (1 - part["score"])))
                cv2.drawContours(work_frame, [part["cnt"]], -1, color, 2)
                
                # Egyedi pontszám kiírása a darab közepére
                M = cv2.moments(part["cnt"])
                if M["m00"] != 0:
                    cX, cY = int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"])
                    cv2.putText(work_frame, f"S:{part['score']}", (cX-25, cY), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                if matches == inv["count"]: break

    # 5. ÖSSZESÍTETT MUTATÓK MEGJELENÍTÉSE
    avg_index = round(total_score / 7, 2) if found_count > 0 else 0.0
    status_color = (0, 255, 0) if found_count == 7 and avg_index > 0.85 else (0, 0, 255)
    
    cv2.putText(work_frame, f"OSSZESITETT INDEX: {avg_index}", (30, 45), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, status_color, 2)
    cv2.putText(work_frame, f"DARABSZAM: {found_count}/7", (30, 85), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    return work_frame

if __name__ == "__main__":
    cap = cv2.VideoCapture(1) # Próbáld a 0-át, ha nincs kép
    cv2.namedWindow("QS Precision Monitor", cv2.WINDOW_NORMAL)
    
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        output_img = analyze_frame_no_roi(frame)
        cv2.imshow("QS Precision Monitor", output_img)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()
