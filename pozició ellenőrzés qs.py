import cv2
import numpy as np
import time

# --- KONFIGURÁCIÓ ---
SHAPE_INVENTORY = [
    {"name": "Ful/Kicsi", "count": 2, "min": 8000, "max": 11000},
    {"name": "kozepes 3 szog", "count": 1, "min": 16600, "max": 18200}, 
    {"name": "fej", "count": 1, "min": 15600, "max": 17200}, 
    {"name": "nagy3szog", "count": 2, "min": 35000, "max": 40000},
    {"name": "nyak/paralelogramma", "count": 1, "min": 19000, "max": 21000}
]

# Párhuzamosság skálázása
MAX_ANGLE_ERROR = 15.0  # 15 fokos hiba felett a pontszám 0.0
MIN_ANGLE_ERROR = 1.0   # 1 fokon belül a pontszám 1.0

def get_angle(cnt):
    """Visszaadja a kontúr dőlésszögét."""
    rect = cv2.minAreaRect(cnt)
    return rect[2]

def calculate_score(diff):
    """Fokeltérésből 0-1 közötti pontszámot csinál."""
    if diff <= MIN_ANGLE_ERROR: return 1.0
    if diff >= MAX_ANGLE_ERROR: return 0.0
    # Lineáris leképezés
    score = 1.0 - ((diff - MIN_ANGLE_ERROR) / (MAX_ANGLE_ERROR - MIN_ANGLE_ERROR))
    return round(score, 2)

def enhance_image(image):
    """Szoftveres telítettség és fényerő emelés."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    s = cv2.add(s, 20)
    v = cv2.add(v, 40)
    return cv2.cvtColor(cv2.merge((h, s, v)), cv2.COLOR_HSV2BGR)

def analyze_frame_no_roi(frame):
    results = {"status": "FAIL", "avg_score": 0.0}
    
    # 1. TUNING ÉS MASZK
    enhanced = enhance_image(frame)
    hsv = cv2.cvtColor(enhanced, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([145, 40, 85]), np.array([171, 250, 255]))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8))
    
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if not contours or hierarchy is None: return results, enhanced

    # 2. BÁZIS (RÓZSASZÍN KERET) AZONOSÍTÁSA
    base_idx = -1
    max_area = 0
    for i, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        if area > max_area:
            max_area = area
            base_idx = i

    if base_idx == -1: return results, enhanced

    base_angle = get_angle(contours[base_idx])
    cv2.drawContours(enhanced, [contours[base_idx]], -1, (255, 0, 0), 2) # KÉK

    # 3. BELSŐ ELEMEK GYŰJTÉSE
    detected_parts = []
    for i, cnt in enumerate(contours):
        if hierarchy[0][i][3] == base_idx:
            area = cv2.contourArea(cnt)
            if area > 1000:
                p_angle = get_angle(cnt)
                # Szögeltérés normalizálása 0-45 fok közé
                diff = abs(p_angle - base_angle) % 90
                if diff > 45: diff = abs(90 - diff)
                
                score = calculate_score(diff)
                detected_parts.append({"cnt": cnt, "area": area, "score": score, "id": False})

    # 4. PÁROSÍTÁS ÉS PONTOZÁS
    total_score = 0
    found_count = 0
    for inv in SHAPE_INVENTORY:
        matches = 0
        for part in detected_parts:
            if not part["id"] and inv["min"] <= part["area"] <= inv["max"]:
                part["id"] = True
                matches += 1
                total_score += part["score"]
                found_count += 1
                
                # Dinamikus szín: Zöld (1.0) -> Piros (0.0)
                color = (0, int(255 * part["score"]), int(255 * (1 - part["score"])))
                cv2.drawContours(enhanced, [part["cnt"]], -1, color, 2)
                
                # Pontszám kiírása a darabra
                M = cv2.moments(part["cnt"])
                if M["m00"] != 0:
                    cX, cY = int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"])
                    cv2.putText(enhanced, f"P:{part['score']}", (cX-25, cY), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                if matches == inv["count"]: break

    # 5. VÉGEREDMÉNY
    avg = round(total_score / 7, 2) if found_count > 0 else 0.0
    status_color = (0, 255, 0) if found_count == 7 and avg > 0.85 else (0, 0, 255)
    
    cv2.putText(enhanced, f"PARALLEL INDEX: {avg}", (30, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, status_color, 2)
    cv2.putText(enhanced, f"FOUND: {found_count}/7", (30, 90), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    return results, enhanced

if __name__ == "__main__":
    cap = cv2.VideoCapture(1)
    cv2.namedWindow("QS Precision Monitor", cv2.WINDOW_NORMAL)
    while True:
        ret, frame = cap.read()
        if not ret: break
        _, output = analyze_frame_no_roi(frame)
        cv2.imshow("QS Precision Monitor", output)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
    cap.release()
    cv2.destroyAllWindows()