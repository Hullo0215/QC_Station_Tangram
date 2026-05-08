import cv2
import numpy as np
import time

# --- KONFIGURÁCIÓ ---
# Az inventory-t a legutóbbi sikeres méréseid alapján lőttem be
SHAPE_INVENTORY = [
    {"name": "Ful/Kicsi", "count": 2, "min": 8000, "max": 11000},
    {"name": "kozepes 3 szög", "count": 1, "min": 16600, "max": 18200}, 
    {"name": "fej", "count": 1, "min": 15600, "max": 17200}, 
    {"name": "nagy3szog", "count": 2, "min": 35000, "max": 42000},
    {"name": "nyak/paralelogramma", "count": 1, "min": 19000, "max": 22000}
]

def get_stable_angle(cnt):
    """Visszaadja a kontúr fő irányát fokban, korrigálva az OpenCV ugrásait."""
    rect = cv2.minAreaRect(cnt)
    (w, h) = rect[1]
    angle = rect[2]
    if w < h:
        angle += 90
    return angle

def get_parallel_score(base_edge_angle, part_angle):
    """Kiszámolja a párhuzamosságot (1.0 = tökéletes, 0.0 = 15+ fok eltérés)."""
    diff = abs(base_edge_angle - part_angle) % 90
    if diff > 45:
        diff = abs(90 - diff)
    
    # 15 fokos skála a perspektivikus torzítás miatt
    score = 1.0 - (diff / 15.0)
    return max(0.0, round(score, 2))

def enhance_image(image):
    """Képminőség javítása a maszkoláshoz (Saturation és Value emelés)."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    s = cv2.add(s, 20) 
    v = cv2.add(v, 40) 
    return cv2.cvtColor(cv2.merge((h, s, v)), cv2.COLOR_HSV2BGR)

def analyze_frame_no_roi(frame):
    # 1. ELŐFELDOLGOZÁS
    enhanced = enhance_image(frame)
    work_frame = enhanced.copy()
    hsv = cv2.cvtColor(work_frame, cv2.COLOR_BGR2HSV)
    
    # Rózsaszín maszk a cica alaphoz
    mask = cv2.inRange(hsv, np.array([145, 40, 85]), np.array([171, 250, 255]))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8))
    
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if not contours or hierarchy is None:
        return 0.0, work_frame

    # 2. BÁZIS (KÉK KONTÚR) ÉS ÉLEI
    base_idx = -1
    max_a = 0
    for i, c in enumerate(contours):
        area = cv2.contourArea(c)
        if area > max_a:
            max_a = area
            base_idx = i
    
    if base_idx == -1: return 0.0, work_frame
    
    # Keret egyszerűsítése szakaszokra a helyi méréshez
    epsilon = 0.015 * cv2.arcLength(contours[base_idx], True)
    approx_base = cv2.approxPolyDP(contours[base_idx], epsilon, True)
    cv2.drawContours(work_frame, [contours[base_idx]], -1, (255, 0, 0), 2)

    base_edges = []
    for i in range(len(approx_base)):
        p1, p2 = approx_base[i][0], approx_base[(i + 1) % len(approx_base)][0]
        angle = np.degrees(np.arctan2(p2[1] - p1[1], p2[0] - p1[0]))
        base_edges.append({"p1": p1, "p2": p2, "angle": angle})

    # 3. ALKATRÉSZEK DETEKTÁLÁSA
    detected_parts = []
    for i, cnt in enumerate(contours):
        if hierarchy[0][i][3] == base_idx:
            area = cv2.contourArea(cnt)
            if area > 1000:
                M = cv2.moments(cnt)
                if M["m00"] == 0: continue
                cX, cY = int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"])
                
                # Legközelebbi alap-él megkeresése a darab középpontjához képest
                min_dist = float('inf')
                closest_edge_angle = 0
                for edge in base_edges:
                    mid_p = ((edge["p1"][0] + edge["p2"][0])//2, (edge["p1"][1] + edge["p2"][1])//2)
                    dist = np.sqrt((cX - mid_p[0])**2 + (cY - mid_p[1])**2)
                    if dist < min_dist:
                        min_dist = dist
                        closest_edge_angle = edge["angle"]
                
                part_angle = get_stable_angle(cnt)
                score = get_parallel_score(closest_edge_angle, part_angle)
                detected_parts.append({"cnt": cnt, "area": area, "score": score, "used": False, "center": (cX, cY)})

    # 4. INVENTORY CHECK ÉS PONTOZÁS
    total_score = 0
    found_count = 0
    for inv in SHAPE_INVENTORY:
        matches = 0
        for p in detected_parts:
            if not p["used"] and inv["min"] <= p["area"] <= inv["max"]:
                p["used"] = True
                matches += 1
                total_score += p["score"]
                found_count += 1
                
                # Színátmenet: Zöld (Jó) -> Piros (Rossz)
                color = (0, int(255 * p["score"]), int(255 * (1 - p["score"])))
                cv2.drawContours(work_frame, [p["cnt"]], -1, color, 2)
                cv2.putText(work_frame, f"S:{p['score']}", (p["center"][0]-25, p["center"][1]), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                if matches == inv["count"]: break

    # 5. ÖSSZESÍTETT EREDMÉNY KIJELZÉSE
    avg_index = round(total_score / 7, 2) if found_count > 0 else 0.0
    status_color = (0, 255, 0) if found_count == 7 and avg_index > 0.8 else (0, 0, 255)
    
    cv2.putText(work_frame, f"OSSZESITETT INDEX: {avg_index}", (30, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, status_color, 2)
    cv2.putText(work_frame, f"TALALAT: {found_count}/7", (30, 90), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    return avg_index, work_frame

if __name__ == "__main__":
    cap = cv2.VideoCapture(0) # Ha nem működik, próbáld 1-gyel
    cv2.namedWindow("QS Precision Monitor", cv2.WINDOW_NORMAL)
    while True:
        ret, frame = cap.read()
        if not ret: break
        idx, img = analyze_frame_no_roi(frame)
        cv2.imshow("QS Precision Monitor", img)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
    cap.release()
    cv2.destroyAllWindows()
