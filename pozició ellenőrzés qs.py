import cv2
import numpy as np
import time

# --- KONFIGURÁCIÓ ---
SHAPE_INVENTORY = [
    {"name": "Ful/Kicsi", "count": 2, "min": 8000, "max": 11000},
    {"name": "kozepes 3 szög", "count": 1, "min": 16600, "max": 18200}, 
    {"name": "fej", "count": 1, "min": 15600, "max": 17200}, 
    {"name": "nagy3szog", "count": 2, "min": 35000, "max": 40000},
    {"name": "nyak/paralelogramma", "count": 1, "min": 19000, "max": 22000}
]

def get_stable_angle(cnt):
    """Visszaadja a kontúr fő irányát, korrigálva az OpenCV szögugrásait."""
    rect = cv2.minAreaRect(cnt)
    (x, y), (w, h), angle = rect
    if w < h:
        angle = angle + 90
    return angle

def get_parallel_score(edge_angle, part_angle):
    """Kiszámolja a párhuzamosságot a legközelebbi élhez képest (0.0 - 1.0)."""
    diff = abs(edge_angle - part_angle) % 90
    if diff > 45:
        diff = abs(90 - diff)
    
    # 12 fokos eltérésnél már 0.0 a pontszám
    score = 1.0 - (diff / 12.0)
    return max(0.0, round(score, 2))

def enhance_image(image):
    """Szoftveres kontraszt és telítettség fokozás."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    s = cv2.add(s, 20)
    v = cv2.add(v, 40)
    return cv2.cvtColor(cv2.merge((h, s, v)), cv2.COLOR_HSV2BGR)

def analyze_frame_no_roi(frame):
    # 1. Előfeldolgozás
    enhanced = enhance_image(frame)
    work_frame = enhanced.copy()
    hsv = cv2.cvtColor(work_frame, cv2.COLOR_BGR2HSV)
    
    mask = cv2.inRange(hsv, np.array([145, 40, 85]), np.array([171, 250, 255]))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8))
    
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if not contours or hierarchy is None:
        return 0, work_frame

    # 2. Alap (Rózsaszín keret) megkeresése és éleinek kinyerése
    base_idx = -1
    max_a = 0
    for i, c in enumerate(contours):
        area = cv2.contourArea(c)
        if area > max_a:
            max_a = area
            base_idx = i
    
    if base_idx == -1: return 0, work_frame
    
    # Alap körvonalának egyszerűsítése szakaszokra
    epsilon = 0.01 * cv2.arcLength(contours[base_idx], True)
    approx_base = cv2.approxPolyDP(contours[base_idx], epsilon, True)
    cv2.drawContours(work_frame, [contours[base_idx]], -1, (255, 0, 0), 2)

    base_edges = []
    for i in range(len(approx_base)):
        p1 = approx_base[i][0]
        p2 = approx_base[(i + 1) % len(approx_base)][0]
        edge_angle = np.degrees(np.arctan2(p2[1] - p1[1], p2[0] - p1[0]))
        base_edges.append((p1, p2, edge_angle))

    # 3. Alkatrészek vizsgálata a legközelebbi él alapján
    detected_parts = []
    for i, cnt in enumerate(contours):
        if hierarchy[0][i][3] == base_idx:
            area = cv2.contourArea(cnt)
            if area > 1000:
                M = cv2.moments(cnt)
                if M["m00"] == 0: continue
                cX, cY = int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"])
                
                # Legközelebbi alap-él megkeresése
                min_dist = float('inf')
                closest_angle = 0
                for p1, p2, e_angle in base_edges:
                    # Távolság a pont és a szakasz között
                    dist = abs(cv2.pointPolygonTest(np.array([p1, p2]), (cX, cY), True))
                    if dist < min_dist:
                        min_dist = dist
                        closest_angle = e_angle
                
                part_angle = get_stable_angle(cnt)
                score = get_parallel_score(closest_angle, part_angle)
                detected_parts.append({"cnt": cnt, "area": area, "score": score, "used": False, "center": (cX, cY)})

    # 4. Azonosítás és pontozás
    total_score = 0
    found_count = 0
    temp_inventory = [dict(item) for item in SHAPE_INVENTORY]
    
    for inv in temp_inventory:
        matches = 0
        for p in detected_parts:
            if not p["used"] and inv["min"] <= p["area"] <= inv["max"]:
                p["used"] = True
                matches += 1
                total_score += p["score"]
                found_count += 1
                
                # Szín és felirat
                color = (0, int(255 * p["score"]), int(255 * (1 - p["score"])))
                cv2.drawContours(work_frame, [p["cnt"]], -1, color, 2)
                cv2.putText(work_frame, f"S:{p['score']}", (p["center"][0]-20, p["center"][1]), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                if matches == inv["count"]: break

    # 5. Összesített Index
    final_index = round(total_score / 7, 2) if found_count > 0 else 0.0
    status_color = (0, 255, 0) if found_count == 7 and final_index > 0.85 else (0, 0, 255)
    
    cv2.putText(work_frame, f"OSSZESITETT INDEX: {final_index}", (30, 45), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, status_color, 2)
    cv2.putText(work_frame, f"TALALAT: {found_count}/7", (30, 85), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    return final_index, work_frame

if __name__ == "__main__":
    cap = cv2.VideoCapture(0) # Módosítsd 1-re, ha külső kamerát használspe
    cv2.namedWindow("QS Precision Monitor", cv2.WINDOW_NORMAL)
    
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        idx, output = analyze_frame_no_roi(frame)
        cv2.imshow("QS Precision Monitor", output)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()
