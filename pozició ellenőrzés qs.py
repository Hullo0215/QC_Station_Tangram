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

def get_longest_edge_angle(cnt):
    """Kiszámolja a kontúr leghosszabb szakaszának szögét."""
    # Kontúr közelítése poligonnal (egyszerűsítés)
    epsilon = 0.02 * cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, epsilon, True)
    
    max_len = 0
    best_angle = 0
    
    # Élek vizsgálata
    for i in range(len(approx)):
        p1 = approx[i][0]
        p2 = approx[(i + 1) % len(approx)][0]
        
        # Hossz számítás
        length = np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
        
        if length > max_len:
            max_len = length
            # Szög kiszámítása radiánból fokba
            best_angle = np.degrees(np.arctan2(p2[1] - p1[1], p2[0] - p1[0]))
            
    return best_angle

def get_parallel_score(base_angle, part_angle):
    """0.0 - 1.0 közötti pontszám a szögeltérés alapján."""
    # A modulo 90 segít, hogy a párhuzamos/merőleges élek ne okozzanak hibát
    diff = abs(base_angle - part_angle) % 90
    if diff > 45:
        diff = abs(90 - diff)
    
    # 12 fokos hiba felett 0 pont
    score = 1.0 - (diff / 12.0)
    return max(0.0, round(score, 2))

def enhance_image(image):
    """Képjavítás a maszkolás előtt."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    s = cv2.add(s, 20) 
    v = cv2.add(v, 40) 
    return cv2.cvtColor(cv2.merge((h, s, v)), cv2.COLOR_HSV2BGR)

def analyze_frame_no_roi(frame):
    enhanced = enhance_image(frame)
    work_frame = enhanced.copy()
    hsv = cv2.cvtColor(work_frame, cv2.COLOR_BGR2HSV)
    
    # Maszk a rózsaszín alaphoz
    mask = cv2.inRange(hsv, np.array([145, 40, 85]), np.array([171, 250, 255]))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8))
    
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if not contours or hierarchy is None: return 0.0, work_frame

    # 1. Alap (Cica forma) megkeresése
    base_idx = -1
    max_area = 0
    for i, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        if area > max_area:
            max_area = area
            base_idx = i
    
    if base_idx == -1: return 0.0, work_frame

    # Az alap éleinek kinyerése
    epsilon_base = 0.015 * cv2.arcLength(contours[base_idx], True)
    approx_base = cv2.approxPolyDP(contours[base_idx], epsilon_base, True)
    cv2.drawContours(work_frame, [contours[base_idx]], -1, (255, 0, 0), 2)

    base_edges = []
    for i in range(len(approx_base)):
        p1, p2 = approx_base[i][0], approx_base[(i + 1) % len(approx_base)][0]
        angle = np.degrees(np.arctan2(p2[1] - p1[1], p2[0] - p1[0]))
        base_edges.append({"p1": p1, "p2": p2, "angle": angle})

    # 2. Alkatrészek mérése
    detected_parts = []
    for i, cnt in enumerate(contours):
        if hierarchy[0][i][3] == base_idx: # Csak ami az alapon belül van
            area = cv2.contourArea(cnt)
            if area > 1000:
                M = cv2.moments(cnt)
                if M["m00"] == 0: continue
                cX, cY = int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"])
                
                # A darab leghosszabb éle
                part_angle = get_longest_edge_angle(cnt)
                
                # Hozzá legközelebbi alap-fal szöge
                min_dist = float('inf')
                closest_base_angle = 0
                for edge in base_edges:
                    mid_p = ((edge["p1"][0] + edge["p2"][0])//2, (edge["p1"][1] + edge["p2"][1])//2)
                    dist = np.sqrt((cX - mid_p[0])**2 + (cY - mid_p[1])**2)
                    if dist < min_dist:
                        min_dist = dist
                        closest_base_angle = edge["angle"]
                
                score = get_parallel_score(closest_base_angle, part_angle)
                detected_parts.append({"cnt": cnt, "area": area, "score": score, "used": False, "center": (cX, cY)})

    # 3. Összesítés
    total_score = 0
    found_count = 0
    for inv in SHAPE_INVENTORY:
        m = 0
        for p in detected_parts:
            if not p["used"] and inv["min"] <= p["area"] <= inv["max"]:
                p["used"] = True
                m += 1
                total_score += p["score"]
                found_count += 1
                
                # Vizualizáció: Színátmenet a pontszám alapján
                color = (0, int(255 * p["score"]), int(255 * (1 - p["score"])))
                cv2.drawContours(work_frame, [p["cnt"]], -1, color, 2)
                cv2.putText(work_frame, f"S:{p['score']}", (p["center"][0]-25, p["center"][1]), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                if m == inv["count"]: break

    avg_idx = round(total_score / 7, 2) if found_count > 0 else 0.0
    cv2.putText(work_frame, f"OSSZESITETT INDEX: {avg_idx}", (30, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    
    return avg_idx, work_frame

if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret: break
        _, img = analyze_frame_no_roi(frame)
        cv2.imshow("Longest Edge Score", img)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
    cap.release()
    cv2.destroyAllWindows()
