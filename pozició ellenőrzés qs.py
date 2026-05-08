import cv2
import numpy as np
from collections import deque

# --- KONFIGURÁCIÓ ---
SHAPE_INVENTORY = [
    {"name": "Ful/Kicsi", "count": 2, "min": 8000, "max": 11000},
    {"name": "kozepes 3 szog", "count": 1, "min": 16600, "max": 18200}, 
    {"name": "fej", "count": 1, "min": 15600, "max": 17200}, 
    {"name": "nagy3szog", "count": 2, "min": 35000, "max": 42000},
    {"name": "nyak/paralelogramma", "count": 1, "min": 19000, "max": 22000}
]

score_history = deque(maxlen=10)

def get_edges_from_contour(cnt, epsilon_coeff=0.02):
    """Szakaszokra bontja a kontúrt."""
    epsilon = epsilon_coeff * cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, epsilon, True)
    edges = []
    if len(approx) < 2: return edges
    for i in range(len(approx)):
        p1, p2 = approx[i][0], approx[(i + 1) % len(approx)][0]
        angle = np.degrees(np.arctan2(p2[1] - p1[1], p2[0] - p1[0]))
        mid_p = ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)
        edges.append({"angle": angle, "mid": mid_p})
    return edges

def get_parallel_score(angle1, angle2):
    """Párhuzamosság számítása."""
    diff = abs(angle1 - angle2) % 90
    if diff > 45: diff = abs(90 - diff)
    score = 1.0 - (diff / 12.0)
    return max(0.0, round(score, 2))

def enhance_image(image):
    """Képjavítás a maszkoláshoz."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    s = cv2.add(s, 20)
    v = cv2.add(v, 40)
    return cv2.cvtColor(cv2.merge((h, s, v)), cv2.COLOR_HSV2BGR)

def analyze_frame(frame):
    if frame is None: return 0.0, None
    
    enhanced = enhance_image(frame)
    work_frame = enhanced.copy()
    hsv = cv2.cvtColor(work_frame, cv2.COLOR_BGR2HSV)
    
    # Maszk a rózsaszín alaphoz (módosítható ha túl szigorú)
    mask = cv2.inRange(hsv, np.array([140, 40, 80]), np.array([175, 255, 255]))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7,7), np.uint8))
    
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if not contours or hierarchy is None:
        return 0.0, work_frame

    # Alap megkeresése
    base_idx = -1
    max_a = 0
    for i, c in enumerate(contours):
        area = cv2.contourArea(c)
        if area > max_a:
            max_a, base_idx = area, i
    
    if base_idx == -1: return 0.0, work_frame
    
    base_edges = get_edges_from_contour(contours[base_idx], 0.015)
    cv2.drawContours(work_frame, [contours[base_idx]], -1, (255, 0, 0), 2)

    detected_parts = []
    for i, cnt in enumerate(contours):
        if hierarchy[0][i][3] == base_idx:
            area = cv2.contourArea(cnt)
            if area > 1000:
                p_edges = get_edges_from_contour(cnt, 0.02)
                best_s, min_d = 0, float('inf')
                
                # Legközelebbi él-pár keresése
                for pe in p_edges:
                    for be in base_edges:
                        dist = np.sqrt((pe["mid"][0]-be["mid"][0])**2 + (pe["mid"][1]-be["mid"][1])**2)
                        if dist < min_d:
                            min_d, best_s = dist, get_parallel_score(pe["angle"], be["angle"])
                
                M = cv2.moments(cnt)
                if M["m00"] != 0:
                    cX, cY = int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"])
                    detected_parts.append({"cnt": cnt, "area": area, "score": best_s, "used": False, "center": (cX, cY)})

    # Pontozás és inventory
    current_total, found = 0, 0
    for inv in SHAPE_INVENTORY:
        matches = 0
        for p in detected_parts:
            if not p["used"] and inv["min"] <= p["area"] <= inv["max"]:
                p["used"] = True
                matches += 1
                current_total += p["score"]
                found += 1
                color = (0, int(255*p["score"]), int(255*(1-p["score"])))
                cv2.drawContours(work_frame, [p["cnt"]], -1, color, 2)
                cv2.putText(work_frame, f"S:{p['score']}", (p["center"][0]-15, p["center"][1]), 0, 0.5, (255,255,255), 1)
                if matches == inv["count"]: break

    f_idx = round(current_total / 7, 2) if found > 0 else 0.0
    if found > 0: score_history.append(f_idx)
    m_avg = round(sum(score_history)/len(score_history), 2) if score_history else 0.0

    cv2.putText(work_frame, f"AVG: {m_avg}", (30, 50), 0, 1.0, (0, 255, 255), 2)
    cv2.putText(work_frame, f"FOUND: {found}/7", (30, 90), 0, 0.7, (255, 255, 255), 1)
    
    return m_avg, work_frame

# --- KAMERA INDÍTÁSA HIBAKEZELÉSSEL ---
def start_camera():
    # Megpróbáljuk több indexen (0, 1, 2) hátha máshol van a kamera
    for index in [0, 1, 2]:
        print(f"Kamera probalkozas: Index {index}...")
        cap = cv2.VideoCapture(index)
        if cap.isOpened():
            print(f"Siker! Kamera megtalalva az indexen: {index}")
            return cap
    return None

if __name__ == "__main__":
    cap = start_camera()
    
    if cap is None:
        print("HIBA: Nem sikerult kameraszenzort nyitni. Ellenorizd a csatlakozast!")
    else:
        cv2.namedWindow("QS Precision Monitor")
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Hiba a kepkocka olvasasakor.")
                break
            
            _, output = analyze_frame(frame)
            if output is not None:
                cv2.imshow("QS Precision Monitor", output)
            
            if cv2.waitKey(300) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
