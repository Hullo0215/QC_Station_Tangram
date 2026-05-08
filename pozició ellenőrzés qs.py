import cv2
import numpy as np
from collections import deque

# --- KONFIGURÁCIÓ ---
# Az alkatrészek méret-tartományai (a korábbi méréseid alapján)
SHAPE_INVENTORY = [
    {"name": "Ful/Kicsi", "count": 2, "min": 8000, "max": 11000},
    {"name": "kozepes 3 szög", "count": 1, "min": 16600, "max": 18200}, 
    {"name": "fej", "count": 1, "min": 15600, "max": 17200}, 
    {"name": "nagy3szog", "count": 2, "min": 35000, "max": 42000},
    {"name": "nyak/paralelogramma", "count": 1, "min": 19000, "max": 22000}
]

# 10 eseményes memória a mozgóátlag kiszámításához
score_history = deque(maxlen=10)

def get_edges_from_contour(cnt, epsilon_coeff=0.02):
    """
    Felbontja a kontúrt egyenes szakaszokra (élekre).
    Visszaadja az élek kezdő- és végpontját, szögét és felezőpontját.
    """
    epsilon = epsilon_coeff * cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, epsilon, True)
    edges = []
    for i in range(len(approx)):
        p1 = approx[i][0]
        p2 = approx[(i + 1) % len(approx)][0]
        # Szög kiszámítása fokban
        angle = np.degrees(np.arctan2(p2[1] - p1[1], p2[0] - p1[0]))
        # Az él felezőpontja a távolságméréshez
        mid_p = ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)
        edges.append({"p1": p1, "p2": p2, "angle": angle, "mid": mid_p})
    return edges

def get_parallel_score(angle1, angle2):
    """
    Kiszámolja a párhuzamosságot 0.0 és 1.0 között.
    A 90 fokos modulo segít a merőleges/párhuzamos élek kezelésében.
    """
    diff = abs(angle1 - angle2) % 90
    if diff > 45:
        diff = abs(90 - diff)
    
    # 12 fokos eltérés felett a pontszám 0.0
    score = 1.0 - (diff / 12.0)
    return max(0.0, round(score, 2))

def enhance_image(image):
    """Szoftveres kontraszt és telítettség fokozás a stabilabb maszkoláshoz."""
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
    
    # Rózsaszín maszk az alapkeret (cica) felismeréséhez
    mask = cv2.inRange(hsv, np.array([145, 40, 85]), np.array([171, 250, 255]))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8))
    
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if not contours or hierarchy is None: 
        return 0.0, work_frame

    # 2. ALAP KERET (CICA) MEGHATÁROZÁSA
    base_idx = -1
    max_a = 0
    for i, c in enumerate(contours):
        area = cv2.contourArea(c)
        if area > max_a:
            max_a = area
            base_idx = i
    
    if base_idx == -1: return 0.0, work_frame
    
    # Az alap éleinek listázása
    base_edges = get_edges_from_contour(contours[base_idx], 0.015)
    cv2.drawContours(work_frame, [contours[base_idx]], -1, (255, 0, 0), 2)

    # 3. ALKATRÉSZEK VIZSGÁLATA (ÉL-PÁR ALAPÚ LOGIKA)
    detected_parts = []
    for i, cnt in enumerate(contours):
        # Csak a bázis kontúron belüli gyerek-kontúrokat nézzük
        if hierarchy[0][i][3] == base_idx:
            area = cv2.contourArea(cnt)
            if area > 1000:
                part_edges = get_edges_from_contour(cnt, 0.02)
                
                # Megkeressük azt az él-párt (alkatrész széle vs. keret széle),
                # ami a legközelebb van egymáshoz fizikailag.
                best_pair_score = 0
                min_dist = float('inf')
                
                for pe in part_edges:
                    for be in base_edges:
                        # Távolságmérés az élek középpontjai között
                        dist = np.sqrt((pe["mid"][0] - be["mid"][0])**2 + (pe["mid"][1] - be["mid"][1])**2)
                        if dist < min_dist:
                            min_dist = dist
                            best_pair_score = get_parallel_score(pe["angle"], be["angle"])
                
                M = cv2.moments(cnt)
                cX, cY = (int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"])) if M["m00"] != 0 else (0,0)
                detected_parts.append({"cnt": cnt, "area": area, "score": best_pair_score, "used": False, "center": (cX, cY)})

    # 4. KIÉRTÉKELÉS ÉS PONTOZÁS
    current_frame_total = 0
    found_count = 0
    for inv in SHAPE_INVENTORY:
        m = 0
        for p in detected_parts:
            if not p["used"] and inv["min"] <= p["area"] <= inv["max"]:
                p["used"] = True
                m += 1
                current_frame_total += p["score"]
                found_count += 1
                
                # Vizualizáció: S mint Score (párhuzamosság)
                # Színátmenet: Zöld (jó) -> Piros (rossz)
                color = (0, int(255 * p["score"]), int(255 * (1 - p["score"])))
                cv2.drawContours(work_frame, [p["cnt"]], -1, color, 2)
                cv2.putText(work_frame, f"S:{p['score']}", (p["center"][0]-20, p["center"][1]), 0, 0.5, (255, 255, 255), 1)
                if m == inv["count"]: break

    # 5. MOZGÓÁTLAG SZÁMÍTÁSA (10 esemény)
    frame_index = round(current_frame_total / 7, 2) if found_count > 0 else 0.0
    
    # Csak akkor mentjük a memóriába, ha van értékelhető adat a képen
    if found_count > 0:
        score_history.append(frame_index)
    
    moving_avg = round(sum(score_history) / len(score_history), 2) if score_history else 0.0

    # 6. ADATOK KIÍRATÁSA
    # Sárga színnel a stabilizált mozgóátlag
    cv2.putText(work_frame, f"PILLANATNYI: {frame_index}", (30, 50), 0, 0.7, (255, 255, 255), 2)
    cv2.putText(work_frame, f"MOZGO ATLAG (10): {moving_avg}", (30, 90), 0, 1.0, (0, 255, 255), 2)
    cv2.putText(work_frame, f"TALALAT: {found_count}/7", (30, 130), 0, 0.7, (200, 200, 200), 1)
    
    return moving_avg, work_frame

if __name__ == "__main__":
    # Videó forrás megnyitása (0 vagy 1 a kamera azonosítója)
    cap = cv2.VideoCapture(0)
    cv2.namedWindow("QS Precision Monitor - Moving Average", cv2.WINDOW_NORMAL)
    
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        _, output_img = analyze_frame_no_roi(frame)
        cv2.imshow
