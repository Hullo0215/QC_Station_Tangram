import cv2
import numpy as np
import time

# --- KONFIGURÁCIÓ (2x felskálázott értékekkel) ---
SHAPE_INVENTORY = [
    {"name": "Ful/Kicsi", "count": 2, "min": 6500, "max": 8800},
    {"name": "(Bordo)", "count": 2, "min": 13600, "max": 15200}, 
    {"name": "nagy3szog", "count": 2, "min": 25000, "max": 36000},
    {"name": "nyak/paralelogramma", "count": 1, "min": 15000, "max": 19000}
]

def enhance_image(image):
    """Szoftveres kontraszt és telítettség fokozás a bordó elem kiemeléséhez"""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    # Telítettség (Saturation) és fényerő (Value) emelése
    s = cv2.add(s, 45) # Erősebb színek
    v = cv2.add(v, 25) # Világosabb sötét részek

    final_hsv = cv2.merge((h, s, v))
    return cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)

def analyze_frame_no_roi(frame):
    results = {"status": "FAIL", "found_count": 0, "missing_items": []}

    # 1. ELŐFELDOLGOZÁS ÉS TUNING
    enhanced = enhance_image(frame)
    work_frame = cv2.resize(enhanced, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    hsv = cv2.cvtColor(work_frame, cv2.COLOR_BGR2HSV)
    
    # Maszkolás
    lower_pink = np.array([130, 40, 20]) 
    upper_pink = np.array([180, 255, 255])
    mask = cv2.inRange(hsv, lower_pink, upper_pink)

    # Morfológia (Zárás): befoltozza a bordó elem belső lyukait
    kernel = np.ones((11, 11), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if contours is None or hierarchy is None:
        return results, work_frame

    # 2. CICA ALAP KERESÉSE
    cica_base_idx = -1
    max_area = 0
    for i, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        if area > max_area:
            max_area = area
            cica_base_idx = i

    if cica_base_idx == -1:
        return results, work_frame

    # 3. ALKATRÉSZEK LISTÁZÁSA
    detected_parts = []
    for i, cnt in enumerate(contours):
        if hierarchy[0][i][3] == cica_base_idx: # Csak a cica belsejében lévő kontúrok
            area = cv2.contourArea(cnt)
            if area > 1000:
                detected_parts.append({"cnt": cnt, "area": area, "identified": False})

    # 4. KIÉRTÉKELÉS (Inventory check)
    # Lemásoljuk az inventory-t, hogy követni tudjuk mi fogyott el
    temp_inventory = [dict(item) for item in SHAPE_INVENTORY]
    
    for inv_item in temp_inventory:
        found_for_type = 0
        for part in detected_parts:
            if not part["identified"] and inv_item["min"] <= part["area"] <= inv_item["max"]:
                part["identified"] = True
                found_for_type += 1
                if found_for_type == inv_item["count"]:
                    break
        
        if found_for_type < inv_item["count"]:
            results["missing_items"].append(f"{inv_item['name']} ({inv_item['count'] - found_for_type}db)")

    # 5. MEGJELENÍTÉS (Zöld = Azonosított, Piros = Selejt/Ismeretlen)
    cv2.drawContours(work_frame, [contours[cica_base_idx]], -1, (255, 0, 0), 2) # Alap kékkel
    
    ok_count = 0
    for part in detected_parts:
        color = (0, 255, 0) if part["identified"] else (0, 0, 255)
        if part["identified"]: ok_count += 1
        
        cv2.drawContours(work_frame, [part["cnt"]], -1, color, 2)
        
        # Terület kiírása fehérrel
        M = cv2.moments(part["cnt"])
        if M["m00"] != 0:
            cX, cY = int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"])
            cv2.putText(work_frame, str(int(part["area"])), (cX-35, cY), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # 6. STÁTUSZ KIÍRÁSA A KÉPERNYŐRE
    if not results["missing_items"] and ok_count == 7:
        status_text = "OK - MINDEN HELYES"
        status_color = (0, 255, 0)
    else:
        status_text = "SELEJT - HIBA!"
        status_color = (0, 0, 255)
        # Hiányzó elemek listázása a bal oldalon
        y_pos = 100
        for missing in results["missing_items"]:
            cv2.putText(work_frame, f"HIANY: {missing}", (30, y_pos), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            y_pos += 40

    cv2.putText(work_frame, f"STATUS: {status_text} ({ok_count}/7)", (30, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, status_color, 3)

    return results, work_frame

if __name__ == "__main__":
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("Hiba: Kamera nem talalhato!")
    
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        data, output_img = analyze_frame_no_roi(frame)
        cv2.imshow("QS Station - Javitott Kiertekeles", output_img)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()