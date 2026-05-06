import cv2
import numpy as np
import time

# --- KONFIGURÁCIÓ ---
# Itt kell majd megadnod a "jó" termék alapján a koordinátákat (Relatív a cica közepéhez)
# Példa adatok (X eltolás, Y eltolás, Név)
EXPECTED_PARTS = [
    {"name": "Bal fül", "pos": (-40, -80), "min_area": 500},
    {"name": "Jobb fül", "pos": (40, -80), "min_area": 500},
    {"name": "Arc", "pos": (0, -40), "min_area": 1000},
    {"name": "nyak", "pos": (0,0 ), "min_area": 800}
    {"name": "Test-felső", "pos": (50, 20), "min_area": 800},
    {"name": "test-alsó", "pos": (50, 80), "min_area": 800}
    {"name": "talp", "pos":(1,1), "min_area":500}
    # ... ide írd be mind a 7-et
]

def analyze_frame(frame):
    results = {
        "timestamp": time.time(),
        "status": "FAIL",
        "found_count": 0,
        "missing_items": [],
        "image_processed": None
    }

    # 1. ELŐFELDOLGOZÁS ÉS PALETTA (ROI) KERESÉSE
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_blue = np.array([90, 50, 50])   # Kék tartomány alja
    upper_blue = np.array([130, 255, 255]) # Kék tartomány teteje
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return results, frame

    # Legnagyobb kék terület = Paletta
    palette_cnt = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(palette_cnt)
    roi = frame[y:y+h, x:x+w]
    
    # 2. CICA ALAP KERESÉSE A PALETTÁN BELÜL
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray_roi, 150, 255, cv2.THRESH_BINARY)
    cica_contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if not cica_contours:
        return results, roi

    # A legnagyobb világos kontúr a cica rózsaszín alapja
    cica_base_cnt = max(cica_contours, key=cv2.contourArea)
    M_base = cv2.moments(cica_base_cnt)
    if M_base["m00"] == 0: return results, roi
    
    base_center_x = int(M_base["m10"] / M_base["m00"])
    base_center_y = int(M_base["m01"] / M_base["m00"])

    # 3. ALKATRÉSZEK ELLENŐRZÉSE
    detected_parts_info = []
    for cnt in cica_contours:
        area = cv2.contourArea(cnt)
        # Csak a belső darabokat nézzük (kisebbek mint az alap, de nem zaj)
        if 300 < area < cv2.contourArea(cica_base_cnt):
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"]) - base_center_x
                cY = int(M["m01"] / M["m00"]) - base_center_y
                detected_parts_info.append({"rel_pos": (cX, cY), "area": area, "cnt": cnt})

    # Beazonosítás a MASTER_PARTS lista alapján
    found_names = []
    for expected in EXPECTED_PARTS:
        found_this_part = False
        for detected in detected_parts_info:
            dist = np.sqrt((detected["rel_pos"][0] - expected["pos"][0])**2 + 
                           (detected["rel_pos"][1] - expected["pos"][1])**2)
            
            if dist < 40: # 40 pixel tűréshatár
                found_this_part = True
                found_names.append(expected["name"])
                cv2.drawContours(roi, [detected["cnt"]], -1, (0, 255, 0), 2)
                break
        
        if not found_this_part:
            results["missing_items"].append(expected["name"])

    # 4. EREDMÉNYEK ÖSSZESÍTÉSE
    results["found_count"] = len(found_names)
    if len(results["missing_items"]) == 0 and results["found_count"] >= 7:
        results["status"] = "OK"

    # Vizuális visszajelzés
    color = (0, 255, 0) if results["status"] == "OK" else (0, 0, 255)
    cv2.rectangle(roi, (0,0), (w-1, h-1), color, 3)
    cv2.putText(roi, f"Status: {results['status']} - Missing: {len(results['missing_items'])}", 
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    return results, roi

if __name__ == "__main__":
    cap = cv2.VideoCapture(1) # 1 = másodlagos webkamera
    
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        data, processed_img = analyze_frame(frame)
        cv2.imshow("Elo teszt", processed_img)
        
        if cv2.waitKey(1) & 0xFF == ord('q'): # 'q' gombra kilép
            break
            
    cap.release()
    cv2.destroyAllWindows()