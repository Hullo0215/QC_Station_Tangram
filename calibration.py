import cv2
import numpy as np

def calibrate_v4_anti_glare():
    # 0 = alap kamera, 1 = külső[cite: 1, 2]
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret: break

        # 1. UPSCALING (A jobb felbontásért)
        frame = cv2.resize(frame, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

        # 2. ENYHE LÁGYÍTÁS (Ez "elkeni" a zavaró csillogást)[cite: 3]
        frame_blur = cv2.GaussianBlur(frame, (5, 5), 0)

        # 3. HSV SZÍNSZŰRÉS (A csillogás fehér, a cica rózsaszín - a szín segít!)[cite: 1, 2]
        hsv = cv2.cvtColor(frame_blur, cv2.COLOR_BGR2HSV)
        
        # Rózsaszín tartomány (Lehet, hogy állítani kell a laborban!)
        # Ha túl érzékeny, a 40-es értéket (Saturation) emeld 70-re.
        lower_pink = np.array([140, 40, 40]) 
        upper_pink = np.array([180, 255, 255])
        mask = cv2.inRange(hsv, lower_pink, upper_pink)

        # 4. ZAJSZŰRÉS (A maradék becsillanás-pöttyök eltüntetése)[cite: 3]
        kernel = np.ones((5,5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        # 5. KONTÚROK ÉS HIERARCHIA[cite: 1, 3]
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        if contours and hierarchy is not None:
            # Alap (Cica) keresése[cite: 3]
            cica_base_idx = -1
            max_area = 0
            for i, cnt in enumerate(contours):
                area = cv2.contourArea(cnt)
                if area > max_area:
                    max_area = area
                    cica_base_idx = i

            if cica_base_idx != -1:
                # Alap kirajzolása kékkel[cite: 3]
                cv2.drawContours(frame, [contours[cica_base_idx]], -1, (255, 0, 0), 2)
                
                for i, cnt in enumerate(contours):
                    # Hierarchia: Csak ami a cicán belül van[cite: 1, 3]
                    if hierarchy[0][i][3] == cica_base_idx:
                        area = cv2.contourArea(cnt)
                        
                        # LEHETŐSÉG: Nagyon kicsi határ (50), hogy biztosan lásd a számokat![cite: 3]
                        if area > 50:
                            cv2.drawContours(frame, [cnt], -1, (0, 255, 0), 2)
                            
                            M = cv2.moments(cnt)
                            if M["m00"] != 0:
                                cX = int(M["m10"] / M["m00"])
                                cY = int(M["m01"] / M["m00"])
                                # Terület kiírása[cite: 3]
                                cv2.putText(frame, str(int(area)), (cX, cY), 
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Két ablak: az egyik a kész kép, a másik a "maszk" (amit a gép lát fekete-fehérben)
        cv2.imshow("Meres", frame)
        cv2.imshow("Gep ezt latja (Maszk)", mask)

        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    calibrate_v4_anti_glare()