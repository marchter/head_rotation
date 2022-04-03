import customtkinter as customtkinter
import cv2
import mediapipe as mp
import numpy as np
from PIL import Image, ImageTk
import tkinter as tk

# TODO: mit https://python-gtk-3-tutorial.readthedocs.io/en/latest/introduction.html#simple-example probieren
# wrapped in a function for n arbitrary
from future.moves.tkinter import ttk


def trim_to_n(number, n):
    negative = False
    if number < 0:
        negative = True
        number = number * -1
    limit = 10 ** (n + 1)
    while number >= limit:
        number = number / 10
    if negative:
        return number * -1
    return number


mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.1, min_tracking_confidence=0.1)
cap = cv2.VideoCapture(0)
max_y_l = 0
max_y_r = 0


def get_frames():
    global max_y_l
    global max_y_r
    success, image = cap.read()

    # Flip the image horizontally for a later selfie-view display
    # Also convert the color space from BGR to RGB
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)

    # To improve performance
    image.flags.writeable = False

    # Get the result
    results = face_mesh.process(image)

    # To improve performance
    image.flags.writeable = True

    # Convert the color space from RGB to BGR
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    img_h, img_w, img_c = image.shape
    face_3d = []
    face_2d = []

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            for idx, lm in enumerate(face_landmarks.landmark):
                if idx == 33 or idx == 263 or idx == 1 or idx == 61 or idx == 291 or idx == 199:
                    if idx == 1:
                        nose_2d = (lm.x * img_w, lm.y * img_h)
                        nose_3d = (lm.x * img_w, lm.y * img_h, lm.z * 8000)

                    x, y = int(lm.x * img_w), int(lm.y * img_h)

                    # Get the 2D Coordinates
                    face_2d.append([x, y])

                    # Get the 3D Coordinates
                    face_3d.append([x, y, lm.z])

                    # Convert it to the NumPy array
            face_2d = np.array(face_2d, dtype=np.float64)

            # Convert it to the NumPy array
            face_3d = np.array(face_3d, dtype=np.float64)

            # The camera matrix
            focal_length = 1 * img_w

            cam_matrix = np.array([[focal_length, 0, img_h / 2],
                                   [0, focal_length, img_w / 2],
                                   [0, 0, 1]])

            # The Distance Matrix
            dist_matrix = np.zeros((4, 1), dtype=np.float64)

            # Solve PnP
            success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)

            # Get rotational matrix
            rmat, jac = cv2.Rodrigues(rot_vec)

            # Get angles
            angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

            # Get the y rotation degree
            x = angles[0] * 360
            y = angles[1] * 360

            # print(y)

            # See where the user's head tilting
            text = str(int(y))
            if int(y) < max_y_l:
                max_y_l = int(y)
            elif int(y) > max_y_r:
                max_y_r = int(y)

            #            if y < -10:
            #                text = "Looking Left"
            #            elif y > 10:
            #                text = "Looking Right"
            #            elif x < -10:
            #                text = "Looking Down"
            #            else:
            #                text = "Forward"

            # Display the nose direction
            nose_3d_projection, jacobian = cv2.projectPoints(nose_3d, rot_vec, trans_vec, cam_matrix, dist_matrix)

            p1 = (int(nose_2d[0]), int(nose_2d[1]))
            p2 = (int(nose_3d_projection[0][0][0]), int(nose_3d_projection[0][0][1]))

            cv2.line(image, p1, p2, (255, 0, 0), 2)

            # Add the text on the image
            cv2.putText(image, text, (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (159, 6, 16), 2)

            # cv2.putText(image, "max. linksdrehung: " + str(abs(max_y_l)), (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1,
            #            (159, 6, 16), 2)
            # cv2.putText(image, "max. rechtsdrehung: " + str(max_y_r), (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 1,
            #            (159, 6, 16), 2)

    # cv2.imshow('Head Pose Estimation', image)

    return image



win = customtkinter.CTk()

win.geometry("1900x900")

label = tk.Label(win)
label.grid(row=0, column=0)


def show_frames():
    cv2image = cv2.cvtColor(get_frames(), cv2.COLOR_BGR2RGB)
    img = Image.fromarray(cv2image)
    imgtk = ImageTk.PhotoImage(image=img)
    label.imgtk = imgtk
    label.configure(image=imgtk)
    label.after(20, show_frames)
    label.grid(row=0, column=0)

    l = tk.Label(text="Deine Maximale Kopfdrehung nach links:  " + str(abs(max_y_l)) + "\n Deine Maximale Kopfdrehung nach rechts: " + str(abs(max_y_r)),font=("Arial", 20), background="gray", borderwidth=1, relief="ridge")
    l.grid(row=0, column=1, padx = 10)



def reset_max():
    global max_y_l
    global max_y_r
    max_y_l = max_y_r = 0


def layout():
    button = customtkinter.CTkButton(win, text='Maximalwerte zurücksetzen', command=reset_max, border_width=1, border_color="white")
    button.grid(row=2, column=0)

    image = Image.open("logo.png")
    image = ImageTk.PhotoImage(image)
    imageLable = customtkinter.CTkButton(win, image=image, text="", width=400, height=60, border_width=1, border_color="white")
    imageLable.grid(row=3, column=0, pady=10)

    show_frames()



layout()
win.mainloop()
