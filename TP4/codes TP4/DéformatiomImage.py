import cv2
import numpy as np
"""import common.Draw as dw
import common.ImageProcessing as ip
import cv2"""
import fonctions.TraitementImage as ti

import fonctions.dessin as ds

dessin = False
couleur_final = (0, 255, 0)
couleur_dessin = (0, 0, 255)
largeur, hauteur= 320,480
points = []




def on_mouse(event, x, y, flags, param):
    global points, dessin, img, img_bk, tiroc, warped_image
    if event == cv2.EVENT_LBUTTONDOWN:
        dessin = True
        add_point(points, (x, y))
        if len(points) == 4:
            dess_polygone(img, points, (x, y), True)
            dessin = False
            img_bk = tiroc.copy()
            warped_image = tiroc.perspective_warp(points, largeur, hauteur)
            points.clear()
            tiroc.show("Perspective Warping", image=warped_image)
    elif event == cv2.EVENT_MOUSEMOVE:
        if dessin == True:
            img = img_bk.copy()
            draw_polygon(img, points, (x, y))

def add_point(points, curt_pt):
    print("Adding point #%d with position(%d,%d)"
          % (len(points), curt_pt[0], curt_pt[1]))
    points.append(curt_pt)

def dess_polygone(img, points, curt_pt, is_final=False):
    if (len(points) > 0):
        if is_final == False:
            ds.dess_polylignes(img, np.array([points]), False, couleur_final)
            ds.dess_ligne(img, points[-1], curt_pt, couleur_dessin)
        else:
            ds.dess_polylignes(img, np.array([points]), True, couleur_final)
        for point in points:
            ds.dess_cercle(img, point, 2, final_color, 2)
            ds.dess_text(img, str(point), point, color=couleur_final, font_scale=0.5)

def print_instruction(img):
    txtInstruction = "Left click to specify four points to warp image. ESC to exit, 's' to save"
    ds.dess_text(img,txtInstruction, (10, 20), 0.5, (255, 255, 255))
    print(txtInstruction)


if __name__ == "__main__":
    global img, img_bk, tiroc, warped_image

    title = "Original Image"
    tiroc = ti.TraitementImage(title, 'ressources/book_perspect1.jpg')  
    img = tiroc.image
    print_instruction(img)
    #img_bk = tiroc.copy()
    cv2.setMouseCallback(title, on_mouse)
    tiroc.show()

    while True:
        tiroc.show(image=img)
        ch = cv2.waitKey(10)
        if (ch & 0xFF) == 27:
            break
        elif ch == ord('s'):
            # press 's' key to save image
            filepath = 'ressources/book_sauv.jpg'
            cv2.imwrite(filepath, warped_image)
            print("File saved to " + filepath)
    cv2.destroyAllWindows()



